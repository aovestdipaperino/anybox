//! Run-time library exports: magic division, generic access masks, and capturing a context.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission, and adapted
//! to this crate's handler interface.

use super::value;
use crate::kernel::{Call, CallResult, KResult, Kernel};

/// `RtlExtendedMagicDivide(Dividend, MagicDivisor, ShiftCount)`: division by multiplying with a
/// reciprocal: the high half of the 128-bit product, shifted, with the dividend's sign.
pub fn extended_magic_divide(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let dividend = c.arg64(0).cast_signed();
    let magic = c.arg64(2);
    let shift = c.arg(4) & 0x3F;
    let product = u128::from(dividend.unsigned_abs()) * u128::from(magic);
    let quotient = ((product >> 64) as u64) >> shift;
    let quotient = if dividend < 0 {
        quotient.wrapping_neg()
    } else {
        quotient
    };
    Ok(CallResult::Value64(
        quotient as u32,
        (quotient >> 32) as u32,
    ))
}

/// `RtlMapGenericMask(AccessMask, GenericMapping)`: replaces the generic rights (read, write,
/// execute, all: bits 31 to 28) with the specific rights the mapping gives each.
pub fn map_generic_mask(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (mask_at, mapping) = (c.arg(0), c.arg(1));
    let mut mask = k.r32(mask_at)?;
    for (index, bit) in (0u32..).zip([31u32, 30, 29, 28]) {
        if mask & 1 << bit != 0 {
            mask |= k.r32(mapping + 4 * index)?;
        }
    }
    k.w32(mask_at, mask & 0x0FFF_FFFF)?;
    value(0)
}

/// `CONTEXT` offsets: the flags, then the general registers after the 0x204-byte FPU area.
const CONTEXT_FLAGS: u32 = 0x00;
const CONTEXT_EBP: u32 = 0x220;
const CONTEXT_EIP: u32 = 0x224;
const CONTEXT_SEG_CS: u32 = 0x228;
const CONTEXT_EFLAGS: u32 = 0x22C;
const CONTEXT_ESP: u32 = 0x230;
const CONTEXT_SEG_SS: u32 = 0x234;
const CONTEXT_SIZE: usize = 0x238;
/// `CONTEXT_CONTROL | CONTEXT_INTEGER` on x86.
const CONTEXT_CONTROL_INTEGER: u32 = 0x0001_0003;

/// `RtlCaptureContext(ContextRecord)`: the caller's context as it returns from this call. The
/// kernel sees the stack and the return address; the other registers are recorded as zero.
///
/// The console's stores the caller's EAX, ECX, EDX, EBX, ESI, EDI, flags, CS and SS, takes EIP,
/// EBP and ESP from the caller's EBP frame, and leaves `ContextFlags` and the floating point
/// area alone. The trap interface passes only ESP, ECX and EDX, so this cannot match it.
pub fn capture_context(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let record = c.arg(0);
    let return_address = k.r32(c.esp)?;
    k.write(record, &[0; CONTEXT_SIZE])?;
    k.w32(record + CONTEXT_FLAGS, CONTEXT_CONTROL_INTEGER)?;
    k.w32(record + CONTEXT_EIP, return_address)?;
    // The stdcall argument is popped on the return.
    k.w32(record + CONTEXT_ESP, c.esp.wrapping_add(8))?;
    k.w32(record + CONTEXT_EBP, 0)?;
    k.w32(record + CONTEXT_SEG_CS, 0x08)?;
    k.w32(record + CONTEXT_SEG_SS, 0x10)?;
    k.w32(record + CONTEXT_EFLAGS, 0x202)?;
    value(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::handlers::lookup;
    use crate::kernel::testing::kernel;

    const SCRATCH: u32 = crate::mem::HEAP_BASE + 0x2F_0000;

    fn call(k: &mut Kernel, name: &str, args: &[u32], esp: u32) -> CallResult {
        let handler = lookup(name).unwrap_or_else(|| panic!("no handler for {name}"));
        handler(
            k,
            &Call {
                args: args.to_vec(),
                esp,
            },
        )
        .unwrap()
    }

    fn magic_divide(k: &mut Kernel, dividend: i64, magic: u64, shift: u32) -> i64 {
        let (d, m) = (dividend.cast_unsigned(), magic);
        let args = [
            d as u32,
            (d >> 32) as u32,
            m as u32,
            (m >> 32) as u32,
            shift,
        ];
        let CallResult::Value64(low, high) = call(k, "RtlExtendedMagicDivide", &args, 0) else {
            panic!("not a 64-bit result");
        };
        (u64::from(low) | u64::from(high) << 32).cast_signed()
    }

    #[test]
    fn magic_division_divides_by_the_reciprocal_keeping_the_sign() {
        let mut k = kernel();
        // Division by 10 000: the magic number and shift a compiler uses.
        let (magic, shift) = (0xD1B7_1758_E219_652C, 13);
        assert_eq!(magic_divide(&mut k, 123_456_789, magic, shift), 12_345);
        assert_eq!(magic_divide(&mut k, -123_456_789, magic, shift), -12_345);
        assert_eq!(magic_divide(&mut k, 9_999, magic, shift), 0);
    }

    #[test]
    fn generic_rights_map_to_specific_ones() {
        let mut k = kernel();
        let (mask, mapping) = (SCRATCH, SCRATCH + 0x10);
        for (i, rights) in (0u32..).zip([0x1, 0x2, 0x4, 0xF]) {
            k.w32(mapping + 4 * i, rights).unwrap();
        }
        // Generic read and execute, and a specific bit already there.
        k.w32(mask, 0xA000_0100).unwrap();
        call(&mut k, "RtlMapGenericMask", &[mask, mapping], 0);
        assert_eq!(k.r32(mask), Ok(0x105));
        k.w32(mask, 0x1000_0000).unwrap();
        call(&mut k, "RtlMapGenericMask", &[mask, mapping], 0);
        assert_eq!(k.r32(mask), Ok(0xF), "generic all");
    }

    #[test]
    fn a_captured_context_returns_to_the_caller() {
        let mut k = kernel();
        let (record, esp) = (SCRATCH, SCRATCH + 0x800);
        k.write(record, &[0xEE; CONTEXT_SIZE]).unwrap();
        k.w32(esp, 0x0001_2345).unwrap();
        k.w32(esp + 4, record).unwrap();
        call(&mut k, "RtlCaptureContext", &[record], esp);
        assert_eq!(k.r32(record + CONTEXT_FLAGS), Ok(CONTEXT_CONTROL_INTEGER));
        assert_eq!(k.r32(record + CONTEXT_EIP), Ok(0x0001_2345));
        assert_eq!(k.r32(record + CONTEXT_ESP), Ok(esp + 8));
        assert_eq!(k.r32(record + CONTEXT_EFLAGS), Ok(0x202));
        assert_eq!(k.r32(record + 0x10), Ok(0), "the rest is zeroed");
    }
}
