//! SSE instructions: the eight XMM registers as 128-bit values in the prelude, with the moves,
//! scalar and packed single-precision arithmetic, logic, conversions, compares, shuffles and
//! unpacks the XDK's Direct3D library uses.

use std::fmt::Write as _;

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::{LiftError, effective_address, read_operand, unsupported, write_operand};

/// The C statements for an SSE instruction, or `None` when `instruction` is not one.
pub(super) fn emit(instruction: &Instruction, address: u32) -> Result<Option<String>, LiftError> {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    let mnemonic = instruction.mnemonic();
    let statement = match mnemonic {
        // The non-temporal stores only hint the cache.
        Movaps | Movups | Movapd | Movupd | Movdqa | Movdqu | Movntps | Movntpd | Movntdq => {
            let source = vector(instruction, address, 1)?;
            store_vector(instruction, address, 0, &source)?
        }
        Movss => move_scalar(instruction, address)?,
        Movlps | Movhps | Movlpd | Movhpd => move_half(instruction, address, mnemonic)?,
        Movhlps => format!(
            "{{ xmm_t s = {}; {}.q[0] = s.q[1]; }}",
            vector(instruction, address, 1)?,
            xmm(instruction, address, 0)?
        ),
        Movlhps => format!(
            "{{ xmm_t s = {}; {}.q[1] = s.q[0]; }}",
            vector(instruction, address, 1)?,
            xmm(instruction, address, 0)?
        ),
        Addss | Subss | Mulss | Divss | Minss | Maxss | Sqrtss | Rcpss | Rsqrtss => {
            scalar(instruction, address, mnemonic)?
        }
        Addps | Subps | Mulps | Divps | Minps | Maxps | Sqrtps | Rcpps | Rsqrtps => {
            packed(instruction, address, mnemonic)?
        }
        Andps | Andnps | Orps | Xorps | Andpd | Andnpd | Orpd | Xorpd | Pand | Pandn | Por
        | Pxor => bitwise(instruction, address, mnemonic)?,
        Cvttss2si | Cvtss2si => {
            let value = scalar_source(instruction, address, 1)?;
            let converted = format!("cvt_ss2si({value}, {})", u8::from(mnemonic == Cvttss2si));
            write_operand(address, instruction, 0, 32, &converted)?
        }
        Cvtps2pi | Cvttps2pi | Cvtpi2ps => packed_conversion(instruction, address, mnemonic)?,
        Cvtsi2ss => {
            let value = read_operand(address, instruction, 1, 32)?;
            format!(
                "{}.f[0] = (float)(int32_t)({value});",
                xmm(instruction, address, 0)?
            )
        }
        Comiss | Ucomiss => {
            let a = format!("{}.f[0]", xmm(instruction, address, 0)?);
            let b = scalar_source(instruction, address, 1)?;
            format!("fcompare_flags((double){a}, (double){b});")
        }
        Shufps => {
            let order = immediate(instruction, 2);
            let d = xmm(instruction, address, 0)?;
            let s = vector(instruction, address, 1)?;
            let pick = |lane: u32| (order >> (2 * lane)) & 3;
            format!(
                "{{ xmm_t a = {d}, b = {s}, r; r.u[0] = a.u[{}]; r.u[1] = a.u[{}]; r.u[2] = b.u[{}]; r.u[3] = b.u[{}]; {d} = r; }}",
                pick(0),
                pick(1),
                pick(2),
                pick(3)
            )
        }
        Unpcklps | Unpckhps => {
            let d = xmm(instruction, address, 0)?;
            let s = vector(instruction, address, 1)?;
            let (x, y) = if mnemonic == Unpcklps { (0, 1) } else { (2, 3) };
            format!(
                "{{ xmm_t a = {d}, b = {s}, r; r.u[0] = a.u[{x}]; r.u[1] = b.u[{x}]; r.u[2] = a.u[{y}]; r.u[3] = b.u[{y}]; {d} = r; }}"
            )
        }
        Movmskps => {
            let s = vector(instruction, address, 1)?;
            write_operand(
                address,
                instruction,
                0,
                32,
                &format!(
                    "({{ xmm_t v = {s}; (v.u[0] >> 31) | ((v.u[1] >> 31) << 1) | ((v.u[2] >> 31) << 2) | ((v.u[3] >> 31) << 3); }})"
                ),
            )?
        }
        Movd => move_dword(instruction, address)?,
        Cmpss | Cmpps => compare_mask(instruction, address, mnemonic == Cmpss)?,
        Ldmxcsr => format!(
            "mxcsr_x = rd32({}) ^ 0x1F80u;",
            effective_address(instruction)
        ),
        Stmxcsr => write_operand(address, instruction, 0, 32, "mxcsr_get()")?,
        _ => return Ok(None),
    };
    Ok(Some(statement))
}

/// The index of an XMM register operand.
fn xmm_index(instruction: &Instruction, operand: u32) -> Option<u32> {
    let register = instruction.op_register(operand);
    let number = (register as u32).checked_sub(Register::XMM0 as u32)?;
    (number < 8).then_some(number)
}

/// `cvtps2pi`, `cvttps2pi` and `cvtpi2ps`: two floats to two 32-bit integers in an MMX register
/// (rounded per MXCSR, or truncated), and two integers from an MMX register or eight bytes of
/// memory to the low two floats.
fn packed_conversion(
    instruction: &Instruction,
    address: u32,
    mnemonic: Mnemonic,
) -> Result<String, LiftError> {
    if mnemonic == Mnemonic::Cvtpi2ps {
        let source = if instruction.op_kind(1) == OpKind::Memory {
            format!("ldq64({})", effective_address(instruction))
        } else {
            format!("MM[{}]", mm_register(instruction, address, 1)?)
        };
        let d = xmm(instruction, address, 0)?;
        return Ok(format!(
            "{{ uint64_t s = {source}; {d}.f[0] = (float)(int32_t)(uint32_t)s; {d}.f[1] = (float)(int32_t)(uint32_t)(s >> 32); }}"
        ));
    }
    let truncate = u8::from(mnemonic == Mnemonic::Cvttps2pi);
    let (low, high) = if instruction.op_kind(1) == OpKind::Memory {
        let at = effective_address(instruction);
        (format!("ldf32x({at})"), format!("ldf32x(({at}) + 4u)"))
    } else {
        let source = xmm(instruction, address, 1)?;
        (format!("{source}.f[0]"), format!("{source}.f[1]"))
    };
    Ok(format!(
        "MM[{}] = (uint64_t)cvt_ss2si({low}, {truncate}) | ((uint64_t)cvt_ss2si({high}, {truncate}) << 32);",
        mm_register(instruction, address, 0)?
    ))
}

/// The index of the MMX register operand `operand`.
fn mm_register(instruction: &Instruction, address: u32, operand: u32) -> Result<u32, LiftError> {
    (instruction.op_register(operand) as u32)
        .checked_sub(Register::MM0 as u32)
        .filter(|number| *number < 8)
        .ok_or_else(|| unsupported(address, "an MMX operand that is not MM0-7"))
}

/// The C lvalue of XMM register operand `operand`.
fn xmm(instruction: &Instruction, address: u32, operand: u32) -> Result<String, LiftError> {
    let index = xmm_index(instruction, operand)
        .ok_or_else(|| unsupported(address, "an XMM operand that is not XMM0-7"))?;
    Ok(format!("XMM[{index}]"))
}

/// A 128-bit value from operand `operand`: an XMM register, or 16 bytes of memory.
fn vector(instruction: &Instruction, address: u32, operand: u32) -> Result<String, LiftError> {
    if instruction.op_kind(operand) == OpKind::Memory {
        return Ok(format!("ldx128({})", effective_address(instruction)));
    }
    xmm(instruction, address, operand)
}

/// A single-precision value from operand `operand`: lane 0 of an XMM register, or 4 bytes.
fn scalar_source(
    instruction: &Instruction,
    address: u32,
    operand: u32,
) -> Result<String, LiftError> {
    if instruction.op_kind(operand) == OpKind::Memory {
        return Ok(format!("ldf32x({})", effective_address(instruction)));
    }
    Ok(format!("{}.f[0]", xmm(instruction, address, operand)?))
}

/// Stores a 128-bit value to operand `operand`.
fn store_vector(
    instruction: &Instruction,
    address: u32,
    operand: u32,
    value: &str,
) -> Result<String, LiftError> {
    if instruction.op_kind(operand) == OpKind::Memory {
        return Ok(format!(
            "{{ xmm_t v = {value}; stx128({}, v); }}",
            effective_address(instruction)
        ));
    }
    Ok(format!(
        "{} = {value};",
        xmm(instruction, address, operand)?
    ))
}

/// `movss`: from memory it clears the upper lanes; between registers it moves lane 0 only.
fn move_scalar(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    if instruction.op_kind(0) == OpKind::Memory {
        let source = xmm(instruction, address, 1)?;
        return Ok(format!(
            "stf32x({}, {source}.f[0]);",
            effective_address(instruction)
        ));
    }
    let destination = xmm(instruction, address, 0)?;
    if instruction.op_kind(1) == OpKind::Memory {
        return Ok(format!(
            "{{ xmm_t v = {{0}}; v.f[0] = ldf32x({}); {destination} = v; }}",
            effective_address(instruction)
        ));
    }
    Ok(format!(
        "{destination}.f[0] = {}.f[0];",
        xmm(instruction, address, 1)?
    ))
}

/// `movlps`, `movhps` and the double forms: 64 bits between memory and one half of a register.
fn move_half(
    instruction: &Instruction,
    address: u32,
    mnemonic: Mnemonic,
) -> Result<String, LiftError> {
    let half = usize::from(matches!(mnemonic, Mnemonic::Movhps | Mnemonic::Movhpd));
    let ea = effective_address(instruction);
    if instruction.op_kind(0) == OpKind::Memory {
        let source = xmm(instruction, address, 1)?;
        return Ok(format!("stq64({ea}, {source}.q[{half}]);"));
    }
    Ok(format!(
        "{}.q[{half}] = ldq64({ea});",
        xmm(instruction, address, 0)?
    ))
}

/// The C expression of one lane of a single-precision operation.
fn lane_operation(mnemonic: Mnemonic, a: &str, b: &str) -> String {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    match mnemonic {
        Addss | Addps => format!("sres({a} + {b}, {a}, {b})"),
        Subss | Subps => format!("sres({a} - {b}, {a}, {b})"),
        Mulss | Mulps => format!("sres({a} * {b}, {a}, {b})"),
        Divss | Divps => format!("sres({a} / {b}, {a}, {b})"),
        Minss | Minps => format!("(({a} < {b}) ? {a} : {b})"),
        Maxss | Maxps => format!("(({a} > {b}) ? {a} : {b})"),
        Sqrtss | Sqrtps => format!("sres(sqrtf({b}), {b}, {b})"),
        Rcpss | Rcpps => format!("sres(1.0f / {b}, {b}, {b})"),
        _ => format!("sres(1.0f / sqrtf({b}), {b}, {b})"),
    }
}

/// A scalar single-precision operation on lane 0.
fn scalar(
    instruction: &Instruction,
    address: u32,
    mnemonic: Mnemonic,
) -> Result<String, LiftError> {
    let destination = xmm(instruction, address, 0)?;
    let b = scalar_source(instruction, address, 1)?;
    let a = format!("{destination}.f[0]");
    let value = lane_operation(mnemonic, &a, "t_b");
    Ok(format!(
        "{{ float t_b = {b}; {destination}.f[0] = {value}; }}"
    ))
}

/// A packed single-precision operation on all four lanes.
fn packed(
    instruction: &Instruction,
    address: u32,
    mnemonic: Mnemonic,
) -> Result<String, LiftError> {
    let destination = xmm(instruction, address, 0)?;
    let source = vector(instruction, address, 1)?;
    let mut lanes = String::new();
    for lane in 0..4 {
        let value = lane_operation(mnemonic, &format!("a.f[{lane}]"), &format!("b.f[{lane}]"));
        write!(lanes, " r.f[{lane}] = {value};").unwrap();
    }
    Ok(format!(
        "{{ xmm_t a = {destination}, b = {source}, r;{lanes} {destination} = r; }}"
    ))
}

/// The bitwise operations, on 32-bit lanes.
fn bitwise(
    instruction: &Instruction,
    address: u32,
    mnemonic: Mnemonic,
) -> Result<String, LiftError> {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    let destination = xmm(instruction, address, 0)?;
    let source = vector(instruction, address, 1)?;
    let operator = |lane: u32| match mnemonic {
        Andps | Andpd | Pand => format!("a.u[{lane}] & b.u[{lane}]"),
        Andnps | Andnpd | Pandn => format!("~a.u[{lane}] & b.u[{lane}]"),
        Orps | Orpd | Por => format!("a.u[{lane}] | b.u[{lane}]"),
        _ => format!("a.u[{lane}] ^ b.u[{lane}]"),
    };
    let mut lanes = String::new();
    for lane in 0..4 {
        write!(lanes, " a.u[{lane}] = {};", operator(lane)).unwrap();
    }
    Ok(format!(
        "{{ xmm_t a = {destination}, b = {source};{lanes} {destination} = a; }}"
    ))
}

/// `movd` between a general register or memory and an XMM register's low dword.
fn move_dword(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    if let Some(index) = xmm_index(instruction, 0) {
        let value = read_operand(address, instruction, 1, 32)?;
        return Ok(format!(
            "{{ xmm_t v = {{0}}; v.u[0] = {value}; XMM[{index}] = v; }}"
        ));
    }
    let source = xmm(instruction, address, 1)?;
    write_operand(address, instruction, 0, 32, &format!("{source}.u[0]"))
}

/// `cmpss` and `cmpps`: a lane becomes all ones where the predicate holds.
fn compare_mask(
    instruction: &Instruction,
    address: u32,
    scalar: bool,
) -> Result<String, LiftError> {
    let destination = xmm(instruction, address, 0)?;
    let source = if scalar {
        format!(
            "({{ xmm_t t = {{0}}; t.f[0] = {}; t; }})",
            scalar_source(instruction, address, 1)?
        )
    } else {
        vector(instruction, address, 1)?
    };
    let predicate = |lane: u32| {
        let (a, b) = (format!("a.f[{lane}]"), format!("b.f[{lane}]"));
        match immediate(instruction, 2) & 7 {
            0 => format!("{a} == {b}"),
            1 => format!("{a} < {b}"),
            2 => format!("{a} <= {b}"),
            3 => format!("({a} != {a} || {b} != {b})"),
            4 => format!("!({a} == {b})"),
            5 => format!("!({a} < {b})"),
            6 => format!("!({a} <= {b})"),
            _ => format!("({a} == {a} && {b} == {b})"),
        }
    };
    let lanes = if scalar { 1 } else { 4 };
    let mut body = String::new();
    for lane in 0..lanes {
        write!(
            body,
            " a.u[{lane}] = ({}) ? 0xFFFFFFFFu : 0u;",
            predicate(lane)
        )
        .unwrap();
    }
    Ok(format!(
        "{{ xmm_t a = {destination}, b = {source};{body} {destination} = a; }}"
    ))
}

/// The immediate at operand `operand`, as a small integer.
fn immediate(instruction: &Instruction, operand: u32) -> u32 {
    u32::try_from(instruction.immediate(operand) & 0xFF).unwrap_or(0)
}
