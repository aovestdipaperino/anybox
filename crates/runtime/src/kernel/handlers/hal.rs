//! HAL, AV and debugging exports (spec 8.12, 8.13).

use super::value;
use crate::boot::{DATA_EXPORT_SIZE, DATA_EXPORTS};
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// `HalGetInterruptVector(level, *irql)`.
pub fn get_interrupt_vector(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let level = c.arg(0);
    if level > 27 {
        return value(0);
    }
    if c.arg(1) != 0 {
        k.w8(c.arg(1), (27 - level) as u8)?;
    }
    value(0x30 + level)
}

/// `HalReadWritePCISpace(bus, slot, reg, buf, len, write)`: writes are dropped; bus 1 slot 0 is
/// the GPU, other devices read `0xFF`.
pub fn read_write_pci_space(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (bus, slot, reg, buf, len, write) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5));
    if write != 0 {
        return value(len);
    }
    let bytes: Vec<u8> = (0..len)
        .map(|i| {
            let at = reg + i;
            if bus == 1 && slot == 0 {
                let dword: u32 = match at & !3 {
                    0x00 => 0x02A0_10DE,
                    0x08 => 0x0300_00A1,
                    _ => 0,
                };
                (dword >> (8 * (at & 3))) as u8
            } else {
                0xFF
            }
        })
        .collect();
    k.write(buf, &bytes)?;
    value(len)
}

/// The SMC's video mode, as `HalBootSMCVideoMode` holds it.
fn smc_video_mode(k: &Kernel) -> u32 {
    k.r32(DATA_EXPORTS + 356 * DATA_EXPORT_SIZE).unwrap_or(6)
}

/// `HalReadSMBusValue(addr, cmd, word, *value)`.
pub fn read_smbus_value(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (address, command) = (c.arg(0) & 0xFE, c.arg(1));
    let result = match (address, command) {
        (0x20, 1) => u32::from(b'P'),
        (0x20, 3) => 0x60,
        (0x20, 4) => smc_video_mode(k),
        (0x20, 9) => 45,
        (0x20, 0x0A) => 40,
        (0xA8, cmd) => u32::from(k.eeprom.get(cmd as usize).copied().unwrap_or(0)),
        _ => 0,
    };
    k.w32_opt(c.arg(3), result)?;
    value(status::SUCCESS)
}

/// `HalReadSMCTrayState(*state, *count)`: closed, media present.
pub fn read_smc_tray_state(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w32_opt(c.arg(0), 0x60)?;
    k.w32_opt(c.arg(1), 0)?;
    value(status::SUCCESS)
}

pub fn halt(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    Ok(CallResult::Halt)
}

pub fn av_get_saved_data_address(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(k.av_saved_data)
}

pub fn av_set_saved_data_address(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.av_saved_data = c.arg(0);
    value(0)
}

/// `AvSetDisplayMode(base, step, mode, format, pitch, fb)`: recorded; success.
pub fn av_set_display_mode(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.display_mode = std::array::from_fn(|i| c.arg(i));
    value(0)
}

/// `AvSendTVEncoderOption(base, option, param, *result)`: option 6 queries the AV capabilities.
pub fn av_send_tv_encoder_option(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if c.arg(1) != 6 {
        return value(0);
    }
    let pack = match smc_video_mode(k) {
        0 => 3,
        1 => 4,
        2 => 5,
        3 => 2,
        4 => 6,
        6 => 1,
        _ => 0,
    };
    let region = u32::from_le_bytes(k.eeprom[0x58..0x5C].try_into().unwrap_or_default());
    let standard = if region & 0xF00 != 0 {
        region & (0xF00 | 0xC0_0000)
    } else if crate::kernel::time::refresh_is_50() {
        0x400 | 0x80_0000
    } else {
        0x100 | 0x40_0000
    };
    let user = u32::from_le_bytes(k.eeprom[0x94..0x98].try_into().unwrap_or_default());
    k.w32_opt(
        c.arg(3),
        pack | standard | (user & !(0xFF | 0xF00 | 0xC0_0000)),
    )?;
    value(0)
}

/// `DbgPrint(fmt)`: the text, at most 512 bytes, not formatted.
pub fn dbg_print(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mut text = k.c_string(c.arg(0)).unwrap_or_default();
    text.truncate(512);
    eprint!("DbgPrint: {}", String::from_utf8_lossy(&text));
    if !text.ends_with(b"\n") {
        eprintln!();
    }
    value(0)
}
