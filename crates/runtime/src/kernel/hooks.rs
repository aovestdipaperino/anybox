//! The XDK functions the runtime replaces (spec 5.6, 9): the hook plan, the patches that send
//! the title's own calls to the traps, and the replacements.
//!
//! Every replacement has the `Handler` signature, including those that cannot fail.
#![allow(clippy::unnecessary_wraps)]

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{Call, CallResult, HOOK_BASE, KResult, Kernel, THREAD_RETURN};

/// The signature list: those converted from `XbSymbolDatabase`, then those read from titles' own
/// code. `anybox` plans from the copy beside `libruntime.a`, which the build writes the same way.
pub const SIGNATURES: &str = concat!(
    include_str!("../../data/signatures.txt"),
    include_str!("../../data/signatures-titles.txt")
);

/// How many hooks and vtable slots fit before the thread-return trap.
pub const MAX_HOOKS: usize = ((THREAD_RETURN - HOOK_BASE) / 4) as usize;

/// `ERROR_DEVICE_NOT_CONNECTED`.
const DEVICE_NOT_CONNECTED: u32 = 1167;
/// The handle `XInputOpen` gives the gamepad in port 0.
const GAMEPAD_HANDLE: u32 = 0x5844_0001;

/// The trap address of hook or vtable slot `index`.
#[must_use]
pub fn trap(index: usize) -> u32 {
    HOOK_BASE + 4 * u32::try_from(index).unwrap_or(u32::MAX)
}

/// One controller in port 0 (spec 9.1).
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gamepad {
    /// The device types the title opens gamepads with.
    pub types: Vec<u32>,
    /// Whether the insertion has been reported.
    pub reported: bool,
    /// Increases on every change of the state.
    pub packet: u32,
    pub buttons: u16,
    /// A, B, X, Y, Black, White, left trigger, right trigger.
    pub analog: [u8; 8],
    /// LX, LY, RX, RY.
    pub sticks: [i16; 4],
}

/// The hooks of a title and the vtable slots that follow them.
#[derive(Debug, Default)]
pub struct Hooks {
    /// The plan, in trap order.
    pub plan: Vec<xdk::Hook>,
    /// Vtable slots after the plan: class and slot number.
    pub slots: Vec<(&'static str, u32)>,
    /// Each class's vtable, built once.
    pub vtables: HashMap<&'static str, u32>,
    pub gamepad: Gamepad,
}

/// Plans the hooks of `xbe` (the same plan `anybox` made), patches each hooked function's first
/// five bytes with a near jump to its trap, and finds the gamepad device types.
///
/// # Errors
///
/// A message when the signature list is malformed or there are too many hooks.
pub fn install(kernel: &mut Kernel, xbe: &xbe::Xbe) -> Result<(), String> {
    let entries: &'static [xdk::Entry] = Box::leak(
        xdk::parse(SIGNATURES)
            .map_err(|e| e.to_string())?
            .into_boxed_slice(),
    );
    let sections: Vec<xdk::Section<'_>> = xbe
        .sections()
        .iter()
        .map(|section| xdk::Section {
            name: section.name(),
            address: section.virtual_address(),
            virtual_size: section.virtual_size(),
            preload: section.flags().preload(),
            data: xbe.section_data(section),
        })
        .collect();
    let stubbed = std::env::var("ANYXBOX_STUB").unwrap_or_else(|_| "DSOUND".to_owned());
    let stubbed: Vec<&str> = stubbed.split(',').filter(|name| !name.is_empty()).collect();
    let plan = xdk::plan(&sections, entries, &stubbed, MAX_HOOKS)?;
    if std::env::var_os("ANYXBOX_VERBOSE").is_some() {
        for (name, matches) in &plan.ambiguous {
            eprintln!("anyxbox: {name} is ambiguous ({matches} matches); not hooked");
        }
    }
    for (index, hook) in plan.hooks.iter().enumerate() {
        let rel = trap(index).wrapping_sub(hook.address.wrapping_add(5));
        let mut jump = [0xE9, 0, 0, 0, 0];
        jump[1..].copy_from_slice(&rel.to_le_bytes());
        kernel
            .write(hook.address, &jump)
            .map_err(|f| format!("patching a hook at {:#010x}", f.0))?;
    }
    let open = plan
        .hooks
        .iter()
        .find(|hook| hook.name == Some("XInputOpen"))
        .map(|hook| hook.address);
    if let Some(open) = open {
        kernel.hooks.gamepad.types = gamepad_types(&sections, open);
    }
    kernel.hooks.plan = plan.hooks;
    Ok(())
}

/// The immediates of every `push imm32` directly followed by a `call rel32` to `open`.
fn gamepad_types(sections: &[xdk::Section<'_>], open: u32) -> Vec<u32> {
    let mut types = Vec::new();
    for section in sections.iter().filter(|s| s.preload) {
        for (offset, window) in section.data.windows(10).enumerate() {
            if window[0] != 0x68 || window[5] != 0xE8 {
                continue;
            }
            let site = section.address + u32::try_from(offset).unwrap_or(0) + 5;
            let rel = u32::from_le_bytes([window[6], window[7], window[8], window[9]]);
            if site.wrapping_add(5).wrapping_add(rel) == open {
                let kind = u32::from_le_bytes([window[1], window[2], window[3], window[4]]);
                if !types.contains(&kind) {
                    types.push(kind);
                }
            }
        }
    }
    types
}

/// A vtable slot the runtime knows: its name, handler, and the bytes it pops.
#[must_use]
pub fn slot(class: &str, slot: u32) -> Option<(&'static str, super::Handler, u16)> {
    super::sound::slot(class, slot)
}

/// What a hook trap names.
#[derive(Debug, PartialEq, Eq)]
pub enum Target {
    /// A planned function: its name (or `None` for a stub) and the bytes it pops.
    Planned(Option<&'static str>, u16),
    /// A vtable slot: class and slot number.
    Slot(&'static str, u32),
    Unknown,
}

impl Hooks {
    /// What trap `index` stands for.
    #[must_use]
    pub fn target(&self, index: usize) -> Target {
        if let Some(hook) = self.plan.get(index) {
            return Target::Planned(hook.name, hook.arg_bytes);
        }
        match self.slots.get(index - self.plan.len()) {
            Some(&(class, slot)) => Target::Slot(class, slot),
            None => Target::Unknown,
        }
    }
}

/// The replacement of a hooked function, by name, and the bytes it pops.
#[must_use]
pub fn handler(name: &str) -> Option<super::Handler> {
    let handler: super::Handler = match name {
        "XInitDevices" | "XInputClose" | "XInputPoll" => |_, _| Ok(CallResult::Value(0)),
        "XGetDevices" => get_devices,
        "XGetDeviceChanges" => get_device_changes,
        "XInputOpen" => input_open,
        "XInputGetState" => input_get_state,
        "XInputSetState" => input_set_state,
        "XInputGetCapabilities" => input_get_capabilities,
        _ => return super::sound::handler(name),
    };
    Some(handler)
}

/// `XGetDevices(type)`: port 0 for a gamepad type, which counts as the insertion.
fn get_devices(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if !k.hooks.gamepad.types.contains(&c.arg(0)) {
        return Ok(CallResult::Value(0));
    }
    k.hooks.gamepad.reported = true;
    Ok(CallResult::Value(1))
}

/// `XGetDeviceChanges(type, *insertions, *removals)`: the gamepad's insertion, once.
fn get_device_changes(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let insert = k.hooks.gamepad.types.contains(&c.arg(0)) && !k.hooks.gamepad.reported;
    if insert {
        k.hooks.gamepad.reported = true;
    }
    k.w32(c.arg(1), u32::from(insert))?;
    k.w32(c.arg(2), 0)?;
    Ok(CallResult::Value(u32::from(insert)))
}

/// `XInputOpen(type, port, slot, params)`.
fn input_open(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let gamepad = k.hooks.gamepad.types.contains(&c.arg(0)) && c.arg(1) == 0;
    Ok(CallResult::Value(if gamepad { GAMEPAD_HANDLE } else { 0 }))
}

/// `XInputGetState(h, *state)`: packet number, buttons, analog buttons, sticks.
fn input_get_state(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if c.arg(0) != GAMEPAD_HANDLE {
        return Ok(CallResult::Value(DEVICE_NOT_CONNECTED));
    }
    let pad = &k.hooks.gamepad;
    let mut state = Vec::with_capacity(22);
    state.extend_from_slice(&pad.packet.to_le_bytes());
    state.extend_from_slice(&pad.buttons.to_le_bytes());
    state.extend_from_slice(&pad.analog);
    for stick in pad.sticks {
        state.extend_from_slice(&stick.to_le_bytes());
    }
    k.write(c.arg(1), &state)?;
    Ok(CallResult::Value(0))
}

/// `XInputSetState(h, *feedback)`: the feedback completes at once.
fn input_set_state(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w32(c.arg(1), 0)?;
    Ok(CallResult::Value(0))
}

/// Bytes of `XINPUT_CAPABILITIES`: subtype, a reserved byte and word, the input mask (an
/// `XINPUT_GAMEPAD`: buttons, eight analog buttons, four stick axes) at `+4`, and the rumble mask
/// (two motor words) at `+22`.
const CAPABILITIES_BYTES: usize = 26;

/// `XInputGetCapabilities(h, *caps)`: a gamepad with every button and both motors. Exactly the
/// structure is written: a title may keep other data right after it (`BLiNX` keeps the controller
/// handle there).
fn input_get_capabilities(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if c.arg(0) != GAMEPAD_HANDLE {
        return Ok(CallResult::Value(DEVICE_NOT_CONNECTED));
    }
    let mut caps = [0u8; CAPABILITIES_BYTES];
    caps[0] = 1;
    caps[4..6].copy_from_slice(&0x00FFu16.to_le_bytes());
    caps[6..].fill(0xFF);
    k.write(c.arg(1), &caps)?;
    Ok(CallResult::Value(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_signatures_parse() {
        let entries = xdk::parse(SIGNATURES).unwrap();
        assert!(entries.iter().any(|e| e.name == "XInitDevices" && e.hook));
    }

    #[test]
    fn gamepad_types_are_the_pushes_before_calls_to_xinput_open() {
        // push 0x12345678 ; call 0x2000, at 0x1000.
        let rel = 0x2000u32.wrapping_sub(0x1005 + 5);
        let mut code = vec![0x68, 0x78, 0x56, 0x34, 0x12, 0xE8];
        code.extend_from_slice(&rel.to_le_bytes());
        let sections = [xdk::Section {
            name: ".text",
            address: 0x1000,
            virtual_size: 10,
            preload: true,
            data: &code,
        }];
        assert_eq!(gamepad_types(&sections, 0x2000), [0x1234_5678]);
        assert_eq!(gamepad_types(&sections, 0x3000), Vec::<u32>::new());
    }

    #[test]
    fn capabilities_leave_what_follows_the_structure_alone() {
        let mut k = crate::kernel::testing::kernel();
        let at = crate::mem::HEAP_BASE + 0x100;
        // The title's controller handle, right after its XINPUT_CAPABILITIES.
        k.w32(at + 26, GAMEPAD_HANDLE).unwrap();
        let call = Call {
            args: vec![GAMEPAD_HANDLE, at],
            esp: 0,
        };
        assert_eq!(
            input_get_capabilities(&mut k, &call),
            Ok(CallResult::Value(0))
        );
        assert_eq!(k.r32(at + 26), Ok(GAMEPAD_HANDLE));
        assert_eq!(k.r16(at + 4), Ok(0x00FF));
        assert_eq!(k.r32(at + 22), Ok(u32::MAX), "both motors");
    }

    #[test]
    fn the_signature_lists_tell_dsound_thunks_of_one_shape_apart() {
        // The `XbSymbolDatabase` list and the one read from titles parse as one, with no name twice.
        let entries = xdk::parse(SIGNATURES).unwrap();
        for name in [
            "IDirectSoundBuffer::SetHeadroom",
            "IDirectSoundBuffer::SetMixBins",
            "IDirectSoundBuffer::SetMixBinVolumes",
            "IDirectSoundBuffer::SetFrequency",
            "IDirectSoundBuffer::Lock",
            "IDirectSound::CreateSoundBuffer",
        ] {
            let entry = entries.iter().find(|e| e.name == name).unwrap();
            assert!(entry.hook && handler(name).is_some(), "{name}");
        }
    }

    #[test]
    fn the_gamepad_is_inserted_once() {
        let mut k = crate::kernel::testing::kernel();
        k.hooks.gamepad.types = vec![0x500];
        let at = crate::mem::HEAP_BASE + 0x100;
        let call = |kind| Call {
            args: vec![kind, at, at + 4],
            esp: 0,
        };
        assert_eq!(
            get_device_changes(&mut k, &call(0x500)),
            Ok(CallResult::Value(1))
        );
        assert_eq!(k.r32(at), Ok(1));
        assert_eq!(
            get_device_changes(&mut k, &call(0x500)),
            Ok(CallResult::Value(0))
        );
        assert_eq!(k.r32(at), Ok(0));
        assert_eq!(
            get_device_changes(&mut k, &call(0x600)),
            Ok(CallResult::Value(0))
        );
    }
}
