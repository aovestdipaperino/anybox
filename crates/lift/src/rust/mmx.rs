//! MMX instructions: the eight 64-bit MMX registers as their own state in the prelude, with the
//! packed integer arithmetic, shifts, logic, unpacks, packs and moves the XDK's movie decoder uses.

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::{LiftError, effective_address, read_operand, unsupported, write_operand};

/// The Rust statements for an MMX instruction, or `None` when `instruction` is not one.
pub(super) fn emit(instruction: &Instruction, address: u32) -> Result<Option<String>, LiftError> {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    let mnemonic = instruction.mnemonic();
    // Only the MMX forms: SSE2 forms of the same mnemonics take XMM registers.
    let uses_mmx = (0..instruction.op_count()).any(|i| mm_index(instruction, i).is_some());
    if !uses_mmx && mnemonic != Emms {
        return Ok(None);
    }
    let statement = match mnemonic {
        Emms => String::new(),
        // `movntq` is `movq` to memory with a cache hint.
        Movq | Movntq => move_quad(instruction, address)?,
        Movd => move_dword(instruction, address)?,
        Psllw | Pslld | Psllq | Psrlw | Psrld | Psrlq | Psraw | Psrad => {
            let helper = format!("mmx_{}", format!("{mnemonic:?}").to_lowercase());
            let destination = mm(instruction, address, 0)?;
            let count = match instruction.op_kind(1) {
                OpKind::Immediate8 => format!("{}u64", instruction.immediate8()),
                _ => quad(instruction, address, 1)?,
            };
            format!("{destination} = {helper}({destination}, {count});")
        }
        // The SSE extensions to MMX: a word to a register, a word from one, the byte sign bits.
        Pextrw => {
            let shift = 16 * (u32::from(instruction.immediate8()) & 3);
            let source = mm(instruction, address, 1)?;
            write_operand(
                address,
                instruction,
                0,
                32,
                &format!("(({source} >> {shift}) & 0xFFFF) as u32"),
            )?
        }
        Pinsrw => {
            let shift = 16 * (u32::from(instruction.immediate8()) & 3);
            let destination = mm(instruction, address, 0)?;
            let bits = if instruction.op_kind(1) == OpKind::Memory {
                16
            } else {
                32
            };
            let word = read_operand(address, instruction, 1, bits)?;
            format!(
                "{destination} = ({destination} & !(0xFFFFu64 << {shift})) | (u64::from(({word}) & 0xFFFF) << {shift});"
            )
        }
        Pmovmskb => {
            let source = mm(instruction, address, 1)?;
            write_operand(
                address,
                instruction,
                0,
                32,
                &format!("mmx_pmovmskb({source})"),
            )?
        }
        Pshufw => {
            let source = quad(instruction, address, 1)?;
            let order = instruction.immediate8();
            format!(
                "{} = mmx_pshufw({source}, {order});",
                mm(instruction, address, 0)?
            )
        }
        _ => match binary_helper(mnemonic) {
            Some(helper) => {
                let destination = mm(instruction, address, 0)?;
                let source = quad(instruction, address, 1)?;
                format!("{destination} = {helper}({destination}, {source});")
            }
            None => return Ok(None),
        },
    };
    Ok(Some(statement))
}

/// The prelude's helper of a two-operand MMX operation.
fn binary_helper(mnemonic: Mnemonic) -> Option<&'static str> {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    Some(match mnemonic {
        Paddb => "mmx_paddb",
        Paddw => "mmx_paddw",
        Paddd => "mmx_paddd",
        Paddq => "mmx_paddq",
        Paddsb => "mmx_paddsb",
        Paddsw => "mmx_paddsw",
        Paddusb => "mmx_paddusb",
        Paddusw => "mmx_paddusw",
        Psubb => "mmx_psubb",
        Psubw => "mmx_psubw",
        Psubd => "mmx_psubd",
        Psubq => "mmx_psubq",
        Psubsb => "mmx_psubsb",
        Psubsw => "mmx_psubsw",
        Psubusb => "mmx_psubusb",
        Psubusw => "mmx_psubusw",
        Pand => "mmx_pand",
        Pandn => "mmx_pandn",
        Por => "mmx_por",
        Pxor => "mmx_pxor",
        Pmullw => "mmx_pmullw",
        Pmulhw => "mmx_pmulhw",
        Pmulhuw => "mmx_pmulhuw",
        Pmaddwd => "mmx_pmaddwd",
        Pavgb => "mmx_pavgb",
        Pavgw => "mmx_pavgw",
        Pminub => "mmx_pminub",
        Pmaxub => "mmx_pmaxub",
        Pminsw => "mmx_pminsw",
        Pmaxsw => "mmx_pmaxsw",
        Pcmpeqb => "mmx_pcmpeqb",
        Pcmpeqw => "mmx_pcmpeqw",
        Pcmpeqd => "mmx_pcmpeqd",
        Pcmpgtb => "mmx_pcmpgtb",
        Pcmpgtw => "mmx_pcmpgtw",
        Pcmpgtd => "mmx_pcmpgtd",
        Punpcklbw => "mmx_punpcklbw",
        Punpcklwd => "mmx_punpcklwd",
        Punpckldq => "mmx_punpckldq",
        Punpckhbw => "mmx_punpckhbw",
        Punpckhwd => "mmx_punpckhwd",
        Punpckhdq => "mmx_punpckhdq",
        Packuswb => "mmx_packuswb",
        Packsswb => "mmx_packsswb",
        Packssdw => "mmx_packssdw",
        Psadbw => "mmx_psadbw",
        _ => return None,
    })
}

/// The index of an MMX register operand.
fn mm_index(instruction: &Instruction, operand: u32) -> Option<u32> {
    if instruction.op_kind(operand) != OpKind::Register {
        return None;
    }
    let number = (instruction.op_register(operand) as u32).checked_sub(Register::MM0 as u32)?;
    (number < 8).then_some(number)
}

/// The Rust place of MMX register operand `operand`.
fn mm(instruction: &Instruction, address: u32, operand: u32) -> Result<String, LiftError> {
    let index = mm_index(instruction, operand)
        .ok_or_else(|| unsupported(address, "an MMX operand that is not MM0-7"))?;
    Ok(format!("(*mm(c, {index}))"))
}

/// A 64-bit value from operand `operand`: an MMX register, or 8 bytes of memory.
fn quad(instruction: &Instruction, address: u32, operand: u32) -> Result<String, LiftError> {
    if instruction.op_kind(operand) == OpKind::Memory {
        return Ok(format!("ldq64({})", effective_address(instruction)));
    }
    mm(instruction, address, operand)
}

/// `movq` between MMX registers and memory.
fn move_quad(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    if instruction.op_kind(0) == OpKind::Memory {
        return Ok(format!(
            "stq64({}, {});",
            effective_address(instruction),
            mm(instruction, address, 1)?
        ));
    }
    Ok(format!(
        "{} = {};",
        mm(instruction, address, 0)?,
        quad(instruction, address, 1)?
    ))
}

/// `movd` between an MMX register's low dword and a general register or memory.
fn move_dword(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    if let Some(index) = mm_index(instruction, 0) {
        let value = read_operand(address, instruction, 1, 32)?;
        return Ok(format!("*mm(c, {index}) = u64::from({value});"));
    }
    let source = mm(instruction, address, 1)?;
    write_operand(address, instruction, 0, 32, &format!("({source} as u32)"))
}
