//! The x87 floating point instructions: the register stack, loads, stores, arithmetic,
//! comparisons and the status word. The stack holds doubles (see the prelude).

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::{LiftError, effective_address, unsupported, write_operand};

/// The C statements for an x87 instruction, or `None` when `instruction` is not one.
pub(super) fn emit(instruction: &Instruction, address: u32) -> Result<Option<String>, LiftError> {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    let statement = match instruction.mnemonic() {
        Fld => load(instruction, address)?,
        Fst | Fstp => store(instruction, address)?,
        Fild => integer_load(instruction),
        Fist | Fistp | Fisttp => integer_store(instruction, address)?,
        Fadd | Faddp => arithmetic(instruction, address, "+", false)?,
        Fsub | Fsubp => arithmetic(instruction, address, "-", false)?,
        Fsubr | Fsubrp => arithmetic(instruction, address, "-", true)?,
        Fmul | Fmulp => arithmetic(instruction, address, "*", false)?,
        Fdiv | Fdivp => arithmetic(instruction, address, "/", false)?,
        Fdivr | Fdivrp => arithmetic(instruction, address, "/", true)?,
        Fxch => exchange(instruction),
        Fcom | Fcomp | Fcompp | Fucom | Fucomp | Fucompp => compare(instruction, address)?,
        Fcomi | Fcomip | Fucomi | Fucomip => compare_flags(instruction),
        Fnstsw => {
            let status = "fstatus()".to_string();
            write_operand(address, instruction, 0, 16, &status)?
        }
        Fnclex | Fclex | Wait | Ffree => String::new(),
        Ffreep => "fpop();".to_string(),
        Fdecstp => "fpu_top = (fpu_top + 7) & 7;".to_string(),
        Fincstp => "fpu_top = (fpu_top + 1) & 7;".to_string(),
        // `fninit` sets the control word to 0x037F, 64-bit precision.
        Fninit | Finit => "fpu_top = 0; fpu_cc = 0; fpu_cwx = 0x0100;".to_string(),
        Fsin => "if (ftrig_ok(ST(0))) ST(0) = fres(sin(ST(0)), ST(0), ST(0));".to_string(),
        Fcos => "if (ftrig_ok(ST(0))) ST(0) = fres(cos(ST(0)), ST(0), ST(0));".to_string(),
        Fsincos => "{ double t = ST(0); if (ftrig_ok(t)) { ST(0) = fres(sin(t), t, t); fpush(fres(cos(t), t, t)); } }"
            .to_string(),
        Fptan => "{ double t = ST(0); if (ftrig_ok(t)) { ST(0) = fres(tan(t), t, t); fpush(1.0); } }"
            .to_string(),
        Fpatan => "ST(1) = fres(atan2(ST(1), ST(0)), ST(1), ST(0)); fpop();".to_string(),
        Fldpi => "fpush(3.14159265358979323846);".to_string(),
        Fldl2e => "fpush(1.4426950408889634074);".to_string(),
        Fldl2t => "fpush(3.3219280948873623479);".to_string(),
        Fldlg2 => "fpush(0.30102999566398119521);".to_string(),
        Fldln2 => "fpush(0.69314718055994530942);".to_string(),
        Frndint => "ST(0) = fround_cw(ST(0));".to_string(),
        Fscale => "ST(0) = fscale_op(ST(0), ST(1));".to_string(),
        Fyl2x => "ST(1) = fres(ST(1) * log2(ST(0)), ST(1), ST(0)); fpop();".to_string(),
        F2xm1 => "ST(0) = fres(expm1(ST(0) * 0.69314718055994530942), ST(0), ST(0));".to_string(),
        Fprem => "fprem_op(0);".to_string(),
        Fprem1 => "fprem_op(1);".to_string(),
        Fxam => "fxam(ST(0));".to_string(),
        Ficom | Ficomp => integer_compare(instruction, address)?,
        Fcmovb | Fcmove | Fcmovbe | Fcmovu | Fcmovnb | Fcmovne | Fcmovnbe | Fcmovnu => {
            let test = match instruction.mnemonic() {
                Fcmovb => "F_CF",
                Fcmove => "F_ZF",
                Fcmovbe => "(F_CF || F_ZF)",
                Fcmovu => "F_PF",
                Fcmovnb => "!F_CF",
                Fcmovne => "!F_ZF",
                Fcmovnbe => "(!F_CF && !F_ZF)",
                _ => "!F_PF",
            };
            let index = stack_index(instruction, 1).unwrap_or(1);
            format!("if ({test}) ST(0) = ST({index});")
        }
        Fiadd => integer_arithmetic(instruction, address, "+", false)?,
        Fisub => integer_arithmetic(instruction, address, "-", false)?,
        Fisubr => integer_arithmetic(instruction, address, "-", true)?,
        Fimul => integer_arithmetic(instruction, address, "*", false)?,
        Fidiv => integer_arithmetic(instruction, address, "/", false)?,
        Fidivr => integer_arithmetic(instruction, address, "/", true)?,
        Fldz => "fpush(0.0);".to_string(),
        Fld1 => "fpush(1.0);".to_string(),
        Fchs => "ST(0) = -ST(0);".to_string(),
        Fabs => "ST(0) = fabs(ST(0));".to_string(),
        Fsqrt => "ST(0) = fres(sqrt(ST(0)), ST(0), ST(0));".to_string(),
        Ftst => "fcompare(ST(0), 0.0);".to_string(),
        // Control words and waits change nothing modelled; the stack keeps no empty tags, so
        // freeing a register changes nothing the code reads.
        Fldcw => format!("fpu_cwx = (uint32_t)rd16({}) ^ 0x027Fu;", effective_address(instruction)),
        Fnstcw => write_operand(address, instruction, 0, 16, "fpu_cw()")?,
        Fnstenv | Fstenv => format!("fstenv_op({});", effective_address(instruction)),
        Fldenv => format!("fldenv_op({});", effective_address(instruction)),
        Fnsave | Fsave => format!("fsave_op({});", effective_address(instruction)),
        Frstor => format!("frstor_op({});", effective_address(instruction)),
        Sahf => {
            "{ uint32_t ah = (R_EAX >> 8) & 0xFF; F_SF = (ah >> 7) & 1; F_ZF = (ah >> 6) & 1; F_CF = ah & 1; }"
                .to_string()
        }
        _ => return Ok(None),
    };
    Ok(Some(statement))
}

/// The memory width of the instruction's memory operand, in bytes.
fn memory_bytes(instruction: &Instruction) -> usize {
    instruction.memory_size().size()
}

/// The stack index named by a `ST(i)` register operand.
fn stack_index(instruction: &Instruction, index: u32) -> Option<u32> {
    #[allow(clippy::enum_glob_use)]
    use Register::*;
    match instruction.op_register(index) {
        ST0 => Some(0),
        ST1 => Some(1),
        ST2 => Some(2),
        ST3 => Some(3),
        ST4 => Some(4),
        ST5 => Some(5),
        ST6 => Some(6),
        ST7 => Some(7),
        _ => Option::None,
    }
}

/// The memory-operand value of an x87 floating operand, read at its width.
fn load_floating(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let ea = effective_address(instruction);
    match memory_bytes(instruction) {
        4 => Ok(format!("ldf32({ea})")),
        8 => Ok(format!("ldf64({ea})")),
        10 => Ok(format!("ld80({ea})")),
        other => Err(unsupported(
            address,
            format!("x87 operand of {other} bytes"),
        )),
    }
}

/// `fld`: pushes a memory value or a copy of `ST(i)`.
fn load(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    if instruction.op_kind(0) == OpKind::Memory {
        let value = load_floating(instruction, address)?;
        return Ok(format!("fpush({value});"));
    }
    let index = stack_index(instruction, 0).ok_or_else(|| unsupported(address, "fld operand"))?;
    Ok(format!("{{ double t = ST({index}); fpush(t); }}"))
}

/// `fst` and `fstp`: stores `ST(0)` to memory or to `ST(i)`, then pops for `fstp`.
fn store(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let pop = if instruction.mnemonic() == Mnemonic::Fstp {
        "fpop();"
    } else {
        ""
    };
    if instruction.op_kind(0) != OpKind::Memory {
        let index =
            stack_index(instruction, 0).ok_or_else(|| unsupported(address, "fst operand"))?;
        return Ok(format!("ST({index}) = ST(0); {pop}"));
    }
    let ea = effective_address(instruction);
    let statement = match memory_bytes(instruction) {
        4 => format!("stf32({ea}, ST(0));"),
        8 => format!("stf64({ea}, ST(0));"),
        10 => format!("st80({ea}, ST(0));"),
        other => return Err(unsupported(address, format!("x87 store of {other} bytes"))),
    };
    Ok(format!("{statement} {pop}"))
}

/// `fild`: pushes a 16, 32 or 64-bit integer from memory.
fn integer_load(instruction: &Instruction) -> String {
    let ea = effective_address(instruction);
    let value = match memory_bytes(instruction) {
        2 => format!("(double)(int16_t)rd16({ea})"),
        4 => format!("(double)(int32_t)rd32({ea})"),
        _ => format!("(double)(int64_t)((uint64_t)rd32({ea}) | ((uint64_t)rd32({ea} + 4) << 32))"),
    };
    format!("fpush({value});")
}

/// `fist` and `fistp`: stores `ST(0)` rounded to an integer, then pops for `fistp`.
fn integer_store(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let ea = effective_address(instruction);
    let pop = if matches!(instruction.mnemonic(), Mnemonic::Fistp | Mnemonic::Fisttp) {
        "fpop();"
    } else {
        ""
    };
    // `fisttp` truncates; the others round as the control word says. A value out of range, or a
    // NaN, stores the most negative integer.
    let rounded = if instruction.mnemonic() == Mnemonic::Fisttp {
        "trunc(ST(0))"
    } else {
        "fround_cw(ST(0))"
    };
    let statement = match memory_bytes(instruction) {
        2 => format!("wr16({ea}, (uint16_t)fint({rounded}, 16));"),
        4 => format!("wr32({ea}, (uint32_t)fint({rounded}, 32));"),
        8 => format!(
            "{{ int64_t v = fint({rounded}, 64); wr32({ea}, (uint32_t)v); wr32({ea} + 4, (uint32_t)((uint64_t)v >> 32)); }}"
        ),
        other => {
            return Err(unsupported(
                address,
                format!("x87 integer store of {other} bytes"),
            ));
        }
    };
    Ok(format!("{statement} {pop}"))
}

/// The arithmetic forms: `op m32/m64 ST(0)`, `op ST(0), ST(i)`, `op ST(i), ST(0)`, and the
/// popping forms. `reversed` is the `r` variant, where the stack operand is the subtrahend or the
/// divisor. The destination is written before a pop, so it refers to the stack as it was.
fn arithmetic(
    instruction: &Instruction,
    address: u32,
    operator: &str,
    reversed: bool,
) -> Result<String, LiftError> {
    let (destination, source, pop) = match instruction.op_count() {
        0 => (1, "ST(0)".to_string(), true),
        _ if instruction.op_kind(0) == OpKind::Memory => {
            (0, load_floating(instruction, address)?, false)
        }
        _ => {
            let destination = stack_index(instruction, 0)
                .ok_or_else(|| unsupported(address, "x87 arithmetic operand"))?;
            let source = stack_index(instruction, 1)
                .ok_or_else(|| unsupported(address, "x87 arithmetic operand"))?;
            let pop = is_popping(instruction.mnemonic());
            (destination, format!("ST({source})"), pop)
        }
    };
    let current = format!("ST({destination})");
    let (left, right) = if reversed {
        (source.as_str(), current.as_str())
    } else {
        (current.as_str(), source.as_str())
    };
    let pop = if pop { "fpop();" } else { "" };
    Ok(format!(
        "{{ double l = {left}, r = {right}; ST({destination}) = fres(l {operator} r, l, r); }} {pop}"
    ))
}

/// `fxch ST(i)`; without an operand it exchanges `ST(0)` and `ST(1)`.
fn exchange(instruction: &Instruction) -> String {
    // The decoder gives `fxch st(i)` as `st0, st(i)`: the partner is the last operand.
    let index = match instruction.op_count() {
        0 => 1,
        count => stack_index(instruction, count - 1).unwrap_or(1),
    };
    format!("{{ double t = ST(0); ST(0) = ST({index}); ST({index}) = t; }}")
}

/// `fcom`, `fcomp`, `fcompp`: compares `ST(0)` with `ST(1)`, `ST(i)` or memory, and sets the
/// status word's condition bits, then pops once for `fcomp` and twice for `fcompp`.
fn compare(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let source = match instruction.op_count() {
        0 => "ST(1)".to_string(),
        _ if instruction.op_kind(0) == OpKind::Memory => load_floating(instruction, address)?,
        // The decoder gives `fcom st(i)` as `st0, st(i)`: the other register is the last operand.
        count => format!(
            "ST({})",
            stack_index(instruction, count - 1)
                .ok_or_else(|| unsupported(address, "fcom operand"))?
        ),
    };
    let pops = match instruction.mnemonic() {
        Mnemonic::Fcomp | Mnemonic::Fucomp => "fpop();",
        Mnemonic::Fcompp | Mnemonic::Fucompp => "fpop(); fpop();",
        _ => "",
    };
    Ok(format!("fcompare(ST(0), {source}); {pops}"))
}

/// `fcomi` and `fucomi` and their popping forms: compares `ST(0)` with `ST(i)` and sets the
/// integer flags, ZF for equal and CF for less or unordered.
fn compare_flags(instruction: &Instruction) -> String {
    let index = stack_index(instruction, 1).unwrap_or(1);
    let pop = if matches!(instruction.mnemonic(), Mnemonic::Fcomip | Mnemonic::Fucomip) {
        "fpop();"
    } else {
        ""
    };
    format!("fcompare_flags(ST(0), ST({index})); {pop}")
}

/// The arithmetic mnemonics that pop the stack after they store.
fn is_popping(mnemonic: Mnemonic) -> bool {
    matches!(
        mnemonic,
        Mnemonic::Faddp
            | Mnemonic::Fsubp
            | Mnemonic::Fsubrp
            | Mnemonic::Fmulp
            | Mnemonic::Fdivp
            | Mnemonic::Fdivrp
    )
}

/// `fiadd`, `fisub`, `fimul`, `fidiv` and their reversed forms: `ST(0)` with a 16 or 32-bit
/// integer in memory.
fn integer_arithmetic(
    instruction: &Instruction,
    address: u32,
    operator: &str,
    reversed: bool,
) -> Result<String, LiftError> {
    let ea = effective_address(instruction);
    let source = match memory_bytes(instruction) {
        2 => format!("(double)(int16_t)rd16({ea})"),
        4 => format!("(double)(int32_t)rd32({ea})"),
        other => {
            return Err(unsupported(
                address,
                format!("x87 integer operand of {other} bytes"),
            ));
        }
    };
    let (left, right) = if reversed {
        (source.as_str(), "ST(0)")
    } else {
        ("ST(0)", source.as_str())
    };
    Ok(format!(
        "{{ double l = {left}, r = {right}; ST(0) = fres(l {operator} r, l, r); }}"
    ))
}

/// `ficom` and `ficomp`: compares `ST(0)` with a 16 or 32-bit integer in memory.
fn integer_compare(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let ea = effective_address(instruction);
    let source = match memory_bytes(instruction) {
        2 => format!("(double)(int16_t)rd16({ea})"),
        4 => format!("(double)(int32_t)rd32({ea})"),
        other => {
            return Err(unsupported(
                address,
                format!("x87 integer operand of {other} bytes"),
            ));
        }
    };
    let pop = if instruction.mnemonic() == Mnemonic::Ficomp {
        "fpop();"
    } else {
        ""
    };
    Ok(format!("fcompare(ST(0), {source}); {pop}"))
}
