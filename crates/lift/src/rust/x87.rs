//! The x87 floating point instructions: the register stack, loads, stores, arithmetic,
//! comparisons and the status word. The stack holds doubles (see the prelude).

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::{LiftError, effective_address, unsupported, write_operand};

/// `ST(0)` as a place.
const ST0: &str = "*st(c, 0)";

/// The Rust statements for an x87 instruction, or `None` when `instruction` is not one.
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
        Fnstsw => write_operand(address, instruction, 0, 16, "fstatus(c)")?,
        Fnclex | Fclex | Wait | Ffree => String::new(),
        Ffreep => "fpop(c);".to_string(),
        Fdecstp => "(*c).fpu_top = ((*c).fpu_top + 7) & 7;".to_string(),
        Fincstp => "(*c).fpu_top = ((*c).fpu_top + 1) & 7;".to_string(),
        // `fninit` sets the control word to 0x037F, 64-bit precision.
        Fninit | Finit => "(*c).fpu_top = 0; (*c).fpu_cc = 0; (*c).fpu_cwx = 0x0100;".to_string(),
        Fsin => format!("if ftrig_ok(c, {ST0}) {{ {ST0} = fres(c, sin({ST0}), {ST0}, {ST0}); }}"),
        Fcos => format!("if ftrig_ok(c, {ST0}) {{ {ST0} = fres(c, cos({ST0}), {ST0}, {ST0}); }}"),
        Fsincos => format!(
            "{{ let t: f64 = {ST0}; if ftrig_ok(c, t) {{ {ST0} = fres(c, sin(t), t, t); fpush(c, fres(c, cos(t), t, t)); }} }}"
        ),
        Fptan => format!(
            "{{ let t: f64 = {ST0}; if ftrig_ok(c, t) {{ {ST0} = fres(c, tan(t), t, t); fpush(c, 1.0); }} }}"
        ),
        Fpatan => {
            "*st(c, 1) = fres(c, atan2(*st(c, 1), *st(c, 0)), *st(c, 1), *st(c, 0)); fpop(c);"
                .to_string()
        }
        Fldpi => "fpush(c, core::f64::consts::PI);".to_string(),
        Fldl2e => "fpush(c, core::f64::consts::LOG2_E);".to_string(),
        Fldl2t => "fpush(c, core::f64::consts::LOG2_10);".to_string(),
        Fldlg2 => "fpush(c, core::f64::consts::LOG10_2);".to_string(),
        Fldln2 => "fpush(c, core::f64::consts::LN_2);".to_string(),
        Frndint => format!("{ST0} = fround_cw(c, {ST0});"),
        Fscale => format!("{ST0} = fscale_op({ST0}, *st(c, 1));"),
        Fyl2x => "*st(c, 1) = fres(c, *st(c, 1) * log2(*st(c, 0)), *st(c, 1), *st(c, 0)); fpop(c);"
            .to_string(),
        F2xm1 => format!("{ST0} = fres(c, expm1({ST0} * core::f64::consts::LN_2), {ST0}, {ST0});"),
        Fprem => "fprem_op(c, false);".to_string(),
        Fprem1 => "fprem_op(c, true);".to_string(),
        Fxam => format!("fxam(c, {ST0});"),
        Ficom | Ficomp => integer_compare(instruction, address)?,
        Fcmovb | Fcmove | Fcmovbe | Fcmovu | Fcmovnb | Fcmovne | Fcmovnbe | Fcmovnu => {
            let test = match instruction.mnemonic() {
                Fcmovb => "(*c).cf != 0",
                Fcmove => "(*c).zf != 0",
                Fcmovbe => "((*c).cf != 0 || (*c).zf != 0)",
                Fcmovu => "(*c).pf != 0",
                Fcmovnb => "(*c).cf == 0",
                Fcmovne => "(*c).zf == 0",
                Fcmovnbe => "((*c).cf == 0 && (*c).zf == 0)",
                _ => "(*c).pf == 0",
            };
            let index = stack_index(instruction, 1).unwrap_or(1);
            format!("if {test} {{ {ST0} = *st(c, {index}); }}")
        }
        Fiadd => integer_arithmetic(instruction, address, "+", false)?,
        Fisub => integer_arithmetic(instruction, address, "-", false)?,
        Fisubr => integer_arithmetic(instruction, address, "-", true)?,
        Fimul => integer_arithmetic(instruction, address, "*", false)?,
        Fidiv => integer_arithmetic(instruction, address, "/", false)?,
        Fidivr => integer_arithmetic(instruction, address, "/", true)?,
        Fldz => "fpush(c, 0.0);".to_string(),
        Fld1 => "fpush(c, 1.0);".to_string(),
        Fchs => format!("{ST0} = -{ST0};"),
        Fabs => format!("{ST0} = f64::from_bits(({ST0}).to_bits() & !(1u64 << 63));"),
        Fsqrt => format!("{ST0} = fres(c, sqrt({ST0}), {ST0}, {ST0});"),
        Ftst => format!("fcompare(c, {ST0}, 0.0);"),
        // Control words and waits change nothing modelled; the stack keeps no empty tags, so
        // freeing a register changes nothing the code reads.
        Fldcw => format!(
            "(*c).fpu_cwx = rd16({}) ^ 0x027F;",
            effective_address(instruction)
        ),
        Fnstcw => write_operand(address, instruction, 0, 16, "fpu_cw(c)")?,
        Fnstenv | Fstenv => format!("fstenv_op(c, {});", effective_address(instruction)),
        Fldenv => format!("fldenv_op(c, {});", effective_address(instruction)),
        Fnsave | Fsave => format!("fsave_op(c, {});", effective_address(instruction)),
        Frstor => format!("frstor_op(c, {});", effective_address(instruction)),
        Sahf => "{ let ah: u32 = ((*c).eax >> 8) & 0xFF; (*c).sf = ((ah >> 7) & 1) as u8; \
                 (*c).zf = ((ah >> 6) & 1) as u8; (*c).cf = (ah & 1) as u8; }"
            .to_string(),
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
        return Ok(format!("fpush(c, {value});"));
    }
    let index = stack_index(instruction, 0).ok_or_else(|| unsupported(address, "fld operand"))?;
    Ok(format!("{{ let t: f64 = *st(c, {index}); fpush(c, t); }}"))
}

/// `fst` and `fstp`: stores `ST(0)` to memory or to `ST(i)`, then pops for `fstp`.
fn store(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let pop = if instruction.mnemonic() == Mnemonic::Fstp {
        " fpop(c);"
    } else {
        ""
    };
    if instruction.op_kind(0) != OpKind::Memory {
        let index =
            stack_index(instruction, 0).ok_or_else(|| unsupported(address, "fst operand"))?;
        return Ok(format!("*st(c, {index}) = {ST0};{pop}"));
    }
    let ea = effective_address(instruction);
    let statement = match memory_bytes(instruction) {
        4 => format!("stf32({ea}, {ST0});"),
        8 => format!("stf64({ea}, {ST0});"),
        10 => format!("st80({ea}, {ST0});"),
        other => return Err(unsupported(address, format!("x87 store of {other} bytes"))),
    };
    Ok(format!("{statement}{pop}"))
}

/// A 16 or 32-bit integer in memory, as a double; `None` for another width.
fn integer_source(instruction: &Instruction) -> Option<String> {
    let ea = effective_address(instruction);
    match memory_bytes(instruction) {
        2 => Some(format!("f64::from(rd16({ea}) as u16 as i16)")),
        4 => Some(format!("f64::from(rd32({ea}) as i32)")),
        _ => None,
    }
}

/// `fild`: pushes a 16, 32 or 64-bit integer from memory.
fn integer_load(instruction: &Instruction) -> String {
    let value = integer_source(instruction).unwrap_or_else(|| {
        let ea = effective_address(instruction);
        format!(
            "((u64::from(rd32({ea})) | (u64::from(rd32({ea}.wrapping_add(4))) << 32)) as i64 as f64)"
        )
    });
    format!("fpush(c, {value});")
}

/// `fist` and `fistp`: stores `ST(0)` rounded to an integer, then pops for `fistp`.
fn integer_store(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let ea = effective_address(instruction);
    let pop = if matches!(instruction.mnemonic(), Mnemonic::Fistp | Mnemonic::Fisttp) {
        " fpop(c);"
    } else {
        ""
    };
    // `fisttp` truncates; the others round as the control word says. A value out of range, or a
    // NaN, stores the most negative integer.
    let rounded = if instruction.mnemonic() == Mnemonic::Fisttp {
        format!("trunc({ST0})")
    } else {
        format!("fround_cw(c, {ST0})")
    };
    let statement = match memory_bytes(instruction) {
        2 => format!("wr16({ea}, fint({rounded}, 16) as u32);"),
        4 => format!("wr32({ea}, fint({rounded}, 32) as u32);"),
        8 => format!(
            "{{ let v: i64 = fint({rounded}, 64); wr32({ea}, v as u32); wr32({ea}.wrapping_add(4), ((v as u64) >> 32) as u32); }}"
        ),
        other => {
            return Err(unsupported(
                address,
                format!("x87 integer store of {other} bytes"),
            ));
        }
    };
    Ok(format!("{statement}{pop}"))
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
        0 => (1, ST0.to_string(), true),
        _ if instruction.op_kind(0) == OpKind::Memory => {
            (0, load_floating(instruction, address)?, false)
        }
        _ => {
            let destination = stack_index(instruction, 0)
                .ok_or_else(|| unsupported(address, "x87 arithmetic operand"))?;
            let source = stack_index(instruction, 1)
                .ok_or_else(|| unsupported(address, "x87 arithmetic operand"))?;
            let pop = is_popping(instruction.mnemonic());
            (destination, format!("*st(c, {source})"), pop)
        }
    };
    let current = format!("*st(c, {destination})");
    let (left, right) = if reversed {
        (source.as_str(), current.as_str())
    } else {
        (current.as_str(), source.as_str())
    };
    let pop = if pop { " fpop(c);" } else { "" };
    Ok(format!(
        "{{ let l: f64 = {left}; let r: f64 = {right}; *st(c, {destination}) = fres(c, l {operator} r, l, r); }}{pop}"
    ))
}

/// `fxch ST(i)`; without an operand it exchanges `ST(0)` and `ST(1)`.
fn exchange(instruction: &Instruction) -> String {
    // The decoder gives `fxch st(i)` as `st0, st(i)`: the partner is the last operand.
    let index = match instruction.op_count() {
        0 => 1,
        count => stack_index(instruction, count - 1).unwrap_or(1),
    };
    format!("{{ let t: f64 = {ST0}; {ST0} = *st(c, {index}); *st(c, {index}) = t; }}")
}

/// `fcom`, `fcomp`, `fcompp`: compares `ST(0)` with `ST(1)`, `ST(i)` or memory, and sets the
/// status word's condition bits, then pops once for `fcomp` and twice for `fcompp`.
fn compare(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let source = match instruction.op_count() {
        0 => "*st(c, 1)".to_string(),
        _ if instruction.op_kind(0) == OpKind::Memory => load_floating(instruction, address)?,
        // The decoder gives `fcom st(i)` as `st0, st(i)`: the other register is the last operand.
        count => format!(
            "*st(c, {})",
            stack_index(instruction, count - 1)
                .ok_or_else(|| unsupported(address, "fcom operand"))?
        ),
    };
    let pops = match instruction.mnemonic() {
        Mnemonic::Fcomp | Mnemonic::Fucomp => " fpop(c);",
        Mnemonic::Fcompp | Mnemonic::Fucompp => " fpop(c); fpop(c);",
        _ => "",
    };
    Ok(format!("fcompare(c, {ST0}, {source});{pops}"))
}

/// `fcomi` and `fucomi` and their popping forms: compares `ST(0)` with `ST(i)` and sets the
/// integer flags, ZF for equal and CF for less or unordered.
fn compare_flags(instruction: &Instruction) -> String {
    let index = stack_index(instruction, 1).unwrap_or(1);
    let pop = if matches!(instruction.mnemonic(), Mnemonic::Fcomip | Mnemonic::Fucomip) {
        " fpop(c);"
    } else {
        ""
    };
    format!("fcompare_flags(c, {ST0}, *st(c, {index}));{pop}")
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
    let source = integer_source(instruction).ok_or_else(|| {
        unsupported(
            address,
            format!("x87 integer operand of {} bytes", memory_bytes(instruction)),
        )
    })?;
    let (left, right) = if reversed {
        (source.as_str(), ST0)
    } else {
        (ST0, source.as_str())
    };
    Ok(format!(
        "{{ let l: f64 = {left}; let r: f64 = {right}; {ST0} = fres(c, l {operator} r, l, r); }}"
    ))
}

/// `ficom` and `ficomp`: compares `ST(0)` with a 16 or 32-bit integer in memory.
fn integer_compare(instruction: &Instruction, address: u32) -> Result<String, LiftError> {
    let source = integer_source(instruction).ok_or_else(|| {
        unsupported(
            address,
            format!("x87 integer operand of {} bytes", memory_bytes(instruction)),
        )
    })?;
    let pop = if instruction.mnemonic() == Mnemonic::Ficomp {
        " fpop(c);"
    } else {
        ""
    };
    Ok(format!("fcompare(c, {ST0}, {source});{pop}"))
}
