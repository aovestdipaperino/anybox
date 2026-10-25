//! The C emitter: each function becomes a C function, compiled by clang and linked with the
//! runtime. [`PRELUDE`] starts every translation unit.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use iced_x86::{FlowControl, Instruction, Mnemonic, OpKind, Register};

use super::{
    Discovered, Image, LiftError, Root, addr, condition_code, find_setjmps, is_tail_call,
    unsupported, walk,
};

mod mmx;
mod sse;
mod x87;

/// The C definitions every lifted unit needs: the guest state and the helpers.
pub const PRELUDE: &str = r#"#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <setjmp.h>
#include <sys/mman.h>

/* One translation unit defines the globals; the others declare them. */
#ifdef LIFT_DEFINE_GLOBALS
#define LIFT_EXTERN
#else
#define LIFT_EXTERN extern
#endif
LIFT_EXTERN uint8_t *G;
/* With LIFT_TRACE_ON, every instruction reports its address and the registers before it ran,
   for comparison with the emulator. */
#ifdef LIFT_TRACE_ON
void anyxbox_trace(uint32_t at, uint32_t eax, uint32_t ecx, uint32_t edx, uint32_t ebx,
                   uint32_t esp, uint32_t ebp, uint32_t esi, uint32_t edi);
#define LIFT_TRACE(a) anyxbox_trace(a, R_EAX, R_ECX, R_EDX, R_EBX, R_ESP, R_EBP, R_ESI, R_EDI)
#else
#define LIFT_TRACE(a)
#endif
LIFT_EXTERN uint32_t FS_BASE;

/* Defined by the program that links the lifted code: prints what it knows of the guest's state. */
void lift_trap_hook(void);
static void lift_trap(uint32_t at, const char *what) {
    fprintf(stderr, "lifted code trapped at %#x: %s\n", at, what);
    lift_trap_hook();
    abort();
}
/* Time slices: the runtime sets the flag every millisecond, and loops check it on their backward
   branches, so a thread that spins without kernel calls still lets time pass and others run. A
   poll that calls the runtime records its site first: a thread hibernated there resumes at it. */
extern volatile int anyxbox_preempt_flag;
void anyxbox_preempt(void);
#define LIFT_POLL(s) do { if (__builtin_expect(anyxbox_preempt_flag, 0)) { LIFT_SITE(s, LIFT_K_POLL); anyxbox_preempt(); } } while (0)
/* A jump to another function's entry is a tail call, so a loop that runs through several functions
   keeps one host frame, even unoptimized. A function that holds a `setjmp` context calls and
   returns instead: its frame must outlive the jump. `s` is the site, for that call. */
#define LIFT_TAIL(f, s) __attribute__((musttail)) return f()
#define LIFT_CALL_RETURN(f, s) do { LIFT_IN(s, LIFT_K_RET, f()); return; } while (0)
/* The GPU's registers (0xFD000000-0xFDFFFFFF) are a device, not memory: the runtime serves them. */
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size);
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v);
#define LIFT_IS_MMIO(a) (__builtin_expect(((a) >> 24) == 0xFDu, 0))
static inline uint8_t rd8(uint32_t a) { if (LIFT_IS_MMIO(a)) return (uint8_t)anyxbox_mmio_read(a, 1); return G[a]; }
static inline uint16_t rd16(uint32_t a) { if (LIFT_IS_MMIO(a)) return (uint16_t)anyxbox_mmio_read(a, 2); uint16_t v; memcpy(&v, G + a, 2); return v; }
static inline uint32_t rd32(uint32_t a) { if (LIFT_IS_MMIO(a)) return anyxbox_mmio_read(a, 4); uint32_t v; memcpy(&v, G + a, 4); return v; }
static inline void wr8(uint32_t a, uint8_t v) { if (LIFT_IS_MMIO(a)) { anyxbox_mmio_write(a, 1, v); return; } G[a] = v; }
static inline void wr16(uint32_t a, uint16_t v) { if (LIFT_IS_MMIO(a)) { anyxbox_mmio_write(a, 2, v); return; } memcpy(G + a, &v, 2); }
static inline void wr32(uint32_t a, uint32_t v) { if (LIFT_IS_MMIO(a)) { anyxbox_mmio_write(a, 4, v); return; } memcpy(G + a, &v, 4); }
/* x87 floating point. The register stack holds doubles: 64-bit precision where the title
   keeps 80-bit extended values, which matches for values that fit in a double. */
/* SSE: the eight XMM registers. */
typedef union { float f[4]; uint32_t u[4]; uint64_t q[2]; } xmm_t;
/* Where each host frame of a guest thread is: the lifted function running in it, and the site it
   was last at (a call, a call that returns as soon as its callee does, or a loop's poll). A
   hibernated thread is resumed by calling down these frames again. */
typedef struct { uint32_t fn, kind, site; } lift_frame;
#define LIFT_K_CALL 0u
#define LIFT_K_RET 1u
#define LIFT_K_POLL 2u
/* The guest CPU, per host thread: each guest thread runs on its own host thread. It is one
   thread-local block rather than a variable per register, so a function looks up its address
   once (`_tlv_get_addr` on macOS) instead of once per register it touches. */
typedef struct {
    uint32_t eax, ecx, edx, ebx, esp, ebp, esi, edi;
    uint8_t cf, zf, sf, of, pf, df;
    uint32_t fpu_top, fpu_cc;
    /* The x87 control word and MXCSR, each kept as its difference from the value the console's
       kernel starts a thread with (0x027F: 53-bit precision, every exception masked; 0x1F80), so
       a zeroed CPU is a new thread's. */
    uint32_t fpu_cwx, mxcsr_x;
    double fpu[8];
    xmm_t xmm[8];
    uint64_t mm[8];
    /* The frame records and the depth of the current frame; while resuming, the number of frames
       to resume down. */
    lift_frame *frames;
    uint32_t depth, resuming, resume_depth;
} lift_cpu_t;
LIFT_EXTERN __thread lift_cpu_t lift_cpu;
#define R_EAX (lift_cpu.eax)
#define R_ECX (lift_cpu.ecx)
#define R_EDX (lift_cpu.edx)
#define R_EBX (lift_cpu.ebx)
#define R_ESP (lift_cpu.esp)
#define R_EBP (lift_cpu.ebp)
#define R_ESI (lift_cpu.esi)
#define R_EDI (lift_cpu.edi)
#define F_CF (lift_cpu.cf)
#define F_ZF (lift_cpu.zf)
#define F_SF (lift_cpu.sf)
#define F_OF (lift_cpu.of)
#define F_PF (lift_cpu.pf)
#define F_DF (lift_cpu.df)
#define fpu_top (lift_cpu.fpu_top)
#define fpu_cc (lift_cpu.fpu_cc)
#define fpu_cwx (lift_cpu.fpu_cwx)
#define mxcsr_x (lift_cpu.mxcsr_x)
#define fpu (lift_cpu.fpu)
#define XMM (lift_cpu.xmm)
#define MM (lift_cpu.mm)
#define LIFT_ENTER(e) (lift_cpu.frames[lift_cpu.depth].fn = (e))
#define LIFT_SITE(s, k) do { lift_frame *lift_f = &lift_cpu.frames[lift_cpu.depth]; lift_f->site = (s); lift_f->kind = (k); } while (0)
#define LIFT_IN(s, k, call) do { LIFT_SITE(s, k); lift_cpu.depth++; call; lift_cpu.depth--; } while (0)
#define LIFT_RESUMING() __builtin_expect(lift_cpu.resuming, 0)
#define LIFT_KEY(s, k) ((uint64_t)(s) << 2 | (k))
/* Room for this many frames, with a guard page after them, so a runaway depth faults. */
#define LIFT_FRAMES 131072u
#define LIFT_FRAMES_LEN ((size_t)LIFT_FRAMES * sizeof(lift_frame))
#define LIFT_FRAMES_GUARD ((size_t)16384)
static inline void lift_frames_init(void) {
    if (lift_cpu.frames) return;
    size_t len = LIFT_FRAMES_LEN, page = LIFT_FRAMES_GUARD;
    uint8_t *at = mmap(NULL, len + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (at == MAP_FAILED) {
        fprintf(stderr, "no memory for the frame records\n");
        abort();
    }
    mprotect(at + len, page, PROT_NONE);
    lift_cpu.frames = (lift_frame *)at;
}
/* Gives back frame records `lift_frames_init` made, with their guard page. */
static inline void lift_frames_free(lift_frame *frames) {
    if (frames) munmap(frames, LIFT_FRAMES_LEN + LIFT_FRAMES_GUARD);
}
/* The title's `setjmp` and `longjmp` (MSVC's). A lifted function that calls `setjmp` keeps a host
   `sigsetjmp` context, and `longjmp` ends with a jump to the saved return address in the middle of
   that function, which `rt_call` turns into a host `siglongjmp` to the context. A context lives
   until its function returns (the cleanup attribute); the contexts of the frames a `longjmp`
   skips are dropped by resetting the top of the stack. */
typedef struct { uint32_t buf, resume, depth, site; int active; sigjmp_buf env; } lift_jmp;
LIFT_EXTERN __thread lift_jmp *lift_jmp_stack[256];
LIFT_EXTERN __thread uint32_t lift_jmp_top;
/* The contexts a resumed thread registers again, as a hibernation saved them: the first four
   words of each `lift_jmp`. */
typedef struct { uint32_t buf, resume, depth, site; } lift_jmp_saved;
LIFT_EXTERN __thread lift_jmp_saved lift_jmp_restore[256];
LIFT_EXTERN __thread uint32_t lift_jmp_restore_count;
static inline void lift_jmp_register(lift_jmp *j, uint32_t buf, uint32_t resume, uint32_t site) {
    j->buf = buf;
    j->resume = resume;
    j->depth = lift_cpu.depth;
    j->site = site;
    if (!j->active) {
        if (lift_jmp_top >= 256) lift_trap(resume, "setjmp nested too deeply");
        lift_jmp_stack[lift_jmp_top++] = j;
        j->active = 1;
    }
}
/* While resuming a frame: the saved context of `setjmp` site `site` in it, if it had one. */
static int lift_resume_jmp(uint32_t site, uint32_t *buf, uint32_t *resume) {
    for (uint32_t i = 0; i < lift_jmp_restore_count; i++) {
        lift_jmp_saved *s = &lift_jmp_restore[i];
        if (s->depth == lift_cpu.depth && s->site == site) {
            *buf = s->buf;
            *resume = s->resume;
            return 1;
        }
    }
    return 0;
}
static void lift_jmp_exit(lift_jmp *j) {
    if (!j->active) return;
    for (uint32_t i = lift_jmp_top; i > 0; i--) {
        if (lift_jmp_stack[i - 1] == j) {
            for (; i < lift_jmp_top; i++) lift_jmp_stack[i - 1] = lift_jmp_stack[i];
            lift_jmp_top--;
            break;
        }
    }
    j->active = 0;
}
/* Jumps to the `setjmp` that saved the address `target` (and, first choice, the buffer in EDX,
   which is what `longjmp` jumps through), if there is one; returns when there is none. */
static void lift_longjmp(uint32_t target) {
    for (int exact = 1; exact >= 0; exact--) {
        for (uint32_t i = lift_jmp_top; i > 0; i--) {
            lift_jmp *j = lift_jmp_stack[i - 1];
            if (j->resume == target && (!exact || j->buf == R_EDX)) {
                lift_jmp_top = i;
                lift_cpu.depth = j->depth;
                siglongjmp(j->env, 1);
            }
        }
    }
}
#define ST(i) fpu[(fpu_top + (i)) & 7]
static void fpush(double v) { fpu_top = (fpu_top + 7) & 7; fpu[fpu_top] = v; }
static void fpop(void) { fpu_top = (fpu_top + 1) & 7; }
static inline uint32_t fpu_cw(void) { return fpu_cwx ^ 0x027Fu; }
static inline uint32_t mxcsr_get(void) { return mxcsr_x ^ 0x1F80u; }
/* The result of an x87 operation that produced a NaN from operands `a` and `b`: a NaN operand
   made quiet (the quiet one beats a signaling one, the larger significand beats the smaller),
   or the negative "real indefinite" when neither operand was a NaN. */
static double fnan2(double a, double b) {
    const uint64_t quiet = 1ull << 51, frac = (1ull << 52) - 1;
    uint64_t ua, ub, u;
    memcpy(&ua, &a, 8); memcpy(&ub, &b, 8);
    int an = a != a, bn = b != b;
    if (an && bn) {
        int as = !(ua & quiet), bs = !(ub & quiet);
        if (as != bs) u = as ? ub : ua;
        else if ((ua & frac) == (ub & frac)) u = (ua >> 63) ? ub : ua;
        else u = (ua & frac) > (ub & frac) ? ua : ub;
    } else if (an || bn) {
        u = an ? ua : ub;
    } else {
        u = 0xFFF8000000000000ull;
    }
    u |= quiet;
    double r; memcpy(&r, &u, 8); return r;
}
/* Rounds to the control word's precision: 24 bits when it says single (the exponent range stays
   wide, so values outside a float's normal range are left alone); doubles are the stack's own. */
static inline double fprec(double v) {
    if (__builtin_expect((fpu_cwx & 0x300u) != 0x200u, 1)) return v;
    if (fabs(v) >= 1.17549435e-38 && fabs(v) <= 3.40282347e38) return (double)(float)v;
    return v;
}
/* The result of an arithmetic operation on `a` and `b`, as the x87 delivers it. */
static inline double fres(double r, double a, double b) {
    return __builtin_expect(r == r, 1) ? fprec(r) : fnan2(a, b);
}
/* Rounds to an integer as the control word's rounding mode says. */
static inline double fround_cw(double v) {
    switch ((fpu_cw() >> 10) & 3) {
    case 0: return nearbyint(v);
    case 1: return floor(v);
    case 2: return ceil(v);
    default: return trunc(v);
    }
}
/* An integral value as a signed integer of `bits` bits; out of range or a NaN gives the "integer
   indefinite", the most negative value, as the x87 and SSE conversions do. */
static inline int64_t fint(double r, int bits) {
    double lim = ldexp(1.0, bits - 1);
    if (!(r >= -lim && r < lim)) return (int64_t)(~0ull << (bits - 1));
    return (int64_t)r;
}
static inline uint32_t cvt_ss2si(float v, int truncate) {
    double r;
    if (truncate) r = trunc((double)v);
    else switch ((mxcsr_get() >> 13) & 3) {
        case 0: r = nearbyint((double)v); break;
        case 1: r = floor((double)v); break;
        case 2: r = ceil((double)v); break;
        default: r = trunc((double)v); break;
    }
    return (uint32_t)(int32_t)fint(r, 32);
}
/* The result of a single-precision SSE operation that produced a NaN from `a` and `b`: the first
   operand if it is a NaN, else the second, made quiet; the negative default NaN otherwise. */
static float snan2(float a, float b) {
    uint32_t u;
    if (a != a) memcpy(&u, &a, 4);
    else if (b != b) memcpy(&u, &b, 4);
    else u = 0xFFC00000u;
    u |= 0x00400000u;
    float r; memcpy(&r, &u, 4); return r;
}
static inline float sres(float r, float a, float b) {
    return __builtin_expect(r == r, 1) ? r : snan2(a, b);
}
static void fxam(double v) {
    uint32_t c = isnan(v) ? 0x0100u : isinf(v) ? 0x0500u : v == 0.0 ? 0x4000u : 0x0400u;
    fpu_cc = c | (signbit(v) ? 0x0200u : 0u);
}
/* FPREM and FPREM1: the remainder, with the quotient's low three bits in C0, C3 and C1, and C2
   clear (the reduction is always complete). */
static void fprem_op(int nearest) {
    double a = ST(0), b = ST(1), r;
    unsigned q = 0;
    if (a != a || b != b || isinf(a) || b == 0.0) { ST(0) = fres(NAN, a, b); fpu_cc = 0; return; }
    if (nearest) { int qi; r = remquo(a, b, &qi); q = (unsigned)abs(qi) & 7u; }
    else {
        r = fmod(a, b);
        double n = fabs((a - r) / b);
        q = n < 9.2e18 ? (unsigned)((uint64_t)n & 7u) : 0u;
    }
    ST(0) = r;
    fpu_cc = ((q & 4u) ? 0x0100u : 0u) | ((q & 2u) ? 0x4000u : 0u) | ((q & 1u) ? 0x0200u : 0u);
}
static double fscale_op(double x, double s) {
    if (x != x || s != s) return fnan2(x, s);
    if (isinf(s)) return s > 0 ? (x == 0.0 ? fnan2(x, x) : x * INFINITY) : (isinf(x) ? fnan2(x, x) : x * 0.0);
    if (s > 100000.0) s = 100000.0;
    if (s < -100000.0) s = -100000.0;
    return ldexp(x, (int)trunc(s));
}
/* The sine, cosine and tangent are defined below 2^63 in magnitude; beyond it C2 is set and the
   stack is left alone. Returns whether the argument is in range. */
static int ftrig_ok(double v) {
    if (fabs(v) >= 9.223372036854775808e18) { fpu_cc |= 0x0400u; return 0; }
    fpu_cc &= ~0x0400u; return 1;
}
static inline xmm_t ldx128(uint32_t a) { xmm_t v; memcpy(&v, G + a, 16); return v; }
static inline void stx128(uint32_t a, xmm_t v) { memcpy(G + a, &v, 16); }
static inline float ldf32x(uint32_t a) { float f; memcpy(&f, G + a, 4); return f; }
static inline void stf32x(uint32_t a, float f) { memcpy(G + a, &f, 4); }
static inline uint64_t ldq64(uint32_t a) { uint64_t v; memcpy(&v, G + a, 8); return v; }
static inline void stq64(uint32_t a, uint64_t v) { memcpy(G + a, &v, 8); }
/* MMX: eight 64-bit registers, and the packed operations on them, one helper per lane width. */
#define MMX_LANES(T) (8 / sizeof(T))
#define MMX_BIN(name, T, W, expr) static inline uint64_t name(uint64_t a, uint64_t b) { \
    union { uint64_t q; T e[MMX_LANES(T)]; } x = {a}, y = {b}, r; \
    for (unsigned i = 0; i < MMX_LANES(T); i++) { W p = x.e[i], q = y.e[i]; r.e[i] = (T)(expr); } \
    return r.q; }
static inline int64_t mmx_clamp(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : v > hi ? hi : v; }
MMX_BIN(mmx_paddb, uint8_t, uint32_t, p + q)
MMX_BIN(mmx_paddw, uint16_t, uint32_t, p + q)
MMX_BIN(mmx_paddd, uint32_t, uint32_t, p + q)
MMX_BIN(mmx_paddq, uint64_t, uint64_t, p + q)
MMX_BIN(mmx_paddsb, int8_t, int32_t, mmx_clamp(p + q, -128, 127))
MMX_BIN(mmx_paddsw, int16_t, int32_t, mmx_clamp(p + q, -32768, 32767))
MMX_BIN(mmx_paddusb, uint8_t, int32_t, mmx_clamp(p + q, 0, 255))
MMX_BIN(mmx_paddusw, uint16_t, int32_t, mmx_clamp(p + q, 0, 65535))
MMX_BIN(mmx_psubb, uint8_t, uint32_t, p - q)
MMX_BIN(mmx_psubw, uint16_t, uint32_t, p - q)
MMX_BIN(mmx_psubd, uint32_t, uint32_t, p - q)
MMX_BIN(mmx_psubq, uint64_t, uint64_t, p - q)
MMX_BIN(mmx_psubsb, int8_t, int32_t, mmx_clamp(p - q, -128, 127))
MMX_BIN(mmx_psubsw, int16_t, int32_t, mmx_clamp(p - q, -32768, 32767))
MMX_BIN(mmx_psubusb, uint8_t, int32_t, mmx_clamp(p - q, 0, 255))
MMX_BIN(mmx_psubusw, uint16_t, int32_t, mmx_clamp(p - q, 0, 65535))
MMX_BIN(mmx_pmullw, uint16_t, int32_t, (int16_t)p * (int16_t)q)
MMX_BIN(mmx_pmulhw, int16_t, int32_t, (p * q) >> 16)
MMX_BIN(mmx_pmulhuw, uint16_t, uint32_t, (p * q) >> 16)
MMX_BIN(mmx_pavgb, uint8_t, uint32_t, (p + q + 1) >> 1)
MMX_BIN(mmx_pavgw, uint16_t, uint32_t, (p + q + 1) >> 1)
MMX_BIN(mmx_pminub, uint8_t, uint32_t, p < q ? p : q)
MMX_BIN(mmx_pmaxub, uint8_t, uint32_t, p > q ? p : q)
MMX_BIN(mmx_pminsw, int16_t, int32_t, p < q ? p : q)
MMX_BIN(mmx_pmaxsw, int16_t, int32_t, p > q ? p : q)
MMX_BIN(mmx_pcmpeqb, uint8_t, uint32_t, p == q ? 0xFF : 0)
MMX_BIN(mmx_pcmpeqw, uint16_t, uint32_t, p == q ? 0xFFFF : 0)
MMX_BIN(mmx_pcmpeqd, uint32_t, uint32_t, p == q ? 0xFFFFFFFFu : 0)
MMX_BIN(mmx_pcmpgtb, int8_t, int32_t, p > q ? -1 : 0)
MMX_BIN(mmx_pcmpgtw, int16_t, int32_t, p > q ? -1 : 0)
MMX_BIN(mmx_pcmpgtd, int32_t, int64_t, p > q ? -1 : 0)
static inline uint64_t mmx_pand(uint64_t a, uint64_t b) { return a & b; }
static inline uint64_t mmx_pandn(uint64_t a, uint64_t b) { return ~a & b; }
static inline uint64_t mmx_por(uint64_t a, uint64_t b) { return a | b; }
static inline uint64_t mmx_pxor(uint64_t a, uint64_t b) { return a ^ b; }
static inline uint64_t mmx_pmaddwd(uint64_t a, uint64_t b) {
    union { uint64_t q; int16_t w[4]; } x = {a}, y = {b};
    union { uint64_t q; int32_t d[2]; } r;
    for (int i = 0; i < 2; i++) r.d[i] = (int32_t)((int64_t)x.w[2*i] * y.w[2*i] + (int64_t)x.w[2*i+1] * y.w[2*i+1]);
    return r.q;
}
static inline uint64_t mmx_psadbw(uint64_t a, uint64_t b) {
    union { uint64_t q; uint8_t e[8]; } x = {a}, y = {b};
    uint64_t sum = 0;
    for (int i = 0; i < 8; i++) sum += x.e[i] > y.e[i] ? x.e[i] - y.e[i] : y.e[i] - x.e[i];
    return sum;
}
#define MMX_UNPACK(name, T, half) static inline uint64_t name(uint64_t a, uint64_t b) { \
    union { uint64_t q; T e[MMX_LANES(T)]; } x = {a}, y = {b}, r; \
    unsigned n = MMX_LANES(T) / 2; \
    for (unsigned i = 0; i < n; i++) { r.e[2*i] = x.e[half * n + i]; r.e[2*i+1] = y.e[half * n + i]; } \
    return r.q; }
MMX_UNPACK(mmx_punpcklbw, uint8_t, 0)
MMX_UNPACK(mmx_punpcklwd, uint16_t, 0)
MMX_UNPACK(mmx_punpckldq, uint32_t, 0)
MMX_UNPACK(mmx_punpckhbw, uint8_t, 1)
MMX_UNPACK(mmx_punpckhwd, uint16_t, 1)
MMX_UNPACK(mmx_punpckhdq, uint32_t, 1)
static inline uint64_t mmx_packuswb(uint64_t a, uint64_t b) {
    union { uint64_t q; int16_t w[4]; } x = {a}, y = {b};
    union { uint64_t q; uint8_t e[8]; } r;
    for (int i = 0; i < 4; i++) { r.e[i] = (uint8_t)mmx_clamp(x.w[i], 0, 255); r.e[i+4] = (uint8_t)mmx_clamp(y.w[i], 0, 255); }
    return r.q;
}
static inline uint64_t mmx_packsswb(uint64_t a, uint64_t b) {
    union { uint64_t q; int16_t w[4]; } x = {a}, y = {b};
    union { uint64_t q; int8_t e[8]; } r;
    for (int i = 0; i < 4; i++) { r.e[i] = (int8_t)mmx_clamp(x.w[i], -128, 127); r.e[i+4] = (int8_t)mmx_clamp(y.w[i], -128, 127); }
    return r.q;
}
static inline uint64_t mmx_packssdw(uint64_t a, uint64_t b) {
    union { uint64_t q; int32_t d[2]; } x = {a}, y = {b};
    union { uint64_t q; int16_t w[4]; } r;
    for (int i = 0; i < 2; i++) { r.w[i] = (int16_t)mmx_clamp(x.d[i], -32768, 32767); r.w[i+2] = (int16_t)mmx_clamp(y.d[i], -32768, 32767); }
    return r.q;
}
#define MMX_SHIFT(name, T, S, op) static inline uint64_t name(uint64_t a, uint64_t count) { \
    union { uint64_t q; T e[MMX_LANES(T)]; } x = {a}; \
    unsigned bits = 8 * sizeof(T); \
    for (unsigned i = 0; i < MMX_LANES(T); i++) { \
        if (count >= bits) x.e[i] = op == 2 ? (T)((S)x.e[i] >> (bits - 1)) : 0; \
        else x.e[i] = op == 0 ? (T)(x.e[i] << count) : op == 1 ? (T)(x.e[i] >> count) : (T)((S)x.e[i] >> count); } \
    return x.q; }
MMX_SHIFT(mmx_psllw, uint16_t, int16_t, 0)
MMX_SHIFT(mmx_pslld, uint32_t, int32_t, 0)
MMX_SHIFT(mmx_psllq, uint64_t, int64_t, 0)
MMX_SHIFT(mmx_psrlw, uint16_t, int16_t, 1)
MMX_SHIFT(mmx_psrld, uint32_t, int32_t, 1)
MMX_SHIFT(mmx_psrlq, uint64_t, int64_t, 1)
MMX_SHIFT(mmx_psraw, uint16_t, int16_t, 2)
MMX_SHIFT(mmx_psrad, uint32_t, int32_t, 2)
static inline uint64_t mmx_pshufw(uint64_t a, uint32_t order) {
    union { uint64_t q; uint16_t w[4]; } x = {a}, r;
    for (int i = 0; i < 4; i++) r.w[i] = x.w[(order >> (2 * i)) & 3];
    return r.q;
}
static double ldf32(uint32_t a) { uint32_t b = rd32(a); float f; memcpy(&f, &b, 4); return f; }
static double ldf64(uint32_t a) {
    uint64_t b = (uint64_t)rd32(a) | ((uint64_t)rd32(a + 4) << 32);
    double d; memcpy(&d, &b, 8); return d;
}
static void stf32(uint32_t a, double v) { float f = (float)v; uint32_t b; memcpy(&b, &f, 4); wr32(a, b); }
static void stf64(uint32_t a, double v) {
    uint64_t b; memcpy(&b, &v, 8);
    wr32(a, (uint32_t)b); wr32(a + 4, (uint32_t)(b >> 32));
}
static double ld80(uint32_t a) {
    uint64_t m = (uint64_t)rd32(a) | ((uint64_t)rd32(a + 4) << 32);
    uint32_t se = rd16(a + 8);
    int sign = (se >> 15) & 1, e = se & 0x7FFF;
    if (e == 0 && m == 0) return sign ? -0.0 : 0.0;
    if (e == 0x7FFF) {
        if ((m << 1) == 0) return sign ? -INFINITY : INFINITY;
        uint64_t u = ((uint64_t)sign << 63) | (0x7FFull << 52) | (1ull << 51) | ((m >> 11) & ((1ull << 51) - 1));
        double d; memcpy(&d, &u, 8); return d;
    }
    double v = ldexp((double)m, e - 16383 - 63);
    return sign ? -v : v;
}
static void st80(uint32_t a, double v) {
    int sign = signbit(v) != 0;
    double m_abs = fabs(v);
    uint64_t m = 0; uint32_t e = 0;
    if (m_abs == INFINITY) { m = 1ull << 63; e = 0x7FFF; }
    else if (m_abs != 0 && m_abs == m_abs) {
        int exp2; double f = frexp(m_abs, &exp2);
        m = (uint64_t)ldexp(f, 64);
        e = (uint32_t)(exp2 - 1 + 16383);
    } else if (m_abs != m_abs) {
        uint64_t u; memcpy(&u, &v, 8);
        m = (3ull << 62) | ((u & ((1ull << 51) - 1)) << 11); e = 0x7FFF;
    }
    wr32(a, (uint32_t)m); wr32(a + 4, (uint32_t)(m >> 32));
    wr16(a + 8, (uint16_t)((sign << 15) | e));
}
static void fcompare(double a, double b) {
    fpu_cc = (a != a || b != b) ? 0x4500u : (a < b ? 0x0100u : (a == b ? 0x4000u : 0u));
}
static void fcompare_flags(double a, double b) {
    fcompare(a, b);
    F_ZF = (fpu_cc & 0x4000u) != 0;
    F_CF = (fpu_cc & 0x0100u) != 0;
    F_SF = 0; F_OF = 0;
    F_PF = (fpu_cc & 0x4500u) == 0x4500u;
}
static uint32_t fstatus(void) { return (fpu_top << 11) | fpu_cc; }
/* `fnstenv` and `fnsave`'s 28-byte environment, in the 32-bit layout: the control, status and tag
   words, the tags all valid as the stack keeps none, and no last instruction or operand. */
static void fstenv_op(uint32_t a) {
    wr32(a, 0xFFFF0000u | fpu_cw()); wr32(a + 4, 0xFFFF0000u | fstatus()); wr32(a + 8, 0xFFFF0000u);
    for (uint32_t i = 12; i < 28; i += 4) wr32(a + i, 0);
}
static void fldenv_op(uint32_t a) {
    uint32_t sw = rd16(a + 4);
    fpu_cwx = (uint32_t)rd16(a) ^ 0x027Fu; fpu_top = (sw >> 11) & 7; fpu_cc = sw & ~0x3800u;
}
/* `fnsave` stores the environment and ST(0) to ST(7), then initializes as `fninit` does;
   `frstor` loads them back. */
static void fsave_op(uint32_t a) {
    fstenv_op(a);
    for (uint32_t i = 0; i < 8; i++) st80(a + 28 + 10 * i, ST(i));
    fpu_top = 0; fpu_cc = 0; fpu_cwx = 0x0100;
}
static void frstor_op(uint32_t a) {
    fldenv_op(a);
    for (uint32_t i = 0; i < 8; i++) ST(i) = ld80(a + 28 + 10 * i);
}
static inline void push32(uint32_t v) { R_ESP -= 4; wr32(R_ESP, v); }
static inline uint32_t pop32(void) { uint32_t v = rd32(R_ESP); R_ESP += 4; return v; }

/* A string step: forwards, or backwards when the direction flag is set. */
static inline uint32_t dstep(uint32_t n) { return F_DF ? (uint32_t)(-(int32_t)n) : n; }
/* The time-stamp counter: host time at the Xbox CPU's 733 MHz, plus the runtime's offset, which a
   resume sets so the counter goes on from where the hibernated title left it. */
extern volatile int64_t anyxbox_tsc_offset;
static uint64_t lift_rdtsc(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t host = ((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec) * 733u / 1000u;
    return host + (uint64_t)anyxbox_tsc_offset;
}
static inline uint32_t mask_of(uint32_t bits) { return bits == 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u); }
static inline uint32_t sign_of(uint32_t bits) { return 1u << (bits - 1); }
static inline void set_zs(uint32_t r, uint32_t bits) {
    F_ZF = (r & mask_of(bits)) == 0;
    F_SF = (r & sign_of(bits)) != 0;
    F_PF = !__builtin_parity(r & 0xFF);
}
static uint32_t alu_add(uint32_t a, uint32_t b, uint32_t bits) {
    uint32_t m = mask_of(bits);
    a &= m; b &= m;
    uint64_t wide = (uint64_t)a + (uint64_t)b;
    uint32_t r = (uint32_t)wide & m;
    F_CF = (wide >> bits) & 1;
    F_OF = (((a ^ r) & (b ^ r)) & sign_of(bits)) != 0;
    set_zs(r, bits);
    return r;
}
static uint32_t alu_sub(uint32_t a, uint32_t b, uint32_t bits) {
    uint32_t m = mask_of(bits);
    a &= m; b &= m;
    uint32_t r = (a - b) & m;
    F_CF = a < b;
    F_OF = (((a ^ b) & (a ^ r)) & sign_of(bits)) != 0;
    set_zs(r, bits);
    return r;
}
static uint32_t alu_logic(uint32_t r, uint32_t bits) {
    r &= mask_of(bits);
    F_CF = 0; F_OF = 0;
    set_zs(r, bits);
    return r;
}
static uint32_t alu_inc(uint32_t a, uint32_t bits) {
    uint8_t cf = F_CF;
    uint32_t r = alu_add(a, 1, bits);
    F_CF = cf;
    return r;
}
static uint32_t alu_dec(uint32_t a, uint32_t bits) {
    uint8_t cf = F_CF;
    uint32_t r = alu_sub(a, 1, bits);
    F_CF = cf;
    return r;
}
static uint32_t alu_neg(uint32_t a, uint32_t bits) {
    uint32_t m = mask_of(bits);
    uint32_t r = alu_sub(0, a, bits);
    F_CF = (a & m) != 0;
    return r & m;
}
static uint32_t alu_adc(uint32_t a, uint32_t b, uint32_t bits) {
    uint32_t m = mask_of(bits);
    a &= m; b &= m;
    uint64_t wide = (uint64_t)a + (uint64_t)b + F_CF;
    uint32_t r = (uint32_t)wide & m;
    F_CF = (wide >> bits) & 1;
    F_OF = (((a ^ r) & (b ^ r)) & sign_of(bits)) != 0;
    set_zs(r, bits);
    return r;
}
static uint32_t alu_sbb(uint32_t a, uint32_t b, uint32_t bits) {
    uint32_t m = mask_of(bits);
    a &= m; b &= m;
    uint32_t borrow = F_CF;
    uint32_t r = (a - b - borrow) & m;
    F_CF = (uint64_t)a < (uint64_t)b + borrow;
    F_OF = (((a ^ b) & (a ^ r)) & sign_of(bits)) != 0;
    set_zs(r, bits);
    return r;
}
static uint32_t alu_bsf(uint32_t src, uint32_t dst) {
    F_ZF = src == 0;
    return src == 0 ? dst : (uint32_t)__builtin_ctz(src);
}
static uint32_t alu_bsr(uint32_t src, uint32_t dst) {
    F_ZF = src == 0;
    return src == 0 ? dst : (uint32_t)(31 - __builtin_clz(src));
}
static uint32_t alu_rol(uint32_t a, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    a &= m;
    if (count == 0) return a;
    uint32_t n = count % bits;
    uint32_t r = n ? (((a << n) | (a >> (bits - n))) & m) : a;
    F_CF = r & 1;
    if (count == 1) F_OF = ((r >> (bits - 1)) & 1) ^ F_CF;
    return r;
}
static uint32_t alu_ror(uint32_t a, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    a &= m;
    if (count == 0) return a;
    uint32_t n = count % bits;
    uint32_t r = n ? (((a >> n) | (a << (bits - n))) & m) : a;
    F_CF = (r >> (bits - 1)) & 1;
    if (count == 1) F_OF = ((r >> (bits - 1)) ^ (r >> (bits - 2))) & 1;
    return r;
}
static uint32_t alu_shld(uint32_t dst, uint32_t src, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    dst &= m; src &= m;
    if (count == 0 || count > bits) return dst;
    F_CF = (dst >> (bits - count)) & 1;
    uint32_t r = ((dst << count) | (count == bits ? 0 : (src >> (bits - count)))) & m;
    if (count == 1) F_OF = ((r >> (bits - 1)) & 1) ^ F_CF;
    set_zs(r, bits);
    return r;
}
static uint32_t alu_shrd(uint32_t dst, uint32_t src, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    dst &= m; src &= m;
    if (count == 0 || count > bits) return dst;
    F_CF = (dst >> (count - 1)) & 1;
    uint32_t r = ((dst >> count) | (count == bits ? 0 : (src << (bits - count)))) & m;
    if (count == 1) F_OF = ((r >> (bits - 1)) ^ (dst >> (bits - 1))) & 1;
    set_zs(r, bits);
    return r;
}
static uint32_t eflags_get(void) {
    return 2u | F_CF | ((uint32_t)F_PF << 2) | ((uint32_t)F_ZF << 6) | ((uint32_t)F_SF << 7)
        | 0x200u | ((uint32_t)F_DF << 10) | ((uint32_t)F_OF << 11);
}
static void eflags_set(uint32_t f) {
    F_CF = f & 1; F_PF = (f >> 2) & 1; F_ZF = (f >> 6) & 1; F_SF = (f >> 7) & 1;
    F_DF = (f >> 10) & 1; F_OF = (f >> 11) & 1;
}
static uint32_t alu_rcl(uint32_t a, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    a &= m;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t top = (a >> (bits - 1)) & 1;
        a = ((a << 1) | F_CF) & m;
        F_CF = top;
    }
    if (count == 1) F_OF = ((a >> (bits - 1)) & 1) ^ F_CF;
    return a;
}
static uint32_t alu_rcr(uint32_t a, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    a &= m;
    if (count == 1) F_OF = ((a >> (bits - 1)) & 1) ^ F_CF;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t low = a & 1;
        a = (a >> 1) | ((uint32_t)F_CF << (bits - 1));
        F_CF = low;
    }
    return a & m;
}
static uint32_t alu_shl(uint32_t a, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    if (count == 0) return a & m;
    a &= m;
    F_CF = count <= bits ? (a >> (bits - count)) & 1 : 0;
    uint32_t r = (count >= bits) ? 0 : ((a << count) & m);
    F_OF = count == 1 ? (((r & sign_of(bits)) != 0) ^ F_CF) : 0;
    set_zs(r, bits);
    return r;
}
static uint32_t alu_shr(uint32_t a, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    if (count == 0) return a & m;
    a &= m;
    F_CF = count <= bits ? (a >> (count - 1)) & 1 : 0;
    uint32_t r = count >= bits ? 0 : (a >> count);
    F_OF = count == 1 ? ((a & sign_of(bits)) != 0) : 0;
    set_zs(r, bits);
    return r;
}
static uint32_t alu_sar(uint32_t a, uint32_t count, uint32_t bits) {
    uint32_t m = mask_of(bits);
    count &= 31;
    if (count == 0) return a & m;
    int32_t s = (int32_t)(a << (32 - bits)) >> (32 - bits);
    F_CF = count <= bits ? ((uint32_t)s >> (count - 1)) & 1 : (s < 0);
    int32_t r = count >= bits ? (s < 0 ? -1 : 0) : (s >> count);
    F_OF = 0;
    set_zs((uint32_t)r, bits);
    return (uint32_t)r & m;
}
static uint32_t alu_imul(uint32_t a, uint32_t b, uint32_t bits) {
    int64_t wide = (int64_t)((int32_t)(a << (32 - bits)) >> (32 - bits)) *
                   (int64_t)((int32_t)(b << (32 - bits)) >> (32 - bits));
    uint32_t r = (uint32_t)wide & mask_of(bits);
    int64_t back = (int64_t)((int32_t)(r << (32 - bits)) >> (32 - bits));
    F_CF = F_OF = (back != wide);
    set_zs(r, bits);
    return r;
}
static void alu_mul(uint32_t a) {
    uint64_t p = (uint64_t)R_EAX * (uint64_t)a;
    R_EAX = (uint32_t)p;
    R_EDX = (uint32_t)(p >> 32);
    F_CF = F_OF = (R_EDX != 0);
}
static void alu_imul1(uint32_t a) {
    int64_t p = (int64_t)(int32_t)R_EAX * (int64_t)(int32_t)a;
    R_EAX = (uint32_t)p;
    R_EDX = (uint32_t)((uint64_t)p >> 32);
    F_CF = F_OF = ((int64_t)(int32_t)R_EAX != p);
}
static void alu_div(uint32_t a, uint32_t at) {
    if (a == 0) lift_trap(at, "divide by zero");
    uint64_t n = ((uint64_t)R_EDX << 32) | R_EAX;
    R_EAX = (uint32_t)(n / a);
    R_EDX = (uint32_t)(n % a);
}
static void alu_idiv(uint32_t a, uint32_t at) {
    int32_t d = (int32_t)a;
    if (d == 0) lift_trap(at, "divide by zero");
    int64_t n = (int64_t)(((uint64_t)R_EDX << 32) | R_EAX);
    R_EAX = (uint32_t)(n / d);
    R_EDX = (uint32_t)(n % d);
}
/* The 8- and 16-bit forms of mul, imul, div and idiv: AL or AX times the operand, or AX or DX:AX
   divided by it. A quotient that does not fit, or a zero divisor, is the console's #DE. */
static void alu_mul8(uint32_t a) {
    uint32_t p = (R_EAX & 0xFFu) * (a & 0xFFu);
    R_EAX = (R_EAX & ~0xFFFFu) | p;
    F_CF = F_OF = ((p >> 8) != 0);
}
static void alu_mul16(uint32_t a) {
    uint32_t p = (R_EAX & 0xFFFFu) * (a & 0xFFFFu);
    R_EAX = (R_EAX & ~0xFFFFu) | (p & 0xFFFFu);
    R_EDX = (R_EDX & ~0xFFFFu) | (p >> 16);
    F_CF = F_OF = ((p >> 16) != 0);
}
static void alu_imul1_8(uint32_t a) {
    int32_t p = (int32_t)(int8_t)R_EAX * (int32_t)(int8_t)a;
    R_EAX = (R_EAX & ~0xFFFFu) | ((uint32_t)p & 0xFFFFu);
    F_CF = F_OF = ((int32_t)(int8_t)p != p);
}
static void alu_imul1_16(uint32_t a) {
    int32_t p = (int32_t)(int16_t)R_EAX * (int32_t)(int16_t)a;
    R_EAX = (R_EAX & ~0xFFFFu) | ((uint32_t)p & 0xFFFFu);
    R_EDX = (R_EDX & ~0xFFFFu) | (((uint32_t)p >> 16) & 0xFFFFu);
    F_CF = F_OF = ((int32_t)(int16_t)p != p);
}
static void alu_div8(uint32_t a, uint32_t at) {
    uint32_t d = a & 0xFFu, n = R_EAX & 0xFFFFu;
    if (d == 0) lift_trap(at, "divide by zero");
    if (n / d > 0xFFu) lift_trap(at, "divide overflow");
    R_EAX = (R_EAX & ~0xFFFFu) | (n / d) | ((n % d) << 8);
}
static void alu_div16(uint32_t a, uint32_t at) {
    uint32_t d = a & 0xFFFFu, n = ((R_EDX & 0xFFFFu) << 16) | (R_EAX & 0xFFFFu);
    if (d == 0) lift_trap(at, "divide by zero");
    if (n / d > 0xFFFFu) lift_trap(at, "divide overflow");
    R_EAX = (R_EAX & ~0xFFFFu) | (n / d);
    R_EDX = (R_EDX & ~0xFFFFu) | (n % d);
}
static void alu_idiv8(uint32_t a, uint32_t at) {
    int32_t d = (int8_t)a, n = (int16_t)R_EAX;
    if (d == 0) lift_trap(at, "divide by zero");
    int32_t q = n / d, r = n % d;
    if (q < -128 || q > 127) lift_trap(at, "divide overflow");
    R_EAX = (R_EAX & ~0xFFFFu) | ((uint32_t)q & 0xFFu) | (((uint32_t)r & 0xFFu) << 8);
}
static void alu_idiv16(uint32_t a, uint32_t at) {
    int32_t d = (int16_t)a, n = (int32_t)(((R_EDX & 0xFFFFu) << 16) | (R_EAX & 0xFFFFu));
    if (d == 0) lift_trap(at, "divide by zero");
    int32_t q = n / d, r = n % d;
    if (q < -32768 || q > 32767) lift_trap(at, "divide overflow");
    R_EAX = (R_EAX & ~0xFFFFu) | ((uint32_t)q & 0xFFFFu);
    R_EDX = (R_EDX & ~0xFFFFu) | ((uint32_t)r & 0xFFFFu);
}
/* pmovmskb: the sign bit of each byte of an MMX value. */
static uint32_t mmx_pmovmskb(uint64_t v) {
    uint32_t m = 0;
    for (int i = 0; i < 8; i++) m |= (uint32_t)((v >> (8 * i + 7)) & 1u) << i;
    return m;
}
static inline uint8_t cond(int code) {
    switch (code) {
    case 0: return F_OF;
    case 1: return !F_OF;
    case 2: return F_CF;
    case 3: return !F_CF;
    case 4: return F_ZF;
    case 5: return !F_ZF;
    case 6: return F_CF | F_ZF;
    case 7: return !F_CF & !F_ZF;
    case 8: return F_SF;
    case 9: return !F_SF;
    case 10: return F_PF;
    case 11: return !F_PF;
    case 12: return F_SF != F_OF;
    case 13: return F_SF == F_OF;
    case 14: return F_ZF | (F_SF != F_OF);
    default: return !F_ZF & (F_SF == F_OF);
    }
}
"#;

/// A lifted program, in pieces: the shared header, one body per function, and the call dispatch.
#[derive(Debug, Default)]
pub struct Program {
    prototypes: String,
    bodies: Vec<String>,
    dispatch: String,
    /// The number of functions that lifted.
    pub lifted: usize,
    /// Places inside lifted functions that trap if they run, because the lifter could not
    /// translate them.
    pub traps: Vec<LiftError>,
    /// The functions that could not be lifted. Each has a stub that traps with the reason when it
    /// runs.
    pub failures: Vec<LiftError>,
}

impl Program {
    /// The header every translation unit includes: the prelude, and the declarations of the
    /// dispatch hook, the dispatcher and every function.
    #[must_use]
    pub fn header(&self) -> String {
        format!(
            "{PRELUDE}\nvoid lift_native(uint32_t target, uint32_t at);\nvoid rt_call(uint32_t target, uint32_t at);\n\
             void lift_resume_call(void);\nvoid lift_resume_root(void);\nvoid lift_resume_innermost(void);\n{}",
            self.prototypes
        )
    }

    /// The C source of each function.
    #[must_use]
    pub fn bodies(&self) -> &[String] {
        &self.bodies
    }

    /// The dispatcher that maps a guest address to its function. It needs the header.
    #[must_use]
    pub fn dispatch(&self) -> &str {
        &self.dispatch
    }

    /// The whole program as one translation unit, which also defines the globals.
    #[must_use]
    pub fn source(&self) -> String {
        format!(
            "#define LIFT_DEFINE_GLOBALS\n{}\n{}{}",
            self.header(),
            self.bodies.concat(),
            self.dispatch
        )
    }
}

/// Lifts the functions reachable from `roots`: each root and every function it calls directly.
///
/// Each root is a function of its own: a body that reaches another root, by a jump or by running
/// into it, tail-calls that root's function instead of holding a copy of its code. Roots found by
/// scanning often fall inside other functions, and without this each would copy the rest of its
/// function, which made one title's C fifteen times the size of its code.
///
/// A function the lifter cannot translate does not stop the others. It gets a stub that traps
/// with the reason, and it is listed in [`Program::failures`]. A function in `hooks` (function
/// address to trap address) is compiled as a call to its trap. With `trace`, every instruction
/// reports its address and registers when the unit is built with `LIFT_TRACE_ON`.
#[must_use]
pub fn lift_program(
    image: &Image,
    roots: &[u32],
    hooks: &BTreeMap<u32, u32>,
    trace: bool,
) -> Program {
    lift_program_with_progress(image, roots, hooks, trace, &mut |_, _| {})
}

/// [`lift_program`], calling `progress` after each function with how many are done and how many
/// are known so far. The second grows as lifted functions name callees not seen before.
#[must_use]
pub fn lift_program_with_progress(
    image: &Image,
    roots: &[u32],
    hooks: &BTreeMap<u32, u32>,
    trace: bool,
    progress: &mut dyn FnMut(usize, usize),
) -> Program {
    let setjmps = find_setjmps(image);
    let mut prototypes = String::new();
    let mut bodies = Vec::new();
    let mut traps = Vec::new();
    let walked = walk(image, roots, hooks, progress, &mut |function, root| {
        writeln!(prototypes, "void fn_{function:08x}(void);").unwrap();
        bodies.push(match root {
            Root::Hooked(trap) => format!(
                "void fn_{function:08x}(void) {{ LIFT_ENTER({function:#010x}u); \
                 if (LIFT_RESUMING()) {{ lift_resume_call(); return; }} \
                 LIFT_IN({function:#010x}u, LIFT_K_RET, rt_call({trap:#010x}u, {function:#010x}u)); }}\n"
            ),
            Root::Lifted(found) => {
                let body = lift_function(function, &found, &setjmps, trace);
                traps.extend(body.traps);
                body.c
            }
            Root::Failed(error) => {
                let trap = trap_statement(function, &error.message);
                format!("void fn_{function:08x}(void) {{ {trap} }}\n")
            }
        });
    });
    let dispatch = dispatcher(&walked.lifted);
    Program {
        prototypes,
        bodies,
        dispatch,
        lifted: walked.lifted.len(),
        traps,
        failures: walked.failures,
    }
}

/// The C of `rt_call`, which maps a guest address to its function.
///
/// The functions are a sorted table searched by bisection, with a cache of the addresses called
/// last. A `switch` over every function compiles to the same search, but clang takes hours over
/// one with a few hundred thousand cases, where the table compiles in moments. The cache keeps
/// each slot in one atomic word (the address above, the table index plus one below), so threads
/// need no lock: a slot is either a whole entry or one that does not match.
fn dispatcher(lifted: &BTreeSet<u32>) -> String {
    let mut c = String::from(
        "#include <stdatomic.h>\n\
         typedef struct { uint32_t address; void (*function)(void); } lift_entry;\n\
         static const lift_entry lift_table[] = {\n",
    );
    for function in lifted {
        writeln!(c, "    {{{function:#010x}u, fn_{function:08x}}},").unwrap();
    }
    if lifted.is_empty() {
        c.push_str("    {0u, 0},\n");
    }
    write!(
        c,
        "}};\n\
         #define LIFT_COUNT {}u\n\
         static _Atomic uint64_t lift_cache[4096];\n\
         /* The function lifted at `target`, or none. */\n\
         static inline void (*lift_find(uint32_t target))(void) {{\n\
         \x20   uint32_t slot = (target ^ (target >> 12)) & 4095u;\n\
         \x20   uint64_t seen = atomic_load_explicit(&lift_cache[slot], memory_order_relaxed);\n\
         \x20   if ((uint32_t)(seen >> 32) == target && (uint32_t)seen != 0) return lift_table[(uint32_t)seen - 1].function;\n\
         \x20   uint32_t low = 0, high = LIFT_COUNT;\n\
         \x20   while (low < high) {{\n\
         \x20       uint32_t mid = low + (high - low) / 2;\n\
         \x20       if (lift_table[mid].address < target) low = mid + 1; else high = mid;\n\
         \x20   }}\n\
         \x20   if (low < LIFT_COUNT && lift_table[low].address == target) {{\n\
         \x20       atomic_store_explicit(&lift_cache[slot], (uint64_t)target << 32 | (low + 1), memory_order_relaxed);\n\
         \x20       return lift_table[low].function;\n\
         \x20   }}\n\
         \x20   return 0;\n\
         }}\n\
         void rt_call(uint32_t target, uint32_t at) {{\n\
         \x20   void (*function)(void) = lift_find(target);\n\
         \x20   if (function) {{\n\
         \x20       function();\n\
         \x20       return;\n\
         \x20   }}\n\
         \x20   /* A `longjmp` lands inside a function: that is not an entry, and the context knows it. */\n\
         \x20   lift_longjmp(target);\n\
         \x20   /* Anything else is a kernel call or a missing function: the native hook decides. */\n\
         \x20   lift_native(target, at);\n\
         }}\n\
         /* Resuming a frame's call: calls the function in the next frame down, which resumes at its\n\
         \x20  own site, or, below the last frame, finishes the parked contact with the runtime. */\n\
         void lift_resume_call(void) {{\n\
         \x20   uint32_t depth = ++lift_cpu.depth;\n\
         \x20   if (depth < lift_cpu.resume_depth) {{\n\
         \x20       void (*function)(void) = lift_find(lift_cpu.frames[depth].fn);\n\
         \x20       if (!function) lift_trap(lift_cpu.frames[depth].fn, \"resuming a function that is not lifted\");\n\
         \x20       function();\n\
         \x20   }} else {{\n\
         \x20       lift_resume_innermost();\n\
         \x20   }}\n\
         \x20   lift_cpu.depth = depth - 1;\n\
         }}\n\
         /* Resumes a thread from its outermost frame. */\n\
         void lift_resume_root(void) {{\n\
         \x20   lift_cpu.depth = 0;\n\
         \x20   void (*function)(void) = lift_find(lift_cpu.frames[0].fn);\n\
         \x20   if (!function) lift_trap(lift_cpu.frames[0].fn, \"resuming a function that is not lifted\");\n\
         \x20   function();\n\
         }}\n",
        lifted.len()
    )
    .unwrap();
    c
}

/// Lifts the functions reachable from `entry`, and returns their C source with the prelude first.
///
/// # Errors
///
/// Returns the first function the lifter cannot translate.
pub fn lift(image: &Image, entry: u32) -> Result<String, LiftError> {
    let program = lift_program(image, &[entry], &BTreeMap::new(), false);
    match program.failures.first() {
        Some(error) => Err(error.clone()),
        None => Ok(program.source()),
    }
}

struct Body {
    c: String,
    /// Places in the function the lifter could not translate; each traps if it runs.
    traps: Vec<LiftError>,
}

/// A C statement that stops the program at `address` with `message`.
fn trap_statement(address: u32, message: &str) -> String {
    let message = message.replace(['"', '\\', '\n'], "'");
    format!("lift_trap({address:#010x}u, \"{message}\");")
}

/// Emits one function that discovery found, in address order, with a `goto` wherever the next
/// instruction in that order is not the fall-through.
// Emission keeps several tables of its own; splitting it would pass them all around.
#[allow(clippy::too_many_lines)]
fn lift_function(entry: u32, found: &Discovered, setjmps: &BTreeSet<u32>, trace: bool) -> Body {
    let Discovered {
        insts,
        faults,
        hooked,
        tables,
        ..
    } = found;
    let mut traps = found.traps();
    let body_set = found.body();
    let name = format!("fn_{entry:08x}");
    let mut c = String::new();
    writeln!(c, "void {name}(void) {{").unwrap();
    writeln!(c, "    LIFT_ENTER({entry:#010x}u);").unwrap();
    // Where the declarations of the `setjmp` contexts go, once the body says how many it needs.
    let declarations_at = c.len();
    let mut setjmp_sites = 0usize;
    // The sites a resume can re-enter the function at, the addresses of its tail calls, and of its
    // `setjmp` calls in order.
    let mut sites: BTreeSet<(u32, Site)> = BTreeSet::new();
    let mut tails: BTreeSet<u32> = BTreeSet::new();
    let mut setjmp_addresses: Vec<u32> = Vec::new();
    let addresses: Vec<u32> = insts.keys().copied().collect();
    // The body is emitted in address order, but the function starts at its entry, which need not
    // be the lowest address when a branch reaches earlier code.
    if addresses.first() != Some(&entry) {
        writeln!(c, "    goto L_{entry:08x};").unwrap();
    }
    for (index, &address) in addresses.iter().enumerate() {
        let instruction = insts[&address];
        // Every instruction gets a label: a fall-through `goto` may land on any of them.
        if trace {
            writeln!(c, "L_{address:08x}:; LIFT_TRACE({address:#010x}u);").unwrap();
        } else {
            writeln!(c, "L_{address:08x}:;").unwrap();
        }
        let flow = instruction.flow_control();
        let falls_through = !matches!(
            flow,
            FlowControl::Return
                | FlowControl::UnconditionalBranch
                | FlowControl::IndirectBranch
                | FlowControl::Interrupt
                | FlowControl::Exception
        );
        // A direct call to `setjmp` is a host `sigsetjmp` in this function's own frame.
        let setjmp_site = (instruction.flow_control() == FlowControl::Call
            && matches!(
                instruction.op0_kind(),
                OpKind::NearBranch32 | OpKind::NearBranch64
            )
            && setjmps.contains(&addr(instruction.near_branch_target())))
        .then(|| {
            setjmp_sites += 1;
            setjmp_addresses.push(address);
            setjmp_sites - 1
        });
        let statement = match emit(
            &instruction,
            address,
            &body_set,
            tables.get(&address),
            setjmp_site,
        ) {
            Ok(statement) => statement,
            Err(error) => {
                let statement = trap_statement(address, &error.message);
                traps.push(error);
                statement
            }
        };
        for (marker, site) in [
            ("LIFT_K_CALL", Site::Call),
            ("LIFT_K_RET", Site::Ret),
            ("LIFT_POLL(", Site::Poll),
        ] {
            if statement.contains(marker) {
                sites.insert((address, site));
            }
        }
        if statement.contains("LIFT_TAIL(") {
            tails.insert(address);
        }
        writeln!(c, "    {statement}").unwrap();
        if falls_through {
            let next = addr(instruction.next_ip());
            if !body_set.contains(&next) {
                // It runs into another function's entry.
                writeln!(c, "    LIFT_TAIL(fn_{next:08x}, {address:#010x}u);").unwrap();
                tails.insert(address);
            } else if addresses.get(index + 1) != Some(&next) {
                writeln!(c, "    goto L_{next:08x};").unwrap();
            }
        }
    }
    for (&address, &trap) in hooked {
        sites.insert((address, Site::Ret));
        writeln!(
            c,
            "L_{address:08x}:; LIFT_IN({address:#010x}u, LIFT_K_RET, rt_call({trap:#010x}u, {address:#010x}u)); return;"
        )
        .unwrap();
    }
    for (&address, message) in faults {
        let trap = trap_statement(address, message);
        writeln!(c, "L_{address:08x}:; {trap}").unwrap();
    }
    writeln!(c, "}}").unwrap();
    let mut declarations = String::new();
    if setjmp_sites > 0 {
        c = c.replace("LIFT_TAIL(", "LIFT_CALL_RETURN(");
        // Each tail call is now a call whose caller returns after it: a site to resume at.
        sites.extend(tails.iter().map(|&address| (address, Site::Ret)));
        for site in 0..setjmp_sites {
            writeln!(
                declarations,
                "    lift_jmp lj{site} __attribute__((cleanup(lift_jmp_exit))) = {{0}};"
            )
            .unwrap();
        }
    }
    declarations.push_str(&resume_prologue(entry, &sites, &setjmp_addresses));
    c.insert_str(declarations_at, &declarations);
    Body { c, traps }
}

/// The kind of site a host frame can be resumed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Site {
    /// A call: resumes the callee, then goes on after the call.
    Call,
    /// A call whose caller returns as soon as it does.
    Ret,
    /// A loop's poll: goes on after it.
    Poll,
}

impl Site {
    /// The `LIFT_K_*` constant.
    fn kind(self) -> &'static str {
        match self {
            Self::Call => "LIFT_K_CALL",
            Self::Ret => "LIFT_K_RET",
            Self::Poll => "LIFT_K_POLL",
        }
    }
}

/// The prologue that resumes a function at a site: it registers the function's `setjmp` contexts
/// again (the call at `setjmps[i]` holds context `i`), then calls down to, or goes on from, the
/// site its frame record names.
fn resume_prologue(entry: u32, sites: &BTreeSet<(u32, Site)>, setjmps: &[u32]) -> String {
    let mut c = String::from("    if (LIFT_RESUMING()) {\n");
    for (index, address) in setjmps.iter().enumerate() {
        writeln!(
            c,
            "        {{ uint32_t b, r; if (lift_resume_jmp({index}u, &b, &r)) {{ \
             lift_jmp_register(&lj{index}, b, r, {index}u); \
             if (sigsetjmp(lj{index}.env, 0)) goto C_{address:08x}; }} }}"
        )
        .unwrap();
    }
    c.push_str(
        "        switch (LIFT_KEY(lift_cpu.frames[lift_cpu.depth].site, lift_cpu.frames[lift_cpu.depth].kind)) {\n",
    );
    for &(address, site) in sites {
        let action = match site {
            Site::Call => format!("lift_resume_call(); goto C_{address:08x};"),
            Site::Ret => "lift_resume_call(); return;".to_owned(),
            Site::Poll => format!("lift_resume_innermost(); goto P_{address:08x};"),
        };
        writeln!(
            c,
            "        case LIFT_KEY({address:#010x}u, {}): {action}",
            site.kind()
        )
        .unwrap();
    }
    writeln!(
        c,
        "        }}\n        lift_trap({entry:#010x}u, \"resuming at a site this function does not have\");\n    }}"
    )
    .unwrap();
    c
}

/// The guest register a 32-bit name refers to, and how it sits in that register.
fn register_slot(register: Register) -> Option<(&'static str, u32, u32)> {
    Some(match register {
        Register::EAX => ("R_EAX", 0, 32),
        Register::AX => ("R_EAX", 0, 16),
        Register::AL => ("R_EAX", 0, 8),
        Register::AH => ("R_EAX", 8, 8),
        Register::ECX => ("R_ECX", 0, 32),
        Register::CX => ("R_ECX", 0, 16),
        Register::CL => ("R_ECX", 0, 8),
        Register::CH => ("R_ECX", 8, 8),
        Register::EDX => ("R_EDX", 0, 32),
        Register::DX => ("R_EDX", 0, 16),
        Register::DL => ("R_EDX", 0, 8),
        Register::DH => ("R_EDX", 8, 8),
        Register::EBX => ("R_EBX", 0, 32),
        Register::BX => ("R_EBX", 0, 16),
        Register::BL => ("R_EBX", 0, 8),
        Register::BH => ("R_EBX", 8, 8),
        Register::ESP => ("R_ESP", 0, 32),
        Register::SP => ("R_ESP", 0, 16),
        Register::EBP => ("R_EBP", 0, 32),
        Register::BP => ("R_EBP", 0, 16),
        Register::ESI => ("R_ESI", 0, 32),
        Register::SI => ("R_ESI", 0, 16),
        Register::EDI => ("R_EDI", 0, 32),
        Register::DI => ("R_EDI", 0, 16),
        _ => return None,
    })
}

/// The C expression reading register `register`, zero-extended to 32 bits.
fn read_register(address: u32, register: Register) -> Result<String, LiftError> {
    let (base, shift, bits) = register_slot(register)
        .ok_or_else(|| unsupported(address, format!("register {register:?}")))?;
    Ok(match bits {
        32 => base.to_string(),
        16 => format!("({base} & 0xFFFFu)"),
        _ => format!("(({base} >> {shift}) & 0xFFu)"),
    })
}

/// The C statement writing `value` to register `register`.
fn write_register(address: u32, register: Register, value: &str) -> Result<String, LiftError> {
    let (base, shift, bits) = register_slot(register)
        .ok_or_else(|| unsupported(address, format!("register {register:?}")))?;
    Ok(match (bits, shift) {
        (32, _) => format!("{base} = (uint32_t)({value});"),
        (16, _) => format!("{base} = ({base} & 0xFFFF0000u) | (((uint32_t)({value})) & 0xFFFFu);"),
        (_, 0) => format!("{base} = ({base} & 0xFFFFFF00u) | (((uint32_t)({value})) & 0xFFu);"),
        _ => format!("{base} = ({base} & 0xFFFF00FFu) | ((((uint32_t)({value})) & 0xFFu) << 8);"),
    })
}

fn register_bits(address: u32, register: Register) -> Result<u32, LiftError> {
    register_slot(register)
        .map(|(_, _, bits)| bits)
        .ok_or_else(|| unsupported(address, format!("register {register:?}")))
}

/// The bit width of operand `index`, from its register or its memory size.
fn operand_bits(address: u32, instruction: &Instruction, index: u32) -> Result<u32, LiftError> {
    match instruction.op_kind(index) {
        OpKind::Register => register_bits(address, instruction.op_register(index)),
        OpKind::Memory => {
            Ok(u32::try_from(instruction.memory_size().size()).unwrap_or(u32::MAX) * 8)
        }
        _ => Err(unsupported(address, "operand width of an immediate")),
    }
}

/// The C expression computing the effective address of the memory operand of `instruction`.
fn effective_address(instruction: &Instruction) -> String {
    let mut parts = Vec::new();
    let base = instruction.memory_base();
    if base != Register::None {
        parts.push(read_register(0, base).unwrap_or_else(|_| "0".into()));
    }
    let index = instruction.memory_index();
    if index != Register::None {
        let scale = instruction.memory_index_scale();
        let reg = read_register(0, index).unwrap_or_else(|_| "0".into());
        parts.push(if scale == 1 {
            reg
        } else {
            format!("({reg} * {scale}u)")
        });
    }
    let displacement = addr(instruction.memory_displacement64());
    if displacement != 0 || parts.is_empty() {
        parts.push(format!("{displacement}u"));
    }
    if instruction.segment_prefix() == Register::FS {
        parts.push("FS_BASE".into());
    }
    format!("({})", parts.join(" + "))
}

/// The C expression reading operand `index`, masked to `bits`. An immediate is taken at `bits`.
fn read_operand(
    address: u32,
    instruction: &Instruction,
    index: u32,
    bits: u32,
) -> Result<String, LiftError> {
    let mask = if bits == 32 {
        "0xFFFFFFFFu".to_string()
    } else {
        format!("{}u", (1u64 << bits) - 1)
    };
    Ok(match instruction.op_kind(index) {
        OpKind::Register => {
            let value = read_register(address, instruction.op_register(index))?;
            format!("(({value}) & {mask})")
        }
        OpKind::Memory => {
            let ea = effective_address(instruction);
            match bits {
                8 => format!("((uint32_t)rd8({ea}))"),
                16 => format!("((uint32_t)rd16({ea}))"),
                _ => format!("rd32({ea})"),
            }
        }
        OpKind::Immediate8
        | OpKind::Immediate16
        | OpKind::Immediate32
        | OpKind::Immediate8to32
        | OpKind::Immediate8to16
        | OpKind::Immediate32to64 => {
            let value = addr(instruction.immediate(index));
            format!("(({value}u) & {mask})")
        }
        other => return Err(unsupported(address, format!("operand kind {other:?}"))),
    })
}

/// The C statement writing `value` to operand `index` at `bits`.
fn write_operand(
    address: u32,
    instruction: &Instruction,
    index: u32,
    bits: u32,
    value: &str,
) -> Result<String, LiftError> {
    match instruction.op_kind(index) {
        OpKind::Register => write_register(address, instruction.op_register(index), value),
        OpKind::Memory => {
            let ea = effective_address(instruction);
            Ok(match bits {
                8 => format!("wr8({ea}, (uint8_t)({value}));"),
                16 => format!("wr16({ea}, (uint16_t)({value}));"),
                _ => format!("wr32({ea}, (uint32_t)({value}));"),
            })
        }
        other => Err(unsupported(address, format!("cannot write to {other:?}"))),
    }
}

/// The C expression sign-extending the low `bits` of `value`.
fn sign_extend(value: &str, bits: u32) -> String {
    format!(
        "((uint32_t)((int32_t)(({value}) << {shift}) >> {shift}))",
        shift = 32 - bits
    )
}

/// A string instruction, once or under `rep`.
fn string_loop(instruction: &Instruction, body: &str, advance: &str) -> String {
    if instruction.has_rep_prefix()
        || instruction.has_repe_prefix()
        || instruction.has_repne_prefix()
    {
        format!("while (R_ECX) {{ {body} {advance} R_ECX--; }}")
    } else {
        format!("{body} {advance}")
    }
}

/// The C statements for one instruction.
///
/// `body` holds the addresses of the function's own instructions: a jump to any other address is
/// a tail call.
// One arm per instruction form: the table is long because the instruction set is.
#[allow(clippy::too_many_lines)]
fn emit(
    instruction: &Instruction,
    address: u32,
    body: &BTreeSet<u32>,
    table: Option<&Vec<(u32, u32)>>,
    setjmp_site: Option<usize>,
) -> Result<String, LiftError> {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    if let Some(statement) = x87::emit(instruction, address)? {
        return Ok(statement);
    }
    if let Some(statement) = mmx::emit(instruction, address)? {
        return Ok(statement);
    }
    if let Some(statement) = sse::emit(instruction, address)? {
        return Ok(statement);
    }
    let next = addr(instruction.next_ip());
    let size = if instruction.op_count() > 0 && instruction.op_kind(0) != OpKind::NearBranch32 {
        operand_bits(address, instruction, 0).unwrap_or(32)
    } else {
        32
    };
    let mnemonic = instruction.mnemonic();
    if let Some(code) = condition_code(mnemonic) {
        return Ok(match mnemonic {
            m if format!("{m:?}").starts_with('J') => {
                let target = addr(instruction.near_branch_target());
                if body.contains(&target) {
                    if target <= address {
                        format!(
                            "if (cond({code})) {{ LIFT_POLL({address:#010x}u); P_{address:08x}:; goto L_{target:08x}; }}"
                        )
                    } else {
                        format!("if (cond({code})) goto L_{target:08x};")
                    }
                } else {
                    let poll = if target <= address {
                        format!("LIFT_POLL({address:#010x}u); P_{address:08x}:; ")
                    } else {
                        String::new()
                    };
                    format!(
                        "if (cond({code})) {{ {poll}LIFT_TAIL(fn_{target:08x}, {address:#010x}u); }}"
                    )
                }
            }
            m if format!("{m:?}").starts_with("Set") => {
                write_operand(address, instruction, 0, 8, &format!("cond({code})"))?
            }
            _ => {
                let value = read_operand(address, instruction, 1, size)?;
                let dst = write_operand(address, instruction, 0, size, &value)?;
                format!("if (cond({code})) {{ {dst} }}")
            }
        });
    }
    Ok(match mnemonic {
        // No effect natively: interrupts are not modelled; port writes reach no device, as in the
        // emulator; caches, prefetches, fences and spin hints change nothing one host image sees.
        Nop | Cli | Sti | Out | Wbinvd | Invd | Invlpg | Clflush | Prefetchnta | Prefetcht0
        | Prefetcht1 | Prefetcht2 | Prefetchw | Sfence | Lfence | Mfence | Pause | Emms => {
            String::new()
        }
        // Port reads give zero, as in the emulator.
        In => write_operand(address, instruction, 0, size, "0u")?,
        Cld => "F_DF = 0;".to_string(),
        Clc => "F_CF = 0;".to_string(),
        Stc => "F_CF = 1;".to_string(),
        Cmc => "F_CF ^= 1;".to_string(),
        Xlatb => "R_EAX = (R_EAX & 0xFFFFFF00u) | rd8(R_EBX + (R_EAX & 0xFFu));".to_string(),
        Enter if instruction.immediate8_2nd() == 0 => format!(
            "push32(R_EBP); R_EBP = R_ESP; R_ESP -= {}u;",
            instruction.immediate16()
        ),
        Loop | Loope | Loopne | Jecxz => {
            let target = addr(instruction.near_branch_target());
            let test = match mnemonic {
                Loop => "(--R_ECX != 0)",
                Loope => "(--R_ECX != 0 && F_ZF)",
                Loopne => "(--R_ECX != 0 && !F_ZF)",
                _ => "(R_ECX == 0)",
            };
            let poll = if target <= address {
                format!("LIFT_POLL({address:#010x}u); P_{address:08x}:; ")
            } else {
                String::new()
            };
            if body.contains(&target) {
                format!("if {test} {{ {poll}goto L_{target:08x}; }}")
            } else {
                format!("if {test} {{ {poll}LIFT_TAIL(fn_{target:08x}, {address:#010x}u); }}")
            }
        }
        Rdtsc => "{ uint64_t t = lift_rdtsc(); R_EAX = (uint32_t)t; R_EDX = (uint32_t)(t >> 32); }"
            .to_string(),
        Std => "F_DF = 1;".to_string(),
        Stosb | Stosw | Stosd => {
            let (bits, step) = match mnemonic {
                Stosb => (8, 1),
                Stosw => (16, 2),
                _ => (32, 4),
            };
            let value = read_register(
                address,
                if bits == 8 {
                    Register::AL
                } else if bits == 16 {
                    Register::AX
                } else {
                    Register::EAX
                },
            )?;
            let store = match bits {
                8 => format!("wr8(R_EDI, (uint8_t)({value}));"),
                16 => format!("wr16(R_EDI, (uint16_t)({value}));"),
                _ => format!("wr32(R_EDI, {value});"),
            };
            string_loop(instruction, &store, &format!("R_EDI += dstep({step}u);"))
        }
        Cmpsb | Cmpsw | Cmpsd => {
            let (bits, step) = match mnemonic {
                Cmpsb => (8, 1),
                Cmpsw => (16, 2),
                _ => (32, 4),
            };
            let load_a = match bits {
                8 => "rd8(R_ESI)",
                16 => "rd16(R_ESI)",
                _ => "rd32(R_ESI)",
            };
            let load_b = match bits {
                8 => "rd8(R_EDI)",
                16 => "rd16(R_EDI)",
                _ => "rd32(R_EDI)",
            };
            let body = format!("(void)alu_sub({load_a}, {load_b}, {bits});");
            let advance = format!("R_ESI += dstep({step}u); R_EDI += dstep({step}u);");
            if instruction.has_repe_prefix() {
                format!("while (R_ECX) {{ {body} {advance} R_ECX--; if (!F_ZF) break; }}")
            } else if instruction.has_repne_prefix() {
                format!("while (R_ECX) {{ {body} {advance} R_ECX--; if (F_ZF) break; }}")
            } else {
                format!("{body} {advance}")
            }
        }
        Movsb | Movsw | Movsd => {
            let (bits, step) = match mnemonic {
                Movsb => (8, 1),
                Movsw => (16, 2),
                _ => (32, 4),
            };
            let load = match bits {
                8 => "rd8(R_ESI)",
                16 => "rd16(R_ESI)",
                _ => "rd32(R_ESI)",
            };
            let store = match bits {
                8 => format!("wr8(R_EDI, (uint8_t)({load}));"),
                16 => format!("wr16(R_EDI, (uint16_t)({load}));"),
                _ => format!("wr32(R_EDI, {load});"),
            };
            string_loop(
                instruction,
                &store,
                &format!("R_ESI += dstep({step}u); R_EDI += dstep({step}u);"),
            )
        }
        Int3 | Int | Ud2 | Hlt => format!("lift_trap({address}u, \"interrupt or halt\");"),
        // `movnti` is a store with a cache hint.
        Mov | Movnti => {
            let value = read_operand(address, instruction, 1, size)?;
            write_operand(address, instruction, 0, size, &value)?
        }
        Movzx => {
            let src_bits = operand_bits(address, instruction, 1)?;
            let value = read_operand(address, instruction, 1, src_bits)?;
            write_operand(address, instruction, 0, size, &value)?
        }
        Movsx => {
            let src_bits = operand_bits(address, instruction, 1)?;
            let value = sign_extend(&read_operand(address, instruction, 1, src_bits)?, src_bits);
            write_operand(address, instruction, 0, size, &value)?
        }
        Lea => {
            let value = format!("(uint32_t){}", effective_address(instruction));
            write_operand(address, instruction, 0, size, &value)?
        }
        Xchg => {
            let a = read_operand(address, instruction, 0, size)?;
            let b = read_operand(address, instruction, 1, size)?;
            let w0 = write_operand(address, instruction, 0, size, "t_b")?;
            let w1 = write_operand(address, instruction, 1, size, "t_a")?;
            format!("{{ uint32_t t_a = {a}; uint32_t t_b = {b}; {w0} {w1} }}")
        }
        Push => {
            let value = read_operand(address, instruction, 0, 32)?;
            format!("push32({value});")
        }
        Pop => write_operand(address, instruction, 0, 32, "pop32()")?,
        Bsf | Bsr => {
            let src = read_operand(address, instruction, 1, size)?;
            let dst = read_operand(address, instruction, 0, size)?;
            let helper = if mnemonic == Bsf {
                "alu_bsf"
            } else {
                "alu_bsr"
            };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}({src}, {dst})"),
            )?
        }
        Adc | Sbb => {
            let a = read_operand(address, instruction, 0, size)?;
            let b = read_operand(address, instruction, 1, size)?;
            let helper = if mnemonic == Adc {
                "alu_adc"
            } else {
                "alu_sbb"
            };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}({a}, {b}, {size})"),
            )?
        }
        Add | Sub | Cmp | And | Or | Xor | Test => {
            let a = read_operand(address, instruction, 0, size)?;
            let b = read_operand(address, instruction, 1, size)?;
            let (expr, store) = match mnemonic {
                Add => (format!("alu_add({a}, {b}, {size})"), true),
                Sub => (format!("alu_sub({a}, {b}, {size})"), true),
                Cmp => (format!("alu_sub({a}, {b}, {size})"), false),
                And => (format!("alu_logic({a} & {b}, {size})"), true),
                Or => (format!("alu_logic({a} | {b}, {size})"), true),
                Xor => (format!("alu_logic({a} ^ {b}, {size})"), true),
                _ => (format!("alu_logic({a} & {b}, {size})"), false),
            };
            if store {
                write_operand(address, instruction, 0, size, &expr)?
            } else {
                format!("(void)({expr});")
            }
        }
        Inc | Dec | Neg | Not => {
            let a = read_operand(address, instruction, 0, size)?;
            let expr = match mnemonic {
                Inc => format!("alu_inc({a}, {size})"),
                Dec => format!("alu_dec({a}, {size})"),
                Neg => format!("alu_neg({a}, {size})"),
                _ => format!(
                    "(~({a})) & {}u",
                    if size == 32 {
                        0xFFFF_FFFFu64
                    } else {
                        (1u64 << size) - 1
                    }
                ),
            };
            write_operand(address, instruction, 0, size, &expr)?
        }
        Rol | Ror => {
            let a = read_operand(address, instruction, 0, size)?;
            let count = read_operand(address, instruction, 1, 8)?;
            let helper = if mnemonic == Rol { "alu_rol" } else { "alu_ror" };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}({a}, {count}, {size})"),
            )?
        }
        Shld | Shrd => {
            let dst = read_operand(address, instruction, 0, size)?;
            let src = read_operand(address, instruction, 1, size)?;
            let count = read_operand(address, instruction, 2, 8)?;
            let helper = if mnemonic == Shld { "alu_shld" } else { "alu_shrd" };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}({dst}, {src}, {count}, {size})"),
            )?
        }
        Cwde => "R_EAX = (uint32_t)(int32_t)(int16_t)R_EAX;".to_string(),
        Cbw => "R_EAX = (R_EAX & 0xFFFF0000u) | (uint16_t)(int16_t)(int8_t)R_EAX;".to_string(),
        Cwd => "R_EDX = (R_EDX & 0xFFFF0000u) | (((int16_t)R_EAX < 0) ? 0xFFFFu : 0u);"
            .to_string(),
        Bswap => {
            let value = read_register(address, instruction.op0_register())?;
            write_register(
                address,
                instruction.op0_register(),
                &format!("__builtin_bswap32({value})"),
            )?
        }
        Pushfd => "push32(eflags_get());".to_string(),
        Pushad => "{ uint32_t t_esp = R_ESP; push32(R_EAX); push32(R_ECX); push32(R_EDX); push32(R_EBX); push32(t_esp); push32(R_EBP); push32(R_ESI); push32(R_EDI); }".to_string(),
        Popad => "{ R_EDI = pop32(); R_ESI = pop32(); R_EBP = pop32(); (void)pop32(); R_EBX = pop32(); R_EDX = pop32(); R_ECX = pop32(); R_EAX = pop32(); }".to_string(),
        Popfd => "eflags_set(pop32());".to_string(),
        Lahf => "R_EAX = (R_EAX & 0xFFFF00FFu) | ((((uint32_t)F_SF << 7) | ((uint32_t)F_ZF << 6) | ((uint32_t)F_PF << 2) | 2u | F_CF) << 8);".to_string(),
        Xadd => {
            let dst = read_operand(address, instruction, 0, size)?;
            let src = read_operand(address, instruction, 1, size)?;
            let write_src = write_operand(address, instruction, 1, size, "t_a")?;
            let write_dst = write_operand(address, instruction, 0, size, "t_r")?;
            format!("{{ uint32_t t_a = {dst}, t_b = {src}; uint32_t t_r = alu_add(t_a, t_b, {size}); {write_src} {write_dst} }}")
        }
        Cmpxchg => {
            let accumulator = match size {
                8 => Register::AL,
                16 => Register::AX,
                _ => Register::EAX,
            };
            let acc = read_register(address, accumulator)?;
            let dst = read_operand(address, instruction, 0, size)?;
            let src = read_operand(address, instruction, 1, size)?;
            let store = write_operand(address, instruction, 0, size, &src)?;
            let load = write_register(address, accumulator, "t_d")?;
            format!("{{ uint32_t t_d = {dst}; (void)alu_sub({acc}, t_d, {size}); if (F_ZF) {{ {store} }} else {{ {load} }} }}")
        }
        Scasb | Scasw | Scasd => {
            let (bits, step, accumulator, load) = match mnemonic {
                Scasb => (8, 1, Register::AL, "rd8(R_EDI)"),
                Scasw => (16, 2, Register::AX, "rd16(R_EDI)"),
                _ => (32, 4, Register::EAX, "rd32(R_EDI)"),
            };
            let acc = read_register(address, accumulator)?;
            let body = format!("(void)alu_sub({acc}, {load}, {bits});");
            let advance = format!("R_EDI += dstep({step}u);");
            if instruction.has_repe_prefix() {
                format!("while (R_ECX) {{ {body} {advance} R_ECX--; if (!F_ZF) break; }}")
            } else if instruction.has_repne_prefix() {
                format!("while (R_ECX) {{ {body} {advance} R_ECX--; if (F_ZF) break; }}")
            } else {
                format!("{body} {advance}")
            }
        }
        Lodsb | Lodsw | Lodsd => {
            let (accumulator, load, step) = match mnemonic {
                Lodsb => (Register::AL, "rd8(R_ESI)", 1),
                Lodsw => (Register::AX, "rd16(R_ESI)", 2),
                _ => (Register::EAX, "rd32(R_ESI)", 4),
            };
            let store = write_register(address, accumulator, load)?;
            string_loop(instruction, &store, &format!("R_ESI += dstep({step}u);"))
        }
        Rcl | Rcr => {
            let a = read_operand(address, instruction, 0, size)?;
            let count = read_operand(address, instruction, 1, 8)?;
            let helper = if mnemonic == Rcl {
                "alu_rcl"
            } else {
                "alu_rcr"
            };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}({a}, {count}, {size})"),
            )?
        }
        Shl | Sal | Shr | Sar => {
            let a = read_operand(address, instruction, 0, size)?;
            let count = read_operand(address, instruction, 1, 8)?;
            let helper = match mnemonic {
                Shl | Sal => "alu_shl",
                Shr => "alu_shr",
                _ => "alu_sar",
            };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}({a}, {count}, {size})"),
            )?
        }
        Imul => match instruction.op_count() {
            1 => {
                let a = read_operand(address, instruction, 0, size)?;
                let helper = match size {
                    8 => "alu_imul1_8",
                    16 => "alu_imul1_16",
                    _ => "alu_imul1",
                };
                format!("{helper}({a});")
            }
            2 => {
                let a = read_operand(address, instruction, 0, size)?;
                let b = read_operand(address, instruction, 1, size)?;
                write_operand(
                    address,
                    instruction,
                    0,
                    size,
                    &format!("alu_imul({a}, {b}, {size})"),
                )?
            }
            3 => {
                let b = read_operand(address, instruction, 1, size)?;
                let c = read_operand(address, instruction, 2, size)?;
                write_operand(
                    address,
                    instruction,
                    0,
                    size,
                    &format!("alu_imul({b}, {c}, {size})"),
                )?
            }
            _ => return Err(unsupported(address, "imul with this operand form")),
        },
        Mul | Div | Idiv => {
            let a = read_operand(address, instruction, 0, size)?;
            let suffix = match size {
                8 => "8",
                16 => "16",
                _ => "",
            };
            match mnemonic {
                Mul => format!("alu_mul{suffix}({a});"),
                Div => format!("alu_div{suffix}({a}, {address}u);"),
                _ => format!("alu_idiv{suffix}({a}, {address}u);"),
            }
        }
        Cdq => "R_EDX = (uint32_t)((int32_t)R_EAX >> 31);".to_string(),
        Leave => "R_ESP = R_EBP; R_EBP = pop32();".to_string(),
        Ret => {
            let extra = if instruction.op_count() == 1 {
                addr(instruction.immediate(0))
            } else {
                0
            };
            format!("R_ESP += 4u + {extra}u; return;")
        }
        Call => match instruction.op0_kind() {
            OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = addr(instruction.near_branch_target());
                match setjmp_site {
                    // The first argument, the buffer, is above the return address just pushed.
                    // A `longjmp` to it arrives at the `else`, with the registers and EAX already
                    // set by the title's own code, and the next instruction runs.
                    Some(index) => format!(
                        "push32({next}u); lift_jmp_register(&lj{index}, rd32(R_ESP + 4u), {next}u, {index}u); \
                         if (sigsetjmp(lj{index}.env, 0) == 0) {{ LIFT_IN({address:#010x}u, LIFT_K_CALL, fn_{target:08x}()); }} \
                         C_{address:08x}:;"
                    ),
                    None => format!(
                        "push32({next}u); LIFT_IN({address:#010x}u, LIFT_K_CALL, fn_{target:08x}()); C_{address:08x}:;"
                    ),
                }
            }
            _ => {
                let target = read_operand(address, instruction, 0, 32)?;
                format!(
                    "{{ uint32_t t_target = {target}; push32({next}u); \
                     LIFT_IN({address:#010x}u, LIFT_K_CALL, rt_call(t_target, {address}u)); }} C_{address:08x}:;"
                )
            }
        },
        Jmp if instruction.op0_kind() != OpKind::NearBranch32
            && instruction.op0_kind() != OpKind::NearBranch64
            && table.is_none()
            && is_tail_call(instruction) =>
        {
            // A jump through an import's slot: the callee returns to our caller, so we return after it.
            let target = read_operand(address, instruction, 0, 32)?;
            format!(
                "{{ uint32_t t_target = {target}; LIFT_IN({address:#010x}u, LIFT_K_RET, rt_call(t_target, {address}u)); return; }}"
            )
        }
        Jmp if instruction.op0_kind() != OpKind::NearBranch32
            && instruction.op0_kind() != OpKind::NearBranch64 =>
        {
            let entries =
                table.ok_or_else(|| unsupported(address, "indirect jump without a table"))?;
            let index = read_register(address, instruction.memory_index())?;
            let mut cases = String::new();
            for (value, target) in entries {
                if body.contains(target) {
                    write!(cases, " case {value}u: goto L_{target:08x};").unwrap();
                } else {
                    write!(
                        cases,
                        " case {value}u: {{ LIFT_TAIL(fn_{target:08x}, {address:#010x}u); }}"
                    )
                    .unwrap();
                }
            }
            format!(
                "switch ({index}) {{{cases} default: lift_trap({address}u, \"jump table index out of range\"); }}"
            )
        }
        Jmp => {
            let target = addr(instruction.near_branch_target());
            if body.contains(&target) {
                if target <= address {
                    format!("LIFT_POLL({address:#010x}u); P_{address:08x}:; goto L_{target:08x};")
                } else {
                    format!("goto L_{target:08x};")
                }
            } else if target <= address {
                format!(
                    "LIFT_POLL({address:#010x}u); P_{address:08x}:; LIFT_TAIL(fn_{target:08x}, {address:#010x}u);"
                )
            } else {
                format!("LIFT_TAIL(fn_{target:08x}, {address:#010x}u);")
            }
        }
        other => return Err(unsupported(address, format!("instruction {other:?}"))),
    })
}
