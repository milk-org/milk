// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_func.h
 * @brief Built-in calculator function catalog and scalar math helper declarations.
 *
 * Purpose:     each built-in function is defined once, so the parser, the resolver, the planner,
 *              the C code generator, both scalar evaluators (session and shell) and completion
 *              agree on what a name means; a removed name fails with its replacement.
 * Does:        mc_fn_kind_t classifies the functions by how they run (per pixel, reduction,
 *              library pre-stage, BLAS/LAPACK/FFT, string, program directive); mc_fn_id_t
 *              names each one (MC_FID_<name>), for the code that treats a function specially.
 *              An mc_func_t entry holds the id, the argument and LHS-target counts, the cast type, whether the
 *              shell's evaluator (mcalc_local()) has it, the scalar C callbacks (fn1/fn2/fn3)
 *              and the names the float32 and float64 kernels call (c_f32/c_f64). Lookups by
 *              name or name slice; the replacements of removed aliases (ln, fabs, ...) and of
 *              the `_PI`-style constants; the reserved constants pi, nan and inf; the scalar
 *              helpers, with the kernels' formulas. mc_func.c also implements mc_truth()
 *              (front/mc_ast.h) and mcalc_print_catalog() (mcalc.h, `--list`).
 * Used by:     the calculator's front end, planner, JIT and scalar evaluators (through
 *              front/mc_ast.h); ext/mc_ext_registry.c and mc_eval.c (an extension or a def may
 *              not take a built-in name); the session: milk_server_complete.c (completion of
 *              function names), milk_server_vars.c (reserved constants).
 * Uses:        libc and libm; mc_func.c also ImageStreamIO/ImageStruct.h (the cast types) and
 *              the extension registry (milk_calc_ext.h).
 * See also:    mc_func.c; jit/mc_prelude.h (the kernels' helpers); docs/guide/calculator.md,
 *              "Functions"; docs/guide/scripting.md, "Migrating Old Scripts" (removed names).
 * Constraints: mc_func.c is also compiled into libmilkbash.so, the thin client every shell
 *              loads, which does not link libmilkcalc: it may call only the calculator files
 *              that target also compiles (ext/mc_ext_registry.c, run/mc_scalar_ops.c, ...).
 */

#ifndef MILK_MC_FUNC_H
#define MILK_MC_FUNC_H

#include <stddef.h>
#include <stdint.h>

typedef enum
{
    MC_FN_SCALAR = 0, /* elementwise scalar/image math & casts (sin, pow, clamp, f32...) */
    MC_FN_WHERE,      /* where(cond, t, f) */
    MC_FN_COORD,      /* ix, iy, ir, itheta */
    MC_FN_PRNG,       /* rand, randn */
    MC_FN_CPLX,       /* real, imag, arg, conj, cplx, expi */
    MC_FN_SHIFT,      /* shift, roll */
    MC_FN_LUT,        /* lut(x, tbl[, x0, dx]) */
    MC_FN_GATHER,     /* pixremap, interp2 */
    MC_FN_LINSPACE,   /* linspace(a, b, n) */
    MC_FN_REDUCE_D,   /* itot, imean, istd, irms, ivar, imin, imax, norm, asum */
    MC_FN_REDUCE_L,   /* icount, iargmax, iargmin, iamax */
    MC_FN_DOT,        /* dot(a, b) */
    MC_FN_ISTATS,     /* istats(im) */
    MC_FN_ICENTROID,  /* icentroid(im[, th]) */
    MC_FN_EAGER_RED,  /* imedian, perc, trace */
    MC_FN_STENCIL_LIB, /* medfilt3/5/7, gauss, conv2, imin3, imax3, boxmean, cumsum, pixunmap, rprof */
    MC_FN_LINALG, /* matmul..zeros, svd, eigh, einsum, fft, ifft, fftshift, *stack, cat, rot90 */
    MC_FN_STR,    /* strlen, toupper, tolower, substr, replace, hex, oct, bin */
    MC_FN_PROG    /* post, hist, prev, ema */
} mc_fn_kind_t;

/* One id per built-in function, MC_FID_<name>: code that treats a function specially compares
 * ids, not names. MC_FID_NONE is no built-in (an extension, a def). */
typedef enum
{
    MC_FID_NONE = 0,
    MC_FID_sin,
    MC_FID_cos,
    MC_FID_tan,
    MC_FID_asin,
    MC_FID_acos,
    MC_FID_atan,
    MC_FID_sinh,
    MC_FID_cosh,
    MC_FID_tanh,
    MC_FID_exp,
    MC_FID_exp2,
    MC_FID_log,
    MC_FID_log10,
    MC_FID_log2,
    MC_FID_sqrt,
    MC_FID_cbrt,
    MC_FID_erf,
    MC_FID_erfc,
    MC_FID_abs,
    MC_FID_ceil,
    MC_FID_floor,
    MC_FID_round,
    MC_FID_trunc,
    MC_FID_posi,
    MC_FID_sign,
    MC_FID_isnan,
    MC_FID_isfinite,
    MC_FID_wrap_pi,
    MC_FID_sinc,
    MC_FID_f32,
    MC_FID_f64,
    MC_FID_u8,
    MC_FID_i8,
    MC_FID_u16,
    MC_FID_i16,
    MC_FID_u32,
    MC_FID_i32,
    MC_FID_u64,
    MC_FID_i64,
    MC_FID_pow,
    MC_FID_atan2,
    MC_FID_hypot,
    MC_FID_fmod,
    MC_FID_min,
    MC_FID_max,
    MC_FID_clamp,
    MC_FID_where,
    MC_FID_ix,
    MC_FID_iy,
    MC_FID_ir,
    MC_FID_itheta,
    MC_FID_rand,
    MC_FID_randn,
    MC_FID_real,
    MC_FID_imag,
    MC_FID_arg,
    MC_FID_conj,
    MC_FID_cplx,
    MC_FID_expi,
    MC_FID_linspace,
    MC_FID_shift,
    MC_FID_roll,
    MC_FID_lut,
    MC_FID_pixremap,
    MC_FID_interp2,
    MC_FID_itot,
    MC_FID_imean,
    MC_FID_istd,
    MC_FID_irms,
    MC_FID_ivar,
    MC_FID_imin,
    MC_FID_imax,
    MC_FID_norm,
    MC_FID_asum,
    MC_FID_icount,
    MC_FID_iargmax,
    MC_FID_iargmin,
    MC_FID_iamax,
    MC_FID_dot,
    MC_FID_istats,
    MC_FID_icentroid,
    MC_FID_imedian,
    MC_FID_perc,
    MC_FID_trace,
    MC_FID_medfilt3,
    MC_FID_medfilt5,
    MC_FID_medfilt7,
    MC_FID_gauss,
    MC_FID_conv2,
    MC_FID_imin3,
    MC_FID_imax3,
    MC_FID_boxmean,
    MC_FID_cumsum,
    MC_FID_pixunmap,
    MC_FID_rprof,
    MC_FID_matmul,
    MC_FID_gemm,
    MC_FID_gemv,
    MC_FID_ger,
    MC_FID_outer,
    MC_FID_syrk,
    MC_FID_symv,
    MC_FID_axpy,
    MC_FID_scal,
    MC_FID_trsv,
    MC_FID_trsm,
    MC_FID_gesv,
    MC_FID_posv,
    MC_FID_gels,
    MC_FID_solve,
    MC_FID_inv,
    MC_FID_pinv,
    MC_FID_svd,
    MC_FID_eigh,
    MC_FID_einsum,
    MC_FID_fft,
    MC_FID_ifft,
    MC_FID_fftshift,
    MC_FID_eye,
    MC_FID_diag,
    MC_FID_transpose,
    MC_FID_reshape,
    MC_FID_vec,
    MC_FID_zeros,
    MC_FID_hstack,
    MC_FID_vstack,
    MC_FID_dstack,
    MC_FID_cat,
    MC_FID_rot90,
    MC_FID_strlen,
    MC_FID_toupper,
    MC_FID_tolower,
    MC_FID_substr,
    MC_FID_replace,
    MC_FID_hex,
    MC_FID_oct,
    MC_FID_bin,
    MC_FID_post,
    MC_FID_hist,
    MC_FID_prev,
    MC_FID_ema,
    MC_FID_COUNT
} mc_fn_id_t;

typedef struct
{
    const char  *name;             /* function identifier (without '(') */
    mc_fn_id_t   id;               /* MC_FID_<name> */
    int          min_args;         /* minimum argument count */
    int          max_args;         /* maximum argument count (-1 for variadic) */
    int          min_lhs;          /* minimum LHS targets when at statement root (normally 1) */
    int          max_lhs;          /* maximum LHS targets when unpacked (1..4) */
    mc_fn_kind_t kind;             /* semantic category */
    uint8_t      cast_dt;          /* _DATATYPE_* for explicit cast functions, else 0 */
    uint8_t      in_local;         /* 1 if supported by mcalc_local() fast path */
    double (*fn1)(double);         /* unary scalar C implementation, if any */
    double (*fn2)(double, double); /* binary scalar C implementation, if any */
    double (*fn3)(double, double, double); /* ternary scalar C implementation, if any */
    const char *c_f32;                     /* C prelude / expression name in float32 pass */
    const char *c_f64;                     /* C prelude / expression name in float64 pass */
    const char *signature;                 /* call signature, for --list */
    const char *help;                      /* one-line description, for --list */
    /* The kinds of the arguments, checked by the resolver (NULL: not checked here). A letter per
     * argument: I image, n number, v number or string, x number or image, S string, . any.
     * [..] is an optional group, given whole or not at all; * repeats the letter before it;
     * | separates alternatives. */
    const char *args;
} mc_func_t;

/** mc_func_find() - look up a built-in function by exact name (without '('). */
const mc_func_t *mc_func_find(const char *name);

/** mc_func_name() - name of built-in function @id ("" for MC_FID_NONE), for messages. */
const char *mc_func_name(mc_fn_id_t id);

/** mc_func_replacement() - the function that replaced removed alias @name, or NULL. */
const char *mc_func_replacement(const char *name);

/** mc_const_removed() - hint when @name is one of the removed `_PI`-style constants, or NULL. */
const char *mc_const_removed(const char *name);

/** mc_is_reserved_const() - 1 if @name is the constant pi, nan or inf (reserved names). */
int mc_is_reserved_const(const char *name);

/** mc_is_stream_prefix() - 1 if @name is S or L, the stream prefix letters: not variable names. */
int mc_is_stream_prefix(const char *name);

/** mc_func_table() - return the built-in function table and write its length to @out_count. */
const mc_func_t *mc_func_table(int *out_count);

/** mc_fmodd() - fmod(@x, @y) by the kernels' exact FMA formula; NaN when |@x / @y| >= 2^52. */
double mc_fmodd(double x, double y);

/** mc_isnan() - IEEE-754 bit-level NaN check (returns 1.0 if @a is NaN, else 0.0). */
double mc_isnan(double a);

/** mc_isfinite() - IEEE-754 bit-level finite check (returns 1.0 if @a is finite, else 0.0). */
double mc_isfinite(double a);

/** mc_posi() - Heaviside step (1.0 if @a > 0, else 0.0). */
double mc_posi(double a);

/** mc_sign() - signum (-1.0, 0.0, or +1.0). */
double mc_sign(double a);

/** mc_wrap_pi() - wrap angle @a (radians) into (-pi, pi]; NaN when |@a| >= 2^52 * 2 pi. */
double mc_wrap_pi(double a);

/** mc_sinc() - normalized sinc sin(pi*x)/(pi*x), with mc_sinc(0) == 1.0. */
double mc_sinc(double a);

/** mc_clamp() - clamp @a to [@lo, @hi]. */
double mc_clamp(double a, double lo, double hi);

#endif /* MILK_MC_FUNC_H */
