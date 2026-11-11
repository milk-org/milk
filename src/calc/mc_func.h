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
 *              names each one (MC_ID_<name>), for the code that treats a function specially.
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

/* One id per built-in function, MC_ID_<name>: code that treats a function specially compares
 * ids, not names. MC_ID_NONE is no built-in (an extension, a def). */
typedef enum
{
    MC_ID_NONE = 0,
    MC_ID_sin,
    MC_ID_cos,
    MC_ID_tan,
    MC_ID_asin,
    MC_ID_acos,
    MC_ID_atan,
    MC_ID_sinh,
    MC_ID_cosh,
    MC_ID_tanh,
    MC_ID_exp,
    MC_ID_exp2,
    MC_ID_log,
    MC_ID_log10,
    MC_ID_log2,
    MC_ID_sqrt,
    MC_ID_cbrt,
    MC_ID_erf,
    MC_ID_erfc,
    MC_ID_abs,
    MC_ID_ceil,
    MC_ID_floor,
    MC_ID_round,
    MC_ID_trunc,
    MC_ID_posi,
    MC_ID_sign,
    MC_ID_isnan,
    MC_ID_isfinite,
    MC_ID_wrap_pi,
    MC_ID_sinc,
    MC_ID_f32,
    MC_ID_f64,
    MC_ID_u8,
    MC_ID_i8,
    MC_ID_u16,
    MC_ID_i16,
    MC_ID_u32,
    MC_ID_i32,
    MC_ID_u64,
    MC_ID_i64,
    MC_ID_pow,
    MC_ID_atan2,
    MC_ID_hypot,
    MC_ID_fmod,
    MC_ID_min,
    MC_ID_max,
    MC_ID_clamp,
    MC_ID_where,
    MC_ID_ix,
    MC_ID_iy,
    MC_ID_ir,
    MC_ID_itheta,
    MC_ID_rand,
    MC_ID_randn,
    MC_ID_real,
    MC_ID_imag,
    MC_ID_arg,
    MC_ID_conj,
    MC_ID_cplx,
    MC_ID_expi,
    MC_ID_linspace,
    MC_ID_shift,
    MC_ID_roll,
    MC_ID_lut,
    MC_ID_pixremap,
    MC_ID_interp2,
    MC_ID_itot,
    MC_ID_imean,
    MC_ID_istd,
    MC_ID_irms,
    MC_ID_ivar,
    MC_ID_imin,
    MC_ID_imax,
    MC_ID_norm,
    MC_ID_asum,
    MC_ID_icount,
    MC_ID_iargmax,
    MC_ID_iargmin,
    MC_ID_iamax,
    MC_ID_dot,
    MC_ID_istats,
    MC_ID_icentroid,
    MC_ID_imedian,
    MC_ID_perc,
    MC_ID_trace,
    MC_ID_medfilt3,
    MC_ID_medfilt5,
    MC_ID_medfilt7,
    MC_ID_gauss,
    MC_ID_conv2,
    MC_ID_imin3,
    MC_ID_imax3,
    MC_ID_boxmean,
    MC_ID_cumsum,
    MC_ID_pixunmap,
    MC_ID_rprof,
    MC_ID_matmul,
    MC_ID_gemm,
    MC_ID_gemv,
    MC_ID_ger,
    MC_ID_outer,
    MC_ID_syrk,
    MC_ID_symv,
    MC_ID_axpy,
    MC_ID_scal,
    MC_ID_trsv,
    MC_ID_trsm,
    MC_ID_gesv,
    MC_ID_posv,
    MC_ID_gels,
    MC_ID_solve,
    MC_ID_inv,
    MC_ID_pinv,
    MC_ID_svd,
    MC_ID_eigh,
    MC_ID_einsum,
    MC_ID_fft,
    MC_ID_ifft,
    MC_ID_fftshift,
    MC_ID_eye,
    MC_ID_diag,
    MC_ID_transpose,
    MC_ID_reshape,
    MC_ID_vec,
    MC_ID_zeros,
    MC_ID_hstack,
    MC_ID_vstack,
    MC_ID_dstack,
    MC_ID_cat,
    MC_ID_rot90,
    MC_ID_strlen,
    MC_ID_toupper,
    MC_ID_tolower,
    MC_ID_substr,
    MC_ID_replace,
    MC_ID_hex,
    MC_ID_oct,
    MC_ID_bin,
    MC_ID_post,
    MC_ID_hist,
    MC_ID_prev,
    MC_ID_ema,
    MC_ID_COUNT
} mc_fn_id_t;

typedef struct
{
    const char  *name;             /* function identifier (without '(') */
    mc_fn_id_t   id;               /* MC_ID_<name> */
    int          min_args;         /* minimum argument count */
    int          max_args;         /* maximum argument count (-1 for variadic) */
    int          min_lhs;          /* minimum LHS targets when at statement root (normally 1) */
    int          max_lhs;          /* maximum LHS targets when unpacked (1..4) */
    mc_fn_kind_t kind;             /* semantic category */
    uint8_t      cast_dt;          /* _DATATYPE_* for explicit cast functions, else 0 */
    uint8_t      in_local;         /* 1 if supported by mcalc_local() fast path */
    uint8_t      in_old;           /* 1 if the old calculator supported it (not read) */
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

/** mc_func_name() - name of built-in function @id ("" for MC_ID_NONE), for messages. */
const char *mc_func_name(mc_fn_id_t id);

/** mc_func_replacement() - the function that replaced removed alias @name, or NULL. */
const char *mc_func_replacement(const char *name);

/** mc_const_removed() - hint when @name is one of the removed `_PI`-style constants, or NULL. */
const char *mc_const_removed(const char *name);

/** mc_is_reserved_const() - 1 if @name is the constant pi, nan or inf (reserved names). */
int mc_is_reserved_const(const char *name);

/** mc_is_stream_prefix() - 1 if @name is S or L, the stream prefix letters: not variable names. */
int mc_is_stream_prefix(const char *name);

/** mc_func_find_len() - look up a built-in function by name slice (@name, @len). */
const mc_func_t *mc_func_find_len(const char *name, size_t len);

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
