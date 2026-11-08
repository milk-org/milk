// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mcalc.h
 * @brief Public API for the milk expression and image calculator (`libmilkcalc`).
 *
 * Role & Goal:
 *   Exposes the single-engine calculator entry points used by `milk-cli`, `milk-call`,
 *   `libmilkbash.so`, `libmilkscript`, and the standalone `milk-fpsexec-milk-calc`
 *   compute unit:
 *     - Interactive and bash-mode expression evaluation (`mcalc_eval_bash`,
 *       `mcalc_eval_str`, `mcalc_eval_math_to_val`).
 *     - Client-side zero-RPC fast path for scalar and `s.`/`f.`/`p.` expressions
 *       (`mcalc_local`).
 *     - Multi-statement compiled programs (`mcalc_prog_new`, `mcalc_prog_run`,
 *       `mcalc_prog_explain`) and user-defined inline functions (`mcalc_udef_*`).
 *     - User C extension loading and FPS parameter discovery (`mcalc_ext_*`).
 */

#ifndef MILK_MCALC_H
#define MILK_MCALC_H

#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "milk_calc_ext.h"

/* Output formatting modes for mcalc_eval_bash() and mcalc_eval_str() */
enum
{
    MCALC_OUT_VALUES, /* the values of non-assignments */
    MCALC_OUT_BIND,   /* one line per statement: its value, an assignment's too */
    MCALC_OUT_PIXELS, /* as BIND, with an image's pixels on its line */
    MCALC_OUT_REPL    /* interactive calc mode: <val>, <var> <val>, or [<name> ]<dims> <type> */
};

extern int  mcalc_out_mode;
extern int  mcalc_test_image;
extern int  mcalc_float_digits;
extern char mcalc_lasterr[256];

/* Set from any thread or a signal handler to stop an evaluation at its next program step or loop
 * iteration, which then fails with "interrupted"; the host clears it before each request. */
extern volatile sig_atomic_t mcalc_cancelled;

/** mcalc_eval_bash() - evaluate an expression for @calc or @test in bash mode (1 on ok, 0 on err). */
int mcalc_eval_bash(const char *input, int is_test, int *out_rc);

/** mcalc_eval_str() - evaluate an expression with output written to @out instead of stdout. */
int mcalc_eval_str(const char *input, int is_test, int *out_rc, char *out, size_t outsz);

/** mcalc_eval_math_to_val() - evaluate @input as a pure numeric math expression silently. */
int mcalc_eval_math_to_val(const char *input, int *out_type, long *out_lval, double *out_dval);

/** mcalc_local() - evaluate @expr locally without a server (1 on local eval, 0 if server needed). */
int mcalc_local(const char *expr, int digits, char *out, size_t outsz, int *truth);

/** mcalc_tmp_cleanup() - delete every temporary _tmpcalcN image still tracked. */
void mcalc_tmp_cleanup(void);

typedef void (*mcalc_ext_param_cb_t)(const char                   *label,
                                     const milk_calc_param_spec_t *spec,
                                     int                           is_op,
                                     void                         *ud);

typedef void (*mcalc_ext_scalar_out_cb_t)(const char *var_name, const char *descr, void *ud);

/** mcalc_ext_load() - compile (if .c) and load a user C extension into the calculator. */
int mcalc_ext_load(const char *path, char *err, size_t errsz);

/** mcalc_ext_load_list() - load a comma/colon-separated list of user C extensions. */
int mcalc_ext_load_list(const char *libs_str, char *err, size_t errsz);

/** mcalc_ext_count() - return the number of registered calculator C extensions. */
int mcalc_ext_count(void);

/** mcalc_ext_get() - return registered calculator C extension @idx, or NULL. */
const milk_calc_ext_entry_t *mcalc_ext_get(int idx);

/** mcalc_ext_scan_source() - scan calculator source @src for unbound extension parameters. */
void mcalc_ext_scan_source(const char               *src,
                           mcalc_ext_param_cb_t      param_cb,
                           mcalc_ext_scalar_out_cb_t scalar_out_cb,
                           void                     *ud);

/* A calculator variable bound to a value outside the calculator (a unit's FPS parameter) */
typedef struct
{
    const char *name; /* kept by the caller while the program exists */
    uint64_t    type; /* FPTYPE_INT64 UINT64 FLOAT64 ONOFF (int32_t) or STRING (char[64]) */
    void       *ptr;
} mcalc_binding;

typedef struct mcalc_prog mcalc_prog;

/** mcalc_prog_attach() - in an fpsexec process, work on the runtime's image and variable tables. */
void mcalc_prog_attach(void);

/** mcalc_prog_new() - compile @text into a reusable calculator program, or NULL on error. */
mcalc_prog *mcalc_prog_new(const char          *text,
                           const mcalc_binding *vars,
                           int                  nvars,
                           char                *err,
                           size_t               errsz);

/** mcalc_prog_assigns() - return 1 if a statement in @p assigns bound variable @i. */
int mcalc_prog_assigns(const mcalc_prog *p, int i);

/** mcalc_prog_run() - set bound variables, run @p, copy back assigned ones, and post outputs. */
int mcalc_prog_run(mcalc_prog *p, char *err, size_t errsz);

/** mcalc_torn_count() - return the cumulative count of torn SHM reads (_torn). */
uint64_t mcalc_torn_count(void);

/** mcalc_prog_set_processinfo() - attach the unit's PROCESSINFO (or NULL) for output stream posts. */
void mcalc_prog_set_processinfo(mcalc_prog *p, void *processinfo);

/** mcalc_prog_exec_once() - execute @text once (for .init / .initfile), sharing new images. */
int mcalc_prog_exec_once(const char          *text,
                         const mcalc_binding *vars,
                         int                  nvars,
                         unsigned char       *assigned_out,
                         char                *err,
                         size_t               errsz);

/** mcalc_prog_free() - free a compiled calculator program. */
void mcalc_prog_free(mcalc_prog *p);

/** mcalc_prog_explain() - compile @text as a unit would and print its execution plan to @out. */
int mcalc_prog_explain(const char *text, FILE *out);

/** mcalc_print_catalog() - print the calculator function and extension catalog to stdout. */
int mcalc_print_catalog(const char *pattern, int json_mode);

/* User-defined inline function (def f(x, ...) = expr) entry for state inspection/snapshotting */
#define MCALC_UDEF_MAX 64
#define MCALC_UDEF_MAXARGS 8

struct mc_expr;

typedef struct
{
    char            name[64];
    int             nargs;
    char            arg_names[MCALC_UDEF_MAXARGS][64];
    struct mc_expr *body; /* parsed once; each call inlines a copy */
} mcalc_udef_t;

/** mcalc_udef_clear() - clear all registered user-defined inline calculator functions. */
void mcalc_udef_clear(void);

/** mcalc_udef_count() - return the number of registered user-defined inline functions. */
int mcalc_udef_count(void);

/** mcalc_udef_get() - return registered user-defined inline function @idx, or NULL. */
const mcalc_udef_t *mcalc_udef_get(int idx);

/** mcalc_udef_add() - register or overwrite a user-defined inline function; takes @body. */
int mcalc_udef_add(const char *name, int nargs, const char arg_names[][64], struct mc_expr *body);

/** mcalc_udef_try_parse() - parse and register a `def f(x, ...) = expr` statement (1/0/-1). */
int mcalc_udef_try_parse(const char *stmt);

#endif /* MILK_MCALC_H */
