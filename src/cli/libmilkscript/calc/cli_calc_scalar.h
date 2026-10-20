// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_scalar.h
 * @brief The calculator's scalar rules (operators, numbers, math functions), shared by the engine and
 * the in-shell evaluator of the bash client so that both compute the same results.
 */

#ifndef CLI_CALC_SCALAR_H
#define CLI_CALC_SCALAR_H

#include <stddef.h>

#include "cli_calc_tokenizer.h"

/** Expression result type tag */
typedef enum
{
    VAL_LONG,
    VAL_DOUBLE,
    VAL_STRING,
    VAL_GENERIC /**< For functions that don't return a specific value type */
} val_type;

/** Expression result value */
typedef struct
{
    val_type type;
    long     lval;
    double   dval;
    char     sval[CLI_CALC_TOKEN_MAXLEN];
} val_t;

/** A scalar math function: name with its '(', number of arguments (1 to 3), implementation. */
typedef struct
{
    const char *name;
    int         nargs;
    double (*fn)();
} cli_calc_scalar_fn_t;

/** cli_calc_op_prec() - binding power of binary operator @t (0 lowest), or -1 if @t is none. */
int cli_calc_op_prec(cli_token_type t);

/** cli_calc_op_right_assoc() - 1 if operator @t groups right to left (`^`, `?:`, assignments). */
int cli_calc_op_right_assoc(cli_token_type t);

/** cli_calc_scalar_binop() - number @l @op number @r into *@out; NULL, or the error to report. */
const char *cli_calc_scalar_binop(cli_token_type op, const val_t *l, const val_t *r, val_t *out);

/** cli_calc_scalar_fn() - the scalar math function whose name starts @p ("sin(..."), or NULL. */
const cli_calc_scalar_fn_t *cli_calc_scalar_fn(const char *p);

/** cli_calc_scan_op() - the operator or punctuation that starts @p and its length in *@len, or TOK_EOF. */
cli_token_type cli_calc_scan_op(const char *p, int *len);

/** cli_calc_scan_number() - the number literal that starts @p (decimal, 0x, 0o, 0b) as TOK_LONG or
 * TOK_DOUBLE in *@l / *@d; returns its length, 0 if @p holds none. */
size_t cli_calc_scan_number(const char *p, cli_token_type *type, long *l, double *d);

/** cli_calc_isnan() - IEEE-754 bit-level NaN test (1.0 if @a is NaN, else 0.0), safe under -Ofast. */
double cli_calc_isnan(double a);

/** cli_calc_isfinite() - IEEE-754 bit-level finite test (1.0 if @a is finite, else 0.0), safe under -Ofast. */
double cli_calc_isfinite(double a);

/** cli_calc_local() - evaluate @expr here when it holds only numbers, operators, math functions and
 * stream, FPS, process or sequencer properties: 1 with its value in @out (printed with @digits
 * significant digits) and its truth in *@truth; 0 when the engine must evaluate it (names, images,
 * strings, assignments, several statements, or an error to report). */
int cli_calc_local(const char *expr, int digits, char *out, size_t outsz, int *truth);

#endif /* CLI_CALC_SCALAR_H */
