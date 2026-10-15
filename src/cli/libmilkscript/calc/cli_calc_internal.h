// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_internal.h
 * @brief Internal types and cross-file prototypes for the `calc/` expression evaluator.
 *
 * Role & Goal:
 *   Shares the expression result value (`val_t`), active parser state (`parse_mode`,
 *   `parse_error`, `eval_error`), token stream accessors, and slice/function/binary
 *   evaluation helpers across `cli_calc_parser.c`, `cli_calc_eval.c`,
 *   `cli_calc_slice.c`, `cli_calc_functions.c`, and `cli_calc_binop.c`.
 */

#ifndef CLI_CALC_INTERNAL_H
#define CLI_CALC_INTERNAL_H

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

extern int parse_mode;
extern int parse_error;
extern int eval_error;

cli_token *cur_parse(void);
cli_token *advance_parse(void);
cli_token *cur_eval(void);
cli_token *advance_eval(void);
void       parse_errmsg(const char *msg);

/** Return active current-token accessor (`cur_parse` or `cur_eval`). */
static inline cli_token *(*cli_calc_cur_func(void))(void)
{
    return (parse_mode == 0) ? cur_parse : cur_eval;
}

/** Return active advance-token accessor (`advance_parse` or `advance_eval`). */
static inline cli_token *(*cli_calc_advance_func(void))(void)
{
    return (parse_mode == 0) ? advance_parse : advance_eval;
}

val_t parse_expr(int min_prec);

val_t parse_primary(void);

val_t eval_binop(cli_token_type op, val_t left, val_t right);

val_t parse_funccall(cli_token *ftok);

val_t cli_calc_eval_image_slice_assign(const char *tok_sval, cli_token_type aop, val_t rhs);

val_t cli_calc_eval_image_whole_assign(const char *tok_sval, cli_token_type aop, val_t rhs);

val_t cli_calc_eval_image_slice_read(const char *tok_sval);

double to_double(val_t v);

val_t mk_long(long v);

val_t mk_double(double v);

val_t mk_string(const char *s);

int check_image(const char *name);

const char *alloc_tmpname(void);

#endif /* CLI_CALC_INTERNAL_H */
