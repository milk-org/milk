// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef CLI_CALC_INTERNAL_H
#define CLI_CALC_INTERNAL_H

#include "cli_calc_tokenizer.h"

/** Expression result type tag */
typedef enum
{
    VAL_LONG,
    VAL_DOUBLE,
    VAL_STRING,
    VAL_GENERIC // For functions that don't return a specific value type
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
val_t      parse_expr(int min_prec);

val_t eval_binop(cli_token_type op, val_t left, val_t right);
val_t parse_funccall(cli_token *ftok);

double      to_double(val_t v);
val_t       mk_long(long v);
val_t       mk_double(double v);
val_t       mk_string(const char *s);
int         check_image(const char *name);
const char *alloc_tmpname(void);

val_t parse_expr(int min_prec);
val_t parse_primary(void);

#endif
