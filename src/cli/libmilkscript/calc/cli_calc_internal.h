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

#include "ImageStreamIO/ImageStruct.h"
#include "cli_calc_scalar.h"
#include "cli_calc_tokenizer.h"


extern int  parse_mode;
extern int  parse_error;
extern int  eval_error;
extern char cli_calc_lasterr[256]; /* the last error message (cli_calc_prog.c reports it) */

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

int  check_image(const char *name);
void cli_calc_refresh_shared_image(IMAGE *im);

int   cli_calc_tok_pos(void);
void  cli_calc_tok_seek(int pos);
int   cli_calc_tokens_have_image(void);
val_t eval_existing_var_ref(const char *varname);

/** cli_calc_fuse_expr() - evaluate the element-wise expression here in one pass: 1 done, 0 not fusible, -1 error. */
int cli_calc_fuse_expr(int min_prec, val_t *out);

/** cli_calc_fuse_assign() - `dst aop <expr>` written straight into @dst when possible (same returns). */
int cli_calc_fuse_assign(IMAGE *dst, cli_token_type aop);

/** cli_calc_select() - a new image, per pixel `c != 0 ? t : f` (image t/f, or number tc/fc when NULL), in
 * one pass; its name, or NULL when the images differ in size or are not real. */
const char *cli_calc_select(const IMAGE *c, const IMAGE *t, double tc, const IMAGE *f, double fc);

/** cli_calc_coord() - a new image of @ref's size: coordinate @kind (CALC_COORD_*) of each pixel about (cx, cy). */
const char *cli_calc_coord(const IMAGE *ref, int kind, double cx, double cy);

/** cli_calc_coord_kind() - CALC_COORD_* for function token @t (ix iy ir itheta). */
int cli_calc_coord_kind(const cli_token *t);

enum
{
    CALC_COORD_X,
    CALC_COORD_Y,
    CALC_COORD_R,
    CALC_COORD_THETA
};

/** cli_calc_where() - `c ? t : f` for an image condition @c; @t and @f are images or numbers. */
val_t cli_calc_where(val_t c, val_t t, val_t f);

/** cli_calc_fuse_keep() - compile `dst aop <expr>` at the current token for repeated runs, variables read at
 * each run (free() it); NULL if not element-wise over images of dst's size (tokens rewound). */
void *cli_calc_fuse_keep(IMAGE *dst, cli_token_type aop);

/** cli_calc_inplace() - `dst aop src` (image @src, or number @val if @src is NULL), any datatypes. */
int cli_calc_inplace(IMAGE *dst, cli_token_type aop, const IMAGE *src, double val);

int          cli_calc_inplace_complex(IMAGE *dst, cli_token_type aop, const IMAGE *src, double val);
int          cli_calc_is_complex_dt(uint8_t dt);
const IMAGE *cli_calc_complex_image(const val_t *v);
val_t        cli_calc_complex_binop(cli_token_type op, val_t left, val_t right);
val_t        cli_calc_complex_neg(const IMAGE *z);
val_t        cli_calc_complex_part(const IMAGE *z, int which);
val_t        cli_calc_complex_func(const char *fname, const val_t *args, int nargs);

int cli_calc_matmul_check(const IMAGE *im1,
                          int          transA,
                          const IMAGE *im2,
                          int          transB,
                          const IMAGE *dst);
int cli_calc_matmul_into(const IMAGE *im1, int transA, const IMAGE *im2, int transB, IMAGE *outim);
int cli_calc_transpose_check(const IMAGE *src, const IMAGE *dst);
int cli_calc_transpose_into(const IMAGE *src, IMAGE *dst);

const char *alloc_tmpname(void);

#endif /* CLI_CALC_INTERNAL_H */
