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
 *   `cli_calc_slice.c`, `cli_calc_functions.c`, `cli_calc_binops.c`,
 *   `cli_calc_linalg.c`, and `cli_calc_lapack.c`.
 */

#ifndef CLI_CALC_INTERNAL_H
#define CLI_CALC_INTERNAL_H

#include "ImageStreamIO/ImageStruct.h"
#include "COREMOD_arith/image_coord.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_arith/image_stats.h"
#include "cli_calc_scalar.h"
#include "cli_calc_tokenizer.h"


extern int         parse_mode;
extern int         parse_error;
extern int         eval_error;
extern cli_token   eval_tokens[CLI_CALC_MAX_TOKENS];
extern int         eval_pos;
extern int         eval_ntok;
extern const char *cli_calc_src;
extern char        cli_calc_lasterr[256]; /* the last error message (cli_calc_prog.c reports it) */
extern unsigned long cli_calc_errs_printed; /* error messages parse_errmsg() has printed */

size_t cli_calc_tmp_begin(void);
void   cli_calc_tmp_end(size_t mark);

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

/** cli_calc_slice_view() - slice token `im[...]` as base/stride/extent per axis of @out_im; 0, 1 (binning:
 * not a view) or -1 (error reported with @missing for an absent image). @dyn: bounds hold variables. */
int cli_calc_slice_view(const char *tok_sval,
                        const char *missing,
                        IMAGE     **out_im,
                        int64_t    *base,
                        int64_t     stride[3],
                        uint32_t    vshape[3],
                        int        *naxis,
                        int        *dyn);

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
val_t cli_calc_store_var_or_im(const char *varname, val_t v, int sync_cli_var_always);
val_t cli_calc_eval_existing_var_assign(const char *varname, cli_token_type aop, val_t v);
void  cli_calc_rewrite_local_prefix(const char *input, char *out, size_t outsz);
void *cli_calc_keep_from_eval_tokens(void);
int   cli_calc_eval_multi_assign_bash(const char *input, int *out_rc);

extern int cli_calc_prog_default_shared;
imageID    cli_calc_prog_connect_stream(const char *name);

typedef struct
{
    cli_token *tokens;
    int        pos;
    int        ntok;
    int        mode;
} cli_calc_tok_state;

void cli_calc_tok_push(cli_token *new_tokens, int new_ntok, cli_calc_tok_state *saved);
void cli_calc_tok_pop(const cli_calc_tok_state *saved);
int  cli_calc_tok_range_eq(int b1, int e1, int b2, int e2);

#define CALC_UDEF_MAX 16
#define CALC_UDEF_MAXARGS 6

typedef struct
{
    char name[64];
    int  nargs;
    char arg_names[CALC_UDEF_MAXARGS][64];
    char body[512];
} cli_calc_udef_t;

void cli_calc_udef_clear(void);
int  cli_calc_udef_add(const char *name, int nargs, const char arg_names[][64], const char *body);
int  cli_calc_udef_find(const char *name, size_t len);
const cli_calc_udef_t *cli_calc_udef_get(int idx);
int                    cli_calc_try_parse_udef(const char *stmt);
int                    cli_calc_udef_eval_lookup(const char *name, val_t *out);
val_t                  cli_calc_func_udef(const cli_token *ftok);

const char *cli_calc_shift(const IMAGE *im, int64_t dx, int64_t dy);
const char *cli_calc_lut(const IMAGE *im, const IMAGE *tbl, double x0, double dx);
const char *cli_calc_medfilt(const IMAGE *im, int ksize);
const char *cli_calc_pixremap(const IMAGE *im, const IMAGE *map, double fill);
const char *cli_calc_pixunmap(const IMAGE *im, const IMAGE *map, int64_t nbins);
const char *cli_calc_rprof(const IMAGE *im, double cx, double cy);
const char *cli_calc_gauss(const IMAGE *im, double sigmax, double sigmay);
const char *cli_calc_interp2(const IMAGE *im, const IMAGE *xmap, const IMAGE *ymap);

/** cli_calc_fuse_expr() - evaluate the element-wise expression here in one pass: 1 done, 0 not fusible, -1 error. */
int cli_calc_fuse_expr(int min_prec, val_t *out);

/** cli_calc_fuse_assign() - `dst aop <expr>` written straight into @dst when possible (same returns). */
int cli_calc_fuse_assign(IMAGE *dst, cli_token_type aop);

/** cli_calc_fuse_assign_tok() - cli_calc_fuse_assign() for destination token @dst_tok, which may be a slice
 * (`out[0:63,:] = expr`, written through a view); @dst may then be NULL. */
int cli_calc_fuse_assign_tok(IMAGE *dst, const char *dst_tok, cli_token_type aop);

/** cli_calc_fuse_keep_tok() - cli_calc_fuse_keep() for destination token @dst_tok, which may be a slice. */
void *cli_calc_fuse_keep_tok(IMAGE *dst, const char *dst_tok, cli_token_type aop);

/** cli_calc_kept_free() - free a kept program (and the axis-reduction results it holds). */
void cli_calc_kept_free(void *kept);

/** cli_calc_kept_reads_image() - 1 if kept program @kept (NULL: none) reads image @name in any sub-program. */
int cli_calc_kept_reads_image(const void *kept, const char *name);

/** cli_calc_slice_template() - split slice token @tok into @bare, @tmpl (expression parts as '\001') and
 * the expression texts @parts; 0 and *@nparts on success, -1 if malformed. */
int cli_calc_slice_template(const char *tok,
                            char       *bare,
                            size_t      baresz,
                            char       *tmpl,
                            size_t      tmplsz,
                            char        parts[][64],
                            int         maxparts,
                            int        *nparts);
/** cli_calc_slice_geometry() - base/stride/vshape/naxis of numeric bracket text @btext over @im: 0 view,
 * 1 not a view (a pixel or binned), -1 error (message set). */
int cli_calc_slice_geometry(const IMAGE *im,
                            const char  *btext,
                            int64_t     *base,
                            int64_t      stride[3],
                            uint32_t     vshape[3],
                            int         *naxis);

/** cli_calc_kept_describe() - one line on how kept program @kept runs (pass kind, ops, precision, threads). */
void cli_calc_kept_describe(const void *kept, char *buf, size_t sz);

/** cli_calc_fuse_reject() - note why the expression being compiled is not fusible (first reason kept). */
void cli_calc_fuse_reject(const char *msg);
/** cli_calc_fuse_reject_reset() - forget the noted reason before a new keep attempt. */
void cli_calc_fuse_reject_reset(void);
/** cli_calc_fuse_reject_reason() - the noted reason of the last keep attempt, or "not fusible". */
const char *cli_calc_fuse_reject_reason(void);

/** cli_calc_select() - a new image, per pixel `c != 0 ? t : f` (image t/f, or number tc/fc when NULL), in
 * one pass; its name, or NULL when the images differ in size or are not real. */
const char *cli_calc_select(const IMAGE *c, const IMAGE *t, double tc, const IMAGE *f, double fc);

/* Output posting: at once, or collected during a unit tick and posted once per image by the flush. */
void cli_calc_post_output(IMAGE *im);
void cli_calc_post_defer(int on);

/** cli_calc_torn_count() - kept passes re-run, or run on a frame still being written, so far. */
uint64_t cli_calc_torn_count(void);
void     cli_calc_post_flush(void *processinfo);

/** cli_calc_eval_binop_im() - evaluate a single binary operator on same-shape real images or image+scalar. */
const char *cli_calc_eval_binop_im(cli_token_type op,
                                   const IMAGE   *li,
                                   double         lc,
                                   const IMAGE   *ri,
                                   double         rc);

/** cli_calc_eval_fn_im() - evaluate a 1-, 2-, or 3-arg math function on same-shape real images or scalars. */
const char *cli_calc_eval_fn_im(int nfn,
                                double (*fn)(),
                                const IMAGE *a,
                                double       ac,
                                const IMAGE *b,
                                double       bc,
                                double       c2);

/** cli_calc_coord() - a new image of @ref's size: coordinate @kind (CALC_COORD_*) of each pixel about (cx, cy). */
const char *cli_calc_coord(const IMAGE *ref, int kind, double cx, double cy);

/** cli_calc_coord_kind() - CALC_COORD_* for function token @t (ix iy ir itheta rand randn). */
int cli_calc_coord_kind(const cli_token *t);
int cli_calc_eval_const_scalar(double *out);

#define CALC_COORD_X ARITH_COORD_X
#define CALC_COORD_Y ARITH_COORD_Y
#define CALC_COORD_R ARITH_COORD_R
#define CALC_COORD_THETA ARITH_COORD_THETA
#define CALC_COORD_RAND ARITH_COORD_RAND
#define CALC_COORD_RANDN ARITH_COORD_RANDN

#define cli_calc_hash32 arith_hash32
#define cli_calc_rng_draw arith_rng_draw
#define cli_calc_next_rng_seed arith_next_rng_seed

typedef arith_reduce_op reduce_op;
#define RED_SUM ARITH_RED_SUM
#define RED_MEAN ARITH_RED_MEAN
#define RED_MIN ARITH_RED_MIN
#define RED_MAX ARITH_RED_MAX
#define RED_STD ARITH_RED_STD
#define RED_VAR ARITH_RED_VAR
#define RED_NORM ARITH_RED_NORM
#define RED_ASUM ARITH_RED_ASUM
#define RED_COUNT ARITH_RED_COUNT
#define RED_DOT ARITH_RED_DOT
#define RED_PERC ARITH_RED_PERC

#define cli_calc_read_im_pix arith_read_im_pix

/** cli_calc_fused_reduce_op() - reduce_op for reduction function pointer @fn, or -1. */
int cli_calc_fused_reduce_op(double (*fn)());

/** cli_calc_fuse_reduce() - evaluate reduction @op over fused expression(s) in one pass:
 * 1 done, 0 fallback. */
int cli_calc_fuse_reduce(reduce_op op, int is_long, val_t *out);

/** cli_calc_where() - `c ? t : f` for an image condition @c; @t and @f are images or numbers. */
val_t cli_calc_where(val_t c, val_t t, val_t f);

/** cli_calc_fuse_keep() - compile `dst aop <expr>` at the current token for repeated runs, variables read at
 * each run (free() it); NULL if not element-wise over images of dst's size (tokens rewound). */
void *cli_calc_fuse_keep(IMAGE *dst, cli_token_type aop);

/** cli_calc_fuse_keep_var() - compile `<var> aop <scalar-or-reduction-expr>` for repeated runs. */
void *cli_calc_fuse_keep_var(int vid, cli_token_type aop);

/** cli_calc_fuse_try_merge_kept() - merge @kept2 into @kept1 for single-pass block execution. */
int cli_calc_fuse_try_merge_kept(void *kept1, void *kept2, int elide_dst1);

/** cli_calc_fuse_keep_multi() - compile multi-target unpacking `a, b = ...` or `m, s = istats(im)`. */
void *cli_calc_fuse_keep_multi(int ndst, IMAGE *const *dst_im, const int *dst_vid);

/** cli_calc_inplace() - `dst aop src` (image @src, or number @val if @src is NULL), any datatypes. */
int cli_calc_inplace(IMAGE *dst, cli_token_type aop, const IMAGE *src, double val);

int          cli_calc_inplace_complex(IMAGE *dst, cli_token_type aop, const IMAGE *src, double val);
int          cli_calc_is_complex_dt(uint8_t dt);
const IMAGE *cli_calc_complex_image(const val_t *v);
val_t        cli_calc_complex_binop(cli_token_type op, val_t left, val_t right);
val_t        cli_calc_complex_neg(const IMAGE *z);
val_t        cli_calc_complex_part(const IMAGE *z, int which);
val_t        cli_calc_complex_func(const char *fname, const val_t *args, int nargs);

int   cli_calc_parse_func_argv(int nmin, int nmax, val_t *args, const char *comma_msg);
val_t cli_calc_func_eval_image_reduction(const cli_token *ftok);
val_t cli_calc_func_eval_string_or_format(const cli_token *ftok);
val_t cli_calc_func_replace(void);

int cli_calc_reject_complex(const val_t *v, int n);

/** cli_calc_long_binop() - exact 64-bit @a @op @b into *@out (double on overflow; `/` is not handled: 0);
 * *@err names a zero modulo. Return: 1 if @op was handled. */
int cli_calc_long_binop(cli_token_type op, long a, long b, val_t *out, const char **err);
/** cli_calc_cumsum() - cumulative sum of @im along @axis into a temporary image of its shape. */
const char *cli_calc_cumsum(const IMAGE *im, int axis);

typedef struct
{
    IMAGE  *img[2];
    int     nimg;
    int     kind;
    int     transA;
    int     transB;
    int     allow_reshape;
    double  alpha_c;
    double  beta_c;
    int16_t alpha_s1;
    int16_t beta_s1;
} cli_calc_kept_matrix;

#define cli_calc_flatten2d arith_image_flatten2d
int cli_calc_parse_matmul_arg(val_t *out, int *out_trans);

int cli_calc_compile_kept_matrix(cli_calc_kept_matrix *m,
                                 IMAGE                *dst,
                                 cli_token_type        aop,
                                 int                   keep_vars,
                                 int                   strict_local_dt);
int cli_calc_exec_kept_matrix(const cli_calc_kept_matrix *m, IMAGE *dst);

val_t cli_calc_func_transpose(void);
val_t cli_calc_func_reshape(void);
val_t cli_calc_func_vec(void);
val_t cli_calc_func_zeros(void);
val_t cli_calc_func_eye(void);
val_t cli_calc_func_diag(void);
val_t cli_calc_func_matmul(void);
val_t cli_calc_func_outer(void);
val_t cli_calc_func_gemm(int is_gemv);
val_t cli_calc_func_gemm_op(void);
val_t cli_calc_func_gemv_op(void);
val_t cli_calc_func_ger(void);
val_t cli_calc_func_syrk(void);
val_t cli_calc_func_symv(void);
val_t cli_calc_func_axpy(void);
val_t cli_calc_func_scal(void);

typedef struct
{
    IMAGE   *imA;
    IMAGE   *imB;
    double  *A;
    double  *B;
    uint64_t M;
    uint64_t N;
    uint64_t NRHS;
    long     out_nax;
    uint32_t out_sz[3];
    uint8_t  out_dt;
    int      n_extra;
    double   extra_scalar;
} cli_calc_ab_solver_ctx;

double *cli_calc_copy_im_to_dmat(const IMAGE *im, uint64_t rows, uint64_t cols, int trans);
val_t   cli_calc_write_dmat_to_new_im(const double  *src,
                                      uint64_t       nel,
                                      long           naxis,
                                      const uint32_t outsz[3],
                                      uint8_t        out_dt,
                                      const char    *fn_name);
int     cli_calc_prepare_ab_solver(const char             *fn,
                                   int                     allow_trans,
                                   int                     req_square,
                                   int                     max_extra,
                                   const char             *sq_err,
                                   const char             *ex_err,
                                   cli_calc_ab_solver_ctx *ctx);
int     cli_calc_resolve_rhs_dims(const IMAGE *imA,
                                  int          is_3d_a,
                                  uint64_t     M,
                                  uint64_t     N,
                                  const IMAGE *imB,
                                  uint64_t    *out_rhs,
                                  long        *out_nax,
                                  uint32_t     out_sz[3]);
int     cli_calc_solve_triangular_inplace(const double *restrict A,
                                          uint64_t N,
                                          int      transA,
                                          int      is_upper,
                                          int      is_unit,
                                          double   alpha,
                                          double *restrict X,
                                          uint64_t NRHS);
int cli_calc_solve_lu_inplace(double *restrict A, uint64_t N, double *restrict X, uint64_t NRHS);
int cli_calc_solve_cholesky_inplace(double *restrict A,
                                    uint64_t N,
                                    double *restrict X,
                                    uint64_t NRHS);

val_t cli_calc_func_trsv(int is_trsm);
val_t cli_calc_func_trsv_op(void);
val_t cli_calc_func_trsm_op(void);
val_t cli_calc_func_gesv(void);
val_t cli_calc_func_posv(void);
val_t cli_calc_func_gels(void);
val_t cli_calc_func_solve(void);
val_t cli_calc_func_inv(void);
val_t cli_calc_func_pinv(void);

const char *alloc_tmpname(void);
errno_t     cli_calc_tmp_image(const char *name,
                               long        naxis,
                               uint32_t   *size,
                               uint8_t     dt,
                               int         zero,
                               imageID    *id);

#include "cli_calc_prog.h"
#include "milk_calc_ext.h"

int    cli_calc_ext_find(const char *name);
void   cli_calc_ext_split_call(const char *sval,
                               char       *func_out,
                               size_t      fsz,
                               char       *label_out,
                               size_t      lsz);
void   cli_calc_ext_pack_params(const milk_calc_param_set_t *pset, void *pbuf, const double *vals);
double cli_calc_ext_resolve_param(const char                   *label,
                                  const milk_calc_param_spec_t *spec,
                                  int                          *out_vid);
void  *cli_calc_fuse_keep_user_op(int        ndst,
                                  const char lhs[][STRINGMAXLEN_IMAGE_NAME],
                                  const int *is_local,
                                  int        default_shared);
val_t  cli_calc_eval_user_call(const cli_token *ftok);

#endif /* CLI_CALC_INTERNAL_H */
