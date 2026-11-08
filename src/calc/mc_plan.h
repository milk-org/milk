// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_plan.h
 * @brief Planner, C kernel ABI, view geometry, and program execution structures for `libmilkcalc`.
 *
 * Defines the intermediate representation lowered from a resolved AST:
 *   - `mc_operand_t`: ABI-stable runtime operand descriptor passed to compiled `.so` kernels.
 *   - `mc_view_t`: Resolved n-D slice/stride/binning geometry for an image operand.
 *   - `mc_pass_t`: Fused elementwise pass (operands, outputs, precision, generated C source,
 *     and `dlopen()`-loaded `mc_kernel_fn`).
 *   - `mc_ctor_step_t`: Whole-image library step (BLAS, LAPACK, FFT, spatial filter, reduction
 *     pre-stage, or user C extension).
 *   - `mcalc_prog_t`: Compiled multi-statement program with image slots, bound FPS variables,
 *     and execution steps.
 */

#ifndef MILK_CALC_MC_PLAN_H
#define MILK_CALC_MC_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "ImageStreamIO/ImageStruct.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "mc_ast.h"
#include "mcalc.h"

/* ABI-stable operand descriptor passed to compiled C kernels. */
typedef struct mc_operand
{
    void          *base;          /* typed data pointer + view byte offset */
    int64_t        st[3];         /* element stride per axis (0 broadcast, <0 reversed) */
    uint32_t       vnx, vny, vnz; /* view extents */
    double         scalar;        /* scalar parameter value */
    const uint8_t *mask;          /* mask pointer when masked */
    uint32_t       cnt0_snapshot; /* cnt0 sampled before pass for torn-frame check */
} mc_operand_t;

typedef void (*mc_kernel_fn)(mc_operand_t *ops, uint64_t y0, uint64_t y1);

/* Resolved slice/view geometry for a single image operand. */
typedef struct
{
    imageID        im_id;
    imageID        owned_tmp_id; /* >= 0 if materialized ::Nb temporary */
    uint8_t        dt;
    uint8_t        naxis;
    uint32_t       vnx, vny, vnz;
    int64_t        offset_elem; /* starting element offset from im->array.raw */
    int64_t        st[3];       /* signed element stride per axis */
    int            is_contig;   /* 1 if un-sliced un-reversed contiguous full image */
    const uint8_t *mask_ptr;    /* non-NULL if masked LHS view */
    uint8_t        mask_dt;
} mc_view_t;

/* Planned operand inside an elementwise pass. */
typedef struct
{
    int              slot_idx;         /* index in prog->slots[], or -1 if scalar */
    uint8_t          dt;               /* _DATATYPE_* for image operand, 0 for scalar */
    int              is_scalar;        /* 1 if scalar parameter */
    int              is_out;           /* 1 if output operand */
    int              is_inplace_rw;    /* 1 if +=, -=, *=, /= or slice/mask assign */
    int              is_non_pointwise; /* 1 if shift, roll, lut table, pixremap, interp2 */
    int              is_table_arg;     /* 1 if table/source arg not participating in pass shape */
    int              is_expr_im;       /* 1 if pre-evaluated image sub-expression in slice_expr */
    int              has_slice;        /* 1 if operand is an MC_EXPR_SLICE */
    const mc_expr_t *slice_expr;       /* slice AST node when has_slice == 1 */
    const mc_expr_t *scalar_expr;      /* scalar AST node when is_scalar == 1 */
    int              check_div_kind;   /* 1 = check != 0 (/ or /=), 2 = check != 0 (%) */
    int              comp_gen_idx;     /* >= 0 if comprehension generator variable */
    int              is_x_bcast;       /* 1 if axis-0 broadcast (st[0] == 0) */
} mc_op_desc_t;

/* Planned output write inside an elementwise pass. */
typedef struct
{
    int              op_idx;      /* index in pass->ops[], or -1 if elided local */
    int              mask_op_idx; /* index of mask operand in pass->ops[], or -1 */
    int              slot_idx;    /* index in prog->slots[] */
    int              stmt_idx;    /* statement index in program */
    mc_tok_type_t    assign_op;   /* MC_TOK_ASSIGN, MC_TOK_PLUS_ASSIGN, etc. */
    const mc_expr_t *rhs_expr;    /* RHS expression tree */
    const mc_expr_t *lhs_slice;   /* LHS slice node if slice/mask assignment */
    uint8_t          out_dt;      /* destination datatype */
    int              is_elided;   /* 1 if elided to register local _l<slot_idx> */
    int              defer_post;  /* 1 if shared stream post is deferred */
} mc_pass_out_t;

/* Compiled elementwise pass. */
typedef struct
{
    mc_op_desc_t    *ops;
    int              nops;
    int              cap_ops;
    mc_pass_out_t   *outs;
    int              nouts;
    int              cap_outs;
    int              work_double;            /* 1 = double, 0 = float */
    int              is_complex;             /* 1 = _Complex float/double */
    int              is_1d_contig;           /* 1 = 1D contiguous fast loop */
    int              no_dst_bcast;           /* 1 = do not broadcast pass to dst shape */
    int              uses_coords_or_stencil; /* 1 = needs 2D/3D (x, y, z) loop */
    int              is_comp;                /* 1 = comprehension pass */
    int              prestage_cnt;           /* number of hoisted pre-stage steps */
    const mc_expr_t *comp_expr;              /* MC_EXPR_COMP node when is_comp == 1 */
    uint8_t          naxis;
    uint32_t         vnx, vny, vnz;
    char            *c_src;
    char             so_path[512];
    mc_kernel_fn     fn;
} mc_pass_t;

/* Runtime image slot in a planned program. */
typedef struct
{
    char     name[256];
    char     bare[256];
    int      loc;             /* 0 default, 1 @S:, 2 @L: */
    int      is_local;        /* 1 if declared with `local` */
    int      is_elided;       /* 1 if elided to register local */
    int      existed_at_plan; /* 1 if image existed before mcalc_prog_new */
    imageID  im_id;
    uint64_t checked_gen;
    uint8_t  orig_dt;
    uint8_t  plan_dt;
    uint8_t  plan_naxis;
    uint32_t plan_size[3];
} mc_slot_t;

typedef enum
{
    MC_STEP_SCALAR = 0,
    MC_STEP_PASS,
    MC_STEP_CTOR
} mc_step_kind_t;

/* Constructor / shape operation descriptor. */
typedef struct
{
    int              slot_idx;
    int              is_anon;
    mc_tok_type_t    assign_op;
    const mc_expr_t *lhs_slice;
    const mc_expr_t *target_expr;
    const mc_expr_t *rhs_expr;
    const mc_stmt_t *multi_stmt;
    void            *csr;         /* cached arith_scatter_csr_t* for pixunmap */
    double          *scratch_buf; /* cached scratch buffer for rprof */
    size_t           scratch_cap;
    void            *ext_state;
    void (*ext_state_free)(void *);
    int     ema_init;
    imageID ema_im_id;
} mc_ctor_step_t;

/* Execution step in a planned program. */
typedef struct
{
    mc_step_kind_t kind;
    int            first_stmt_idx;
    int            last_stmt_idx;
    int            is_hist_step; /* 1 if appended by hist() ring rewriter */
    mc_stmt_t     *scalar_stmt;  /* for MC_STEP_SCALAR */
    mc_pass_t      pass;         /* for MC_STEP_PASS */
    mc_ctor_step_t ctor;         /* for MC_STEP_CTOR */
} mc_step_t;

/* Planned program object (`mcalc_prog_t`). */
typedef struct mcalc_prog
{
    char          *src_text;
    mc_stmt_t     *ast_head;
    mc_slot_t     *slots;
    int            nslots;
    int            cap_slots;
    mc_step_t     *steps;
    int            nsteps;
    int            cap_steps;
    int           *stmt_to_step; /* maps statement index -> step index */
    char         **stmt_texts;   /* per-statement source text for explain */
    int            nstmts;
    mcalc_binding *vars;
    long          *var_ids;
    unsigned char *assigned;
    int            nvars;
    void          *processinfo;
    uint64_t       plan_gen;    /* milk_table_generation() at plan time */
    imageID        last_im_id;  /* last produced/updated imageID */
    mc_val_t       last_scalar; /* last scalar value if final step was scalar */
    int            last_was_im; /* 1 if final step produced an image */
} mcalc_prog_t;

/* Run-time C compiles (mc_cc.c), shared by JIT kernels and `-l` extensions: the checked cache
 * directory, a cache key that covers compiler, flags and CPU, and the compile step itself. */
int  mc_cc_cache_dir(char *dir_out, size_t dir_sz, char *err, size_t errsz);
void mc_cc_key(const char *text, char key[33]);
int  mc_cc_build(const char        *c_path,
                 const char        *so_path,
                 const char *const *extra,
                 char              *err,
                 size_t             errsz);

/* Resolve an image or slice expression into a concrete view `mc_view_t`. */
int mc_view_resolve(const mc_expr_t *expr,
                    imageID          base_im_id,
                    int              is_lhs,
                    mc_view_t       *out_v,
                    char            *err,
                    size_t           errsz);

/* Merge operand view extents into pass iteration extents `(vnx, vny, vnz, naxis)`. */
int mc_view_merge_shape(uint8_t         *io_naxis,
                        uint32_t         io_size[3],
                        int              has_lhs_anchor,
                        const mc_view_t *v,
                        char            *err,
                        size_t           errsz);

/* Adjust operand strides (`0` on broadcast axes) to match pass extents. */
void mc_view_apply_broadcast(mc_view_t *v, const uint32_t pass_size[3]);

/* Find a program slot index by bare or qualified name, or -1 if not found. */
int mc_find_slot_by_bare(const mcalc_prog_t *prog, const char *name, const char *bare);

/* Find an image operand index in @pass matching the given parameters, or -1. */
int mc_find_im_op(const mc_pass_t *pass,
                  int              sidx,
                  int              has_slice,
                  const mc_expr_t *slice_expr,
                  int              tbl,
                  int              is_eim);

/* Find a scalar or PRNG operand index in @pass matching the given parameters, or -1. */
int mc_find_scalar_op(const mc_pass_t *pass, const mc_expr_t *se, const mc_expr_t *prng_e, int cg);

/* Return 1 if passes @a and @b can be fused into a single elementwise pass. */
int mc_can_merge_passes(const mcalc_prog_t *prog, const mc_pass_t *a, const mc_pass_t *b);

/* Return 1 if @e contains a `shift`/`roll` on an operand whose shape differs from @dsz. */
int mc_has_bcast_stencil(const mc_expr_t *e, const uint32_t dsz[3]);

/* Prepare or resize destination image for a slot and return its imageID. */
imageID mc_out_prepare(mc_slot_t     *slot,
                       uint8_t        naxis,
                       const uint32_t size[3],
                       uint8_t        dt,
                       int            in_unit,
                       char          *err,
                       size_t         errsz);

/* Allocate or reuse a pooled temporary local image `_tmpcalcN`. */
imageID mc_tmp_image(uint8_t naxis, const uint32_t size[3], uint8_t dt);

/* Return a temporary image `_tmpcalcN` to the 4-slot pool or delete it. */
void mc_tmp_release(imageID tid);

/* Force-release a slot-owned `_tmpcalc*` image when freeing a program. */
void mc_tmp_release_force(imageID tid);

/* Detach a temporary output image from @slot (pinned for `_tmpcalcsub*`, cloned if named). */
imageID mc_tmp_steal_slot(mc_slot_t *slot);

/* Publish an updated image (`md->write=0`, `cnt0++`, post semaphores if shared). */
void mc_out_post(imageID id);

/* Enable or disable deferred semaphore/cnt0 posting for shared outputs. */
void mc_post_defer(int on);

/* Flush all deferred output stream updates in a single batch. */
void mc_post_flush(void *processinfo);

/* Ensure at least one pooled temporary image exists for complex expressions. */
void mc_ensure_pool_warm(uint8_t naxis, const uint32_t size[3], uint8_t dt);

/* Format a one-line summary of image @id into @buf (with optional `name ` prefix). */
void mc_format_image_summary(imageID id, const char *assign_name, char *buf, size_t bufsz);

/* Format all pixels of real image @id as a space-separated heap string (NULL on error). */
char *mc_format_pixel_line(imageID id);

/* Build execution steps and passes for resolved AST `p->ast_head`. */
int mc_plan_build(mcalc_prog_t *p, int has_vars, char *err, size_t errsz);

/* Emit C source code for @pass into a newly heap-allocated string `pass->c_src`. */
int mc_emit_pass(const mcalc_prog_t *prog, mc_pass_t *pass, char *err, size_t errsz);

/* Execute a single elementwise pass @pass on the persistent worker team. */
int mc_exec_pass(mcalc_prog_t *prog, mc_pass_t *pass, int in_unit, char *err, size_t errsz);

/* Execute a constructor/shape step @cs. */
int mc_exec_ctor(mcalc_prog_t *prog, mc_ctor_step_t *cs, int in_unit, char *err, size_t errsz);

/* Store scalar value @v into variable @bare, updating cached variable ID *@io_vid. */
void mc_store_scalar_var(const char *bare, long *io_vid, const mc_val_t *v);

/* Evaluate a scalar reduction call (`MC_FN_REDUCE_*`, `dot`, `imedian`, etc.) into @out. */
int mc_exec_reduce_scalar(const mc_expr_t *call_expr, mc_val_t *out, char *err, size_t errsz);

/* Evaluate two scalar reductions on the same input image in a single pass. */
int mc_exec_reduce_pair(const mc_expr_t *c0,
                        const mc_expr_t *c1,
                        double          *o0,
                        double          *o1,
                        char            *err,
                        size_t           errsz);

/* Evaluate a multi-return reduction (`istats`, `icentroid`, `iargmax`, `iargmin`). */
int mc_exec_multi_reduce(const mc_stmt_t *s, mc_val_t *out, char *err, size_t errsz);

/* Execute a pre-stage / non-elementwise reduction or stencil constructor step. */
int mc_exec_prestage(mcalc_prog_t   *prog,
                     mc_ctor_step_t *cs,
                     imageID         src_id,
                     int             in_unit,
                     char           *err,
                     size_t          errsz);

/* Free any cached CSR or scratch buffers attached to @cs. */
void mc_ctor_step_free(mc_ctor_step_t *cs);

/* Clear the cached compound reduction sub-program table. */
void mc_reduce_cache_clear(void);

/* Mark unused local intermediate pass outputs as elided across @prog. */
void mc_elide_unused_locals(mcalc_prog_t *prog);

/* Source descriptor for scalar and prestage reductions (direct image or L1 sub-pass). */
typedef struct
{
    const IMAGE *im;
    mc_pass_t   *pass;
    mc_operand_t ops[32];
    int          out_op;
    uint8_t      dt;
    uint8_t      naxis;
    uint32_t     ishape[3];
    uint64_t     nel;
    imageID      tmp_id;
} mc_red_src_t;

/* Prepare reduction source @out from expression @e (materializing or L1-binding as needed). */
int mc_prepare_red_src(const mc_expr_t *e,
                       imageID          avoid_id,
                       int              allow_sub_pass,
                       mc_red_src_t    *out,
                       char            *err,
                       size_t           errsz);

/* Release any temporary image allocated by `mc_prepare_red_src`. */
void mc_release_red_src(mc_red_src_t *s);

/* Evaluate a block [off, off + len) of reduction source @s as double into @dst. */
void mc_eval_src_block_d(const mc_red_src_t *s, uint64_t off, int len, double *restrict dst);

/* Evaluate scalar expression @e into *@out. */
int mc_eval_scalar_d(const mc_expr_t *e, double *out, char *err, size_t errsz);

/* Map a reduction function name to its `arith_reduce_op` enum value. */
int mc_map_reduce_op(const char *name);

/* Compute the maximum finite value of reduction source @s into *@out_max. */
int mc_run_stat_max(const mc_red_src_t *s, double *out_max, char *err, size_t errsz);

/* Sync bound FPS/unit variables into `milk_data.variable` before execution. */
void mc_prog_load_vars(mcalc_prog_t *p);

/* Sync assigned bound variables from `milk_data.variable` back to their pointers; -1 with @err if
 * one holds the other kind (a string for a number parameter). */
int mc_prog_store_vars(mcalc_prog_t *p, char *err, size_t errsz);

/* 1 if engine variable @vid is bound to a unit parameter (it keeps its place). */
int mc_var_bound(long vid);

/* Return 1 if all image slots in @p still match their planned datatype and shape. */
int mc_prog_slots_valid(mcalc_prog_t *p, int same_gen);

/* Compile all elementwise passes in @p, batching cache misses into a single compiler call. */
int mc_cc_compile_prog(mcalc_prog_t *p, char *err, size_t errsz);

/* Bind the arguments of a call to extension @ext_id to its inputs and parameters (mc_ext_loader.c). */
int mc_ext_bind_args(int ext_id,
                     int nargs,
                     const char (*kw)[64],
                     int   *in_arg,
                     int   *par_arg,
                     char  *err,
                     size_t errsz);

#endif /* MILK_CALC_MC_PLAN_H */
