// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_internal.h
 * @brief Shared types, inline helpers, and cross-TU prototypes for the fused expression engine.
 */

#ifndef MILK_CLI_CALC_FUSE_INTERNAL_H
#define MILK_CLI_CALC_FUSE_INTERNAL_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "CLIcore.h"
#include "COREMOD_arith/arith_team.h"
#include "COREMOD_arith/image_arith__Cim_Cim__Cim.h"
#include "COREMOD_arith/image_elem_block.h"
#include "COREMOD_arith/image_spatial.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"

#define FUSE_MAXINS 128
#define FUSE_MAXIMG 16
#define FUSE_MAXSTK 8
#define FUSE_BLK 1024
#define FUSE_OMP_MIN 262144
#define FUSE_MAXRED 4
#define FUSE_MAXAX 4
#define FUSE_MAXPROP 8
#define FUSE_MAXREG 4
#define FUSE_MAXDST 4
#define FUSE_MAXEXTRA_DST 3
#define FUSE_MAXUSER 8
#define FUSE_MAXMATMUL 4

typedef enum
{
    AX_KIND_AXIS   = 0,
    AX_KIND_SEG    = 1,
    AX_KIND_RPROF  = 2,
    AX_KIND_GAUSS  = 3,
    AX_KIND_EVAL   = 4,
    AX_KIND_USER   = 5,
    AX_KIND_CUMSUM = 6 /* the sub-program's result, summed cumulatively along ax_c[slot][0] */
} ax_kind_t;

typedef arith_csr_cache_t ax_csr_cache;

typedef enum
{
    FI_IMG, /* push image img */
    FI_BIN, /* binary op; cpos 1: c is left operand, 2: c is right operand, 0: two stack slots */
    FI_NEG,
    FI_NOT,
    FI_BNOT,
    FI_FN1,
    FI_FN2,        /* as FI_BIN, with fn */
    FI_FN3,        /* fn(top, c, c2) */
    FI_SEL,        /* cond ? t : f; cpos bit 0: t on stack (else c), bit 1: f on stack (else c2) */
    FI_COORD,      /* push coordinate cpos (CALC_COORD_*) of image img's pixels about (c, c2) */
    FI_CPART,      /* push complex component cpos (ARITH_CPLX_*) of complex image img */
    FI_SHIFT,      /* push 2D edge-clamped neighbor (x + c, y + c2) of image img */
    FI_LUT,        /* 1D linear interp of top of stack in table img with (x0 = c, dx = c2) */
    FI_MEDFILT,    /* push 2D KxK edge-clamped median (cpos = K in {3,5,7}) of image img */
    FI_GATHER,     /* replace top of stack idx with img[idx] (or fill c when out of range) */
    FI_INTERP2,    /* pop (xmap, ymap) from stack and push 2D bilinear sample of image img */
    FI_STORE_SLOT, /* pop top of stack into register slot stk[cpos] */
    FI_LOAD_SLOT,  /* push register slot stk[cpos] onto stack */
    FI_STORE_DST,  /* store top of stack to extra_dst[img] (pop if cpos == 0) */
    FI_USER_BLK,   /* run 1- or 2-input user block kernel (img = ucall idx, cpos = nin) */
    FI_USER_STENCIL, /* push user 2D stencil kernel (img = operand k, cpos = ucall idx) */
    FI_SVAR,         /* scalar tape: read calculator variable vid */
    FI_PROP,         /* scalar tape: read live SHM property prop_path[vid] */
    FI_RED,          /* scalar tape: fused reduction op (vid = is_long, c[0] = slot) */
    FI_CENT,         /* scalar tape: fused centroid component op (0=cx, 1=cy, 2=flux) */
    FI_CONST,        /* push scalar constant c onto the vector stack */
    FI_POP,          /* drop the top of the vector stack (an elided, unread local) */
    FI_IMGC,         /* push image img op c (cpos 2) or c op image img (cpos 1); op in + - * / */
    FI_IMGIMG,       /* push image img op image cpos; op in + - * / */
    FI_VIMGC,        /* top = top op (image img * c); op in + - * / */
    FI_MATMUL,       /* push block of matrix product matmul[cpos] onto the vector stack */
    FI_EYE           /* push 2D identity matrix (1 on diagonal k = (int64_t)c, else 0) */
} fi_kind;

/** The one-argument functions with a vector form, resolved from the instruction's pointer at emit. */
typedef enum
{
    FN1_OTHER = 0, /* called through the pointer, in double */
    FN1_FABS,
    FN1_SQRT,
    FN1_FLOOR,
    FN1_CEIL,
    FN1_ROUND,
    FN1_SIN,
    FN1_COS,
    FN1_TAN,
    FN1_SINH,
    FN1_COSH,
    FN1_TANH,
    FN1_EXP,
    FN1_EXP2,
    FN1_LOG,
    FN1_LOG10,
    FN1_LOG2,
    FN1_ERF,
    FN1_ERFC,
    FN1_ATAN,
    FN1_ASIN,
    FN1_ACOS,
    FN1_CBRT,
    FN1_POSI,
    FN1_SIGN
} fn1_kind;

typedef struct
{
    fi_kind        k;
    cli_token_type op;
    int            vid;
    int            s[3]; /* > 0: 1-based index in sval[]; 0: use c[] */
    double         c[3];
    uint8_t        cl[3]; /* c[] is an integer literal or long variable */
    uint32_t       vhash; /* FI_SVAR: name_hash32() of the variable */
    uint8_t
        guarded; /* inside a `?:` / where() branch: a zero divisor is selected away, not an error */
    double (*fn)();
} fsins;

typedef struct
{
    fi_kind        k;
    cli_token_type op;
    int            img;
    int            cpos;
    int            s1; /* > 0: 1-based index in sval[] for c */
    int            s2; /* > 0: 1-based index in sval[] for c2 */
    double         c;
    double         c2;
    uint8_t
        guarded; /* inside a `?:` / where() branch: a zero divisor is selected away, not an error */
    uint8_t fn1; /* FI_FN1: the function's fn1_kind, so the kernel switches instead of comparing */
    double (*fn)();
} fins;

/**
 * struct fview - How an image operand is read over the program's iteration space
 */
typedef struct
{
    int      is_view;
    int      is_lut;   /* 1-D lookup table operand: shape not merged into iteration space */
    int      is_shift; /* accessed via FI_SHIFT stencil */
    int      dyn;
    int      internal;  /* an axis-reduction result held by the program, not a table image */
    int64_t  base;      /* source element of the iteration origin */
    int64_t  stride[3]; /* source elements per step along each iteration axis; 0 = broadcast */
    uint32_t vshape[3]; /* the operand's own extent per axis (1 where it broadcasts) */
    int      naxis;     /* axes of the operand's own shape */
    char     spec[200]; /* the slice token, "" for a whole image */
    /* dynamic bounds compiled once: tmpl is the bracket text with each expression replaced by
     * '\001', part i is scalar-tape slot part_s1[i] (> 0) or the constant part_c[i] */
    char     tmpl[200];
    int      npart;
    int16_t  part_s1[9];
    double   part_c[9];
    imageID  kept_id;
    uint8_t  kept_dt;
    uint32_t kept_sz[3];
    char     kept_name[STRINGMAXLEN_IMAGE_NAME];
    uint64_t last_cnt0;
    uint64_t snap_cnt0;
} fview;

typedef struct
{
    int      ext_idx;
    char     label[64];
    int      nparams;
    double   pvals[MILK_CALC_MAX_PARAMS];
    int16_t  ps1[MILK_CALC_MAX_PARAMS];
    uint64_t pbuf[MILK_CALC_PARAM_BYTES / sizeof(uint64_t)];
} fuse_user_call_t;

typedef struct
{
    int      ext_idx;
    int      in_bound[MILK_CALC_MAX_PORTS];
    int      in_img_k[MILK_CALC_MAX_PORTS];
    double   in_sval[MILK_CALC_MAX_PORTS];
    int      in_ss1[MILK_CALC_MAX_PORTS];
    uint32_t in_init_sz[MILK_CALC_MAX_PORTS][3];
    uint8_t  in_init_dt[MILK_CALC_MAX_PORTS];
    double   init_pvals[MILK_CALC_MAX_PARAMS];
    int      ndst;
    char     out_name[MILK_CALC_MAX_PORTS][STRINGMAXLEN_IMAGE_NAME];
    int      out_is_local[MILK_CALC_MAX_PORTS];
    imageID  out_id[MILK_CALC_MAX_PORTS];
    int      out_vid[MILK_CALC_MAX_PORTS];
    uint8_t  out_naxis[MILK_CALC_MAX_PORTS];
    uint32_t out_size[MILK_CALC_MAX_PORTS][3];
    uint8_t  out_dtype[MILK_CALC_MAX_PORTS];
    int      default_shared;
    int      inited;
    void    *state;
} fuse_user_op_t;

typedef struct
{
    int      imgA;
    int      imgB;
    int      tA;
    int      tB;
    uint64_t M;
    uint64_t K;
    uint64_t N;
    int64_t  cached_col;
    void    *col_buf;
    size_t   col_cap;
} fuse_matmul_t;

typedef struct fprog
{
    fins             ins[FUSE_MAXINS];
    int              n;
    fsins            sins[FUSE_MAXINS];
    uint8_t          slong[FUSE_MAXINS]; /* per sins[] result, set by eval_scalars() */
    int64_t          slval[FUSE_MAXINS]; /* the exact value of a long result (slong[i] set) */
    int              ns;
    const IMAGE     *img[FUSE_MAXIMG];
    fview            view[FUSE_MAXIMG];
    int              nimg;
    fview            dstv;      /* destination view (sliced LHS), else is_view 0 */
    uint32_t         ishape[3]; /* iteration space: destination shape or broadcast shape */
    int              inaxis;
    int              anchored; /* ishape is the destination's: operands may not widen it */
    uint64_t         inel;
    struct fprog    *ax_p[FUSE_MAXAX];
    struct fprog    *ax_p2[FUSE_MAXAX];
    int              nax;
    int              ax_kind[FUSE_MAXAX];
    int              ax_axis[FUSE_MAXAX];
    reduce_op        ax_op[FUSE_MAXAX];
    double           ax_c[FUSE_MAXAX][2];
    int              ax_s[FUSE_MAXAX][2];
    ax_csr_cache     ax_csr[FUSE_MAXAX];
    IMAGE            ax_im[FUSE_MAXAX];
    IMAGE_METADATA   ax_md[FUSE_MAXAX];
    void            *ax_data[FUSE_MAXAX];
    size_t           ax_cap[FUSE_MAXAX];
    int              depth;
    int              maxdepth;
    int              nreg;     /* active register slots allocated from top of stk */
    int              max_nreg; /* peak register slots used */
    int              nops;     /* vector operations emitted */
    int              out_double;
    uint8_t          out_dt; /* > 0 when outermost op is an explicit cast f32..i64 */
    int              has_red;
    int              nred;
    struct fprog    *red_p1[FUSE_MAXRED];
    struct fprog    *red_p2[FUSE_MAXRED];
    reduce_op        red_op[FUSE_MAXRED];
    int              red_tok0[FUSE_MAXRED];
    int              red_tok1[FUSE_MAXRED];
    int              has_cent;
    int              cent_th_s1;
    double           cent_th_c;
    char             prop_path[FUSE_MAXPROP][128];
    int              nprop;
    IMAGE           *extra_dst[FUSE_MAXEXTRA_DST];
    int              nextra_dst;
    fuse_user_call_t ucall[FUSE_MAXUSER];
    int              nuser;
    fuse_matmul_t    matmul[FUSE_MAXMATMUL];
    int              nmatmul;
    void            *alias_buf;
    size_t           alias_cap;
} fprog;

/** Compile-time value: a scalar (constant c when s1 == 0, else sins[s1 - 1]), or a vector. */
typedef struct
{
    int    vec;
    int    s1;
    double c;
    int    lng;
} fval;

typedef arith_stat_acc_t fred_acc;
#define fred_merge arith_stat_merge_inline
#define finish_reduce arith_stat_finish_inline

/* index reductions, beyond arith_reduce_op: run by cli_calc_fuse_run_argext(), the index in acc.sum */
#define RED_ARGMAX ((reduce_op) (ARITH_RED_ALL + 1))
#define RED_ARGMIN ((reduce_op) (ARITH_RED_ALL + 2))
#define RED_AMAX ((reduce_op) (ARITH_RED_ALL + 3))
#define RED_TRACE ((reduce_op) (ARITH_RED_ALL + 4))

/** red_is_argext() - 1 for the index reductions. */
static inline int red_is_argext(reduce_op op)
{
    return (int) op >= (int) RED_ARGMAX && (int) op <= (int) RED_AMAX;
}

/** fuse_finish() - the scalar of accumulator @acc for @op: the index for an index reduction. */
static inline double fuse_finish(const fred_acc *acc, reduce_op op, int is_long)
{
    return red_is_argext(op) ? acc->sum : finish_reduce(acc, op, is_long);
}
#define reduce_block_d arith_stat_block_d_inline
#define reduce_block_f arith_stat_block_f_inline

/** emit() - append a vector stack instruction to @p. */
static inline int emit(fprog *p, fins in, int ddepth)
{
    if (p->n >= FUSE_MAXINS)
    {
        cli_calc_fuse_reject("more than 128 vector instructions (FUSE_MAXINS)");
        return -1;
    }
    p->ins[p->n++] = in;
    p->depth += ddepth;
    if (p->depth > p->maxdepth)
    {
        p->maxdepth = p->depth;
    }
    if (in.k != FI_IMG)
    {
        p->nops++;
    }
    if (p->maxdepth + p->max_nreg > FUSE_MAXSTK)
    {
        cli_calc_fuse_reject("expression needs more than 8 vector stack rows (FUSE_MAXSTK)");
        return -1;
    }
    return 0;
}

/** fprog_effective_nel() - compute-weighted element count for OpenMP team sizing. */
static inline uint64_t fprog_effective_nel(const fprog *p)
{
    uint64_t mm_work = 0;
    for (int i = 0; i < p->nmatmul; i++)
    {
        mm_work += p->matmul[i].M * p->matmul[i].K * p->matmul[i].N * 2ULL;
    }
    if (p->inel < 4 * FUSE_BLK && mm_work == 0)
    {
        return p->inel;
    }
    uint64_t cost = 0;
    for (int i = 0; i < p->n; i++)
    {
        const fins *in = &p->ins[i];
        if (in->k == FI_FN1)
        {
            cost += (in->fn1 == FN1_FABS || in->fn1 == FN1_FLOOR || in->fn1 == FN1_CEIL ||
                     in->fn1 == FN1_ROUND || in->fn1 == FN1_POSI || in->fn1 == FN1_SIGN)
                        ? 1
                    : (in->fn1 == FN1_SQRT) ? 2
                                            : 8;
        }
        else if (in->k == FI_VIMGC)
        {
            cost += 2;
        }
        else if (in->k == FI_FN2)
        {
            double (*f)(double, double) = (double (*)(double, double)) in->fn;
            cost += (f == fmin || f == fmax) ? 1 : (f == hypot) ? 3 : 8;
        }
        else if (in->k == FI_COORD &&
                 (in->cpos == CALC_COORD_RAND || in->cpos == CALC_COORD_RANDN ||
                  in->cpos == CALC_COORD_R || in->cpos == CALC_COORD_THETA))
        {
            cost += 6;
        }
        else if (in->k == FI_BIN && in->op == TOK_OP_CARET)
        {
            cost += (in->cpos == 2 && (in->c == 2.0 || in->c == 0.5 || in->c == 1.0)) ? 2 : 8;
        }
        else if (in->k == FI_LUT || in->k == FI_SHIFT || in->k == FI_GATHER)
        {
            cost += 2;
        }
        else if (in->k == FI_INTERP2)
        {
            cost += 4;
        }
        else if (in->k == FI_MEDFILT)
        {
            cost += (in->cpos == 3) ? 6 : (in->cpos == 5) ? 12 : 28;
        }
        else if (in->k == FI_USER_BLK)
        {
            const milk_calc_ext_entry_t *e = cli_calc_ext_get(p->ucall[in->img].ext_idx);
            cost += (e != NULL && e->cost > 0) ? (uint64_t) e->cost : 4;
        }
        else if (in->k == FI_USER_STENCIL)
        {
            const milk_calc_ext_entry_t *e = cli_calc_ext_get(p->ucall[in->cpos].ext_idx);
            cost += (e != NULL && e->cost > 0) ? (uint64_t) e->cost : 8;
        }
        else if (in->k != FI_IMG)
        {
            cost += 1;
        }
    }
    return p->inel * (1 + cost / 4) + mm_work;
}

/** fprog_choose_grid() - choose thread count, block count, and block size for @nel / @eff_nel. */
static inline void fprog_choose_grid(uint64_t  nel,
                                     uint64_t  eff_nel,
                                     int      *out_nth,
                                     uint64_t *out_nblk,
                                     uint64_t *out_blk_sz)
{
    int      nth    = (nel >= 64) ? milk_omp_team_size(eff_nel, FUSE_OMP_MIN) : 1;
    uint64_t blk_sz = FUSE_BLK;
    if (nth > 1 && nel < (uint64_t) nth * FUSE_BLK)
    {
        blk_sz = ((nel / (uint64_t) nth) & ~63ULL);
        if (blk_sz < 64)
        {
            blk_sz = 64;
        }
    }
    uint64_t nblk = (nel + blk_sz - 1) / blk_sz;
    if ((uint64_t) nth > nblk)
    {
        nth = (int) ((nblk > 0) ? nblk : 1);
    }
    *out_nth    = nth;
    *out_nblk   = nblk;
    *out_blk_sz = blk_sz;
}

/** name_hash32() - FNV-1a of @s: identity checked by kept FI_SVAR against its variable slot. */
static inline uint32_t name_hash32(const char *s)
{
    uint32_t h = 2166136261u;
    for (; *s != '\0'; s++)
    {
        h = (h ^ (uint8_t) *s) * 16777619u;
    }
    return h;
}

/** bin_is_long() - comparisons and bitwise ops are long; `+ - * %` of two longs stay long. */
static inline int bin_is_long(cli_token_type op, int al, int bl)
{
    switch (op)
    {
    case TOK_OP_LT:
    case TOK_OP_LE:
    case TOK_OP_GT:
    case TOK_OP_GE:
    case TOK_OP_EQ:
    case TOK_OP_NEQ:
    case TOK_OP_AND:
    case TOK_OP_OR:
    case TOK_OP_BAND:
    case TOK_OP_BOR:
    case TOK_OP_BXOR:
    case TOK_OP_LSHIFT:
    case TOK_OP_RSHIFT:
        return 1;
    case TOK_OP_PLUS:
    case TOK_OP_MINUS:
    case TOK_OP_STAR:
    case TOK_OP_MOD:
        return al && bl;
    default:
        return 0;
    }
}

/** emit_scalar() - append a scalar tape instruction for runtime variable evaluation. */
static inline int emit_scalar(fprog *p, fsins in, fval *out)
{
    if (p->ns >= FUSE_MAXINS)
    {
        cli_calc_fuse_reject("more than 128 scalar instructions (FUSE_MAXINS)");
        return -1;
    }
    p->sins[p->ns++] = in;
    *out             = (fval) { .vec = 0, .s1 = p->ns, .c = 0.0 };
    return 0;
}

/** is_real_datatype() - 1 if @dt is a standard real integer or float32/float64 datatype. */
static inline int is_real_datatype(uint8_t dt)
{
    return dt != _DATATYPE_COMPLEX_FLOAT && dt != _DATATYPE_COMPLEX_DOUBLE &&
           dt != _DATATYPE_HALF && ImageStreamIO_typesize(dt) > 0;
}

/** image_vshape() - @im's extent per axis, 1 beyond its naxis. */
static inline void image_vshape(const IMAGE *im, uint32_t vs[3])
{
    for (int a = 0; a < 3; a++)
    {
        vs[a] = (a < im->md[0].naxis && im->md[0].size[a] > 0) ? im->md[0].size[a] : 1;
    }
}

/** whole_view() - the view of whole image @im: natural strides, no offset. */
static inline void whole_view(const IMAGE *im, fview *v)
{
    memset(v, 0, sizeof(*v));
    image_vshape(im, v->vshape);
    v->naxis     = im->md[0].naxis;
    v->stride[0] = 1;
    v->stride[1] = (int64_t) v->vshape[0];
    v->stride[2] = (int64_t) v->vshape[0] * v->vshape[1];
}

/** scalar_bin() - evaluate a fusible binary operator on two double-precision scalars. */
static inline double scalar_bin(cli_token_type op, double a, double b)
{
    switch (op)
    {
    case TOK_OP_PLUS:
        return a + b;
    case TOK_OP_MINUS:
        return a - b;
    case TOK_OP_STAR:
        return a * b;
    case TOK_OP_SLASH:
        return a / b;
    case TOK_OP_MOD:
        return fmod(a, b);
    case TOK_OP_CARET:
        return pow(a, b);
    case TOK_OP_LT:
        return a < b;
    case TOK_OP_LE:
        return a <= b;
    case TOK_OP_GT:
        return a > b;
    case TOK_OP_GE:
        return a >= b;
    case TOK_OP_EQ:
        return a == b;
    case TOK_OP_NEQ:
        return a != b;
    case TOK_OP_AND:
        return (a != 0.0) && (b != 0.0);
    case TOK_OP_BAND:
        return (double) ((int64_t) a & (int64_t) b);
    case TOK_OP_BOR:
        return (double) ((int64_t) a | (int64_t) b);
    case TOK_OP_BXOR:
        return (double) ((int64_t) a ^ (int64_t) b);
    case TOK_OP_LSHIFT:
    {
        int64_t sh = (int64_t) b;
        return (double) ((sh >= 0 && sh < 64) ? (int64_t) ((uint64_t) (int64_t) a << sh) : 0);
    }
    case TOK_OP_RSHIFT:
    {
        int64_t sh = (int64_t) b;
        return (double) ((sh >= 0 && sh < 64) ? ((int64_t) a >> sh) : (((int64_t) a < 0) ? -1 : 0));
    }
    default:
        return (a != 0.0) || (b != 0.0);
    }
}

/** axis_op_ok() - 1 if @fop supports an axis or mask argument. */
static inline int axis_op_ok(int fop)
{
    return fop == RED_SUM || fop == RED_MEAN || fop == RED_MIN || fop == RED_MAX ||
           fop == RED_STD || fop == RED_VAR || fop == RED_NORM || fop == RED_ASUM;
}

/* VIEW_WALK: visit the row segments of iteration elements [off, off + len) of @p. */
#define VIEW_WALK(p, v, off, len, BODY)                                                          \
    do                                                                                           \
    {                                                                                            \
        const uint64_t nx_ = (p)->ishape[0], ny_ = (p)->ishape[1];                               \
        uint64_t       x_ = (off) % nx_, r_ = (off) / nx_, y_ = r_ % ny_, z_ = r_ / ny_;         \
        int            J = 0;                                                                    \
        while (J < (len))                                                                        \
        {                                                                                        \
            const int N =                                                                        \
                (int) ((nx_ - x_ < (uint64_t) ((len) - J)) ? nx_ - x_ : (uint64_t) ((len) - J)); \
            const int64_t S = (v)->base + (int64_t) x_ * (v)->stride[0] +                        \
                              (int64_t) y_ * (v)->stride[1] + (int64_t) z_ * (v)->stride[2];     \
            (void) S;                                                                            \
            BODY J += N;                                                                         \
            x_ = 0;                                                                              \
            if (++y_ == ny_)                                                                     \
            {                                                                                    \
                y_ = 0;                                                                          \
                z_++;                                                                            \
            }                                                                                    \
        }                                                                                        \
    } while (0)

typedef struct
{
    fprog                p;
    IMAGE               *dst;
    imageID              dst_id;
    uint8_t              dst_dt;
    uint32_t             dst_sz[3];
    char                 dst_name[STRINGMAXLEN_IMAGE_NAME];
    imageID              extra_dst_id[FUSE_MAXEXTRA_DST];
    uint8_t              extra_dst_dt[FUSE_MAXEXTRA_DST];
    uint32_t             extra_dst_sz[FUSE_MAXEXTRA_DST][3];
    char                 extra_dst_name[FUSE_MAXEXTRA_DST][STRINGMAXLEN_IMAGE_NAME];
    int                  dst_vid;
    int                  scalar_s1;
    int                  scalar_is_long;
    double               scalar_c;
    int                  nsdst;
    int                  sdst_vid[FUSE_MAXDST];
    int                  sdst_s1[FUSE_MAXDST];
    int                  sdst_is_long[FUSE_MAXDST];
    double               sdst_c[FUSE_MAXDST];
    char                 sdst_name[FUSE_MAXDST][STRINGMAXLEN_IMAGE_NAME];
    struct timespec      last_check;
    uint64_t             last_gen; /* milk_table_generation() at the last name validation */
    int                  writer_stuck;
    cli_calc_kept_matrix m;
    fuse_user_op_t      *uop;
} fkept;

/* The persistent worker team (COREMOD_arith/arith_team.c): one dispatch per block loop */
typedef arith_team_job_fn fuse_job_fn;
#define cli_calc_team_run arith_team_run
#define cli_calc_team_run_chunks arith_team_run_chunks
#define cli_calc_team_lock arith_team_lock
#define cli_calc_team_unlock arith_team_unlock

/** cli_calc_fuse_fn1_kind() - the fn1_kind of one-argument function @fn (FN1_OTHER when it has no vector form). */
uint8_t cli_calc_fuse_fn1_kind(double (*fn)());
/** cli_calc_fuse_emit_bin() - emit binary operator @bi (both: two stack operands), fusing with the
 * instructions before it into FI_IMGC / FI_IMGIMG / FI_VIMGC where the pattern allows. */
int cli_calc_fuse_emit_bin(fprog *p, fins bi, int both);

/* Compiler & view helpers (cli_calc_fuse.c, cli_calc_fuse_views.c, cli_calc_fuse_ops_*.c) */
int     cli_calc_fuse_add_operand(fprog *p, const IMAGE *im, const char *spec);
int     cli_calc_fuse_add_image(fprog *p, const IMAGE *im);
int     cli_calc_fuse_add_flat_image(fprog *p, const IMAGE *im);
int     cli_calc_fuse_add_complex_image(fprog *p, const IMAGE *im);
int     cli_calc_fuse_add_lut_image(fprog *p, const IMAGE *im);
int     cli_calc_fuse_add_axis_operand(fprog *p, const IMAGE *im);
int     cli_calc_fuse_finalize_views(fprog *p);
int     cli_calc_fuse_reduce_shapes_match(fprog *a, fprog *b);
int     cli_calc_fuse_refresh_views(fprog *p, const double *sval, IMAGE *dst);
int     cli_calc_fuse_bind_slice_tape(fprog *p, fview *v);
void    cli_calc_fuse_anchor_begin(const fview *dv, int naxis);
void    cli_calc_fuse_anchor_end(void);
int     cli_calc_fuse_anchor_save(void);
void    cli_calc_fuse_anchor_restore(int on);
void    cli_calc_fuse_set_keep_vars(int on);
int     cli_calc_fuse_get_keep_vars(void);
void    cli_calc_fuse_reset_fprog(fprog *p);
void    cli_calc_fuse_free_fprog_children(fprog *p);
int     cli_calc_fuse_compile(fprog *p, int min_prec, fval *v);
int     cli_calc_fuse_compile_more(fprog *p, int min_prec, fval *v);
int     cli_calc_fuse_c_expr(fprog *p, int min_prec, fval *out);
int     cli_calc_fuse_c_args(fprog *p, int n, fval *a);
int     cli_calc_fuse_udef_depth(void);
int     cli_calc_fuse_udef_param_img(const char *name);
int     cli_calc_fuse_c_cplx_part(fprog *p, int which, fval *out);
int     cli_calc_fuse_c_func(fprog *p, const cli_token *ft, fval *out);
int     cli_calc_fuse_c_select(fprog *p, fval t, fval f, fval *out);
int     cli_calc_fuse_c_coord(fprog *p, const cli_token *ft, fval *out);
int     cli_calc_fuse_c_reduce(fprog *p, const cli_token *ft, fval *out);
int     cli_calc_fuse_c_shift(fprog *p, fval *out);
int     cli_calc_fuse_c_pixunmap(fprog *p, const cli_token *ft, fval *out);
int     cli_calc_fuse_c_medfilt(fprog *p, const cli_token *ft, fval *out);
int     cli_calc_fuse_c_pixremap(fprog *p, fval *out);
int     cli_calc_fuse_c_gauss(fprog *p, fval *out);
int     cli_calc_fuse_c_cumsum(fprog *p, fval *out);
int     cli_calc_fuse_c_interp2(fprog *p, fval *out);
int     cli_calc_fuse_c_lut(fprog *p, fval *out);
int     cli_calc_fuse_c_linalg(fprog *p, const cli_token *ft, fval *out);
int     cli_calc_fuse_merge_shape(fprog *p, const uint32_t vs[3], int vnax);
int     cli_calc_fuse_restrict_diag(fprog *sub, int64_t k);
uint8_t cli_calc_fuse_axis_result_dt(const fprog *sub);
int     cli_calc_fuse_axred_setup(fprog *p, int i, int onax, const uint32_t osz[3], uint8_t dt);
int     cli_calc_fuse_c_leaf_or_eval_image(fprog *p, int as_lut, int *out_k);

/* User C extension kernels & operators (cli_calc_fuse_user_kern.c, cli_calc_fuse_user_op.c) */
int   cli_calc_fuse_c_user(fprog *p, const cli_token *ft, fval *out);
int   cli_calc_fuse_run_user_prestage(fprog *p, int slot);
void  cli_calc_fuse_user_blk_d(const fprog *p,
                               int          u,
                               int          len,
                               double *restrict l,
                               const double *restrict r);
void  cli_calc_fuse_user_blk_f(const fprog *p,
                               int          u,
                               int          len,
                               float *restrict l,
                               const float *restrict r);
void  cli_calc_fuse_user_stencil_d(const fprog *p,
                                   int          k,
                                   int          u,
                                   uint64_t     off,
                                   int          len,
                                   double *restrict dst);
void  cli_calc_fuse_user_stencil_f(const fprog *p,
                                   int          k,
                                   int          u,
                                   uint64_t     off,
                                   int          len,
                                   float *restrict dst);
void *cli_calc_fuse_keep_user_op(int        ndst,
                                 const char lhs[][STRINGMAXLEN_IMAGE_NAME],
                                 const int *is_local,
                                 int        default_shared);
int   cli_calc_fuse_emit_default_user_param(fprog                        *p,
                                            fuse_user_call_t             *uc,
                                            int                           j,
                                            const milk_calc_param_spec_t *sp);
int   cli_calc_fuse_exec_user_op(fkept *k);
void  cli_calc_fuse_free_user_op(fuse_user_op_t *uop);
val_t cli_calc_eval_user_call(const cli_token *ftok);

/* Kept & cross-statement helpers (cli_calc_fuse_kept.c & cli_calc_fuse_merge.c) */
int   cli_calc_fuse_dst_view(IMAGE **dst, const char *dst_tok, fview *dv);
void  cli_calc_fuse_record_kept_images(fkept *k, IMAGE *dst);
int   cli_calc_fuse_prepend_compound_lhs(fprog         *p,
                                         IMAGE         *dst,
                                         const char    *dst_tok,
                                         cli_token_type aop);
int   cli_calc_fuse_try_merge_kept(void *kept1, void *kept2, int elide_dst1);
void *cli_calc_fuse_keep_multi(int ndst, IMAGE *const *dst_im, const int *dst_vid);

/* Spatial block kernels (cli_calc_fuse_exec_spatial.c & cli_calc_fuse_kernels_medfilt.h) */
void cli_calc_fuse_medfilt_cols_d(int K, int N, double col[7][FUSE_BLK + 6], double *restrict d);
void cli_calc_fuse_medfilt_cols_f(int K, int N, float col[7][FUSE_BLK + 6], float *restrict d);
void cli_calc_fuse_shift_block_d(const fprog *p,
                                 int          k,
                                 int64_t      dx,
                                 int64_t      dy,
                                 uint64_t     off,
                                 int          len,
                                 double *restrict dst);
void cli_calc_fuse_shift_block_f(const fprog *p,
                                 int          k,
                                 int64_t      dx,
                                 int64_t      dy,
                                 uint64_t     off,
                                 int          len,
                                 float *restrict dst);
#define cli_calc_fuse_lut_block_d arith_lut_block_d
#define cli_calc_fuse_lut_block_f arith_lut_block_f
void cli_calc_fuse_medfilt_block_d(const fprog *p,
                                   int          k,
                                   int          K,
                                   uint64_t     off,
                                   int          len,
                                   double *restrict dst);
void cli_calc_fuse_medfilt_block_f(const fprog *p,
                                   int          k,
                                   int          K,
                                   uint64_t     off,
                                   int          len,
                                   float *restrict dst);
#define cli_calc_fuse_gather_block_d arith_gather_block_d
#define cli_calc_fuse_gather_block_f arith_gather_block_f
#define cli_calc_fuse_interp2_block_d arith_interp2_block_d
#define cli_calc_fuse_interp2_block_f arith_interp2_block_f
int  cli_calc_fuse_prep_matmuls(fprog *p, int nth);
void cli_calc_fuse_matmul_block_d(const fprog *p,
                                  int          m,
                                  uint64_t     off,
                                  int          len,
                                  double *restrict dst);
void cli_calc_fuse_matmul_block_f(const fprog *p,
                                  int          m,
                                  uint64_t     off,
                                  int          len,
                                  float *restrict dst);
void cli_calc_fuse_eye_block_d(const fprog *p,
                               int64_t      kdiag,
                               uint64_t     off,
                               int          len,
                               double *restrict dst);
void cli_calc_fuse_eye_block_f(const fprog *p,
                               int64_t      kdiag,
                               uint64_t     off,
                               int          len,
                               float *restrict dst);

/* Execution, pre-stage & reduction helpers (cli_calc_fuse_exec.c, prestage*.c, reduce.c) */
void cli_calc_fuse_load_block(const IMAGE *im, uint64_t off, int len, double *restrict dst);
void cli_calc_fuse_store_block(IMAGE   *im,
                               uint64_t off,
                               int      len,
                               const double *restrict v,
                               int    isc,
                               double c);
void cli_calc_fuse_eval_block_d(const fprog *p, uint64_t off, int len, double (*stk)[FUSE_BLK]);
void cli_calc_fuse_eval_block_f(const fprog *p, uint64_t off, int len, float (*stk)[FUSE_BLK]);
int  cli_calc_fuse_all_float_prog(const fprog *p);
int  cli_calc_fuse_all_float(const fprog *p, const IMAGE *out);
int  cli_calc_fuse_eval_scalars(fprog *p, double *out_sval);
int  cli_calc_fuse_run_axes(fprog *p);
int  cli_calc_fuse_run_seg_reduce(fprog *p, int slot);
int  cli_calc_fuse_run_gauss_prestage(fprog *p, int slot);
int  cli_calc_fuse_run_cumsum_prestage(fprog *p, int slot);
int  cli_calc_fuse_run_argext(fprog *sub, reduce_op op, int64_t *out_idx);
int  cli_calc_fuse_run_program(fprog *p, IMAGE *out);
int  cli_calc_fuse_run_reduce(fprog *p1, fprog *p2, reduce_op op, fred_acc *out_acc);
int  cli_calc_fuse_run_centroid(fprog *sub, double thresh, double out_cent[3]);
#define cli_calc_fuse_map_nbins arith_image_map_nbins
#define cli_calc_fuse_rprof_nbins arith_image_rprof_nbins

#endif /* MILK_CLI_CALC_FUSE_INTERNAL_H */
