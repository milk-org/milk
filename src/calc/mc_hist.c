// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_hist.c
 * @brief Self-referential stream bootstrap, `hist()`/`prev()` ring rewriter, and `--explain`.
 *
 * Implements three compilation-support facilities for `libmilkcalc` programs:
 *   1. **Self-referential recurrence bootstrap (`mc_expr_mentions_self`, `mc_self_init_create`)**:
 *      Detects recurrence assignments where an uncreated target appears on its own RHS (e.g.
 *      `avg = 0.9 * avg + 0.1 * im` or `n = n + 1`) and pre-creates the target image or scalar
 *      variable zero-initialized using the shape/datatype of the non-self operand.
 *   2. **History ring rewriter (`mc_rewrite_history`)**: Scans program source text before lexing
 *      for `hist(stream, N)` declarations and `prev(stream[, k])` references. Allocates or
 *      verifies the backing `(naxis + 1)`-D shared-memory ring buffer `<stream>_hist` of depth `N`
 *      (default `2` when `prev(stream)` is used without an explicit `hist()`), rewrites every
 *      `prev(stream, k)` call into `<stream>_hist[..., (_n + N - k) % N]`, and appends
 *      `<stream>_hist[..., _n % N] = <stream>` update statements at the end of the program.
 *   3. **Execution plan explainer (`mcalc_prog_explain`)**: Compiles and dry-runs a program, then
 *      formats a human-readable breakdown of how each statement is executed (inlined `def`,
 *      fused elementwise pass with dimensions/work-type/pre-stages, constructor step, or scalar
 *      tape).
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_memory/create_image.h"
#include "COREMOD_memory/create_variable.h"
#include "COREMOD_memory/image_ID.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "mc_ast.h"
#include "mc_plan.h"
#include "mcalc.h"

/* Maximum number of distinct streams with a `hist()` / `prev()` ring in a single program. */
#define MC_HIST_MAX 16

/**
 * struct mc_hist_decl_t - Descriptor for a stream with an associated `<name>_hist` ring buffer
 * @name:    Stream identifier
 * @N:       Ring depth in frames (`>= 2`)
 * @naxis:   Dimensionality of the source stream (`1` or `2`; `0` before `hist_ring()` validation)
 * @src_off: Byte span of the statement that declared (or first used) the ring
 * @src_end:
 */
typedef struct
{
    char name[STRINGMAXLEN_IMAGE_NAME];
    int  N;
    int  naxis;
    int  src_off;
    int  src_end;
} mc_hist_decl_t;

/* ============================================================================================== */
/* Self-Referential Recurrence Bootstrap                                                          */
/* ============================================================================================== */

/**
 * mc_expr_mentions_self - Check whether RHS expression @e references LHS target @bare
 * @e:      RHS expression AST node to inspect recursively
 * @bare:   Bare identifier name of the LHS assignment target
 * @out_im: Optional output pointer receiving the first non-self `IMAGE *` referenced in @e
 *
 * Used by the resolver (`mc_resolve.c`) when an assignment target does not yet exist: if @e
 * mentions @bare alongside another existing image `*out_im`, the target can be auto-created with
 * the dimensions and datatype of `*out_im` to bootstrap recurrence relations like
 * `y = 0.95 * y + 0.05 * x`. Arguments inside scalar reductions (`imean(x)`, etc.) do not
 * contribute a shape to `@out_im`.
 *
 * Return: 1 if @e references @bare, 0 otherwise
 */
int mc_expr_mentions_self(const mc_expr_t *e, const char *bare, const void **out_im)
{
    if (e == NULL)
    {
        return 0;
    }
    int hit = 0;
    switch (e->kind)
    {
    case MC_EXPR_IDENT:
    case MC_EXPR_SLICE:
    {
        char nbuf[256];
        char bbuf[MC_IDENT_MAX];
        snprintf(nbuf, sizeof(nbuf), "%s",
                 (e->kind == MC_EXPR_IDENT) ? e->u.ident.name : e->u.slice.name);
        mc_parse_ident_streamname(nbuf, bbuf, NULL);
        if (strcmp(bbuf, bare) == 0)
        {
            hit = 1;
        }
        else if (out_im != NULL && *out_im == NULL)
        {
            imageID id = mc_resolve_or_autoconnect_image(nbuf);
            if (id != -1)
            {
                *out_im = &milk_data.image[id];
            }
        }
        for (int ax = 0; e->kind == MC_EXPR_SLICE && ax < e->u.slice.naxis; ax++)
        {
            hit |= mc_expr_mentions_self(e->u.slice.axes[ax].start, bare, NULL);
            hit |= mc_expr_mentions_self(e->u.slice.axes[ax].stop, bare, NULL);
            hit |= mc_expr_mentions_self(e->u.slice.axes[ax].step, bare, NULL);
        }
        return hit;
    }
    case MC_EXPR_UNARY:
        return mc_expr_mentions_self(e->u.unary.operand, bare, out_im);
    case MC_EXPR_BINARY:
        hit |= mc_expr_mentions_self(e->u.binary.left, bare, out_im);
        hit |= mc_expr_mentions_self(e->u.binary.right, bare, out_im);
        return hit;
    case MC_EXPR_TERNARY:
        hit |= mc_expr_mentions_self(e->u.ternary.cond, bare, out_im);
        hit |= mc_expr_mentions_self(e->u.ternary.t_expr, bare, out_im);
        hit |= mc_expr_mentions_self(e->u.ternary.f_expr, bare, out_im);
        return hit;
    case MC_EXPR_CHAIN:
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            hit |= mc_expr_mentions_self(e->u.chain.operands[i], bare, out_im);
        }
        return hit;
    case MC_EXPR_CALL:
    {
        const mc_func_t *fn     = mc_func_find(e->u.call.name);
        int              is_red = (fn != NULL &&
                                   (fn->kind == MC_FN_REDUCE_D || fn->kind == MC_FN_REDUCE_L ||
                                    fn->kind == MC_FN_ISTATS || fn->kind == MC_FN_ICENTROID ||
                                    fn->kind == MC_FN_EAGER_RED || fn->kind == MC_FN_DOT) &&
                                   e->u.call.nargs < 2);
        for (int i = 0; i < e->u.call.nargs; i++)
        {
            hit |= mc_expr_mentions_self(e->u.call.args[i], bare, is_red ? NULL : out_im);
        }
        return hit;
    }
    case MC_EXPR_ASSIGN:
        return mc_expr_mentions_self(e->u.assign.rhs, bare, out_im);
    default:
        return 0;
    }
}

/**
 * mc_self_init_create - Pre-create a zero-initialized image or variable for a recurrence target
 * @tb:       Bare identifier name of the recurrence target
 * @is_local: 1 to create an `@L:` process-local image, 0 for an `@S:` shared-memory stream
 * @rhs:      Resolved RHS expression providing the inferred result kind, shape, and datatype
 *
 * Return: void
 */
void mc_self_init_create(const char *tb, int is_local, const mc_expr_t *rhs)
{
    if (tb == NULL || tb[0] == '\0' || rhs == NULL)
    {
        return;
    }
    if (rhs->res_kind == MC_RES_IMAGE && rhs->res_naxis > 0 && rhs->res_size[0] > 0)
    {
        uint32_t sz[3] = {
            rhs->res_size[0],
            (rhs->res_naxis >= 2 && rhs->res_size[1] > 0) ? rhs->res_size[1] : 1,
            (rhs->res_naxis >= 3 && rhs->res_size[2] > 0) ? rhs->res_size[2] : 1,
        };
        uint8_t dt  = rhs->res_dt ? rhs->res_dt : _DATATYPE_FLOAT;
        imageID nid = -1;
        create_image_ID(tb, rhs->res_naxis, sz, dt, is_local ? 0 : 1, 10, 0, &nid);
    }
    else if (rhs->res_kind == MC_RES_SCALAR_L)
    {
        create_variable_long_ID(tb, 0);
    }
    else if (rhs->res_kind == MC_RES_SCALAR_D)
    {
        create_variable_ID(tb, 0.0);
    }
}

/* ============================================================================================== */
/* `hist()` / `prev()` Ring Buffer Management and Source Rewriter                                 */
/* ============================================================================================== */

/**
 * hist_ring - Validate source stream `d->name` and create or verify `<name>_hist`
 * @d:     History declaration entry (populated with `d->naxis` on success)
 * @err:   Error message buffer
 * @errsz: Capacity of @err in bytes
 *
 * Source streams must be 1D or 2D (`naxis` 1 or 2) and ring depth `d->N` must be at least 2.
 * Creates `<name>_hist` in shared memory with shape `(nx, N)` for 1D streams or `(nx, ny, N)` for
 * 2D streams, or verifies that an existing `<name>_hist` matches those dimensions.
 *
 * Return: 0 on success, -1 on error
 */
static int hist_ring(mc_hist_decl_t *d, char *err, size_t errsz)
{
    char    ring[STRINGMAXLEN_IMAGE_NAME];
    imageID id = mc_resolve_or_autoconnect_image(d->name);
    snprintf(ring, sizeof(ring), "%.*s_hist", (int) sizeof(ring) - 6, d->name);
    if (id == -1)
    {
        snprintf(err, errsz, "hist: stream '%s' not found", d->name);
        return -1;
    }
    const IMAGE *im = &milk_data.image[id];
    d->naxis        = im->md[0].naxis;
    if (d->naxis < 1 || d->naxis > 2 || d->N < 2)
    {
        snprintf(err, errsz, "hist: '%s' must be 1-D or 2-D and the depth at least 2", d->name);
        return -1;
    }
    uint32_t sz[3] = {
        im->md[0].size[0],
        (d->naxis == 2) ? im->md[0].size[1] : (uint32_t) d->N,
        (d->naxis == 2) ? (uint32_t) d->N : 1,
    };
    imageID rid = mc_resolve_or_autoconnect_image(ring);
    if (rid == -1)
    {
        create_image_ID(ring, d->naxis + 1, sz, im->md[0].datatype, 1, 10, 0, &rid);
        if (rid == -1)
        {
            snprintf(err, errsz, "hist: cannot create stream '%s'", ring);
            return -1;
        }
        int tsz = ImageStreamIO_typesize(im->md[0].datatype);
        if (tsz > 0 && milk_data.image[rid].array.raw != NULL)
        {
            memset(milk_data.image[rid].array.raw, 0,
                   (size_t) milk_data.image[rid].md[0].nelement * (size_t) tsz);
        }
        return 0;
    }
    const IMAGE *r = &milk_data.image[rid];
    if (r->md[0].naxis != d->naxis + 1 || r->md[0].size[0] != sz[0] || r->md[0].size[1] != sz[1] ||
        (d->naxis == 2 && r->md[0].size[2] != sz[2]))
    {
        snprintf(err, errsz, "hist: stream '%s' exists with another shape", ring);
        return -1;
    }
    return 0;
}
/**
 * struct hist_ctx_t - State of the `hist()` / `prev()` rewrite of one program
 * @decl:    Rings found so far
 * @nd:      Number of entries in @decl
 * @in_unit: 1 when compiling a unit (whose `_n` indexes the rings)
 * @off:     Byte span of the statement being rewritten
 * @end:
 * @err:     Error message buffer
 * @errsz:   Capacity of @err in bytes
 */
typedef struct
{
    mc_hist_decl_t decl[MC_HIST_MAX];
    int            nd;
    int            in_unit;
    int            off;
    int            end;
    char          *err;
    size_t         errsz;
} hist_ctx_t;

/* Index of the ring of stream @name, added with depth @N if new; -1 when there are too many */
static int hist_ring_of(hist_ctx_t *h, const char *name, int N)
{
    const char *bare =
        (name[0] == '@' && (name[1] == 'S' || name[1] == 'L') && name[2] == ':') ? name + 3 : name;
    for (int k = 0; k < h->nd; k++)
    {
        if (strcmp(h->decl[k].name, bare) == 0)
        {
            return k;
        }
    }
    if (h->nd == MC_HIST_MAX)
    {
        snprintf(h->err, h->errsz, "more than %d streams with a history", MC_HIST_MAX);
        return -1;
    }
    mc_hist_decl_t *d = &h->decl[h->nd];
    memset(d, 0, sizeof(*d));
    snprintf(d->name, sizeof(d->name), "%s", bare);
    d->N       = N;
    d->src_off = h->off;
    d->src_end = h->end;
    return h->nd++;
}

/* The expression of the one-statement text @text, or NULL */
static mc_expr_t *parse_snippet(const char *text)
{
    static _Thread_local mc_tok_list_t s_tl;
    mc_stmt_t *s = (mc_lex(text, &s_tl) == 0) ? mc_parse_stmt(&s_tl, NULL, 0) : NULL;
    mc_expr_t *e = (s != NULL && s->kind == MC_STMT_EXPR) ? s->u.expr : NULL;
    if (e != NULL)
    {
        s->u.expr = NULL;
    }
    mc_stmt_free(s);
    return e;
}

/* 1 if @e calls @fn with a stream name first */
static int is_ring_call(const mc_expr_t *e, const char *fn)
{
    return e != NULL && e->kind == MC_EXPR_CALL && strcmp(e->u.call.name, fn) == 0 &&
           e->u.call.nargs >= 1 && e->u.call.args[0]->kind == MC_EXPR_IDENT;
}

/* Pass 1: each statement `hist(s, N)`, at any depth, declares a ring and becomes empty */
static int hist_find_decls(hist_ctx_t *h, mc_stmt_t *s)
{
    for (; s != NULL; s = s->next)
    {
        mc_expr_t *e = (s->kind == MC_STMT_EXPR) ? s->u.expr : NULL;
        if (is_ring_call(e, "hist") && e->u.call.nargs == 2 &&
            e->u.call.args[1]->kind == MC_EXPR_LONG)
        {
            if (!h->in_unit)
            {
                snprintf(h->err, h->errsz, "hist() and prev() work only in a milk.calc unit");
                return -1;
            }
            h->off = s->src_off;
            h->end = s->src_end;
            int k  = hist_ring_of(h, e->u.call.args[0]->u.ident.name, 2);
            if (k < 0)
            {
                return -1;
            }
            h->decl[k].N       = (int) e->u.call.args[1]->u.lval;
            h->decl[k].src_off = s->src_off;
            h->decl[k].src_end = s->src_end;
            mc_expr_free(e);
            s->u.expr = NULL;
            s->kind   = MC_STMT_EMPTY;
        }
        for (int b = 0; s->kind == MC_STMT_IF && b < s->u.if_stmt.nbranches; b++)
        {
            if (hist_find_decls(h, s->u.if_stmt.branches[b].body) != 0)
            {
                return -1;
            }
        }
        if (s->kind == MC_STMT_FOR && hist_find_decls(h, s->u.for_stmt.body) != 0)
        {
            return -1;
        }
    }
    return 0;
}

/* Pass 2: each `prev(s[, k])` in @*pe becomes the slice of its ring for k frames ago */
static int hist_expand(hist_ctx_t *h, mc_expr_t **pe)
{
    mc_expr_t *e = *pe;
    if (is_ring_call(e, "prev") && e->u.call.nargs <= 2)
    {
        if (!h->in_unit)
        {
            snprintf(h->err, h->errsz, "hist() and prev() work only in a milk.calc unit");
            return -1;
        }
        int is_lit = (e->u.call.nargs == 1 || e->u.call.args[1]->kind == MC_EXPR_LONG);
        if (!is_lit && hist_expand(h, &e->u.call.args[1]) != 0)
        {
            return -1;
        }
        long lag = (e->u.call.nargs == 2 && is_lit) ? e->u.call.args[1]->u.lval : 1;
        int  k   = hist_ring_of(h, e->u.call.args[0]->u.ident.name, 2);
        if (k < 0)
        {
            return -1;
        }
        mc_hist_decl_t *d = &h->decl[k];
        if (is_lit && (lag < 1 || lag >= d->N))
        {
            snprintf(h->err, h->errsz, "prev(%s, %ld): the ring holds %d frames", d->name, lag,
                     d->N);
            return -1;
        }
        if (d->naxis == 0 && hist_ring(d, h->err, h->errsz) != 0)
        {
            return -1;
        }
        char text[STRINGMAXLEN_IMAGE_NAME + 64];
        if (is_lit)
        {
            snprintf(text, sizeof(text), "%s_hist[%s(_n + %d) %% %d]", d->name,
                     (d->naxis == 2) ? ":,:," : ":,", d->N - (int) lag, d->N);
        }
        else
        {
            snprintf(text, sizeof(text), "%s_hist[%s((_n %% %d) + %d - (0)) %% %d]", d->name,
                     (d->naxis == 2) ? ":,:," : ":,", d->N, d->N, d->N);
        }
        mc_expr_t *sl = parse_snippet(text);
        if (sl == NULL)
        {
            snprintf(h->err, h->errsz, "prev: '%s' is no stream name", d->name);
            return -1;
        }
        if (!is_lit)
        {
            mc_expr_t *sub = sl->u.slice.axes[d->naxis].start->u.binary.left;
            mc_expr_free(sub->u.binary.right);
            sub->u.binary.right = e->u.call.args[1];
            e->u.call.args[1]   = NULL;
        }
        sl->line = e->line;
        sl->col  = e->col;
        mc_expr_free(e);
        *pe = sl;
        return 0;
    }
    mc_expr_t **slots[MC_MAX_ARGS];
    for (int i = 0, n = mc_expr_child_slots(e, slots); i < n; i++)
    {
        if (hist_expand(h, slots[i]) != 0)
        {
            return -1;
        }
    }
    return 0;
}

/* Pass 2 over every expression of statement list @s */
static int hist_expand_stmts(hist_ctx_t *h, mc_stmt_t *s)
{
    for (; s != NULL; s = s->next)
    {
        h->off = s->src_off;
        h->end = s->src_end;
        int rc = 0;
        switch (s->kind)
        {
        case MC_STMT_EXPR:
            rc = hist_expand(h, &s->u.expr);
            break;
        case MC_STMT_MULTI_ASSIGN:
            for (int i = 0; rc == 0 && i < s->u.multi.nrhs; i++)
            {
                rc = hist_expand(h, &s->u.multi.rhs[i]);
            }
            break;
        case MC_STMT_IF:
            for (int b = 0; rc == 0 && b < s->u.if_stmt.nbranches; b++)
            {
                rc = (hist_expand(h, &s->u.if_stmt.branches[b].cond) != 0 ||
                      hist_expand_stmts(h, s->u.if_stmt.branches[b].body) != 0)
                         ? -1
                         : 0;
            }
            break;
        case MC_STMT_FOR:
            rc = (hist_expand(h, &s->u.for_stmt.start) != 0 ||
                  hist_expand(h, &s->u.for_stmt.stop) != 0 ||
                  hist_expand(h, &s->u.for_stmt.step) != 0 ||
                  hist_expand_stmts(h, s->u.for_stmt.body) != 0)
                     ? -1
                     : 0;
            break;
        case MC_STMT_DEF:
            rc = hist_expand(h, &s->def.body);
            break;
        default:
            break;
        }
        if (rc != 0)
        {
            return -1;
        }
    }
    return 0;
}

/**
 * mc_hist_rewrite - Turn the `hist(s, N)` declarations and `prev(s[, k])` calls of a program into rings
 * @head:      Statement list of the program; one ring update per ring is appended to it
 * @in_unit:   1 when compiling a unit, whose `_n` indexes the rings
 * @out_nhist: Receives the number of appended ring updates
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Each `hist(s, N)` statement declares an N-frame ring `<s>_hist` and becomes empty (the statement
 * numbering is kept). Each `prev(s[, k])` becomes the slice `<s>_hist[:, (_n + N - k) % N]`
 * (`[:,:, ...]` for a 2-D stream), N being 2 for a stream without `hist`. The rings are created or
 * checked here, and an update `<s>_hist[:, _n % N] = <s>` per ring is appended, flagged
 * `is_hist_stmt`, with the span of the statement that declared (or first used) the ring.
 *
 * Return: 0 on success, -1 on error (`hist` or `prev` outside a unit, too many rings, bad lag)
 */
int mc_hist_rewrite(mc_stmt_t **head, int in_unit, int *out_nhist, char *err, size_t errsz)
{
    hist_ctx_t h = { .in_unit = in_unit, .err = err, .errsz = errsz };
    *out_nhist   = 0;
    if (hist_find_decls(&h, *head) != 0 || hist_expand_stmts(&h, *head) != 0)
    {
        return -1;
    }
    mc_stmt_t **tail = head;
    while (*tail != NULL)
    {
        tail = &(*tail)->next;
    }
    for (int k = 0; k < h.nd; k++)
    {
        mc_hist_decl_t *d = &h.decl[k];
        char            text[2 * STRINGMAXLEN_IMAGE_NAME + 64];
        if (d->naxis == 0 && hist_ring(d, err, errsz) != 0)
        {
            return -1;
        }
        snprintf(text, sizeof(text), "%s_hist[%s_n %% %d] = %s", d->name,
                 (d->naxis == 2) ? ":,:," : ":,", d->N, d->name);
        mc_stmt_t *s = calloc(1, sizeof(*s));
        if (s == NULL || (s->u.expr = parse_snippet(text)) == NULL)
        {
            free(s);
            snprintf(err, errsz, "hist: '%s' is no stream name", d->name);
            return -1;
        }
        s->kind         = MC_STMT_EXPR;
        s->is_hist_stmt = 1;
        s->src_off      = d->src_off;
        s->src_end      = d->src_end;
        *tail           = s;
        tail            = &s->next;
    }
    *out_nhist = h.nd;
    return 0;
}

/* ============================================================================================== */
/* Program Execution Plan Explainer (`mcalc_prog_explain`)                                        */
/* ============================================================================================== */

/**
 * count_scalar_ops_reds - Count scalar operations and image reductions inside expression @e
 * @e:    AST expression to inspect recursively
 * @ops:  In/out counter incremented for each operator and function call
 * @reds: In/out counter incremented for each scalar reduction call (`imean`, `istats`, etc.)
 *
 * Used by `mcalc_prog_explain()` to summarize `MC_STEP_SCALAR` steps.
 *
 * Return: void
 */
static void count_scalar_ops_reds(const mc_expr_t *e, int *ops, int *reds)
{
    if (e == NULL)
    {
        return;
    }
    if (e->kind == MC_EXPR_ASSIGN)
    {
        count_scalar_ops_reds(e->u.assign.rhs, ops, reds);
        return;
    }
    if (e->kind == MC_EXPR_CALL)
    {
        const mc_func_t *fn = e->u.call.fn;
        if (fn != NULL && (fn->kind == MC_FN_REDUCE_D || fn->kind == MC_FN_REDUCE_L ||
                           fn->kind == MC_FN_DOT || fn->kind == MC_FN_EAGER_RED ||
                           fn->kind == MC_FN_ISTATS || fn->kind == MC_FN_ICENTROID))
        {
            (*reds)++;
            (*ops)++;
            return;
        }
        (*ops)++;
        for (int i = 0; i < e->u.call.nargs; i++)
        {
            count_scalar_ops_reds(e->u.call.args[i], ops, reds);
        }
        return;
    }
    if (e->kind == MC_EXPR_UNARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.unary.operand, ops, reds);
    }
    else if (e->kind == MC_EXPR_BINARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.binary.left, ops, reds);
        count_scalar_ops_reds(e->u.binary.right, ops, reds);
    }
    else if (e->kind == MC_EXPR_TERNARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.ternary.cond, ops, reds);
        count_scalar_ops_reds(e->u.ternary.t_expr, ops, reds);
        count_scalar_ops_reds(e->u.ternary.f_expr, ops, reds);
    }
    else if (e->kind == MC_EXPR_CHAIN)
    {
        (*ops) += e->u.chain.nops;
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            count_scalar_ops_reds(e->u.chain.operands[i], ops, reds);
        }
    }
}

/**
 * mcalc_prog_explain - Compile @text and print its execution plan to @out, without running it
 * @text: Program source text to analyze
 * @out:  Output `FILE *` stream (e.g. `stdout`)
 *
 * Prints each user statement (skipping synthesized `hist()` tail statements) followed by how the
 * planner scheduled it:
 *   - `def, inlined where it is called`
 *   - `kept, merged into the fused pass of statement K`
 *   - `kept: fused pass over NxMxK (<dtype>, 1D|ND, P operands[, Q pre-stages]), writes ...`, or
 *     `over a shape set at run time (<dtype>, P operands...)` when planning cannot know it (an
 *     operand is a target the program creates, or the operand shapes do not match)
 *   - `kept: constructor/library step`
 *   - `kept: scalar tape: N ops, M reductions, 0 pre-stages`
 *
 * Return: 0 on success, 1 on compilation error
 */
int mcalc_prog_explain(const char *text, FILE *out)
{
    char err[512] = "";
    mc_udef_save(); /* its defs stay out of the session */
    mcalc_prog *p = mcalc_prog_new(text, NULL, 0, err, sizeof(err));
    mc_udef_restore();
    if (p == NULL)
    {
        if (out != NULL)
        {
            fprintf(out, "milk: calc error: %s\n", err[0] != '\0' ? err : "compilation error");
        }
        return 1;
    }
    int              nkept = 0;
    const mc_stmt_t *cur_s = p->ast_head;
    for (int i = 0; i < p->nstmts; i++, cur_s = cur_s ? cur_s->next : NULL)
    {
        if (cur_s != NULL && cur_s->is_hist_stmt)
        {
            continue;
        }
        const char *st_txt =
            (p->stmt_texts != NULL && p->stmt_texts[i] != NULL) ? p->stmt_texts[i] : "";
        int  sidx = (p->stmt_to_step != NULL) ? p->stmt_to_step[i] : -1;
        char how[512];
        if (sidx < 0)
        {
            snprintf(how, sizeof(how), "def, inlined where it is called");
        }
        else
        {
            const mc_step_t *st = &p->steps[sidx];
            nkept++;
            if (st->kind == MC_STEP_PASS)
            {
                const char *wtype =
                    st->pass.is_complex
                        ? (st->pass.work_double ? "complex double" : "complex float32")
                        : (st->pass.work_double ? "double" : "float32");
                if (i > st->first_stmt_idx)
                {
                    snprintf(how, sizeof(how), "kept, merged into the fused pass of statement %d",
                             st->first_stmt_idx + 1);
                }
                else
                {
                    char   ps_buf[64]  = "";
                    char   wr_buf[256] = "";
                    size_t wlen        = 0;
                    if (st->pass.prestage_cnt > 0)
                    {
                        snprintf(ps_buf, sizeof(ps_buf), ", %d pre-stage%s", st->pass.prestage_cnt,
                                 (st->pass.prestage_cnt == 1) ? "" : "s");
                    }
                    int nin_ops = 0;
                    for (int k = 0; k < st->pass.nops; k++)
                    {
                        nin_ops += (!st->pass.ops[k].is_out && !st->pass.ops[k].is_scalar);
                    }
                    for (int k = 0; k < st->pass.nouts; k++)
                    {
                        if (!st->pass.outs[k].is_elided && wlen + 32 < sizeof(wr_buf))
                        {
                            wlen += (size_t) snprintf(wr_buf + wlen, sizeof(wr_buf) - wlen, " %s",
                                                      p->slots[st->pass.outs[k].slot_idx].bare);
                        }
                    }
                    const char *lim_note  = (cur_s != NULL && cur_s->kind == MC_STMT_MULTI_ASSIGN &&
                                             cur_s->u.multi.nlhs > 4)
                                                ? " (more than 4 targets (FUSE_MAXDST))"
                                                : "";
                    char        shape[48] = "a shape set at run time";
                    if (st->pass.vnx != 0)
                    {
                        snprintf(shape, sizeof(shape), "%ux%ux%u", st->pass.vnx,
                                 st->pass.vny ? st->pass.vny : 1, st->pass.vnz ? st->pass.vnz : 1);
                    }
                    snprintf(how, sizeof(how),
                             "kept: fused pass over %s (%s, %s%d operand%s%s)%s, writes%s", shape,
                             wtype,
                             (st->pass.vnx == 0)     ? ""
                             : st->pass.is_1d_contig ? "1D, "
                                                     : "ND, ",
                             nin_ops, (nin_ops == 1) ? "" : "s", ps_buf, lim_note, wr_buf);
                }
            }
            else if (st->kind == MC_STEP_CTOR)
            {
                const char *lim_note = (cur_s != NULL && cur_s->kind == MC_STMT_MULTI_ASSIGN &&
                                        cur_s->u.multi.nlhs > 4)
                                           ? " (more than 4 targets (FUSE_MAXDST))"
                                           : "";
                snprintf(how, sizeof(how), "kept: constructor/library step%s", lim_note);
            }
            else
            {
                int nops  = 0;
                int nreds = 0;
                if (st->scalar_stmt != NULL && st->scalar_stmt->kind == MC_STMT_EXPR)
                {
                    count_scalar_ops_reds(st->scalar_stmt->u.expr, &nops, &nreds);
                }
                else if (st->scalar_stmt != NULL && st->scalar_stmt->kind == MC_STMT_MULTI_ASSIGN)
                {
                    for (int j = 0; j < st->scalar_stmt->u.multi.nrhs; j++)
                    {
                        count_scalar_ops_reds(st->scalar_stmt->u.multi.rhs[j], &nops, &nreds);
                    }
                }
                if (nops == 0)
                {
                    nops = 1;
                }
                snprintf(how, sizeof(how),
                         "kept: scalar tape: %d op%s, %d reduction%s, 0 pre-stages", nops,
                         (nops == 1) ? "" : "s", nreds, (nreds == 1) ? "" : "s");
            }
        }
        if (out != NULL)
        {
            fprintf(out, "%3d  %s\n       %s\n", i + 1, st_txt, how);
        }
    }
    if (out != NULL)
    {
        fprintf(out, "%d statement%s kept, 0 evaluated as text\n", nkept, (nkept == 1) ? "" : "s");
    }
    mcalc_prog_free(p);
    return 0;
}
