// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILK_MC_AST_H
#define MILK_MC_AST_H

#include <stddef.h>
#include <stdint.h>

#include "mc_func.h"
#include "mcalc.h"

#define MC_TOK_STR_MAX 1024
#define MC_MAX_TOKENS 512
#define MC_MAX_ARGS 16
#define MC_MAX_LHS 8
#define MC_MAX_CHAIN 8
#define MC_MAX_SLICE_AX 3

typedef enum
{
    MC_TOK_EOF = 0,
    MC_TOK_NEWLINE,
    MC_TOK_SEMI,
    MC_TOK_COMMA,
    MC_TOK_LPAREN,
    MC_TOK_RPAREN,
    MC_TOK_LBRACK,
    MC_TOK_RBRACK,
    MC_TOK_LBRACE,
    MC_TOK_RBRACE,
    MC_TOK_LONG,
    MC_TOK_DOUBLE,
    MC_TOK_STRING,
    MC_TOK_IDENT,
    MC_TOK_ASSIGN,     /* =  */
    MC_TOK_ADD_ASSIGN, /* += */
    MC_TOK_SUB_ASSIGN, /* -= */
    MC_TOK_MUL_ASSIGN, /* *= */
    MC_TOK_DIV_ASSIGN, /* /= */
    MC_TOK_QUEST,      /* ?  */
    MC_TOK_COLON,      /* :  */
    MC_TOK_OR,         /* || */
    MC_TOK_AND,        /* && */
    MC_TOK_NOT,        /* !  */
    MC_TOK_BOR,        /* |  */
    MC_TOK_BXOR,       /* ^^ */
    MC_TOK_BAND,       /* &  */
    MC_TOK_BNOT,       /* ~  */
    MC_TOK_SHL,        /* << */
    MC_TOK_SHR,        /* >> */
    MC_TOK_EQ,         /* == */
    MC_TOK_NE,         /* != */
    MC_TOK_LT,         /* <  */
    MC_TOK_LE,         /* <= */
    MC_TOK_GT,         /* >  */
    MC_TOK_GE,         /* >= */
    MC_TOK_PLUS,       /* +  */
    MC_TOK_MINUS,      /* -  */
    MC_TOK_MUL,        /* *  */
    MC_TOK_DIV,        /* /  */
    MC_TOK_MOD,        /* %  */
    MC_TOK_MATMUL,     /* @  */
    MC_TOK_DOT_T       /* .T */
} mc_tok_type_t;

typedef struct
{
    mc_tok_type_t type;
    int           line;
    int           col;
    int           no_space_after; /* 1 if immediately followed by '(' without whitespace */
    long          lval;
    double        dval;
    char          sval[MC_TOK_STR_MAX];
} mc_tok_t;

typedef struct
{
    mc_tok_t toks[MC_MAX_TOKENS];
    int      ntok;
    int      pos;
    int      err_line;
    int      err_col;
    char     errmsg[256];
} mc_tok_list_t;

/* Runtime scalar / image value representation */
typedef enum
{
    MC_VAL_LONG = 1,
    MC_VAL_DOUBLE,
    MC_VAL_STRING,
    MC_VAL_IMAGE
} mc_val_type_t;

typedef struct
{
    mc_val_type_t type;
    long          lval;
    double        dval;
    char          sval[MC_TOK_STR_MAX];
} mc_val_t;

/* Resolved expression value category */
typedef enum
{
    MC_RES_UNKNOWN = 0,
    MC_RES_SCALAR_L,
    MC_RES_SCALAR_D,
    MC_RES_STRING,
    MC_RES_IMAGE
} mc_res_kind_t;

/* Resolved identifier binding kind */
typedef enum
{
    MC_ID_UNRESOLVED = 0,
    MC_ID_DEF_PARAM, /* bound parameter of active def function */
    MC_ID_CONST,     /* fallback constant pi, nan, inf */
    MC_ID_VAR,       /* numeric/string variable in data.core.variable or host/env */
    MC_ID_IMAGE,     /* existing image in data.core.image */
    MC_ID_PROP,      /* SHM property s.<stream>.<prop>, f.<fps>.<prop>, p.<proc>.<prop> */
    MC_ID_NEW        /* new target identifier on LHS */
} mc_ident_kind_t;

typedef struct mc_expr mc_expr_t;

/* One axis of a bracket slice: start[:stop[:step[b]]] or * or -* */
typedef struct
{
    int        is_full;     /* 1 for '*' or ':' */
    int        is_rev_full; /* 1 for '-*' */
    int        has_colon;   /* 1 if ':' present on this axis */
    int        has_step;    /* 1 if second ':' present */
    int        is_binned;   /* 1 if step has 'b' or 'B' suffix */
    mc_expr_t *start;       /* NULL when omitted before ':' */
    mc_expr_t *stop;        /* NULL when has_colon == 0 or omitted after ':' */
    mc_expr_t *step;        /* NULL when has_step == 0 */
} mc_slice_axis_t;

/* Comprehension generator axis: var = start : stop [: step] */
typedef struct
{
    char       var[64];
    mc_expr_t *start;
    mc_expr_t *stop;
    mc_expr_t *step; /* NULL for default step 1 */
} mc_comp_gen_t;

typedef enum
{
    MC_EXPR_LONG = 0,
    MC_EXPR_DOUBLE,
    MC_EXPR_STRING,
    MC_EXPR_IDENT,
    MC_EXPR_END,     /* 'end' keyword inside slice bracket */
    MC_EXPR_UNARY,   /* -x, ~x, !x */
    MC_EXPR_BINARY,  /* a op b */
    MC_EXPR_CHAIN,   /* chained relational comparison a <= x <= b */
    MC_EXPR_TERNARY, /* cond ? t : f */
    MC_EXPR_CALL,    /* fn(arg1, ...) or udef(arg1, ...) or ext(arg1, k=v, ...) */
    MC_EXPR_SLICE,   /* target[ax0, ...] or im[mask] */
    MC_EXPR_ASSIGN,  /* target op= rhs (as expression or statement root) */
    MC_EXPR_VEC_LIT, /* [e1, e2, ...] 1-D vector literal */
    MC_EXPR_COMP     /* [body for x = a:b, ...] array comprehension */
} mc_expr_kind_t;

struct mc_expr
{
    mc_expr_kind_t kind;
    int            line;
    int            col;

    /* Resolved type and shape */
    mc_res_kind_t      res_kind;
    uint8_t            res_dt;      /* _DATATYPE_* when MC_RES_IMAGE */
    int                res_naxis;   /* 1..3 when MC_RES_IMAGE */
    uint32_t           res_size[3]; /* extents when MC_RES_IMAGE */
    int                needs_image; /* 1 if subtree involves any image operation */
    struct mcalc_prog *sub_prog;    /* cached sub-program when evaluated as image sub-expr */

    union
    {
        long   lval;
        double dval;
        char   sval[MC_TOK_STR_MAX];

        struct
        {
            char            name[MC_TOK_STR_MAX];
            char            bare[256];
            char            loc; /* 'S', 'L', or 0 */
            mc_ident_kind_t id_kind;
            long            var_id;
            long            im_id;
            int             def_param_idx;
            double          const_val;
        } ident;

        struct
        {
            mc_tok_type_t op;
            mc_expr_t    *operand;
        } unary;

        struct
        {
            mc_tok_type_t op;
            mc_expr_t    *left;
            mc_expr_t    *right;
        } binary;

        struct
        {
            int           nops; /* number of comparisons (>= 2) */
            mc_tok_type_t ops[MC_MAX_CHAIN];
            mc_expr_t    *operands[MC_MAX_CHAIN + 1];
        } chain;

        struct
        {
            mc_expr_t *cond;
            mc_expr_t *t_expr;
            mc_expr_t *f_expr;
        } ternary;

        struct
        {
            char             name[128];
            const mc_func_t *fn;      /* built-in function entry, or NULL */
            int              udef_id; /* index in udef table, or -1 */
            int              ext_id;  /* index in milk_calc_ext registry, or -1 */
            int              nargs;
            mc_expr_t       *args[MC_MAX_ARGS];
            char             kw_names[MC_MAX_ARGS][64]; /* optional keyword arg name k=v */
            mc_expr_t       *udef_body;                 /* resolved inline body for udef_id >= 0 */
            int              ema_init;
            double           ema_val;
        } call;

        struct
        {
            char            name[MC_TOK_STR_MAX];
            char            bare[256];
            long            im_id;
            int             naxis;
            mc_slice_axis_t axes[MC_MAX_SLICE_AX];
            int             is_mask_index;   /* 1 when im[mask_expr] */
            int             is_single_pixel; /* 1 when all axes are scalar single indices */
        } slice;

        struct
        {
            mc_tok_type_t op;     /* MC_TOK_ASSIGN, MC_TOK_ADD_ASSIGN, ... */
            mc_expr_t    *target; /* MC_EXPR_IDENT or MC_EXPR_SLICE */
            mc_expr_t    *rhs;
            int           is_local;
        } assign;

        struct
        {
            int        nelem;
            mc_expr_t *elems[MC_MAX_ARGS];
        } vec_lit;

        struct
        {
            mc_expr_t    *body;
            int           ngens;
            mc_comp_gen_t gens[MC_MAX_SLICE_AX];
        } comp;
    } u;
};

typedef enum
{
    MC_STMT_EMPTY = 0,
    MC_STMT_EXPR,         /* single expression or single-LHS assignment */
    MC_STMT_MULTI_ASSIGN, /* a, b[, c, d] = rhs1[, rhs2...] or unpacking call */
    MC_STMT_DEF,          /* def f(x, ...) = body */
    MC_STMT_SKIP,         /* skip (inside unit program) */
    MC_STMT_IF,           /* if / elif / else */
    MC_STMT_FOR           /* for var = start:stop[:step] ... end */
} mc_stmt_kind_t;

typedef struct mc_stmt mc_stmt_t;

typedef struct
{
    mc_expr_t         *cond; /* NULL for else branch */
    mc_stmt_t         *body; /* linked list of statements */
    struct mcalc_prog *sub_prog;
} mc_if_branch_t;

struct mc_stmt
{
    mc_stmt_kind_t kind;
    int            line;
    int            col;
    int            is_local;     /* leading 'local' keyword */
    int            needs_image;  /* 1 if statement requires image engine */
    int            is_hist_stmt; /* 1 if appended by hist() ring rewriter */
    mc_stmt_t     *next;

    union
    {
        mc_expr_t *expr;

        struct
        {
            int        nlhs;
            mc_expr_t *lhs[MC_MAX_LHS];
            int        nrhs;
            mc_expr_t *rhs[MC_MAX_LHS];
        } multi;

        struct
        {
            int            nbranches;
            mc_if_branch_t branches[16];
        } if_stmt;

        struct
        {
            char               var[64];
            mc_expr_t         *start;
            mc_expr_t         *stop;
            mc_expr_t         *step; /* NULL for default 1 */
            mc_stmt_t         *body;
            struct mcalc_prog *sub_prog;
        } for_stmt;
    } u;
};

/** mc_op_prec() - return binary/ternary operator precedence (or 0 if not an infix operator). */
int mc_op_prec(mc_tok_type_t t);

/** mc_is_assign_tok() - return 1 if @t is `=`, `+=`, `-=`, `*=`, or `/=`. */
static inline int mc_is_assign_tok(mc_tok_type_t t)
{
    return t == MC_TOK_ASSIGN || t == MC_TOK_ADD_ASSIGN || t == MC_TOK_SUB_ASSIGN ||
           t == MC_TOK_MUL_ASSIGN || t == MC_TOK_DIV_ASSIGN;
}

/** mc_is_rel_tok() - return 1 if @t is `<`, `<=`, `>`, or `>=`. */
static inline int mc_is_rel_tok(mc_tok_type_t t)
{
    return t == MC_TOK_LT || t == MC_TOK_LE || t == MC_TOK_GT || t == MC_TOK_GE;
}

/** mc_scan_number() - scan a numeric literal at @p; return bytes consumed (0 if not a number). */
size_t mc_scan_number(const char *p, mc_tok_type_t *out_type, long *out_l, double *out_d);

/** mc_lex() - tokenize calculator source @src into @out (returns 0 on ok, -1 on lexical error). */
int mc_lex(const char *src, mc_tok_list_t *out);

/** mc_parse_stmt() - parse a single statement from @tl (returns NULL on syntax error). */
mc_stmt_t *mc_parse_stmt(mc_tok_list_t *tl, char *err, size_t errsz);

/** mc_parse_prog() - parse a multi-statement program / block from @tl (NULL on error). */
mc_stmt_t *mc_parse_prog(mc_tok_list_t *tl, char *err, size_t errsz);

/** mc_expr_free() - recursively free an AST expression node. */
void mc_expr_free(mc_expr_t *e);

/** mc_get_or_build_sub_prog() - return or build cached sub-program attached to @expr. */
struct mcalc_prog *mc_get_or_build_sub_prog(const mc_expr_t *expr, char *err, size_t errsz);

/** mc_stmt_free() - recursively free a linked list of AST statements. */
void mc_stmt_free(mc_stmt_t *s);

/** mc_resolve_prog_ex() - resolve all statements in @prog with unit context flag @in_unit. */
int mc_resolve_prog_ex(mc_stmt_t *prog, int in_unit, char *err, size_t errsz);

/** mc_resolve_stmt() - resolve names, types, shapes, and arities in @s (0 on ok, -1 on error). */
static inline int mc_resolve_stmt(mc_stmt_t *s, char *err, size_t errsz)
{
    return mc_resolve_prog_ex(s, 0, err, errsz);
}

/** mc_resolve_prog() - resolve all statements in program list @prog (0 on ok, -1 on error). */
static inline int mc_resolve_prog(mc_stmt_t *prog, char *err, size_t errsz)
{
    return mc_resolve_prog_ex(prog, 0, err, errsz);
}

/** mcalc_sub_prog_free() - free a cached branch/loop sub-program without freeing its AST. */
void mcalc_sub_prog_free(struct mcalc_prog *p);

/** mc_exec_sub_prog() - compile on first run and execute a branch/loop statement list @head. */
int mc_exec_sub_prog(struct mcalc_prog **io_prog, mc_stmt_t *head, char *err, size_t errsz);

/** mc_prog_set_skip() - set or clear the active unit `skip` flag for the current tick. */
void mc_prog_set_skip(int on);

/** mc_prog_skip_active() - return 1 if `skip` was triggered during the current tick. */
int mc_prog_skip_active(void);

/** mc_prog_in_unit() - return 1 if currently executing inside a `milk.calc` unit program. */
int mc_prog_in_unit(void);

/** mc_scalar_eval_expr() - evaluate a resolved scalar expression @e into @out (0 on ok, -1 on err). */
int mc_scalar_eval_expr(const mc_expr_t *e, mc_val_t *out, char *err, size_t errsz);

/** mc_scalar_set_slice_end() - set active slice axis 'end' bound (-1 when inactive). */
void mc_scalar_set_slice_end(long end_val);

/** mc_scalar_sync_host() - adjust active `mcalc_eval_bash` depth for host string sync. */
int mc_scalar_sync_host(int delta);

/** mc_scalar_exec_stmt() - execute a resolved scalar statement @s into @out (0 on ok, -1 on err). */
int mc_scalar_exec_stmt(const mc_stmt_t *s, mc_val_t *out, char *err, size_t errsz);

/** mc_scalar_exec_prog() - execute a resolved scalar program list @prog (0 on ok, -1 on err). */
int mc_scalar_exec_prog(const mc_stmt_t *prog, char *err, size_t errsz);

/** mc_exec_stmt_list() - resolve, plan, and execute a statement list @head (0 on ok, -1 on err). */
int mc_exec_stmt_list(mc_stmt_t *head, char *err, size_t errsz);

/** mc_scalar_binop() - apply binary operator @op to scalar values @l and @r (NULL on ok, else err). */
const char *mc_scalar_binop(mc_tok_type_t op, const mc_val_t *l, const mc_val_t *r, mc_val_t *out);

/** mc_scalar_pow() - evaluate pow(@a, @b) with exact int64 when b >= 0 fits in int64. */
void mc_scalar_pow(const mc_val_t *a, const mc_val_t *b, mc_val_t *out);

/** mc_val_to_str() - format scalar value @v into string buffer @buf. */
void mc_val_to_str(const mc_val_t *v, char *buf, size_t bufsz, int use_float_digits);

/** mc_scalar_str_fn() - evaluate scalar string function @fn on @args into @out. */
const char *mc_scalar_str_fn(const mc_func_t *fn, const mc_val_t *args, mc_val_t *out);

/** mc_is_numeric_env_var() - return 1 if environment variable @s holds a valid number. */
int mc_is_numeric_env_var(const char *s);

/** mc_resolve_or_autoconnect_image() - look up or auto-connect image stream @name. */
long mc_resolve_or_autoconnect_image(const char *name);

/** mc_parse_ident_streamname() - resolve string alias and parse `@S:`/`@L:` stream modifier. */
void mc_parse_ident_streamname(char *name, char *bare, char *loc);

/** mc_set_expr_from_im() - populate expression result shape/type from image @im_ptr. */
void mc_set_expr_from_im(mc_expr_t *e, const void *im_ptr);

/** mc_expr_mentions_self() - return 1 if @e references identifier @bare, setting @out_im to peer. */
int mc_expr_mentions_self(const mc_expr_t *e, const char *bare, const void **out_im);

/** mc_self_init_create() - create frame-0 zero-initialized stream or variable @tb for @rhs. */
void mc_self_init_create(const char *tb, int is_local, const mc_expr_t *rhs);

/** mc_rewrite_history() - expand `hist(s, N)` and `prev(s[, k])` into ring-buffer slices. */
char *mc_rewrite_history(const char *text, int *out_nhist, char *err, size_t errsz);

#endif /* MILK_MC_AST_H */
