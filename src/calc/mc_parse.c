// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_parse.c
 * @brief Pratt expression parser and recursive-descent statement parser for libmilkcalc.
 *
 * This module implements the second stage of the calculator front-end pipeline:
 * consuming a flat `mc_tok_list_t` from `mc_lex()` and building an unresolved
 * Abstract Syntax Tree (`mc_stmt_t` / `mc_expr_t`) for subsequent type and
 * symbol resolution in `mc_resolve.c`.
 *
 * Key responsibilities:
 * - Precedence-climbing (Pratt) expression parsing (`parse_expr`,
 *   `parse_primary`) with support for unary operators, ternary `c ? a : b`,
 *   chained relational comparisons (`a <= x <= b`), postfix `.T` (desugared to
 *   `transpose(x)`), and infix `@` (desugared to `matmul(a, b)`).
 * - Bracket syntax disambiguation between N-D image slices (`im[0:10:2b, *]`),
 *   1-D vector literals (`[1, 2, 3]`), and array comprehensions
 *   (`[expr for x = a:b[:s], ...]`).
 * - Function call parsing with optional keyword arguments (`k=v`) for user C
 *   extensions.
 * - Statement and control-flow parsing for single-line (`if cond: stmt`),
 *   braced (`if cond { ... }`), and keyword-terminated (`if ... elif ... else
 *   ... end`, `for var = a:b[:s] ... end`) blocks, `local` declarations,
 *   multi-target tuple assignments (`a, b = ...`), `def`, and `skip`.
 */

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mc_ast.h"

/* ========================================================================= */
/* Parser State and Token Stream Helpers                                     */
/* ========================================================================= */

/**
 * struct mc_parse_ctx_t - Mutable state threaded through recursive-descent parsing
 * @tl:               Token list being consumed (`tl->pos` tracks current cursor)
 * @in_slice_bracket: Nesting depth inside `im[...]` slice brackets (enables `end` keyword)
 * @err_flag:         Non-zero once the first syntax error has been recorded
 * @err_line:         1-based source line of the first syntax error
 * @err_col:          1-based source column of the first syntax error
 * @errmsg:           Formatted diagnostic message for the first syntax error
 */
typedef struct
{
    mc_tok_list_t *tl;
    int            in_slice_bracket;
    int            err_flag;
    int            err_line;
    int            err_col;
    char           errmsg[256];
} mc_parse_ctx_t;

/** cur_tok() - Return pointer to the current lookahead token without advancing. */
static inline const mc_tok_t *cur_tok(const mc_parse_ctx_t *ctx)
{
    return &ctx->tl->toks[ctx->tl->pos];
}

/** advance_tok() - Return current token and advance cursor unless at `MC_TOK_EOF`. */
static inline const mc_tok_t *advance_tok(mc_parse_ctx_t *ctx)
{
    const mc_tok_t *t = &ctx->tl->toks[ctx->tl->pos];
    if (t->type != MC_TOK_EOF)
    {
        ctx->tl->pos++;
    }
    return t;
}

/**
 * parse_err - Record the first syntax error encountered during parsing
 * @ctx: Active parser context
 * @tok: Offending token used for line/column location (may be NULL)
 * @fmt: printf-style error message format string
 *
 * Subsequent calls after `ctx->err_flag` is set are ignored so that cascading
 * recovery errors never overwrite the root-cause diagnostic.
 */
__attribute__((format(printf, 3, 4))) static void parse_err(mc_parse_ctx_t *ctx,
                                                            const mc_tok_t *tok,
                                                            const char     *fmt,
                                                            ...)
{
    if (ctx->err_flag)
    {
        return;
    }
    ctx->err_flag = 1;
    ctx->err_line = (tok != NULL) ? tok->line : 1;
    ctx->err_col  = (tok != NULL) ? tok->col : 1;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->errmsg, sizeof(ctx->errmsg), fmt, ap);
    va_end(ap);
}

/**
 * expect_tok - Consume expected token type @t or record a syntax error
 * @ctx: Active parser context
 * @t:   Required token type
 * @msg: Diagnostic message if the current token does not match @t
 */
static void expect_tok(mc_parse_ctx_t *ctx, mc_tok_type_t t, const char *msg)
{
    if (!ctx->err_flag)
    {
        if (cur_tok(ctx)->type != t)
        {
            parse_err(ctx, cur_tok(ctx), "%s", msg);
        }
        else
        {
            advance_tok(ctx);
        }
    }
}

/**
 * alloc_expr - Allocate and zero-initialize an AST expression node
 * @kind: Expression node discriminant
 * @tok:  Token providing source line and column coordinates (may be NULL)
 *
 * Initializes sentinel IDs (`im_id`, `var_id`, `def_param_idx`, `udef_id`,
 * `ext_id`) to -1 for identifier and call nodes.
 *
 * Return: Newly allocated `mc_expr_t` pointer, or NULL on allocation failure
 */
static mc_expr_t *alloc_expr(mc_expr_kind_t kind, const mc_tok_t *tok)
{
    mc_expr_t *e = (mc_expr_t *) calloc(1, sizeof(mc_expr_t));
    if (e != NULL)
    {
        e->kind = kind;
        e->line = (tok != NULL) ? tok->line : 1;
        e->col  = (tok != NULL) ? tok->col : 1;
        if (kind == MC_EXPR_IDENT)
        {
            e->u.ident.im_id         = -1;
            e->u.ident.var_id        = -1;
            e->u.ident.def_param_idx = -1;
        }
        else if (kind == MC_EXPR_CALL)
        {
            e->u.call.udef_id = -1;
            e->u.call.ext_id  = -1;
        }
    }
    return e;
}

static mc_expr_t *parse_expr(mc_parse_ctx_t *ctx, int min_prec);

/**
 * make_call - Synthesize a 1- or 2-argument built-in function call AST node
 * @tok:     Operator token providing source coordinates
 * @fn_name: Built-in function name (e.g. `"transpose"` or `"matmul"`)
 * @arg0:    First argument expression
 * @arg1:    Second argument expression, or NULL for unary calls
 *
 * Used to desugar postfix `.T` into `transpose(x)` and infix `@` into
 * `matmul(a, b)`.
 *
 * Return: Newly allocated `MC_EXPR_CALL` node
 */
static mc_expr_t *make_call(const mc_tok_t *tok,
                            const char     *fn_name,
                            mc_expr_t      *arg0,
                            mc_expr_t      *arg1)
{
    mc_expr_t *c = alloc_expr(MC_EXPR_CALL, tok);
    snprintf(c->u.call.name, sizeof(c->u.call.name), "%s", fn_name);
    c->u.call.fn      = mc_func_find(fn_name);
    c->u.call.udef_id = -1;
    c->u.call.ext_id  = -1;
    c->u.call.nargs   = (arg1 != NULL) ? 2 : 1;
    c->u.call.args[0] = arg0;
    c->u.call.args[1] = arg1;
    return c;
}

/* ========================================================================= */
/* Slice, Vector Literal, and Comprehension Parsing                          */
/* ========================================================================= */

/**
 * parse_one_slice_axis - Parse a single axis specification inside `im[...]`
 * @ctx: Active parser context
 * @ax:  Output slice axis descriptor
 *
 * Supports full-axis wildcards (`:`, `*`, or empty before `,`/`]`), reversed
 * full-axis (`-*`), single scalar index (`expr`), and range specifications
 * (`[start]:[stop][:[step][b|B]]`, including `::step`).
 *
 * Return: 0 on success, -1 on syntax error
 */
static int parse_one_slice_axis(mc_parse_ctx_t *ctx, mc_slice_axis_t *ax)
{
    memset(ax, 0, sizeof(*ax));
    const mc_tok_t *t = cur_tok(ctx);

    /* Empty axis before ',' or ']' means full axis */
    if (t->type == MC_TOK_COMMA || t->type == MC_TOK_RBRACK)
    {
        ax->is_full = 1;
        return 0;
    }

    /* '*' wildcard means full axis */
    if (t->type == MC_TOK_MUL && (ctx->tl->toks[ctx->tl->pos + 1].type == MC_TOK_COMMA ||
                                  ctx->tl->toks[ctx->tl->pos + 1].type == MC_TOK_RBRACK))
    {
        advance_tok(ctx);
        ax->is_full = 1;
        return 0;
    }

    /* '-*' wildcard means reversed full axis */
    if (t->type == MC_TOK_MINUS && ctx->tl->toks[ctx->tl->pos + 1].type == MC_TOK_MUL &&
        (ctx->tl->toks[ctx->tl->pos + 2].type == MC_TOK_COMMA ||
         ctx->tl->toks[ctx->tl->pos + 2].type == MC_TOK_RBRACK))
    {
        advance_tok(ctx);
        advance_tok(ctx);
        ax->is_full     = 1;
        ax->is_rev_full = 1;
        return 0;
    }

    /* Optional start expression before ':' */
    if (cur_tok(ctx)->type != MC_TOK_COLON)
    {
        ax->start = parse_expr(ctx, 0);
        if (ctx->err_flag)
        {
            return -1;
        }
    }

    /* Optional ':stop[:step[b]]' range suffix */
    if (cur_tok(ctx)->type == MC_TOK_COLON)
    {
        advance_tok(ctx);
        ax->has_colon = 1;
        if (cur_tok(ctx)->type != MC_TOK_COLON && cur_tok(ctx)->type != MC_TOK_COMMA &&
            cur_tok(ctx)->type != MC_TOK_RBRACK)
        {
            ax->stop = parse_expr(ctx, 0);
            if (ctx->err_flag)
            {
                return -1;
            }
        }
        if (cur_tok(ctx)->type == MC_TOK_COLON)
        {
            advance_tok(ctx);
            if (cur_tok(ctx)->type == MC_TOK_COLON)
            {
                advance_tok(ctx);
            }
            ax->has_step = 1;
            if (cur_tok(ctx)->type != MC_TOK_COMMA && cur_tok(ctx)->type != MC_TOK_RBRACK)
            {
                ax->step = parse_expr(ctx, 0);
                if (ctx->err_flag)
                {
                    return -1;
                }
                if (cur_tok(ctx)->type == MC_TOK_IDENT &&
                    (strcmp(cur_tok(ctx)->sval, "b") == 0 || strcmp(cur_tok(ctx)->sval, "B") == 0))
                {
                    advance_tok(ctx);
                    ax->is_binned = 1;
                }
            }
        }
        ax->is_full = (ax->start == NULL && ax->stop == NULL && !ax->has_step);
    }
    return 0;
}

/**
 * parse_slice_expr - Parse an image slice or mask-indexing expression `ident[...]`
 * @ctx:    Active parser context (positioned at `[`)
 * @id_tok: Identifier token naming the sliced image
 *
 * Parses up to `MC_MAX_SLICE_AX` (3) comma-separated axis specifications while
 * incrementing `ctx->in_slice_bracket` so that `end` inside bounds parses as
 * `MC_EXPR_END`.
 *
 * Return: Newly allocated `MC_EXPR_SLICE` node
 */
static mc_expr_t *parse_slice_expr(mc_parse_ctx_t *ctx, const mc_tok_t *id_tok)
{
    mc_expr_t *e = alloc_expr(MC_EXPR_SLICE, id_tok);
    snprintf(e->u.slice.name, sizeof(e->u.slice.name), "%s", id_tok->sval);
    advance_tok(ctx);
    ctx->in_slice_bracket++;

    int nax = 0;
    while (!ctx->err_flag && cur_tok(ctx)->type != MC_TOK_RBRACK &&
           cur_tok(ctx)->type != MC_TOK_EOF)
    {
        if (nax >= MC_MAX_SLICE_AX)
        {
            parse_err(ctx, cur_tok(ctx), "Slice supports at most 3 axes");
            break;
        }
        if (parse_one_slice_axis(ctx, &e->u.slice.axes[nax]) != 0)
        {
            break;
        }
        nax++;
        if (cur_tok(ctx)->type != MC_TOK_COMMA)
        {
            break;
        }
        advance_tok(ctx);
    }
    e->u.slice.naxis = nax;
    ctx->in_slice_bracket--;

    if (!ctx->err_flag && (nax == 0 || cur_tok(ctx)->type != MC_TOK_RBRACK))
    {
        parse_err(ctx, cur_tok(ctx), "Malformed slice: missing closing ']'");
    }
    else if (!ctx->err_flag)
    {
        advance_tok(ctx);
    }
    return e;
}

/**
 * parse_bracket_primary - Parse a standalone `[...]` primary expression
 * @ctx:        Active parser context (positioned immediately after `[`)
 * @lbrack_tok: Opening `[` token for source location reporting
 *
 * Parses the first element expression, then inspects the following token:
 * - If followed by `for`, parses an N-D array comprehension
 *   `[body for var = start:stop[:step], ...]` (`MC_EXPR_COMP`).
 * - Otherwise, parses a comma-separated 1-D vector literal `[e0, e1, ...]`
 *   (`MC_EXPR_VEC_LIT`).
 *
 * Return: Newly allocated `MC_EXPR_COMP` or `MC_EXPR_VEC_LIT` node
 */
static mc_expr_t *parse_bracket_primary(mc_parse_ctx_t *ctx, const mc_tok_t *lbrack_tok)
{
    if (cur_tok(ctx)->type == MC_TOK_RBRACK)
    {
        parse_err(ctx, lbrack_tok, "Empty '[ ]' literal");
        return alloc_expr(MC_EXPR_LONG, lbrack_tok);
    }
    mc_expr_t *first = parse_expr(ctx, 0);
    if (ctx->err_flag)
    {
        return first;
    }

    /* Array comprehension: [body for x = start:stop[:step], ...] */
    if (cur_tok(ctx)->type == MC_TOK_IDENT && strcmp(cur_tok(ctx)->sval, "for") == 0)
    {
        advance_tok(ctx);
        mc_expr_t *comp   = alloc_expr(MC_EXPR_COMP, lbrack_tok);
        comp->u.comp.body = first;
        int ngens         = 0;
        while (!ctx->err_flag)
        {
            if (ngens >= MC_MAX_SLICE_AX)
            {
                parse_err(ctx, cur_tok(ctx), "Comprehension supports at most 3 axes");
                break;
            }
            const mc_tok_t *vtok = cur_tok(ctx);
            if (vtok->type != MC_TOK_IDENT)
            {
                parse_err(ctx, vtok, "Expected loop variable name after 'for'");
                break;
            }
            snprintf(comp->u.comp.gens[ngens].var, sizeof(comp->u.comp.gens[ngens].var), "%s",
                     vtok->sval);
            advance_tok(ctx);
            expect_tok(ctx, MC_TOK_ASSIGN, "Expected '=' after comprehension variable");
            comp->u.comp.gens[ngens].start = parse_expr(ctx, 0);
            expect_tok(ctx, MC_TOK_COLON, "Expected ':' in comprehension range");
            comp->u.comp.gens[ngens].stop = parse_expr(ctx, 0);
            if (!ctx->err_flag && cur_tok(ctx)->type == MC_TOK_COLON)
            {
                advance_tok(ctx);
                comp->u.comp.gens[ngens].step = parse_expr(ctx, 0);
            }
            comp->u.comp.ngens = ++ngens;
            if (cur_tok(ctx)->type != MC_TOK_COMMA)
            {
                break;
            }
            advance_tok(ctx);
        }
        expect_tok(ctx, MC_TOK_RBRACK, "Expected ']' at end of comprehension");
        return comp;
    }

    /* 1-D vector literal: [e0, e1, ...] */
    mc_expr_t *vec          = alloc_expr(MC_EXPR_VEC_LIT, lbrack_tok);
    vec->u.vec_lit.elems[0] = first;
    vec->u.vec_lit.nelem    = 1;
    while (!ctx->err_flag && cur_tok(ctx)->type == MC_TOK_COMMA)
    {
        advance_tok(ctx);
        if (vec->u.vec_lit.nelem >= MC_MAX_ARGS)
        {
            parse_err(ctx, cur_tok(ctx), "Too many elements in vector literal");
            break;
        }
        vec->u.vec_lit.elems[vec->u.vec_lit.nelem++] = parse_expr(ctx, 0);
    }
    expect_tok(ctx, MC_TOK_RBRACK, "Expected ']' at end of vector literal");
    return vec;
}

/* ========================================================================= */
/* Primary and Precedence-Climbing Expression Parser                         */
/* ========================================================================= */

/**
 * parse_primary - Parse a primary or prefix-unary expression, plus postfix ops
 * @ctx: Active parser context
 *
 * Handles literals, prefix unary operators (`-`, `~`, `!`), parenthesized
 * sub-expressions, bracketed vector literals / comprehensions, identifiers,
 * function calls (`fn(...)`), slices (`im[...]`), postfix `.T` transposes, and
 * assignment expressions (`target op= rhs`) when the primary is a valid LHS
 * (`MC_EXPR_IDENT` or `MC_EXPR_SLICE`).
 *
 * Return: Newly allocated expression AST node
 */
static mc_expr_t *parse_primary(mc_parse_ctx_t *ctx)
{
    if (ctx->err_flag)
    {
        return alloc_expr(MC_EXPR_LONG, cur_tok(ctx));
    }

    const mc_tok_t *t    = cur_tok(ctx);
    mc_expr_t      *node = NULL;

    if (t->type == MC_TOK_LONG)
    {
        advance_tok(ctx);
        node         = alloc_expr(MC_EXPR_LONG, t);
        node->u.lval = t->lval;
    }
    else if (t->type == MC_TOK_DOUBLE)
    {
        advance_tok(ctx);
        node         = alloc_expr(MC_EXPR_DOUBLE, t);
        node->u.dval = t->dval;
    }
    else if (t->type == MC_TOK_STRING)
    {
        advance_tok(ctx);
        node = alloc_expr(MC_EXPR_STRING, t);
        snprintf(node->u.sval, sizeof(node->u.sval), "%s", t->sval);
    }
    else if (t->type == MC_TOK_MINUS || t->type == MC_TOK_BNOT || t->type == MC_TOK_NOT)
    {
        advance_tok(ctx);
        node                  = alloc_expr(MC_EXPR_UNARY, t);
        node->u.unary.op      = t->type;
        node->u.unary.operand = parse_primary(ctx);
        return node;
    }
    else if (t->type == MC_TOK_LPAREN)
    {
        advance_tok(ctx);
        node = parse_expr(ctx, 0);
        expect_tok(ctx, MC_TOK_RPAREN, "Expected ')'");
    }
    else if (t->type == MC_TOK_LBRACK)
    {
        advance_tok(ctx);
        node = parse_bracket_primary(ctx, t);
    }
    else if (t->type == MC_TOK_IDENT)
    {
        advance_tok(ctx);
        if (ctx->in_slice_bracket > 0 && strcmp(t->sval, "end") == 0 &&
            cur_tok(ctx)->type != MC_TOK_LPAREN)
        {
            node = alloc_expr(MC_EXPR_END, t);
        }
        else if (cur_tok(ctx)->type == MC_TOK_LPAREN && t->no_space_after)
        {
            /* Function call: ident(arg0, arg1, k=v, ...) */
            advance_tok(ctx);
            node = alloc_expr(MC_EXPR_CALL, t);
            snprintf(node->u.call.name, sizeof(node->u.call.name), "%s", t->sval);
            node->u.call.udef_id = -1;
            node->u.call.ext_id  = -1;
            int nargs            = 0;
            while (cur_tok(ctx)->type != MC_TOK_RPAREN && !ctx->err_flag)
            {
                if (nargs >= MC_MAX_ARGS)
                {
                    parse_err(ctx, cur_tok(ctx), "Too many arguments in call to '%s'", t->sval);
                    break;
                }
                mc_expr_t *arg           = parse_expr(ctx, 0);
                node->u.call.args[nargs] = arg;

                /* Unpack keyword argument k=v parsed as an MC_EXPR_ASSIGN */
                if (arg != NULL && arg->kind == MC_EXPR_ASSIGN &&
                    arg->u.assign.op == MC_TOK_ASSIGN &&
                    arg->u.assign.target->kind == MC_EXPR_IDENT)
                {
                    snprintf(node->u.call.kw_names[nargs], sizeof(node->u.call.kw_names[nargs]),
                             "%s", arg->u.assign.target->u.ident.name);
                    node->u.call.args[nargs] = arg->u.assign.rhs;
                    arg->u.assign.rhs        = NULL;
                    mc_expr_free(arg);
                }
                node->u.call.nargs = ++nargs;
                if (cur_tok(ctx)->type != MC_TOK_COMMA)
                {
                    break;
                }
                advance_tok(ctx);
            }
            expect_tok(ctx, MC_TOK_RPAREN, "Expected ')'");
        }
        else if (cur_tok(ctx)->type == MC_TOK_LBRACK && t->no_space_after)
        {
            node = parse_slice_expr(ctx, t);
            if (!ctx->err_flag && cur_tok(ctx)->type == MC_TOK_LPAREN)
            {
                parse_err(ctx, cur_tok(ctx), "unknown function '%s[...]'", t->sval);
            }
        }
        else
        {
            node = alloc_expr(MC_EXPR_IDENT, t);
            snprintf(node->u.ident.name, sizeof(node->u.ident.name), "%s", t->sval);
        }
    }
    else
    {
        parse_err(ctx, t, "Unexpected token");
        return alloc_expr(MC_EXPR_LONG, t);
    }

    /* Postfix matrix transpose (.T -> transpose(node)) */
    while (!ctx->err_flag && cur_tok(ctx)->type == MC_TOK_DOT_T)
    {
        node = make_call(advance_tok(ctx), "transpose", node, NULL);
    }

    /* Assignment expression when primary is an identifier or slice */
    if (!ctx->err_flag && (node->kind == MC_EXPR_IDENT || node->kind == MC_EXPR_SLICE) &&
        mc_is_assign_tok(cur_tok(ctx)->type))
    {
        const mc_tok_t *op_tok = advance_tok(ctx);
        mc_expr_t      *asgn   = alloc_expr(MC_EXPR_ASSIGN, op_tok);
        asgn->u.assign.op      = op_tok->type;
        asgn->u.assign.target  = node;
        asgn->u.assign.rhs     = parse_expr(ctx, 0);
        return asgn;
    }

    return node;
}

/**
 * parse_expr - Precedence-climbing parser for binary, chained, and ternary expressions
 * @ctx:      Active parser context
 * @min_prec: Minimum operator precedence to bind at this recursion level
 *
 * Parses a left-hand primary via `parse_primary()`, then consumes infix
 * operators with precedence `>= @min_prec`. Handles right-associative ternary
 * `? :`, desugars `@` into `matmul(left, right)`, and collapses adjacent
 * relational operators (`<`, `<=`, `>`, `>=`) into a single `MC_EXPR_CHAIN`
 * node (`a <= x <= b`).
 *
 * Return: Root `mc_expr_t` node of the parsed sub-expression
 */
static mc_expr_t *parse_expr(mc_parse_ctx_t *ctx, int min_prec)
{
    mc_expr_t *left = parse_primary(ctx);
    while (!ctx->err_flag)
    {
        const mc_tok_t *op_tok = cur_tok(ctx);
        mc_tok_type_t   op     = op_tok->type;
        int             prec   = mc_op_prec(op);
        if (prec < min_prec || mc_is_assign_tok(op))
        {
            break;
        }
        advance_tok(ctx);

        if (op == MC_TOK_QUEST)
        {
            mc_expr_t *t_expr = parse_expr(ctx, 0);
            expect_tok(ctx, MC_TOK_COLON, "Expected ':' in ternary");
            mc_expr_t *f_expr      = parse_expr(ctx, 0);
            mc_expr_t *tern        = alloc_expr(MC_EXPR_TERNARY, op_tok);
            tern->u.ternary.cond   = left;
            tern->u.ternary.t_expr = t_expr;
            tern->u.ternary.f_expr = f_expr;
            left                   = tern;
            continue;
        }

        int        next_min = prec + 1;
        mc_expr_t *right    = parse_expr(ctx, next_min);
        if (ctx->err_flag)
        {
            mc_expr_free(right);
            return left;
        }
        if (op == MC_TOK_MATMUL)
        {
            left = make_call(op_tok, "matmul", left, right);
            continue;
        }

        /* Chained relational comparison: a < b < c */
        if (mc_is_rel_tok(op) && mc_is_rel_tok(cur_tok(ctx)->type))
        {
            mc_expr_t *ch           = alloc_expr(MC_EXPR_CHAIN, op_tok);
            ch->u.chain.ops[0]      = op;
            ch->u.chain.operands[0] = left;
            ch->u.chain.operands[1] = right;
            ch->u.chain.nops        = 1;
            while (!ctx->err_flag && mc_is_rel_tok(cur_tok(ctx)->type))
            {
                if (ch->u.chain.nops >= MC_MAX_CHAIN)
                {
                    parse_err(ctx, cur_tok(ctx), "Chained comparison too long");
                    break;
                }
                mc_tok_type_t cop             = advance_tok(ctx)->type;
                mc_expr_t    *nx              = parse_expr(ctx, next_min);
                int           idx             = ch->u.chain.nops++;
                ch->u.chain.ops[idx]          = cop;
                ch->u.chain.operands[idx + 1] = nx;
            }
            left = ch;
            continue;
        }

        mc_expr_t *bin      = alloc_expr(MC_EXPR_BINARY, op_tok);
        bin->u.binary.op    = op;
        bin->u.binary.left  = left;
        bin->u.binary.right = right;
        left                = bin;
    }
    return left;
}

/* ========================================================================= */
/* Statement and Control-Flow Block Parser                                   */
/* ========================================================================= */

/** skip_stmt_seps() - Advance past any consecutive newline or `;` tokens. */
static void skip_stmt_seps(mc_parse_ctx_t *ctx)
{
    while (cur_tok(ctx)->type == MC_TOK_NEWLINE || cur_tok(ctx)->type == MC_TOK_SEMI)
    {
        advance_tok(ctx);
    }
}

/**
 * is_block_end_kw - Check whether token @t terminates an unbraced control block
 * @t: Token to inspect
 *
 * Return: 1 if @t is the identifier `end`, `elif`, or `else`, 0 otherwise
 */
static int is_block_end_kw(const mc_tok_t *t)
{
    return t->type == MC_TOK_IDENT &&
           (strcmp(t->sval, "end") == 0 || strcmp(t->sval, "elif") == 0 ||
            strcmp(t->sval, "else") == 0);
}

/**
 * is_multi_lhs_start - Look ahead to check for a multi-target assignment `a, b = ...`
 * @ctx: Active parser context
 *
 * Scans ahead without mutating `ctx->tl->pos` to verify a comma-separated list
 * of 2 to `MC_MAX_LHS` identifiers followed immediately by `=`.
 *
 * Return: 1 if positioned at a multi-LHS assignment, 0 otherwise
 */
static int is_multi_lhs_start(const mc_parse_ctx_t *ctx)
{
    int p = ctx->tl->pos;
    if (ctx->tl->toks[p].type != MC_TOK_IDENT || ctx->tl->toks[p + 1].type != MC_TOK_COMMA)
    {
        return 0;
    }
    int n = 0;
    while (n < MC_MAX_LHS && ctx->tl->toks[p].type == MC_TOK_IDENT)
    {
        n++;
        p++;
        if (ctx->tl->toks[p].type != MC_TOK_COMMA)
        {
            break;
        }
        p++;
    }
    return (n >= 2 && ctx->tl->toks[p].type == MC_TOK_ASSIGN);
}

static mc_stmt_t *parse_stmt_one(mc_parse_ctx_t *ctx);

/**
 * parse_block_body - Parse a sequence of statements inside a block or program
 * @ctx:    Active parser context
 * @braced: 1 if the block is delimited by `{ ... }`, 0 if keyword/EOF-delimited
 *
 * Return: Head of the linked list of parsed `mc_stmt_t` nodes
 */
static mc_stmt_t *parse_block_body(mc_parse_ctx_t *ctx, int braced)
{
    if (braced)
    {
        expect_tok(ctx, MC_TOK_LBRACE, "expected '{'");
    }
    mc_stmt_t  head;
    mc_stmt_t *tail = &head;
    memset(&head, 0, sizeof(head));
    while (!ctx->err_flag)
    {
        skip_stmt_seps(ctx);
        const mc_tok_t *t = cur_tok(ctx);
        if (t->type == MC_TOK_EOF || (braced ? (t->type == MC_TOK_RBRACE) : is_block_end_kw(t)))
        {
            break;
        }
        mc_stmt_t *s = parse_stmt_one(ctx);
        if (s == NULL)
        {
            break;
        }
        tail->next = s;
        tail       = s;
    }
    if (braced)
    {
        expect_tok(ctx, MC_TOK_RBRACE, "unclosed '{'");
    }
    return head.next;
}

/**
 * parse_if_stmt - Parse an `if` / `elif` / `else` conditional statement
 * @ctx:    Active parser context (positioned at `if`)
 * @if_tok: The `if` token for line/column tracking
 *
 * Supports three forms:
 * 1. Single-line `if <cond>: <stmt>`
 * 2. Braced `if <cond> { ... } [elif <cond> { ... }]* [else { ... }]`
 * 3. Keyword-terminated `if <cond> ... [elif <cond> ...]* [else ...] end`
 *
 * Return: Newly allocated `MC_STMT_IF` statement node
 */
static mc_stmt_t *parse_if_stmt(mc_parse_ctx_t *ctx, const mc_tok_t *if_tok)
{
    advance_tok(ctx);
    mc_stmt_t *s = (mc_stmt_t *) calloc(1, sizeof(mc_stmt_t));
    s->kind      = MC_STMT_IF;
    s->line      = if_tok->line;
    s->col       = if_tok->col;

    mc_expr_t *cond = parse_expr(ctx, 0);
    if (ctx->err_flag)
    {
        mc_expr_free(cond);
        return s;
    }

    /* Single-statement form: if <cond>: <stmt> */
    if (cur_tok(ctx)->type == MC_TOK_COLON)
    {
        advance_tok(ctx);
        mc_tok_type_t nt = cur_tok(ctx)->type;
        if (nt == MC_TOK_EOF || nt == MC_TOK_SEMI || nt == MC_TOK_NEWLINE || nt == MC_TOK_RBRACE)
        {
            parse_err(ctx, cur_tok(ctx), "empty statement after 'if ...:'");
            mc_expr_free(cond);
            return s;
        }
        s->u.if_stmt.branches[0].cond = cond;
        s->u.if_stmt.branches[0].body = parse_stmt_one(ctx);
        s->u.if_stmt.nbranches        = 1;
        return s;
    }

    /* Multi-statement block form (braced or 'end'-terminated) */
    mc_tok_type_t bt = cur_tok(ctx)->type;
    if (bt != MC_TOK_LBRACE && bt != MC_TOK_NEWLINE && bt != MC_TOK_SEMI)
    {
        parse_err(ctx, cur_tok(ctx), "if without ':'");
        mc_expr_free(cond);
        return s;
    }
    int braced                    = (bt == MC_TOK_LBRACE);
    s->u.if_stmt.branches[0].cond = cond;
    s->u.if_stmt.branches[0].body = parse_block_body(ctx, braced);
    s->u.if_stmt.nbranches        = 1;

    while (!ctx->err_flag)
    {
        while (braced && cur_tok(ctx)->type == MC_TOK_NEWLINE)
        {
            advance_tok(ctx);
        }
        if (cur_tok(ctx)->type != MC_TOK_IDENT || strcmp(cur_tok(ctx)->sval, "elif") != 0)
        {
            break;
        }
        advance_tok(ctx);
        if (s->u.if_stmt.nbranches >= 15)
        {
            parse_err(ctx, cur_tok(ctx), "Too many 'elif' branches");
            break;
        }
        int b                         = s->u.if_stmt.nbranches++;
        s->u.if_stmt.branches[b].cond = parse_expr(ctx, 0);
        s->u.if_stmt.branches[b].body = parse_block_body(ctx, braced);
    }

    while (!ctx->err_flag && braced && cur_tok(ctx)->type == MC_TOK_NEWLINE)
    {
        advance_tok(ctx);
    }
    if (!ctx->err_flag && cur_tok(ctx)->type == MC_TOK_IDENT &&
        strcmp(cur_tok(ctx)->sval, "else") == 0)
    {
        advance_tok(ctx);
        int b                         = s->u.if_stmt.nbranches++;
        s->u.if_stmt.branches[b].cond = NULL;
        s->u.if_stmt.branches[b].body = parse_block_body(ctx, braced);
    }
    if (!braced && !ctx->err_flag)
    {
        if (cur_tok(ctx)->type != MC_TOK_IDENT || strcmp(cur_tok(ctx)->sval, "end") != 0)
        {
            parse_err(ctx, cur_tok(ctx), "if without end");
        }
        else
        {
            advance_tok(ctx);
        }
    }
    return s;
}

/**
 * parse_for_stmt - Parse a `for <var> = <start>:<stop>[:<step>]` loop statement
 * @ctx:     Active parser context (positioned at `for`)
 * @for_tok: The `for` token for line/column tracking
 *
 * Supports both `{ ... }` braced bodies and `... end` keyword-terminated
 * bodies.
 *
 * Return: Newly allocated `MC_STMT_FOR` statement node
 */
static mc_stmt_t *parse_for_stmt(mc_parse_ctx_t *ctx, const mc_tok_t *for_tok)
{
    advance_tok(ctx);
    mc_stmt_t *s = (mc_stmt_t *) calloc(1, sizeof(mc_stmt_t));
    s->kind      = MC_STMT_FOR;
    s->line      = for_tok->line;
    s->col       = for_tok->col;

    const mc_tok_t *vtok = cur_tok(ctx);
    if (vtok->type != MC_TOK_IDENT)
    {
        parse_err(ctx, vtok, "for syntax: for <var> = <start>:<end>[:<step>]");
        return s;
    }
    snprintf(s->u.for_stmt.var, sizeof(s->u.for_stmt.var), "%s", vtok->sval);
    advance_tok(ctx);
    expect_tok(ctx, MC_TOK_ASSIGN, "for syntax: for <var> = <start>:<end>[:<step>]");
    s->u.for_stmt.start = parse_expr(ctx, 0);
    expect_tok(ctx, MC_TOK_COLON, "for syntax: for <var> = <start>:<end>[:<step>]");
    s->u.for_stmt.stop = parse_expr(ctx, 0);
    if (!ctx->err_flag && cur_tok(ctx)->type == MC_TOK_COLON)
    {
        advance_tok(ctx);
        s->u.for_stmt.step = parse_expr(ctx, 0);
    }

    int braced         = (!ctx->err_flag && cur_tok(ctx)->type == MC_TOK_LBRACE);
    s->u.for_stmt.body = parse_block_body(ctx, braced);
    if (!braced && !ctx->err_flag)
    {
        if (cur_tok(ctx)->type != MC_TOK_IDENT || strcmp(cur_tok(ctx)->sval, "end") != 0)
        {
            parse_err(ctx, cur_tok(ctx), "for without end");
        }
        else
        {
            advance_tok(ctx);
        }
    }
    return s;
}

/**
 * parse_stmt_one - Parse a single statement from the token stream
 * @ctx: Active parser context
 *
 * Dispatches on leading keywords (`skip`, `def`, `if`, `for`, `local`),
 * multi-LHS tuple assignments (`a, b = ...`), or general expression/assignment
 * statements (`MC_STMT_EXPR`). When `local` precedes an assignment, rewrites
 * unqualified LHS target names to carry the `@L:` process-local prefix.
 *
 * Return: Newly allocated `mc_stmt_t` node
 */
static mc_stmt_t *parse_stmt_one(mc_parse_ctx_t *ctx)
{
    skip_stmt_seps(ctx);
    const mc_tok_t *t = cur_tok(ctx);
    if (t->type == MC_TOK_EOF)
    {
        mc_stmt_t *s = (mc_stmt_t *) calloc(1, sizeof(mc_stmt_t));
        s->kind      = MC_STMT_EMPTY;
        return s;
    }

    if (t->type == MC_TOK_IDENT && strcmp(t->sval, "skip") == 0)
    {
        const mc_tok_t *nx = &ctx->tl->toks[ctx->tl->pos + 1];
        if (nx->type == MC_TOK_EOF || nx->type == MC_TOK_NEWLINE || nx->type == MC_TOK_SEMI)
        {
            advance_tok(ctx);
            mc_stmt_t *s = (mc_stmt_t *) calloc(1, sizeof(mc_stmt_t));
            s->kind      = MC_STMT_SKIP;
            s->line      = t->line;
            s->col       = t->col;
            return s;
        }
    }

    if (t->type == MC_TOK_IDENT && strcmp(t->sval, "def") == 0)
    {
        mc_stmt_t *s = (mc_stmt_t *) calloc(1, sizeof(mc_stmt_t));
        s->kind      = MC_STMT_DEF;
        s->line      = t->line;
        s->col       = t->col;
        while (cur_tok(ctx)->type != MC_TOK_EOF && cur_tok(ctx)->type != MC_TOK_NEWLINE &&
               cur_tok(ctx)->type != MC_TOK_SEMI && cur_tok(ctx)->type != MC_TOK_RBRACE)
        {
            advance_tok(ctx);
        }
        return s;
    }

    if (t->type == MC_TOK_IDENT && strcmp(t->sval, "if") == 0)
    {
        const mc_tok_t *nx = &ctx->tl->toks[ctx->tl->pos + 1];
        if (!mc_is_assign_tok(nx->type) && nx->type != MC_TOK_COMMA)
        {
            return parse_if_stmt(ctx, t);
        }
    }

    if (t->type == MC_TOK_IDENT && strcmp(t->sval, "for") == 0 &&
        ctx->tl->toks[ctx->tl->pos + 1].type == MC_TOK_IDENT &&
        ctx->tl->toks[ctx->tl->pos + 2].type == MC_TOK_ASSIGN)
    {
        return parse_for_stmt(ctx, t);
    }

    int is_local = 0;
    if (t->type == MC_TOK_IDENT && strcmp(t->sval, "local") == 0 &&
        ctx->tl->toks[ctx->tl->pos + 1].type == MC_TOK_IDENT)
    {
        advance_tok(ctx);
        is_local = 1;
        t        = cur_tok(ctx);
    }

    /* Multi-target tuple assignment: a, b[, c] = rhs1[, rhs2...] */
    if (is_multi_lhs_start(ctx))
    {
        mc_stmt_t *s = (mc_stmt_t *) calloc(1, sizeof(mc_stmt_t));
        s->kind      = MC_STMT_MULTI_ASSIGN;
        s->line      = t->line;
        s->col       = t->col;
        s->is_local  = is_local;
        int nlhs     = 0;
        while (nlhs < MC_MAX_LHS && cur_tok(ctx)->type == MC_TOK_IDENT)
        {
            const mc_tok_t *lt = advance_tok(ctx);
            mc_expr_t      *le = alloc_expr(MC_EXPR_IDENT, lt);
            snprintf(le->u.ident.name, sizeof(le->u.ident.name),
                     (is_local && lt->sval[0] != '@') ? "@L:%s" : "%s", lt->sval);
            s->u.multi.lhs[nlhs++] = le;
            if (cur_tok(ctx)->type != MC_TOK_COMMA)
            {
                break;
            }
            advance_tok(ctx);
        }
        s->u.multi.nlhs = nlhs;
        advance_tok(ctx);

        int nrhs = 0;
        while (!ctx->err_flag && nrhs < MC_MAX_LHS)
        {
            s->u.multi.rhs[nrhs++] = parse_expr(ctx, 0);
            s->u.multi.nrhs        = nrhs;
            if (cur_tok(ctx)->type != MC_TOK_COMMA)
            {
                break;
            }
            advance_tok(ctx);
        }
        const mc_tok_t *end_t = cur_tok(ctx);
        if (!ctx->err_flag && end_t->type != MC_TOK_EOF && end_t->type != MC_TOK_NEWLINE &&
            end_t->type != MC_TOK_SEMI && end_t->type != MC_TOK_RBRACE && !is_block_end_kw(end_t))
        {
            parse_err(ctx, end_t, "syntax error in multi-target assignment");
        }
        return s;
    }

    mc_stmt_t *s = (mc_stmt_t *) calloc(1, sizeof(mc_stmt_t));
    s->kind      = MC_STMT_EXPR;
    s->line      = t->line;
    s->col       = t->col;
    s->is_local  = is_local;
    s->u.expr    = parse_expr(ctx, 0);

    if (is_local && s->u.expr != NULL && s->u.expr->kind == MC_EXPR_ASSIGN &&
        s->u.expr->u.assign.target->kind == MC_EXPR_IDENT)
    {
        s->u.expr->u.assign.is_local = 1;
        char *nm                     = s->u.expr->u.assign.target->u.ident.name;
        if (nm[0] != '@')
        {
            char tmp[MC_TOK_STR_MAX];
            snprintf(tmp, sizeof(tmp), "@L:%s", nm);
            snprintf(nm, MC_TOK_STR_MAX, "%s", tmp);
        }
    }

    const mc_tok_t *end_t = cur_tok(ctx);
    if (!ctx->err_flag && end_t->type != MC_TOK_EOF && end_t->type != MC_TOK_NEWLINE &&
        end_t->type != MC_TOK_SEMI && end_t->type != MC_TOK_RBRACE && !is_block_end_kw(end_t))
    {
        parse_err(ctx, end_t, "syntax error");
    }
    return s;
}

/* ========================================================================= */
/* Public Parser Entry Points                                                */
/* ========================================================================= */

/**
 * mc_parse_stmt - Parse a single calculator statement from a token list
 * @tl:    Token list produced by `mc_lex()`
 * @err:   Output buffer for syntax error message (may be NULL)
 * @errsz: Size of @err buffer in bytes
 *
 * Verifies that no trailing tokens remain after the statement (apart from
 * trailing `;` or newlines). On failure, frees any partially constructed AST
 * and populates `@tl->err_line`, `@tl->err_col`, and `@err`.
 *
 * Return: Newly allocated `mc_stmt_t` node, or NULL on syntax error
 */
mc_stmt_t *mc_parse_stmt(mc_tok_list_t *tl, char *err, size_t errsz)
{
    mc_parse_ctx_t ctx = { .tl = tl };
    mc_stmt_t     *s   = parse_stmt_one(&ctx);
    skip_stmt_seps(&ctx);
    if (!ctx.err_flag && cur_tok(&ctx)->type != MC_TOK_EOF)
    {
        parse_err(&ctx, cur_tok(&ctx), "syntax error");
    }
    if (ctx.err_flag)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "%s", ctx.errmsg);
        }
        tl->err_line = ctx.err_line;
        tl->err_col  = ctx.err_col;
        mc_stmt_free(s);
        return NULL;
    }
    return s;
}

/**
 * mc_parse_prog - Parse a multi-statement calculator program from a token list
 * @tl:    Token list produced by `mc_lex()`
 * @err:   Output buffer for syntax error message (may be NULL)
 * @errsz: Size of @err buffer in bytes
 *
 * Parses a sequence of statements separated by newlines or `;` up to
 * `MC_TOK_EOF`. On failure, frees the statement list and populates `@err`.
 *
 * Return: Head of linked `mc_stmt_t` list (NULL if empty or on syntax error)
 */
mc_stmt_t *mc_parse_prog(mc_tok_list_t *tl, char *err, size_t errsz)
{
    mc_parse_ctx_t ctx  = { .tl = tl };
    mc_stmt_t     *prog = parse_block_body(&ctx, 0);
    skip_stmt_seps(&ctx);
    if (!ctx.err_flag && cur_tok(&ctx)->type != MC_TOK_EOF)
    {
        parse_err(&ctx, cur_tok(&ctx), "unexpected '%s'", cur_tok(&ctx)->sval);
    }
    if (ctx.err_flag)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "%s", ctx.errmsg);
        }
        tl->err_line = ctx.err_line;
        tl->err_col  = ctx.err_col;
        mc_stmt_free(prog);
        return NULL;
    }
    return prog;
}
