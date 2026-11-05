// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mc_ast.h"

int mc_op_prec(mc_tok_type_t t)
{
    switch (t)
    {
    case MC_TOK_MATMUL:
    case MC_TOK_MUL:
    case MC_TOK_DIV:
    case MC_TOK_MOD:
        return 11;
    case MC_TOK_PLUS:
    case MC_TOK_MINUS:
        return 10;
    case MC_TOK_SHL:
    case MC_TOK_SHR:
        return 9;
    case MC_TOK_LT:
    case MC_TOK_LE:
    case MC_TOK_GT:
    case MC_TOK_GE:
        return 8;
    case MC_TOK_EQ:
    case MC_TOK_NE:
        return 7;
    case MC_TOK_BAND:
        return 6;
    case MC_TOK_BXOR:
        return 5;
    case MC_TOK_BOR:
        return 4;
    case MC_TOK_AND:
        return 3;
    case MC_TOK_OR:
        return 2;
    case MC_TOK_QUEST:
        return 1;
    case MC_TOK_ASSIGN:
    case MC_TOK_ADD_ASSIGN:
    case MC_TOK_SUB_ASSIGN:
    case MC_TOK_MUL_ASSIGN:
    case MC_TOK_DIV_ASSIGN:
        return 0;
    default:
        return -1;
    }
}

size_t mc_scan_number(const char *p, mc_tok_type_t *out_type, long *out_l, double *out_d)
{
    const char *start    = p;
    char       *end      = NULL;
    int         is_float = 0;

    if (!(isdigit((unsigned char) *p) || (*p == '.' && isdigit((unsigned char) p[1]))))
    {
        return 0;
    }
    if (*p == '0' && p[1] != '\0' && !isdigit((unsigned char) p[1]) && p[1] != '.')
    {
        int base = (p[1] == 'x' || p[1] == 'X')   ? 16
                   : (p[1] == 'o' || p[1] == 'O') ? 8
                   : (p[1] == 'b' || p[1] == 'B') ? 2
                                                  : 0;
        if (base != 0)
        {
            *out_type = MC_TOK_LONG;
            *out_l    = strtol((base == 16) ? p : p + 2, &end, base);
            *out_d    = 0.0;
            return (size_t) (end - start);
        }
    }
    while (isdigit((unsigned char) *p))
    {
        p++;
    }
    if (*p == '.')
    {
        is_float = 1;
        p++;
        while (isdigit((unsigned char) *p))
        {
            p++;
        }
    }
    if (*p == 'e' || *p == 'E')
    {
        is_float = 1;
        p++;
        p += (*p == '+' || *p == '-');
        while (isdigit((unsigned char) *p))
        {
            p++;
        }
    }
    *out_type = is_float ? MC_TOK_DOUBLE : MC_TOK_LONG;
    if (is_float)
    {
        *out_d = strtod(start, NULL);
        *out_l = 0;
    }
    else
    {
        *out_l = strtol(start, NULL, 10);
        *out_d = 0.0;
    }
    return (size_t) (p - start);
}

static inline int is_ident_start(int c)
{
    return isalpha(c) || c == '_' || c == '?' || c == '.';
}

static inline int is_ident_char(int c)
{
    return isalnum(c) || c == '_' || c == '.' || c == '$';
}

static int is_ns_prefix_only(const char *start, size_t len)
{
    if (len == 1 && (start[0] == 's' || start[0] == 'l' || start[0] == 'f' || start[0] == 'p'))
    {
        return 1;
    }
    if (len == 2 && (strncmp(start, "@S", 2) == 0 || strncmp(start, "@L", 2) == 0 ||
                     strncmp(start, "@F", 2) == 0 || strncmp(start, "S:", 2) == 0 ||
                     strncmp(start, "L:", 2) == 0))
    {
        return 1;
    }
    if (len == 3 && (strncmp(start, "@S:", 3) == 0 || strncmp(start, "@L:", 3) == 0 ||
                     strncmp(start, "@F:", 3) == 0))
    {
        return 1;
    }
    return 0;
}

static size_t scan_ident(const char *start)
{
    const char *p = start;
    if (*p == '@' && (p[1] == 'S' || p[1] == 'L' || p[1] == 'F') && p[2] == ':')
    {
        p += 3;
    }
    else if ((*p == 'S' || *p == 'L') && p[1] == ':')
    {
        p += 2;
    }
    while (is_ident_char((unsigned char) *p))
    {
        if (p[0] == '.' && p[1] == 'T' && !is_ident_char((unsigned char) p[2]) && p > start &&
            !is_ns_prefix_only(start, (size_t) (p - start)))
        {
            break; /* leave .T to be lexed as MC_TOK_DOT_T */
        }
        p++;
    }
    /* Support labeled extension calls like my_op:stage2( */
    if (*p == ':' && (isalpha((unsigned char) p[1]) || p[1] == '_'))
    {
        const char *q = p + 1;
        while (isalnum((unsigned char) *q) || *q == '_')
        {
            q++;
        }
        if (*q == '(')
        {
            p = q;
        }
    }
    return (size_t) (p - start);
}

static mc_tok_type_t scan_op(const char *p, int *len)
{
    static const struct
    {
        char          s[3];
        mc_tok_type_t t;
    } two[]                            = { { "<=", MC_TOK_LE },         { ">=", MC_TOK_GE },
                                           { "==", MC_TOK_EQ },         { "!=", MC_TOK_NE },
                                           { "&&", MC_TOK_AND },        { "||", MC_TOK_OR },
                                           { "+=", MC_TOK_ADD_ASSIGN }, { "-=", MC_TOK_SUB_ASSIGN },
                                           { "*=", MC_TOK_MUL_ASSIGN }, { "/=", MC_TOK_DIV_ASSIGN },
                                           { "<<", MC_TOK_SHL },        { ">>", MC_TOK_SHR },
                                           { "^^", MC_TOK_BXOR } };
    static const char          one_s[] = "+-*/%<>!()[]{}|&~,=?:;";
    static const mc_tok_type_t one_t[] = {
        MC_TOK_PLUS,   MC_TOK_MINUS,  MC_TOK_MUL,    MC_TOK_DIV,    MC_TOK_MOD,    MC_TOK_LT,
        MC_TOK_GT,     MC_TOK_NOT,    MC_TOK_LPAREN, MC_TOK_RPAREN, MC_TOK_LBRACK, MC_TOK_RBRACK,
        MC_TOK_LBRACE, MC_TOK_RBRACE, MC_TOK_BOR,    MC_TOK_BAND,   MC_TOK_BNOT,   MC_TOK_COMMA,
        MC_TOK_ASSIGN, MC_TOK_QUEST,  MC_TOK_COLON,  MC_TOK_SEMI
    };

    for (size_t i = 0; i < sizeof(two) / sizeof(two[0]); i++)
    {
        if (p[0] == two[i].s[0] && p[1] == two[i].s[1])
        {
            *len = 2;
            return two[i].t;
        }
    }
    const char *c = (*p != '\0') ? strchr(one_s, *p) : NULL;
    if (c != NULL)
    {
        *len = 1;
        return one_t[c - one_s];
    }
    return MC_TOK_EOF;
}

int mc_lex(const char *src, mc_tok_list_t *out)
{
    memset(out, 0, sizeof(*out));
    if (src == NULL)
    {
        return 0;
    }

    const char *p    = src;
    int         line = 1;
    int         col  = 1;
    int         nt   = 0;

    while (*p != '\0' && nt < MC_MAX_TOKENS - 1)
    {
        while (*p == ' ' || *p == '\t' || *p == '\r')
        {
            p++;
            col++;
        }
        if (*p == '#')
        {
            while (*p != '\0' && *p != '\n')
            {
                p++;
                col++;
            }
        }
        if (*p == '\0')
        {
            break;
        }
        if (*p == '\n')
        {
            mc_tok_t *t = &out->toks[nt++];
            t->type     = MC_TOK_NEWLINE;
            t->line     = line;
            t->col      = col;
            p++;
            line++;
            col = 1;
            continue;
        }

        /* Postfix .T after identifier, ')', ']', or another .T */
        if (p[0] == '.' && p[1] == 'T' && !is_ident_char((unsigned char) p[2]) && nt > 0 &&
            (out->toks[nt - 1].type == MC_TOK_IDENT || out->toks[nt - 1].type == MC_TOK_RPAREN ||
             out->toks[nt - 1].type == MC_TOK_RBRACK || out->toks[nt - 1].type == MC_TOK_DOT_T))
        {
            mc_tok_t *t = &out->toks[nt++];
            t->type     = MC_TOK_DOT_T;
            t->line     = line;
            t->col      = col;
            strcpy(t->sval, ".T");
            p += 2;
            col += 2;
            continue;
        }

        /* @S:, @L:, @F: prefix vs @ matrix multiply operator */
        if (*p == '@')
        {
            if (!((p[1] == 'S' || p[1] == 'L' || p[1] == 'F') && p[2] == ':') &&
                !is_ident_start((unsigned char) p[1]))
            {
                mc_tok_t *t = &out->toks[nt++];
                t->type     = MC_TOK_MATMUL;
                t->line     = line;
                t->col      = col;
                strcpy(t->sval, "@");
                p++;
                col++;
                continue;
            }
            if (!((p[1] == 'S' || p[1] == 'L' || p[1] == 'F') && p[2] == ':'))
            {
                /* Infix A@B without space when preceded by an operand */
                if (nt > 0 && (out->toks[nt - 1].type == MC_TOK_IDENT ||
                               out->toks[nt - 1].type == MC_TOK_RPAREN ||
                               out->toks[nt - 1].type == MC_TOK_RBRACK ||
                               out->toks[nt - 1].type == MC_TOK_DOT_T))
                {
                    mc_tok_t *t = &out->toks[nt++];
                    t->type     = MC_TOK_MATMUL;
                    t->line     = line;
                    t->col      = col;
                    strcpy(t->sval, "@");
                    p++;
                    col++;
                    continue;
                }
                p++;
                col++;
            }
        }

        int           oplen  = 0;
        mc_tok_type_t optype = scan_op(p, &oplen);
        if (optype != MC_TOK_EOF)
        {
            mc_tok_t *t = &out->toks[nt++];
            t->type     = optype;
            t->line     = line;
            t->col      = col;
            memcpy(t->sval, p, (size_t) oplen);
            t->sval[oplen]    = '\0';
            t->no_space_after = (p[oplen] != '\0' && !isspace((unsigned char) p[oplen]));
            p += oplen;
            col += oplen;
            continue;
        }

        if (*p == '^')
        {
            out->err_line = line;
            out->err_col  = col;
            snprintf(out->errmsg, sizeof(out->errmsg),
                     "'^' is not an operator: write pow(a, b) (^^ is xor)");
            return -1;
        }

        mc_tok_type_t ntype = MC_TOK_EOF;
        long          lval  = 0;
        double        dval  = 0.0;
        size_t        nlen  = mc_scan_number(p, &ntype, &lval, &dval);
        if (nlen > 0)
        {
            mc_tok_t *t = &out->toks[nt++];
            t->type     = ntype;
            t->line     = line;
            t->col      = col;
            t->lval     = lval;
            t->dval     = dval;
            size_t clen = (nlen < MC_TOK_STR_MAX) ? nlen : (MC_TOK_STR_MAX - 1);
            memcpy(t->sval, p, clen);
            t->sval[clen]     = '\0';
            t->no_space_after = (p[nlen] != '\0' && !isspace((unsigned char) p[nlen]));
            p += nlen;
            col += (int) nlen;
            continue;
        }

        if (*p == '"')
        {
            mc_tok_t *t = &out->toks[nt++];
            t->type     = MC_TOK_STRING;
            t->line     = line;
            t->col      = col;
            p++;
            col++;
            const char *start = p;
            while (*p != '"' && *p != '\0' && *p != '\n')
            {
                p++;
                col++;
            }
            size_t slen = (size_t) (p - start);
            if (slen >= MC_TOK_STR_MAX)
            {
                slen = MC_TOK_STR_MAX - 1;
            }
            memcpy(t->sval, start, slen);
            t->sval[slen] = '\0';
            if (*p == '"')
            {
                p++;
                col++;
            }
            continue;
        }

        if (*p == '@' || is_ident_start((unsigned char) *p))
        {
            size_t idlen = scan_ident(p);
            if (idlen == 0)
            {
                out->err_line = line;
                out->err_col  = col;
                snprintf(out->errmsg, sizeof(out->errmsg), "unexpected character '%c'", *p);
                return -1;
            }
            mc_tok_t *t = &out->toks[nt++];
            t->type     = MC_TOK_IDENT;
            t->line     = line;
            t->col      = col;
            if ((p[0] == 'L' || p[0] == 'S') && p[1] == ':' && idlen + 1 < MC_TOK_STR_MAX)
            {
                t->sval[0] = '@';
                memcpy(t->sval + 1, p, idlen);
                t->sval[idlen + 1] = '\0';
            }
            else
            {
                size_t clen = (idlen < MC_TOK_STR_MAX) ? idlen : (MC_TOK_STR_MAX - 1);
                memcpy(t->sval, p, clen);
                t->sval[clen] = '\0';
            }
            t->no_space_after = (p[idlen] != '\0' && !isspace((unsigned char) p[idlen]));
            p += idlen;
            col += (int) idlen;
            continue;
        }

        out->err_line = line;
        out->err_col  = col;
        snprintf(out->errmsg, sizeof(out->errmsg), "unexpected character '%c'", *p);
        return -1;
    }

    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')
    {
        p++;
    }
    if (*p != '\0')
    {
        out->err_line = line;
        out->err_col  = col;
        snprintf(out->errmsg, sizeof(out->errmsg), "expression too long (max %d tokens)",
                 MC_MAX_TOKENS - 1);
        return -1;
    }

    out->toks[nt].type = MC_TOK_EOF;
    out->toks[nt].line = line;
    out->toks[nt].col  = col;
    out->ntok          = nt;
    return 0;
}

void mc_expr_free(mc_expr_t *e)
{
    if (e == NULL)
    {
        return;
    }
    switch (e->kind)
    {
    case MC_EXPR_UNARY:
        mc_expr_free(e->u.unary.operand);
        break;
    case MC_EXPR_BINARY:
        mc_expr_free(e->u.binary.left);
        mc_expr_free(e->u.binary.right);
        break;
    case MC_EXPR_CHAIN:
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            mc_expr_free(e->u.chain.operands[i]);
        }
        break;
    case MC_EXPR_TERNARY:
        mc_expr_free(e->u.ternary.cond);
        mc_expr_free(e->u.ternary.t_expr);
        mc_expr_free(e->u.ternary.f_expr);
        break;
    case MC_EXPR_CALL:
        for (int i = 0; i < e->u.call.nargs; i++)
        {
            mc_expr_free(e->u.call.args[i]);
        }
        mc_expr_free(e->u.call.udef_body);
        break;
    case MC_EXPR_SLICE:
        for (int i = 0; i < e->u.slice.naxis; i++)
        {
            mc_expr_free(e->u.slice.axes[i].start);
            mc_expr_free(e->u.slice.axes[i].stop);
            mc_expr_free(e->u.slice.axes[i].step);
        }
        break;
    case MC_EXPR_ASSIGN:
        mc_expr_free(e->u.assign.target);
        mc_expr_free(e->u.assign.rhs);
        break;
    case MC_EXPR_VEC_LIT:
        for (int i = 0; i < e->u.vec_lit.nelem; i++)
        {
            mc_expr_free(e->u.vec_lit.elems[i]);
        }
        break;
    case MC_EXPR_COMP:
        mc_expr_free(e->u.comp.body);
        for (int i = 0; i < e->u.comp.ngens; i++)
        {
            mc_expr_free(e->u.comp.gens[i].start);
            mc_expr_free(e->u.comp.gens[i].stop);
            mc_expr_free(e->u.comp.gens[i].step);
        }
        break;
    default:
        break;
    }
    free(e);
}

void mc_stmt_free(mc_stmt_t *s)
{
    while (s != NULL)
    {
        mc_stmt_t *next = s->next;
        switch (s->kind)
        {
        case MC_STMT_EXPR:
            mc_expr_free(s->u.expr);
            break;
        case MC_STMT_MULTI_ASSIGN:
            for (int i = 0; i < s->u.multi.nlhs; i++)
            {
                mc_expr_free(s->u.multi.lhs[i]);
            }
            for (int i = 0; i < s->u.multi.nrhs; i++)
            {
                mc_expr_free(s->u.multi.rhs[i]);
            }
            break;
        case MC_STMT_IF:
            for (int i = 0; i < s->u.if_stmt.nbranches; i++)
            {
                mc_expr_free(s->u.if_stmt.branches[i].cond);
                mc_stmt_free(s->u.if_stmt.branches[i].body);
            }
            break;
        case MC_STMT_FOR:
            mc_expr_free(s->u.for_stmt.start);
            mc_expr_free(s->u.for_stmt.stop);
            mc_expr_free(s->u.for_stmt.step);
            mc_stmt_free(s->u.for_stmt.body);
            break;
        default:
            break;
        }
        free(s);
        s = next;
    }
}
