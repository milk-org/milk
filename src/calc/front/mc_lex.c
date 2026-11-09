// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_lex.c
 * @brief Lexical analyzer, and the AST free, copy and walk functions, for libmilkcalc.
 *
 * Purpose:     turn calculator text, one expression or a whole program, into tokens that keep
 *              their line, column and byte span, so errors point into the text and a statement
 *              can be printed back.
 * Does:        mc_lex() fills a growable token list (the token texts in one buffer, reused at
 *              the next call): `#` comments are dropped, newlines kept as statement
 *              separators, `S:name` and `L:name` written `@S:name` and `@L:name`. `@` is a
 *              stream prefix (`@S:` `@L:` `@F:`) or, after an operand, the matrix product;
 *              `.T` after a name, `)` or `]` is the transpose, but `s.T` is the stream T;
 *              no_space_after tells `f(x)` and `im[0]` from `f (x)` and `[...]`.
 *              mc_scan_number() reads 0x/0o/0b, decimal and floating literals (an integer
 *              beyond 64 bits is a double), mc_scan_ident() reads names and entity paths (a
 *              stream name in `s.my-cam.cnt0` may hold `-`), mc_op_prec() gives the operator
 *              precedence. mc_expr_free(), mc_stmt_free(), mc_expr_clone() and
 *              mc_expr_child_slots() free, copy and walk AST nodes.
 * Used by:     front/mc_parse.c (the tokens); front/mc_resolve.c and plan/mc_history.c (copy,
 *              free, walk); mc_eval.c and run/mc_prog.c (lex, then parse); ext/mc_ext_loader.c
 *              (mcalc_ext_scan_source()); in every shell, mcalc_local() (run/mc_scalar_ops.c)
 *              and `@waitfor` (milk_client_stream.c); the server's milk_server_vars.c (calc
 *              mode). Declared in front/mc_ast.h.
 * Uses:        milk_entity_ns.h (entity path prefixes), ImageStreamIO/ImageStruct.h
 *              (STRINGMAXLEN_IMAGE_NAME), mcalc_sub_prog_free() (run/mc_prog.c).
 * See also:    front/mc_parse.c (the next stage); docs/guide/calculator.md, "Operators".
 * Constraints: also compiled into libmilkbash.so (the thin client every shell loads), which
 *              uses only mc_lex(), the scanners and mc_op_prec(); the AST functions are left
 *              out there (MILK_BASH_CLIENT), as they call mcalc_sub_prog_free().
 */

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStruct.h"
#include "front/mc_ast.h"
#include "milk_entity_ns.h"

/* ========================================================================= */
/* Operator Precedence and Numeric Literal Scanning                          */
/* ========================================================================= */

/**
 * mc_op_prec - Return binding precedence for a binary, ternary, or assignment token
 * @t: Token type to query
 *
 * Higher values bind more tightly. Precedence levels follow C rules from
 * multiplicative (`*`, `/`, `%`, `@` at 11) down to logical OR (`||` at 2),
 * ternary conditional (`?` at 1), and assignment operators (`=`, `+=`, etc.
 * at 0). Non-infix tokens return -1.
 *
 * Return: Precedence level in [0, 11], or -1 if @t is not an infix operator
 */
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

/**
 * mc_scan_number - Scan an integer or floating-point literal at the start of a string
 * @p:        Input pointer positioned at the candidate literal
 * @out_type: Output token type (`MC_TOK_LONG` or `MC_TOK_DOUBLE`)
 * @out_l:    Output integer value when `@out_type == MC_TOK_LONG`
 * @out_d:    Output double value when `@out_type == MC_TOK_DOUBLE`
 *
 * Recognizes hexadecimal (`0x`), octal (`0o`), and binary (`0b`) integer
 * literals as `MC_TOK_LONG`, standard decimal integers as `MC_TOK_LONG`, and
 * numbers containing a decimal point or exponent (`e`/`E`) as `MC_TOK_DOUBLE`.
 * An integer too big for 64 bits is a `MC_TOK_DOUBLE` of its value.
 *
 * Return: Number of bytes consumed from @p, or 0 if @p does not start a number
 */
size_t mc_scan_number(const char *p, mc_tok_type_t *out_type, long *out_l, double *out_d)
{
    const char *start    = p;
    char       *end      = NULL;
    int         is_float = 0;

    if (!(isdigit((unsigned char) *p) || (*p == '.' && isdigit((unsigned char) p[1]))))
    {
        return 0;
    }

    /* Check for 0x (hex), 0o (octal), or 0b (binary) integer radix prefixes */
    if (*p == '0' && p[1] != '\0' && !isdigit((unsigned char) p[1]) && p[1] != '.')
    {
        int base = 0;
        if (p[1] == 'x' || p[1] == 'X')
        {
            base = 16;
        }
        else if (p[1] == 'o' || p[1] == 'O')
        {
            base = 8;
        }
        else if (p[1] == 'b' || p[1] == 'B')
        {
            base = 2;
        }

        if (base != 0)
        {
            errno     = 0;
            *out_type = MC_TOK_LONG;
            *out_l    = strtol((base == 16) ? p : p + 2, &end, base);
            *out_d    = 0.0;
            if (errno == ERANGE)
            {
                *out_type = MC_TOK_DOUBLE;
                *out_l    = 0;
                for (const char *q = p + 2; q < end; q++)
                {
                    *out_d = *out_d * base + (isdigit((unsigned char) *q)
                                                  ? (*q - '0')
                                                  : (tolower((unsigned char) *q) - 'a' + 10));
                }
            }
            return (size_t) (end - start);
        }
    }

    /* Scan integer significand digits */
    while (isdigit((unsigned char) *p))
    {
        p++;
    }

    /* Optional fractional part */
    if (*p == '.')
    {
        is_float = 1;
        p++;
        while (isdigit((unsigned char) *p))
        {
            p++;
        }
    }

    /* Optional scientific notation exponent */
    if (*p == 'e' || *p == 'E')
    {
        is_float = 1;
        p++;
        if (*p == '+' || *p == '-')
        {
            p++;
        }
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
        errno  = 0;
        *out_l = strtol(start, NULL, 10);
        *out_d = 0.0;
        if (errno == ERANGE)
        {
            *out_type = MC_TOK_DOUBLE;
            *out_l    = 0;
            *out_d    = strtod(start, NULL);
        }
    }
    return (size_t) (p - start);
}

/* ========================================================================= */
/* Identifier and Operator Scanning Helpers                                  */
/* ========================================================================= */

/** is_ident_start() - Return non-zero if @c can start a calculator identifier. */
static inline int is_ident_start(int c)
{
    return isalpha(c) || c == '_' || c == '?' || c == '.';
}

/** is_ident_char() - Return non-zero if @c can continue a calculator identifier. */
static inline int is_ident_char(int c)
{
    return isalnum(c) || c == '_' || c == '.' || c == '$';
}

/**
 * is_ns_prefix_only - Check if a scanned prefix is solely a namespace qualifier
 * @start: Pointer to start of identifier token
 * @len:   Number of bytes scanned so far before `.T`
 *
 * When an identifier begins with a namespace prefix (`s.`, `l.`, `f.`, `p.`,
 * `@S:`, `@L:`, `@F:`, `S:`, `L:`), a following `T` (e.g. `s.T` or `@S:T`) is
 * the stream name `T`, not the postfix transpose operator `.T`.
 *
 * Return: 1 if @start[0..@len-1] is only a namespace prefix, 0 otherwise
 */
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

/**
 * mc_scan_ident - Scan an identifier, qualified stream name, or labeled extension call
 * @start: Input pointer positioned at the start of the identifier
 *
 * Handles leading `@S:`, `@L:`, `@F:`, `S:`, and `L:` stream location
 * modifiers, dotted property paths (`s.im.cnt0`), and labeled user extension
 * calls (`my_op:label(...)`), while stopping before a postfix `.T` transpose
 * suffix on a complete identifier.
 *
 * Return: Number of bytes in the identifier token
 */
size_t mc_scan_ident(const char *start)
{
    const char *p = start;

    /* Consume optional @S:, @L:, @F:, S:, or L: stream namespace prefix */
    if (*p == '@' && (p[1] == 'S' || p[1] == 'L' || p[1] == 'F') && p[2] == ':')
    {
        p += 3;
    }
    else if ((*p == 'S' || *p == 'L') && p[1] == ':')
    {
        p += 2;
    }

    for (;;)
    {
        while (is_ident_char((unsigned char) *p))
        {
            if (p[0] == '.' && p[1] == 'T' && !is_ident_char((unsigned char) p[2]) && p > start &&
                !is_ns_prefix_only(start, (size_t) (p - start)))
            {
                break; /* leave .T to be lexed as MC_TOK_DOT_T */
            }
            p++;
        }
        /* An entity path's name may hold '-' (s.my-cam.cnt0) when a ".name" segment follows:
         * s.cam.cnt0-1 and s.cam.x-1.5 stay subtractions */
        const char *dot = memchr(start, '.', (size_t) (p - start));
        if (*p != '-' || dot == NULL || !milk_is_entity_ns_prefix(start, (size_t) (dot - start)))
        {
            break;
        }
        const char *q = p + 1;
        while (isalnum((unsigned char) *q) || *q == '_')
        {
            q++;
        }
        if (q == p + 1 || q[0] != '.' || !(isalpha((unsigned char) q[1]) || q[1] == '_'))
        {
            break;
        }
        p = q;
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

/**
 * scan_op - Match a two-character or single-character operator/punctuation token
 * @p:   Input pointer positioned at the candidate operator
 * @len: Output length in bytes of the matched token (1 or 2)
 *
 * Return: Matched `mc_tok_type_t`, or `MC_TOK_EOF` if @p is not an operator
 */
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

/* ========================================================================= */
/* Main Tokenizer                                                            */
/* ========================================================================= */

/* Make room in @out for token @nt and the EOF tokens after it; -1 when out of memory */
static int tok_reserve(mc_tok_list_t *out, int nt)
{
    if (nt + 3 < out->cap)
    {
        return 0;
    }
    int       cap  = (out->cap > 0) ? 2 * out->cap : 64;
    mc_tok_t *toks = realloc(out->toks, (size_t) cap * sizeof(*toks));
    if (toks == NULL)
    {
        return -1;
    }
    out->toks = toks;
    out->cap  = cap;
    return 0;
}

/* Start token @nt of @out, of type @type, whose text begins at @ts in @src */
static mc_tok_t *tok_new(mc_tok_list_t *out,
                         int            nt,
                         mc_tok_type_t  type,
                         const char    *src,
                         const char    *ts,
                         int            line,
                         int            col)
{
    mc_tok_t *t       = &out->toks[nt];
    t->type           = type;
    t->line           = line;
    t->col            = col;
    t->off            = (int) (ts - src);
    t->len            = 0;
    t->no_space_after = 0;
    t->lval           = 0;
    t->dval           = 0.0;
    t->sval           = NULL;
    return t;
}

/* Set @t's text to @pfx (or nothing) and @n bytes of @s, in @out's string buffer at *@used */
static void tok_text(mc_tok_list_t *out,
                     size_t        *used,
                     mc_tok_t      *t,
                     const char    *pfx,
                     const char    *s,
                     size_t         n)
{
    char  *w  = out->strs + *used;
    size_t pl = (pfx != NULL) ? strlen(pfx) : 0;
    if (pl > 0)
    {
        memcpy(w, pfx, pl);
    }
    if (n > 0)
    {
        memcpy(w + pl, s, n);
    }
    w[pl + n] = '\0';
    t->sval   = w;
    *used += pl + n + 1;
}

/**
 * mc_lex - Tokenize calculator source text into a flat token array
 * @src: Null-terminated source expression or multi-line program (may be NULL)
 * @out: Output token list populated with tokens and terminated by `MC_TOK_EOF`
 *
 * Strips `#` line comments and horizontal whitespace, preserves newlines as
 * `MC_TOK_NEWLINE` statement separators, normalizes bare `S:name` / `L:name`
 * prefixes to `@S:name` / `@L:name`, and records line/column coordinates and the
 * byte span in @src of every token. The token array grows as needed (up to
 * `MC_MAX_TOKENS`); its token texts live in one buffer sized from @src.
 *
 * Return: 0 on success, -1 on lexical error (with `@out->errmsg` populated)
 */
int mc_lex(const char *src, mc_tok_list_t *out)
{
    out->ntok      = 0;
    out->pos       = 0;
    out->err_line  = 0;
    out->err_col   = 0;
    out->errmsg[0] = '\0';

    /* each token text is at most its source bytes plus an '@' and a NUL */
    size_t need = 3 * ((src != NULL) ? strlen(src) : 0) + 16;
    if (need > out->strs_cap)
    {
        char *strs = realloc(out->strs, need);
        if (strs == NULL)
        {
            snprintf(out->errmsg, sizeof(out->errmsg), "out of memory");
            return -1;
        }
        out->strs     = strs;
        out->strs_cap = need;
    }
    if (tok_reserve(out, 0) != 0)
    {
        snprintf(out->errmsg, sizeof(out->errmsg), "out of memory");
        return -1;
    }

    const char *p    = (src != NULL) ? src : "";
    const char *base = p;
    size_t      used = 0;
    int         line = 1;
    int         col  = 1;
    int         nt   = 0;

    while (*p != '\0')
    {
        /* Skip horizontal whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\r')
        {
            p++;
            col++;
        }

        /* Skip '#' comment to end of line */
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
        if (nt + 1 >= MC_MAX_TOKENS)
        {
            out->err_line = line;
            out->err_col  = col;
            snprintf(out->errmsg, sizeof(out->errmsg), "expression too long (max %d tokens)",
                     MC_MAX_TOKENS - 1);
            return -1;
        }
        if (tok_reserve(out, nt) != 0)
        {
            snprintf(out->errmsg, sizeof(out->errmsg), "out of memory");
            return -1;
        }
        const char *ts = p;

        if (*p == '\n')
        {
            mc_tok_t *t = tok_new(out, nt++, MC_TOK_NEWLINE, base, ts, line, col);
            tok_text(out, &used, t, NULL, "", 0);
            t->len = 1;
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
            mc_tok_t *t = tok_new(out, nt++, MC_TOK_DOT_T, base, ts, line, col);
            tok_text(out, &used, t, NULL, ".T", 2);
            t->len = 2;
            p += 2;
            col += 2;
            continue;
        }

        /* Disambiguate @S:, @L:, @F: prefix vs @ matrix multiply operator */
        if (*p == '@')
        {
            int is_pfx = ((p[1] == 'S' || p[1] == 'L' || p[1] == 'F') && p[2] == ':');
            /* A@B without space, after an operand, is the operator too */
            if (!is_pfx && (!is_ident_start((unsigned char) p[1]) ||
                            (nt > 0 && (out->toks[nt - 1].type == MC_TOK_IDENT ||
                                        out->toks[nt - 1].type == MC_TOK_RPAREN ||
                                        out->toks[nt - 1].type == MC_TOK_RBRACK ||
                                        out->toks[nt - 1].type == MC_TOK_DOT_T))))
            {
                mc_tok_t *t = tok_new(out, nt++, MC_TOK_MATMUL, base, ts, line, col);
                tok_text(out, &used, t, NULL, "@", 1);
                t->len = 1;
                p++;
                col++;
                continue;
            }
            if (!is_pfx)
            {
                p++;
                col++;
            }
        }

        /* Operators and punctuation */
        int           oplen  = 0;
        mc_tok_type_t optype = scan_op(p, &oplen);
        if (optype != MC_TOK_EOF)
        {
            mc_tok_t *t = tok_new(out, nt++, optype, base, ts, line, col);
            tok_text(out, &used, t, NULL, p, (size_t) oplen);
            t->no_space_after = (p[oplen] != '\0' && !isspace((unsigned char) p[oplen]));
            p += oplen;
            col += oplen;
            t->len = (int) (p - ts);
            continue;
        }

        /* Reject single '^' with a clear hint toward pow(a, b) or '^^' */
        if (*p == '^')
        {
            out->err_line = line;
            out->err_col  = col;
            snprintf(out->errmsg, sizeof(out->errmsg),
                     "'^' is not an operator: write pow(a, b) (^^ is xor)");
            return -1;
        }

        /* Numeric literals */
        mc_tok_type_t ntype = MC_TOK_EOF;
        long          lval  = 0;
        double        dval  = 0.0;
        size_t        nlen  = mc_scan_number(p, &ntype, &lval, &dval);
        if (nlen > 0)
        {
            mc_tok_t *t = tok_new(out, nt++, ntype, base, ts, line, col);
            t->lval     = lval;
            t->dval     = dval;
            tok_text(out, &used, t, NULL, p, (nlen < MC_TOK_STR_MAX) ? nlen : (MC_TOK_STR_MAX - 1));
            t->no_space_after = (p[nlen] != '\0' && !isspace((unsigned char) p[nlen]));
            p += nlen;
            col += (int) nlen;
            t->len = (int) (p - ts);
            continue;
        }

        /* Double-quoted string literals */
        if (*p == '"')
        {
            mc_tok_t *t = tok_new(out, nt++, MC_TOK_STRING, base, ts, line, col);
            p++;
            col++;
            const char *start = p;
            while (*p != '"' && *p != '\0' && *p != '\n')
            {
                p++;
                col++;
            }
            size_t slen = (size_t) (p - start);
            tok_text(out, &used, t, NULL, start,
                     (slen < MC_TOK_STR_MAX) ? slen : (MC_TOK_STR_MAX - 1));
            if (*p == '"')
            {
                p++;
                col++;
            }
            t->len = (int) (p - ts);
            continue;
        }

        /* Identifiers and stream-prefixed names */
        if (*p == '@' || is_ident_start((unsigned char) *p))
        {
            size_t idlen = mc_scan_ident(p);
            /* A plain name is a variable or image name (at most 79 characters, never cut); an
             * entity path (s.cam.cnt0) may be longer */
            size_t pfx    = (p[0] == '@' && p[1] != '\0' && p[2] == ':')    ? 3
                            : ((p[0] == 'S' || p[0] == 'L') && p[1] == ':') ? 2
                                                                            : 0;
            int    dotted = (idlen > 0 && memchr(p, '.', idlen) != NULL);
            size_t maxlen = dotted ? (size_t) MC_IDENT_MAX - 1 : STRINGMAXLEN_IMAGE_NAME - 1;
            if (idlen == 0 || idlen - (dotted ? 0 : pfx) > maxlen)
            {
                out->err_line = line;
                out->err_col  = col;
                if (idlen == 0)
                {
                    snprintf(out->errmsg, sizeof(out->errmsg), "unexpected character '%c'", *p);
                }
                else
                {
                    snprintf(out->errmsg, sizeof(out->errmsg),
                             "identifier too long (max %zu characters)", maxlen);
                }
                return -1;
            }
            mc_tok_t *t = tok_new(out, nt++, MC_TOK_IDENT, base, ts, line, col);
            tok_text(out, &used, t, ((p[0] == 'L' || p[0] == 'S') && p[1] == ':') ? "@" : NULL, p,
                     idlen);
            t->no_space_after = (p[idlen] != '\0' && !isspace((unsigned char) p[idlen]));
            p += idlen;
            col += (int) idlen;
            t->len = (int) (p - ts);
            continue;
        }

        out->err_line = line;
        out->err_col  = col;
        snprintf(out->errmsg, sizeof(out->errmsg), "unexpected character '%c'", *p);
        return -1;
    }

    for (int k = nt; k < nt + 3; k++) /* EOF, and two more for the parser's lookahead */
    {
        mc_tok_t *t = tok_new(out, k, MC_TOK_EOF, base, p, line, col);
        tok_text(out, &used, t, NULL, "", 0);
    }
    out->ntok = nt;
    return 0;
}

/**
 * mc_tok_list_free - Release the token and text buffers mc_lex() allocated in @tl
 * @tl: Token list (may be NULL); the struct itself is not freed
 *
 * Return: void
 */
void mc_tok_list_free(mc_tok_list_t *tl)
{
    if (tl == NULL)
    {
        return;
    }
    free(tl->toks);
    free(tl->strs);
    tl->toks     = NULL;
    tl->strs     = NULL;
    tl->cap      = 0;
    tl->strs_cap = 0;
    tl->ntok     = 0;
}

/* Not in libmilkbash.so (MILK_BASH_CLIENT): the shell client uses only the lexer and the scanners
 * above, and freeing an AST frees its sub-programs with mcalc_sub_prog_free() (run/mc_prog.c),
 * which that library does not contain. */
#ifndef MILK_BASH_CLIENT

/* ========================================================================= */
/* AST Memory Deallocation                                                   */
/* ========================================================================= */

/**
 * mc_expr_free - Recursively free an expression AST subtree
 * @e: Expression node to free (safe if NULL)
 *
 * Releases any cached sub-program (`e->sub_prog`), recursively frees all child
 * expressions according to `e->kind`, and frees @e itself.
 */
void mc_expr_free(mc_expr_t *e)
{
    if (e == NULL)
    {
        return;
    }
    mcalc_sub_prog_free(e->sub_prog);
    e->sub_prog = NULL;
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

/**
 * mc_expr_child_slots - List the child pointers of expression @e
 * @e:     Expression node
 * @slots: Receives the address of each child pointer (some may hold NULL)
 *
 * Lets a walk visit, or replace, every child of any node kind.
 *
 * Return: Number of slots filled (at most `MC_MAX_ARGS`)
 */
int mc_expr_child_slots(mc_expr_t *e, mc_expr_t **slots[MC_MAX_ARGS])
{
    int n = 0;
    switch ((e != NULL) ? e->kind : MC_EXPR_LONG)
    {
    case MC_EXPR_UNARY:
        slots[n++] = &e->u.unary.operand;
        break;
    case MC_EXPR_BINARY:
        slots[n++] = &e->u.binary.left;
        slots[n++] = &e->u.binary.right;
        break;
    case MC_EXPR_CHAIN:
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            slots[n++] = &e->u.chain.operands[i];
        }
        break;
    case MC_EXPR_TERNARY:
        slots[n++] = &e->u.ternary.cond;
        slots[n++] = &e->u.ternary.t_expr;
        slots[n++] = &e->u.ternary.f_expr;
        break;
    case MC_EXPR_CALL:
        for (int i = 0; i < e->u.call.nargs; i++)
        {
            slots[n++] = &e->u.call.args[i];
        }
        break;
    case MC_EXPR_SLICE:
        for (int i = 0; i < e->u.slice.naxis; i++)
        {
            slots[n++] = &e->u.slice.axes[i].start;
            slots[n++] = &e->u.slice.axes[i].stop;
            slots[n++] = &e->u.slice.axes[i].step;
        }
        break;
    case MC_EXPR_ASSIGN:
        slots[n++] = &e->u.assign.target;
        slots[n++] = &e->u.assign.rhs;
        break;
    case MC_EXPR_VEC_LIT:
        for (int i = 0; i < e->u.vec_lit.nelem; i++)
        {
            slots[n++] = &e->u.vec_lit.elems[i];
        }
        break;
    case MC_EXPR_COMP:
        slots[n++] = &e->u.comp.body;
        for (int i = 0; i < e->u.comp.ngens; i++)
        {
            slots[n++] = &e->u.comp.gens[i].start;
            slots[n++] = &e->u.comp.gens[i].stop;
            slots[n++] = &e->u.comp.gens[i].step;
        }
        break;
    default:
        break;
    }
    return n;
}

/* *@dst = a copy of @src (NULL stays NULL); 0 when out of memory */
static int clone_into(mc_expr_t **dst, const mc_expr_t *src)
{
    *dst = mc_expr_clone(src);
    return src == NULL || *dst != NULL;
}

/**
 * mc_expr_clone - Deep-copy an expression AST subtree
 * @e: Expression to copy (may be NULL)
 *
 * The copy owns its children and has no per-call-site state (cached sub-program, `ema` value),
 * so each inlined `def` call keeps its own.
 *
 * Return: The copy, or NULL if @e is NULL or memory runs out
 */
mc_expr_t *mc_expr_clone(const mc_expr_t *e)
{
    mc_expr_t *c = (e != NULL) ? malloc(sizeof(*c)) : NULL;
    if (c == NULL)
    {
        return NULL;
    }
    *c          = *e;
    c->sub_prog = NULL;
    int ok      = 1;
    switch (e->kind) /* children are detached first: a partial copy frees only its own */
    {
    case MC_EXPR_UNARY:
        ok = clone_into(&c->u.unary.operand, e->u.unary.operand);
        break;
    case MC_EXPR_BINARY:
        c->u.binary.right = NULL;
        ok                = clone_into(&c->u.binary.left, e->u.binary.left) &&
                            clone_into(&c->u.binary.right, e->u.binary.right);
        break;
    case MC_EXPR_CHAIN:
        memset(c->u.chain.operands, 0, sizeof(c->u.chain.operands));
        for (int i = 0; ok && i <= e->u.chain.nops; i++)
        {
            ok = clone_into(&c->u.chain.operands[i], e->u.chain.operands[i]);
        }
        break;
    case MC_EXPR_TERNARY:
        c->u.ternary.t_expr = NULL;
        c->u.ternary.f_expr = NULL;
        ok                  = clone_into(&c->u.ternary.cond, e->u.ternary.cond) &&
                              clone_into(&c->u.ternary.t_expr, e->u.ternary.t_expr) &&
                              clone_into(&c->u.ternary.f_expr, e->u.ternary.f_expr);
        break;
    case MC_EXPR_CALL:
        memset(c->u.call.args, 0, sizeof(c->u.call.args));
        c->u.call.ema_init = 0;
        c->u.call.ema_val  = 0.0;
        for (int i = 0; ok && i < e->u.call.nargs; i++)
        {
            ok = clone_into(&c->u.call.args[i], e->u.call.args[i]);
        }
        break;
    case MC_EXPR_SLICE:
        for (int i = 0; i < MC_MAX_SLICE_AX; i++)
        {
            c->u.slice.axes[i].start = NULL;
            c->u.slice.axes[i].stop  = NULL;
            c->u.slice.axes[i].step  = NULL;
        }
        for (int i = 0; ok && i < e->u.slice.naxis; i++)
        {
            ok = clone_into(&c->u.slice.axes[i].start, e->u.slice.axes[i].start) &&
                 clone_into(&c->u.slice.axes[i].stop, e->u.slice.axes[i].stop) &&
                 clone_into(&c->u.slice.axes[i].step, e->u.slice.axes[i].step);
        }
        break;
    case MC_EXPR_ASSIGN:
        c->u.assign.rhs = NULL;
        ok              = clone_into(&c->u.assign.target, e->u.assign.target) &&
                          clone_into(&c->u.assign.rhs, e->u.assign.rhs);
        break;
    case MC_EXPR_VEC_LIT:
        memset(c->u.vec_lit.elems, 0, sizeof(c->u.vec_lit.elems));
        for (int i = 0; ok && i < e->u.vec_lit.nelem; i++)
        {
            ok = clone_into(&c->u.vec_lit.elems[i], e->u.vec_lit.elems[i]);
        }
        break;
    case MC_EXPR_COMP:
        for (int i = 0; i < MC_MAX_SLICE_AX; i++)
        {
            c->u.comp.gens[i].start = NULL;
            c->u.comp.gens[i].stop  = NULL;
            c->u.comp.gens[i].step  = NULL;
        }
        ok = clone_into(&c->u.comp.body, e->u.comp.body);
        for (int i = 0; ok && i < e->u.comp.ngens; i++)
        {
            ok = clone_into(&c->u.comp.gens[i].start, e->u.comp.gens[i].start) &&
                 clone_into(&c->u.comp.gens[i].stop, e->u.comp.gens[i].stop) &&
                 clone_into(&c->u.comp.gens[i].step, e->u.comp.gens[i].step);
        }
        break;
    default:
        break;
    }
    if (!ok)
    {
        mc_expr_free(c);
        return NULL;
    }
    return c;
}

/**
 * mc_stmt_free - Recursively free a linked list of statement AST nodes
 * @s: Head of statement list to free (safe if NULL)
 *
 * Walks the `s->next` chain, freeing all embedded expressions, nested
 * `if`/`for` statement bodies, and cached sub-programs.
 */
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
                mcalc_sub_prog_free(s->u.if_stmt.branches[i].sub_prog);
                mc_expr_free(s->u.if_stmt.branches[i].cond);
                mc_stmt_free(s->u.if_stmt.branches[i].body);
            }
            break;
        case MC_STMT_FOR:
            mcalc_sub_prog_free(s->u.for_stmt.sub_prog);
            mc_expr_free(s->u.for_stmt.start);
            mc_expr_free(s->u.for_stmt.stop);
            mc_expr_free(s->u.for_stmt.step);
            mc_stmt_free(s->u.for_stmt.body);
            break;
        case MC_STMT_DEF:
            mc_expr_free(s->def.body);
            break;
        default:
            break;
        }
        free(s);
        s = next;
    }
}

#endif /* !MILK_BASH_CLIENT */
