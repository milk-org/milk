// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shell_lex.c
 * @brief Bash-aware context scanner (see milk_shell_lex.h).
 *
 * One forward pass with an explicit frame stack: code (top level, $(...), `...`, ${| ...; },
 * <(...)), quotes, $'...', ${...}, arithmetic. Here-documents and compound keywords inside
 * code frames are handled by `milk_shell_lex_code.c`.
 */

#include <stdlib.h>
#include <string.h>

#include "milk_shell_lex_internal.h"

/**
 * msl_open_expansion - Handle `$(`, `$((`, `${`, `${|`, `$[` or a backtick at @i inside frame @f
 * @L: Active lexer state
 * @i: Current byte offset in `L->s`
 * @f: Active lexical frame
 *
 * Return: Number of bytes consumed, or 0 if @i does not start an expansion
 */
size_t msl_open_expansion(lex_t *L, size_t i, const frame_t *f)
{
    const char *s  = L->s;
    char        c  = s[i];
    char        n1 = (i + 1 < L->len) ? s[i + 1] : '\0';
    char        n2 = (i + 2 < L->len) ? s[i + 2] : '\0';
    int         d  = f->depth;
    int         dq = (f->type == F_DQ) || (f->type == F_PARAM && f->in_dq);

    if (c == '`')
    {
        msl_mark(L, i, MSL_OPEN, d + 1);
        msl_push(L, F_CODE, C_BACKTICK, d + 1, i + 1);
        return 1;
    }
    if (c != '$')
    {
        return 0;
    }
    if (n1 == '(' && n2 == '(')
    {
        for (size_t k = 0; k < 3; k++)
        {
            msl_mark(L, i + k, MSL_ARITH, d);
        }
        msl_push(L, F_ARITH, A_DPAREN, d, i + 3);
        return 3;
    }
    if (n1 == '(')
    {
        msl_mark(L, i, MSL_OPEN, d + 1);
        msl_mark(L, i + 1, MSL_OPEN, d + 1);
        msl_push(L, F_CODE, C_PAREN, d + 1, i + 2);
        return 2;
    }
    if (n1 == '{' && (n2 == '|' || n2 == ' ' || n2 == '\t' || n2 == '\n')) /* bash 5.3 funsub */
    {
        size_t n = (n2 == '|') ? 3 : 2;
        for (size_t k = 0; k < n; k++)
        {
            msl_mark(L, i + k, MSL_OPEN, d + 1);
        }
        msl_push(L, F_CODE, C_BRACE, d + 1, i + n);
        return n;
    }
    if (n1 == '{')
    {
        msl_mark(L, i, MSL_PARAM, d);
        msl_mark(L, i + 1, MSL_PARAM, d);
        frame_t *p = msl_push(L, F_PARAM, 0, d, i + 2);
        p->in_dq   = dq;
        return 2;
    }
    if (n1 == '[')
    {
        msl_mark(L, i, MSL_ARITH, d);
        msl_mark(L, i + 1, MSL_ARITH, d);
        msl_push(L, F_ARITH, A_BRACKET, d, i + 2);
        return 2;
    }
    return 0;
}

/** Advance one step inside a single-quote, `$'...'`, or double-quote frame. */
static size_t scan_frame_quote(lex_t *L, frame_t *f, size_t i)
{
    char   c = L->s[i];
    size_t adv;

    if (f->type == F_SQ)
    {
        msl_mark(L, i, MSL_SQ, f->depth);
        if (c == '\'')
        {
            msl_pop(L);
        }
        return i + 1;
    }
    if (f->type == F_ANSI)
    {
        msl_mark(L, i, MSL_ANSI, f->depth);
        if (c == '\\' && i + 1 < L->len)
        {
            msl_mark(L, i + 1, MSL_ANSI, f->depth);
            return i + 2;
        }
        if (c == '\'')
        {
            msl_pop(L);
        }
        return i + 1;
    }
    if (c == '\\' && i + 1 < L->len)
    {
        msl_mark(L, i, MSL_DQ, f->depth);
        msl_mark(L, i + 1, MSL_DQ, f->depth);
        return i + 2;
    }
    if (c == '"')
    {
        msl_mark(L, i, MSL_DQ, f->depth);
        msl_pop(L);
        return i + 1;
    }
    if ((adv = msl_open_expansion(L, i, f)) > 0)
    {
        return i + adv;
    }
    msl_mark(L, i, MSL_DQ, f->depth);
    return i + 1;
}

/** Advance one step inside a `${...}` parameter expansion frame. */
static size_t scan_frame_param(lex_t *L, frame_t *f, size_t i)
{
    char   c = L->s[i];
    size_t adv;

    if (c == '\\' && i + 1 < L->len)
    {
        msl_mark(L, i, MSL_PARAM, f->depth);
        msl_mark(L, i + 1, MSL_PARAM, f->depth);
        return i + 2;
    }
    if (c == '}' && f->nest == 0)
    {
        msl_mark(L, i, MSL_PARAM, f->depth);
        msl_pop(L);
        return i + 1;
    }
    if (c == '"')
    {
        msl_mark(L, i, MSL_DQ, f->depth);
        msl_push(L, F_DQ, 0, f->depth, i + 1);
        return i + 1;
    }
    if (c == '\'' && !f->in_dq)
    {
        msl_mark(L, i, MSL_SQ, f->depth);
        msl_push(L, F_SQ, 0, f->depth, i + 1);
        return i + 1;
    }
    if ((adv = msl_open_expansion(L, i, f)) > 0)
    {
        return i + adv;
    }
    f->nest += (c == '{') - (c == '}');
    msl_mark(L, i, MSL_PARAM, f->depth);
    return i + 1;
}

/** Advance one step inside a `$((...))`, `((...))`, or `$[...]` arithmetic frame. */
static size_t scan_frame_arith(lex_t *L, frame_t *f, size_t i)
{
    char   c  = L->s[i];
    char   n1 = (i + 1 < L->len) ? L->s[i + 1] : '\0';
    size_t adv;

    if (f->closer == A_DPAREN && c == ')' && f->nest == 0 && n1 == ')')
    {
        msl_mark(L, i, MSL_ARITH, f->depth);
        msl_mark(L, i + 1, MSL_ARITH, f->depth);
        msl_pop(L);
        return i + 2;
    }
    if (f->closer == A_BRACKET && c == ']' && f->nest == 0)
    {
        msl_mark(L, i, MSL_ARITH, f->depth);
        msl_pop(L);
        return i + 1;
    }
    if (c == '"')
    {
        msl_mark(L, i, MSL_DQ, f->depth);
        msl_push(L, F_DQ, 0, f->depth, i + 1);
        return i + 1;
    }
    if ((adv = msl_open_expansion(L, i, f)) > 0)
    {
        return i + adv;
    }
    if (f->closer == A_DPAREN)
    {
        f->nest += (c == '(') - (c == ')' && f->nest > 0);
    }
    else
    {
        f->nest += (c == '[') - (c == ']' && f->nest > 0);
    }
    msl_mark(L, i, MSL_ARITH, f->depth);
    return i + 1;
}

/**
 * msl_scan - Scan a bash command or script buffer and classify every byte
 * @src: Input shell source buffer
 * @len: Number of bytes to scan in @src
 * @m:   Output lexical map populated with per-byte context, depth, and completion flags
 *
 * Performs a single forward pass with an explicit frame stack (`F_CODE`, quotes,
 * parameter and arithmetic expansions, and queued here-documents) to annotate
 * each byte's context and command-substitution depth, and sets `m->incomplete`
 * or `m->unterminated` when constructs remain open at the end of @src.
 *
 * Return: 0 on success, -1 on memory allocation failure
 */
int msl_scan(const char *src, size_t len, msl_map_t *m)
{
    memset(m, 0, sizeof(*m));
    m->ctx   = (unsigned char *) calloc(len + 1, 1);
    m->depth = (unsigned char *) calloc(len + 1, 1);
    if (m->ctx == NULL || m->depth == NULL)
    {
        msl_free(m);
        return -1;
    }
    m->len = len;

    lex_t *L = (lex_t *) calloc(1, sizeof(lex_t));
    if (L == NULL)
    {
        msl_free(m);
        return -1;
    }
    L->s   = src;
    L->len = len;
    L->m   = m;
    msl_push(L, F_CODE, C_TOP, 0, 0);

    size_t i = 0;
    while (i < len)
    {
        frame_t *f = (L->overflow > 0) ? &L->ovf_frame : &L->f[L->nf - 1];
        switch (f->type)
        {
        case F_SQ:
        case F_ANSI:
        case F_DQ:
            i = scan_frame_quote(L, f, i);
            break;
        case F_PARAM:
            i = scan_frame_param(L, f, i);
            break;
        case F_ARITH:
            i = scan_frame_arith(L, f, i);
            break;
        case F_CODE:
        default:
            i = msl_scan_frame_code(L, f, i);
            break;
        }
    }

    if (L->nhd > 0) /* heredoc operator on the last line: body still to come */
    {
        m->incomplete = 1;
    }
    const frame_t *t = &L->f[0];
    m->unterminated  = (L->nf > 1 || L->overflow > 0);
    if (m->unterminated || t->nest > 0 || t->inline_brace > 0 || t->nkw > 0 || L->trailing_op)
    {
        m->incomplete = 1;
    }
    free(L);
    return 0;
}

/**
 * msl_free - Release the context and depth arrays allocated by `msl_scan()`
 * @m: Lexical map structure whose heap buffers should be freed
 *
 * Return: void
 */
void msl_free(msl_map_t *m)
{
    free(m->ctx);
    free(m->depth);
    m->ctx   = NULL;
    m->depth = NULL;
    m->len   = 0;
}

/**
 * milk_shell_incomplete - Check whether a bash input buffer needs a continuation line
 * @src: Null-terminated shell command or multi-line buffer to inspect
 *
 * Return: 1 if @src is syntactically incomplete and requires more input, 0 otherwise
 */
int milk_shell_incomplete(const char *src)
{
    if (src == NULL || src[0] == '\0')
    {
        return 0;
    }
    msl_map_t m;
    if (msl_scan(src, strlen(src), &m) != 0)
    {
        return 0;
    }
    int inc = m.incomplete;
    msl_free(&m);
    return inc;
}
