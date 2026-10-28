// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shell_lex_code.c
 * @brief Code-frame, compound-keyword, and here-document scanning for `milk_shell_lex`.
 */

#include <ctype.h>
#include <string.h>

#include "milk_shell_lex_internal.h"

/** Return 1 if @c is a bash unquoted word delimiter or whitespace. */
static int is_meta(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == ';' || c == '&' || c == '|' || c == '(' ||
           c == ')' || c == '<' || c == '>';
}

/** Return the innermost open compound keyword in code frame @f, or 0. */
static int kw_top(const frame_t *f)
{
    return (f->nkw > 0) ? f->kw[f->nkw - 1] : 0;
}

/** Push an open compound-command tag @k onto code frame @f's keyword stack. */
static void kw_push(frame_t *f, char k)
{
    if (f->nkw < MSL_MAX_KW)
    {
        f->kw[f->nkw++] = k;
    }
}

/** Pop a matching compound-command tag @k from code frame @f's keyword stack. */
static void kw_pop(frame_t *f, char k)
{
    int top = kw_top(f);
    if (top == k || (k == 'c' && (top == 'w' || top == 'p')))
    {
        f->nkw--;
    }
}

/**
 * read_heredoc_op - Parse `<<[-] WORD` at @i and queue the here-document delimiter
 * @L:     Active lexer state
 * @i:     Byte offset of `<<`
 * @depth: Command-substitution depth
 *
 * Return: Number of bytes consumed from @i
 */
static size_t read_heredoc_op(lex_t *L, size_t i, int depth)
{
    const char *s     = L->s;
    size_t      k     = i + 2;
    int         strip = (k < L->len && s[k] == '-') ? (k++, 1) : 0;

    while (k < L->len)
    {
        if (s[k] == ' ' || s[k] == '\t')
        {
            k++;
        }
        else if (s[k] == '\\' && k + 1 < L->len && s[k + 1] == '\n')
        {
            k += 2;
        }
        else
        {
            break;
        }
    }
    char   delim[128];
    size_t dn = 0;
    char   q  = 0;
    while (k < L->len && s[k] != '\n' && (q != 0 || !is_meta(s[k])))
    {
        char c = s[k];
        if (q == '"' && c == '\\' && k + 1 < L->len)
        {
            if (s[k + 1] == '\n')
            {
                k += 2;
                continue;
            }
            if (s[k + 1] == '"' || s[k + 1] == '\\' || s[k + 1] == '$' || s[k + 1] == '`')
            {
                k++;
                if (dn + 1 < sizeof(delim))
                {
                    delim[dn++] = s[k];
                }
                k++;
                continue;
            }
        }
        if (q != 0)
        {
            if (c == q)
            {
                q = 0;
            }
            else if (dn + 1 < sizeof(delim))
            {
                delim[dn++] = c;
            }
        }
        else if (c == '\'' || c == '"')
        {
            q = c;
        }
        else if (c == '\\' && k + 1 < L->len)
        {
            if (s[k + 1] == '\n')
            {
                k += 2;
                continue;
            }
            k++;
            if (dn + 1 < sizeof(delim))
            {
                delim[dn++] = s[k];
            }
        }
        else if (dn + 1 < sizeof(delim))
        {
            delim[dn++] = c;
        }
        k++;
    }
    delim[dn] = '\0';
    for (size_t j = i; j < k; j++)
    {
        msl_mark(L, j, MSL_CODE, depth);
    }
    if (dn > 0 && L->nhd < MSL_MAX_HEREDOC)
    {
        memcpy(L->hd[L->nhd].delim, delim, dn + 1);
        L->hd[L->nhd].strip_tabs = strip;
        L->nhd++;
    }
    return k - i;
}

/**
 * read_heredoc_bodies - Consume queued here-document bodies starting at @p
 * @L:     Active lexer state
 * @p:     Byte offset immediately following the newline
 * @depth: Command-substitution depth
 *
 * Return: Next byte offset after all queued here-document bodies
 */
static size_t read_heredoc_bodies(lex_t *L, size_t p, int depth)
{
    const char *s = L->s;
    for (int h = 0; h < L->nhd; h++)
    {
        size_t dl   = strlen(L->hd[h].delim);
        int    done = 0;
        while (p < L->len && !done)
        {
            const char *eol = memchr(s + p, '\n', L->len - p);
            size_t      e   = (eol != NULL) ? (size_t) (eol - s) : L->len;
            size_t      t   = p;
            while (L->hd[h].strip_tabs && t < e && s[t] == '\t')
            {
                t++;
            }
            done = (e - t == dl && strncmp(s + t, L->hd[h].delim, dl) == 0);
            for (size_t j = p; j < e; j++)
            {
                msl_mark(L, j, MSL_HEREDOC, depth);
            }
            if (eol != NULL)
            {
                msl_mark(L, e, MSL_HEREDOC, depth);
            }
            p = (eol != NULL) ? e + 1 : L->len;
        }
        if (!done)
        {
            L->m->incomplete = 1;
        }
    }
    L->nhd = 0;
    return p;
}

/**
 * code_word - Scan a word in a code frame and update compound-keyword state
 * @L: Active lexer state
 * @f: Active `F_CODE` frame
 * @i: Start offset of the word
 *
 * Return: Number of bytes consumed (at least 1)
 */
static size_t code_word(lex_t *L, frame_t *f, size_t i)
{
    const char *s = L->s;
    size_t      j = i;
    while (j < L->len && !is_meta(s[j]) && s[j] != '\'' && s[j] != '"' && s[j] != '$' &&
           s[j] != '`' && s[j] != '\\' &&
           !(s[j] == '@' && j + 1 < L->len && s[j + 1] == '{' && (j == i || s[j - 1] == '=')))
    {
        j++;
    }
    size_t n = j - i;
    for (size_t k = i; k < j; k++)
    {
        msl_mark(L, k, MSL_CODE, f->depth);
    }
    int at_end = (j >= L->len || is_meta(s[j]));
#define KW(w) (at_end && n == sizeof(w) - 1 && strncmp(s + i, w, n) == 0)
    if (kw_top(f) == 'w')
    {
        if (KW("in"))
        {
            f->kw[f->nkw - 1] = 'p';
            f->cmdpos         = 1;
        }
    }
    else if (kw_top(f) == 'p')
    {
        if (KW("esac") && (j >= L->len || s[j] != ')'))
        {
            kw_pop(f, 'c');
            f->cmdpos = 0;
        }
    }
    else if (f->fn_name)
    {
        f->fn_name = 0;
        f->cmdpos  = 1;
    }
    else if (KW("if"))
    {
        kw_push(f, 'i');
    }
    else if (KW("fi"))
    {
        kw_pop(f, 'i');
        f->cmdpos = 0;
    }
    else if (KW("then") || KW("else") || KW("elif") || KW("do") || KW("!") || KW("time"))
    {
        f->cmdpos = 1;
    }
    else if (KW("case"))
    {
        kw_push(f, 'w');
        f->cmdpos = 0;
    }
    else if (KW("esac"))
    {
        kw_pop(f, 'c');
        f->cmdpos = 0;
    }
    else if (KW("for") || KW("select"))
    {
        kw_push(f, 'l');
        f->cmdpos = 0;
    }
    else if (KW("while") || KW("until"))
    {
        kw_push(f, 'l');
    }
    else if (KW("done"))
    {
        kw_pop(f, 'l');
        f->cmdpos = 0;
    }
    else if (KW("function"))
    {
        f->fn_name = 1;
        f->cmdpos  = 0;
    }
    else if (KW("{"))
    {
        kw_push(f, '{');
    }
    else if (KW("}"))
    {
        kw_pop(f, '{');
        f->cmdpos = 0;
    }
    else
    {
        const char *eq = memchr(s + i, '=', n);
        f->cmdpos = (eq != NULL && eq > s + i && (isalpha((unsigned char) s[i]) || s[i] == '_'));
    }
#undef KW
    return (n > 0) ? n : 1;
}

/** Handle `;`, `&`, and `|` statement separators in an `F_CODE` frame. */
static size_t scan_code_sep(lex_t *L, frame_t *f, size_t i, int top)
{
    const char *src = L->s;
    char        c   = src[i];
    char        n1  = (i + 1 < L->len) ? src[i + 1] : '\0';
    int         d   = f->depth;

    msl_mark(L, i, MSL_CODE, d);
    int two = (n1 == c) || (c == '|' && n1 == '&') || (c == ';' && n1 == '&');
    if (two)
    {
        msl_mark(L, i + 1, MSL_CODE, d);
    }
    if (c == ';' && (n1 == ';' || n1 == '&') && kw_top(f) == 'c')
    {
        int three         = (n1 == ';' && i + 2 < L->len && src[i + 2] == '&');
        f->kw[f->nkw - 1] = 'p';
        if (three)
        {
            msl_mark(L, i + 2, MSL_CODE, d);
            f->cmdpos = 1;
            return i + 3;
        }
    }
    if (top && kw_top(f) != 'p')
    {
        L->trailing_op = (c == '|' || (c == '&' && n1 == '&'));
    }
    f->cmdpos = 1;
    return i + (two ? 2 : 1);
}

/** Handle operators, parentheses, and words in an `F_CODE` frame. */
static size_t scan_code_punct(lex_t *L, frame_t *f, size_t i, int ws, int top)
{
    const char *src = L->s;
    char        c   = src[i];
    char        n1  = (i + 1 < L->len) ? src[i + 1] : '\0';
    char        n2  = (i + 2 < L->len) ? src[i + 2] : '\0';
    int         d   = f->depth;

    if ((c == '<' || c == '>') && n1 == '(')
    {
        msl_mark(L, i, MSL_OPEN, d + 1);
        msl_mark(L, i + 1, MSL_OPEN, d + 1);
        msl_push(L, F_CODE, C_PAREN, d + 1, i + 2);
        return i + 2;
    }
    if (c == '<' && n1 == '<' && n2 != '<')
    {
        f->cmdpos = 0;
        return i + read_heredoc_op(L, i, d);
    }
    if (c == '<' && n1 == '<' && n2 == '<')
    {
        msl_mark(L, i, MSL_CODE, d);
        msl_mark(L, i + 1, MSL_CODE, d);
        msl_mark(L, i + 2, MSL_CODE, d);
        return i + 3;
    }
    if (c == '(' && n1 == '(' && ws)
    {
        msl_mark(L, i, MSL_ARITH, d);
        msl_mark(L, i + 1, MSL_ARITH, d);
        msl_push(L, F_ARITH, A_DPAREN, d, i + 2);
        return i + 2;
    }
    if (c == '(')
    {
        msl_mark(L, i, MSL_CODE, d);
        f->nest += (kw_top(f) != 'p');
        f->cmdpos = 1;
        return i + 1;
    }
    if (c == ')')
    {
        if (f->nest > 0)
        {
            f->nest--;
        }
        else if (kw_top(f) == 'p')
        {
            f->kw[f->nkw - 1] = 'c';
        }
        else if (f->closer == C_PAREN)
        {
            msl_mark(L, i, MSL_CLOSE, d);
            msl_pop(L);
            return i + 1;
        }
        msl_mark(L, i, MSL_CODE, d);
        f->cmdpos = 1;
        return i + 1;
    }
    if (c == ';' || c == '&' || c == '|')
    {
        return scan_code_sep(L, f, i, top);
    }
    if (c == '<' || c == '>')
    {
        msl_mark(L, i, MSL_CODE, d);
        return i + 1;
    }
    if (ws && (f->cmdpos || kw_top(f) == 'w'))
    {
        return i + code_word(L, f, i);
    }
    msl_mark(L, i, MSL_CODE, d);
    f->cmdpos = 0;
    return i + 1;
}

/**
 * msl_scan_frame_code - Advance one step inside an `F_CODE` shell code frame
 * @L: Active lexer state
 * @f: Active `F_CODE` frame
 * @i: Current byte offset
 *
 * Return: Next byte offset in `L->s`
 */
size_t msl_scan_frame_code(lex_t *L, frame_t *f, size_t i)
{
    const char *src = L->s;
    size_t      len = L->len;
    char        c   = src[i];
    char        n1  = (i + 1 < len) ? src[i + 1] : '\0';
    int         d   = f->depth;
    int         ws =
        (i == f->start) || (i > 0 && is_meta(src[i - 1]) && src[i - 1] != '>' && src[i - 1] != '<');
    int    top = (L->nf == 1 && L->overflow == 0);
    size_t adv;

    if ((f->closer == C_BACKTICK && c == '`') ||
        (f->closer == C_BRACE && f->inline_brace == 0 && c == '}' && ws && kw_top(f) != '{'))
    {
        msl_mark(L, i, MSL_CLOSE, d);
        msl_pop(L);
        return i + 1;
    }
    if (c == '\\')
    {
        msl_mark(L, i, MSL_CODE, d);
        if (i + 1 < len)
        {
            msl_mark(L, i + 1, MSL_CODE, d);
            L->m->incomplete |= (n1 == '\n' && i + 2 >= len);
            i += 2;
        }
        else
        {
            L->m->incomplete = 1;
            i++;
        }
        if (n1 != '\n')
        {
            f->cmdpos      = 0;
            L->trailing_op = top ? 0 : L->trailing_op;
        }
        return i;
    }
    if (f->inline_brace == 0 && c == '#' && ws)
    {
        while (i < len && src[i] != '\n')
        {
            msl_mark(L, i++, MSL_COMMENT, d);
        }
        return i;
    }
    if (c == ' ' || c == '\t')
    {
        msl_mark(L, i, MSL_CODE, d);
        return i + 1;
    }
    if (c == '\n')
    {
        msl_mark(L, i++, MSL_CODE, d);
        f->cmdpos = 1;
        return (L->nhd > 0) ? read_heredoc_bodies(L, i, d) : i;
    }
    if (top && f->inline_brace == 0)
    {
        L->trailing_op = 0;
    }
    if (c == '\'' || (c == '$' && n1 == '\'') || c == '"')
    {
        int    qt   = (c == '\'') ? F_SQ : ((c == '$') ? F_ANSI : F_DQ);
        int    qctx = (qt == F_SQ) ? MSL_SQ : ((qt == F_ANSI) ? MSL_ANSI : MSL_DQ);
        size_t qn   = (qt == F_ANSI) ? 2 : 1;
        msl_mark(L, i, qctx, d);
        if (qn == 2)
        {
            msl_mark(L, i + 1, qctx, d);
        }
        msl_push(L, qt, 0, d, i + qn);
        f->cmdpos = 0;
        return i + qn;
    }
    if ((adv = msl_open_expansion(L, i, f)) > 0)
    {
        f->cmdpos = 0;
        return i + adv;
    }
    if (f->inline_brace == 0 && c == '@' && n1 == '{' &&
        (i == 0 || !(isalnum((unsigned char) src[i - 1]) || src[i - 1] == '_')))
    {
        msl_mark(L, i, MSL_CODE, d);
        msl_mark(L, i + 1, MSL_CODE, d);
        f->inline_brace = 1;
        f->cmdpos       = 0;
        return i + 2;
    }
    if (f->inline_brace > 0)
    {
        f->inline_brace += (c == '{') - (c == '}');
        msl_mark(L, i, MSL_CODE, d);
        return i + 1;
    }
    return scan_code_punct(L, f, i, ws, top);
}
