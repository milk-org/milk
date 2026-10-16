// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shell_lex.c
 * @brief Bash-aware context scanner (see milk_shell_lex.h).
 *
 * One forward pass with an explicit frame stack: code (top level, $(...), `...`, ${| ...; },
 * <(...)), quotes, $'...', ${...}, arithmetic. Here-documents are queued at `<<` and their
 * bodies consumed after the next newline. Compound keywords are tracked per code frame so an
 * unfinished `if`/`case`/loop/`{`/`(` makes the text incomplete.
 */

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "milk_shell_lex.h"

#define MSL_MAX_FRAMES 64
#define MSL_MAX_KW 64
#define MSL_MAX_HEREDOC 16

enum frame_type
{
    F_CODE,
    F_SQ,
    F_DQ,
    F_ANSI,
    F_PARAM,
    F_ARITH
};

enum frame_closer
{
    C_TOP,      /* top level: never closed */
    C_PAREN,    /* $( <( >(  -> ) */
    C_BACKTICK, /* `           -> ` */
    C_BRACE,    /* ${| ${<sp>  -> } */
    A_DPAREN,   /* $(( ((      -> )) */
    A_BRACKET   /* $[          -> ] */
};

typedef struct
{
    int    type;
    int    closer;
    int    nest;    /* CODE: open ( subshells; PARAM: { depth; ARITH: ( or [ depth */
    int    cmdpos;  /* CODE: next word is in command position */
    int    fn_name; /* CODE: after `function`, the name word is not a command */
    int    in_dq;   /* PARAM: inside double quotes (single quotes are literal) */
    int    depth;   /* command-substitution depth of this frame's bytes */
    size_t start;
    char   kw[MSL_MAX_KW];
    int    nkw;
} frame_t;

typedef struct
{
    char delim[128];
    int  strip_tabs;
} heredoc_t;

typedef struct
{
    const char *s;
    size_t      len;
    msl_map_t  *m;
    frame_t     f[MSL_MAX_FRAMES];
    int         nf;
    heredoc_t   hd[MSL_MAX_HEREDOC];
    int         nhd;
    int         trailing_op; /* top level ends with | && || */
} lex_t;

static void mark(lex_t *L, size_t i, int ctx, int depth)
{
    if (i < L->len)
    {
        L->m->ctx[i]   = (unsigned char) ctx;
        L->m->depth[i] = (unsigned char) (depth > 255 ? 255 : depth);
    }
}

static frame_t *push(lex_t *L, int type, int closer, int depth, size_t start)
{
    if (L->nf >= MSL_MAX_FRAMES)
    {
        return &L->f[L->nf - 1];
    }
    frame_t *f = &L->f[L->nf++];
    memset(f, 0, sizeof(*f));
    f->type   = type;
    f->closer = closer;
    f->depth  = depth;
    f->start  = start;
    f->cmdpos = 1;
    return f;
}

static void pop(lex_t *L)
{
    if (L->nf > 1)
    {
        L->nf--;
    }
}

static int is_meta(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == ';' || c == '&' || c == '|' || c == '(' ||
           c == ')' || c == '<' || c == '>';
}

static int kw_top(const frame_t *f)
{
    return (f->nkw > 0) ? f->kw[f->nkw - 1] : 0;
}

static int case_open(const frame_t *f)
{
    return kw_top(f) == 'c';
}

static void kw_push(frame_t *f, char k)
{
    if (f->nkw < MSL_MAX_KW)
    {
        f->kw[f->nkw++] = k;
    }
}

static void kw_pop(frame_t *f, char k)
{
    if (kw_top(f) == k)
    {
        f->nkw--;
    }
}

/**
 * open_expansion() - handle `$(`, `$((`, `${`, `${|`, `$[` or a backtick at @i inside frame @f.
 * Returns the number of bytes consumed, or 0 if @i does not start an expansion.
 */
static size_t open_expansion(lex_t *L, size_t i, const frame_t *f)
{
    const char *s  = L->s;
    char        c  = s[i];
    char        n1 = (i + 1 < L->len) ? s[i + 1] : '\0';
    char        n2 = (i + 2 < L->len) ? s[i + 2] : '\0';
    int         d  = f->depth;
    int         dq = (f->type == F_DQ) || (f->type == F_PARAM && f->in_dq);

    if (c == '`')
    {
        mark(L, i, MSL_OPEN, d + 1);
        push(L, F_CODE, C_BACKTICK, d + 1, i + 1);
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
            mark(L, i + k, MSL_ARITH, d);
        }
        push(L, F_ARITH, A_DPAREN, d, i + 3);
        return 3;
    }
    if (n1 == '(')
    {
        mark(L, i, MSL_OPEN, d + 1);
        mark(L, i + 1, MSL_OPEN, d + 1);
        push(L, F_CODE, C_PAREN, d + 1, i + 2);
        return 2;
    }
    if (n1 == '{' && (n2 == '|' || n2 == ' ' || n2 == '\t' || n2 == '\n')) /* bash 5.3 funsub */
    {
        size_t n = (n2 == '|') ? 3 : 2;
        for (size_t k = 0; k < n; k++)
        {
            mark(L, i + k, MSL_OPEN, d + 1);
        }
        push(L, F_CODE, C_BRACE, d + 1, i + n);
        return n;
    }
    if (n1 == '{')
    {
        mark(L, i, MSL_PARAM, d);
        mark(L, i + 1, MSL_PARAM, d);
        frame_t *p = push(L, F_PARAM, 0, d, i + 2);
        p->in_dq   = dq;
        return 2;
    }
    if (n1 == '[')
    {
        mark(L, i, MSL_ARITH, d);
        mark(L, i + 1, MSL_ARITH, d);
        push(L, F_ARITH, A_BRACKET, d, i + 2);
        return 2;
    }
    return 0;
}

/** read_heredoc_op() - parse `<<[-] WORD` at @i; queue the delimiter. Returns bytes consumed. */
static size_t read_heredoc_op(lex_t *L, size_t i, int depth)
{
    const char *s     = L->s;
    size_t      k     = i + 2;
    int         strip = 0;
    if (k < L->len && s[k] == '-')
    {
        strip = 1;
        k++;
    }
    while (k < L->len && (s[k] == ' ' || s[k] == '\t'))
    {
        k++;
    }
    char   delim[128];
    size_t dn = 0;
    char   q  = 0;
    while (k < L->len && s[k] != '\n' && (q != 0 || !is_meta(s[k])))
    {
        char c = s[k];
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
        mark(L, j, MSL_CODE, depth);
    }
    if (dn > 0 && L->nhd < MSL_MAX_HEREDOC)
    {
        memcpy(L->hd[L->nhd].delim, delim, dn + 1);
        L->hd[L->nhd].strip_tabs = strip;
        L->nhd++;
    }
    return k - i;
}

/** read_heredoc_bodies() - consume queued here-document bodies starting at @p. Returns next index. */
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
                mark(L, j, MSL_HEREDOC, depth);
            }
            if (eol != NULL)
            {
                mark(L, e, MSL_HEREDOC, depth);
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
 * code_word() - handle a word at command position in a code frame: track compound keywords.
 * Returns the number of bytes consumed (the word).
 */
static size_t code_word(lex_t *L, frame_t *f, size_t i)
{
    const char *s = L->s;
    size_t      j = i;
    while (j < L->len && !is_meta(s[j]) && s[j] != '\'' && s[j] != '"' && s[j] != '$' &&
           s[j] != '`' && s[j] != '\\')
    {
        j++;
    }
    size_t n = j - i;
    for (size_t k = i; k < j; k++)
    {
        mark(L, k, MSL_CODE, f->depth);
    }
    int at_end = (j >= L->len || is_meta(s[j]));
#define KW(w) (at_end && n == sizeof(w) - 1 && strncmp(s + i, w, n) == 0)
    if (f->fn_name)
    {
        f->fn_name = 0;
        f->cmdpos  = 1; /* `function f {` */
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
        kw_push(f, 'c');
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
        /* NAME=value assignments keep the command position for the next word */
        const char *eq = memchr(s + i, '=', n);
        f->cmdpos = (eq != NULL && eq > s + i && (isalpha((unsigned char) s[i]) || s[i] == '_'));
    }
#undef KW
    return (n > 0) ? n : 1;
}

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
    push(L, F_CODE, C_TOP, 0, 0);

    size_t i = 0;
    while (i < len)
    {
        frame_t *f  = &L->f[L->nf - 1];
        char     c  = src[i];
        char     n1 = (i + 1 < len) ? src[i + 1] : '\0';
        char     n2 = (i + 2 < len) ? src[i + 2] : '\0';
        size_t   adv;

        switch (f->type)
        {
        case F_SQ:
            mark(L, i, MSL_SQ, f->depth);
            if (c == '\'')
            {
                pop(L);
            }
            i++;
            break;

        case F_ANSI:
            mark(L, i, MSL_ANSI, f->depth);
            if (c == '\\' && i + 1 < len)
            {
                mark(L, i + 1, MSL_ANSI, f->depth);
                i += 2;
                break;
            }
            if (c == '\'')
            {
                pop(L);
            }
            i++;
            break;

        case F_DQ:
            if (c == '\\' && i + 1 < len)
            {
                mark(L, i, MSL_DQ, f->depth);
                mark(L, i + 1, MSL_DQ, f->depth);
                i += 2;
                break;
            }
            if (c == '"')
            {
                mark(L, i, MSL_DQ, f->depth);
                pop(L);
                i++;
                break;
            }
            if ((adv = open_expansion(L, i, f)) > 0)
            {
                i += adv;
                break;
            }
            mark(L, i, MSL_DQ, f->depth);
            i++;
            break;

        case F_PARAM:
            if (c == '\\' && i + 1 < len)
            {
                mark(L, i, MSL_PARAM, f->depth);
                mark(L, i + 1, MSL_PARAM, f->depth);
                i += 2;
                break;
            }
            if (c == '}' && f->nest == 0)
            {
                mark(L, i, MSL_PARAM, f->depth);
                pop(L);
                i++;
                break;
            }
            if (c == '"')
            {
                mark(L, i, MSL_DQ, f->depth);
                push(L, F_DQ, 0, f->depth, i + 1);
                i++;
                break;
            }
            if (c == '\'' && !f->in_dq)
            {
                mark(L, i, MSL_SQ, f->depth);
                push(L, F_SQ, 0, f->depth, i + 1);
                i++;
                break;
            }
            if ((adv = open_expansion(L, i, f)) > 0)
            {
                i += adv;
                break;
            }
            f->nest += (c == '{') - (c == '}');
            mark(L, i, MSL_PARAM, f->depth);
            i++;
            break;

        case F_ARITH:
            if (f->closer == A_DPAREN && c == ')' && f->nest == 0 && n1 == ')')
            {
                mark(L, i, MSL_ARITH, f->depth);
                mark(L, i + 1, MSL_ARITH, f->depth);
                pop(L);
                i += 2;
                break;
            }
            if (f->closer == A_BRACKET && c == ']' && f->nest == 0)
            {
                mark(L, i, MSL_ARITH, f->depth);
                pop(L);
                i++;
                break;
            }
            if (c == '"')
            {
                mark(L, i, MSL_DQ, f->depth);
                push(L, F_DQ, 0, f->depth, i + 1);
                i++;
                break;
            }
            if ((adv = open_expansion(L, i, f)) > 0)
            {
                i += adv;
                break;
            }
            if (f->closer == A_DPAREN)
            {
                f->nest += (c == '(') - (c == ')' && f->nest > 0);
            }
            else
            {
                f->nest += (c == '[') - (c == ']' && f->nest > 0);
            }
            mark(L, i, MSL_ARITH, f->depth);
            i++;
            break;

        case F_CODE:
        default:
        {
            int d   = f->depth;
            int ws  = (i == f->start) ||
                      (i > 0 && is_meta(src[i - 1]) && src[i - 1] != '>' && src[i - 1] != '<');
            int top = (L->nf == 1);

            if (f->closer == C_BACKTICK && c == '`')
            {
                mark(L, i, MSL_CLOSE, d);
                pop(L);
                i++;
                break;
            }
            if (f->closer == C_BRACE && c == '}' && ws && kw_top(f) != '{')
            {
                mark(L, i, MSL_CLOSE, d);
                pop(L);
                i++;
                break;
            }
            if (c == '\\')
            {
                mark(L, i, MSL_CODE, d);
                if (i + 1 < len)
                {
                    mark(L, i + 1, MSL_CODE, d);
                    if (n1 == '\n' && i + 2 >= len)
                    {
                        m->incomplete = 1; /* line continuation at the end */
                    }
                    i += 2;
                }
                else
                {
                    m->incomplete = 1; /* trailing backslash: line continues */
                    i++;
                }
                if (n1 != '\n')
                {
                    f->cmdpos = 0;
                    if (top)
                    {
                        L->trailing_op = 0;
                    }
                }
                break;
            }
            if (c == '#' && ws)
            {
                while (i < len && src[i] != '\n')
                {
                    mark(L, i, MSL_COMMENT, d);
                    i++;
                }
                break;
            }
            if (c == ' ' || c == '\t')
            {
                mark(L, i, MSL_CODE, d);
                i++;
                break;
            }
            if (c == '\n') /* a trailing | && || stays pending across the newline */
            {
                mark(L, i, MSL_CODE, d);
                f->cmdpos = 1;
                i++;
                if (L->nhd > 0)
                {
                    i = read_heredoc_bodies(L, i, d);
                }
                break;
            }
            if (top)
            {
                L->trailing_op = 0;
            }
            if (c == '\'')
            {
                mark(L, i, MSL_SQ, d);
                push(L, F_SQ, 0, d, i + 1);
                f->cmdpos = 0;
                i++;
                break;
            }
            if (c == '$' && n1 == '\'')
            {
                mark(L, i, MSL_ANSI, d);
                mark(L, i + 1, MSL_ANSI, d);
                push(L, F_ANSI, 0, d, i + 2);
                f->cmdpos = 0;
                i += 2;
                break;
            }
            if (c == '"')
            {
                mark(L, i, MSL_DQ, d);
                push(L, F_DQ, 0, d, i + 1);
                f->cmdpos = 0;
                i++;
                break;
            }
            if ((adv = open_expansion(L, i, f)) > 0)
            {
                f->cmdpos = 0;
                i += adv;
                break;
            }
            if ((c == '<' || c == '>') && n1 == '(')
            {
                mark(L, i, MSL_OPEN, d + 1);
                mark(L, i + 1, MSL_OPEN, d + 1);
                push(L, F_CODE, C_PAREN, d + 1, i + 2);
                i += 2;
                break;
            }
            if (c == '<' && n1 == '<' && n2 != '<')
            {
                i += read_heredoc_op(L, i, d);
                f->cmdpos = 0;
                break;
            }
            if (c == '<' && n1 == '<' && n2 == '<')
            {
                mark(L, i, MSL_CODE, d);
                mark(L, i + 1, MSL_CODE, d);
                mark(L, i + 2, MSL_CODE, d);
                i += 3;
                break;
            }
            if (c == '(' && n1 == '(' && ws)
            {
                mark(L, i, MSL_ARITH, d);
                mark(L, i + 1, MSL_ARITH, d);
                push(L, F_ARITH, A_DPAREN, d, i + 2);
                i += 2;
                break;
            }
            if (c == '(')
            {
                mark(L, i, MSL_CODE, d);
                f->nest++;
                f->cmdpos = 1;
                i++;
                break;
            }
            if (c == ')')
            {
                if (f->nest > 0)
                {
                    f->nest--;
                    mark(L, i, MSL_CODE, d);
                }
                else if (case_open(f))
                {
                    mark(L, i, MSL_CODE, d); /* case pattern terminator */
                }
                else if (f->closer == C_PAREN)
                {
                    mark(L, i, MSL_CLOSE, d);
                    pop(L);
                    i++;
                    break;
                }
                else
                {
                    mark(L, i, MSL_CODE, d);
                }
                f->cmdpos = 1; /* `f() {`, case arms */
                i++;
                break;
            }
            if (c == ';' || c == '&' || c == '|')
            {
                mark(L, i, MSL_CODE, d);
                int two = (n1 == c) || (c == '|' && n1 == '&') || (c == ';' && n1 == '&');
                if (two)
                {
                    mark(L, i + 1, MSL_CODE, d);
                }
                if (top)
                {
                    L->trailing_op = (c == '|' || (c == '&' && n1 == '&'));
                }
                f->cmdpos = 1;
                i += two ? 2 : 1;
                break;
            }
            if (c == '<' || c == '>')
            {
                mark(L, i, MSL_CODE, d);
                i++;
                break;
            }
            if (ws && f->cmdpos)
            {
                i += code_word(L, f, i);
                break;
            }
            mark(L, i, MSL_CODE, d);
            f->cmdpos = 0;
            i++;
            break;
        }
        }
    }

    if (L->nhd > 0) /* heredoc operator on the last line: body still to come */
    {
        m->incomplete = 1;
    }
    const frame_t *t = &L->f[0];
    m->unterminated  = (L->nf > 1);
    if (L->nf > 1 || t->nest > 0 || t->nkw > 0 || L->trailing_op)
    {
        m->incomplete = 1;
    }
    free(L);
    return 0;
}

void msl_free(msl_map_t *m)
{
    free(m->ctx);
    free(m->depth);
    m->ctx   = NULL;
    m->depth = NULL;
    m->len   = 0;
}

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
