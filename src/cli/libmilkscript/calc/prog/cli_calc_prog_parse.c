// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_prog_parse.c
 * @brief Statement splitting, structured `if`/`for` block parsing, and target classification
 *        for calculator programs.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "cli_calc_internal.h"
#include "cli_calc_prog_internal.h"

/**
 * add_stmt - Trim whitespace and append a statement string to a program
 * @p:     Calculator program being constructed
 * @s:     Start of statement substring
 * @len:   Length of statement substring in bytes
 * @err:   Buffer for error message if `PROG_MAXSTMT` is exceeded
 * @errsz: Size of @err in bytes
 *
 * Return: 0 on success (including empty statements), -1 on limit or allocation failure
 */
static int add_stmt(cli_calc_prog *p, const char *s, size_t len, char *err, size_t errsz)
{
    while (len > 0 && isspace((unsigned char) *s))
    {
        s++;
        len--;
    }
    while (len > 0 && isspace((unsigned char) s[len - 1]))
    {
        len--;
    }
    if (len == 0)
    {
        return 0;
    }
    if (p->n == PROG_MAXSTMT)
    {
        snprintf(err, errsz, "more than %d statements", PROG_MAXSTMT);
        return -1;
    }
    p->st[p->n].block_id = p->next_block_id;
    p->st[p->n].text     = strndup(s, len);
    return (p->st[p->n++].text == NULL) ? -1 : 0;
}

static const char *skip_ws_comments(const char *c, const char *end)
{
    while (c < end)
    {
        while (c < end && (isspace((unsigned char) *c) || *c == ';'))
        {
            c++;
        }
        if (c < end && *c == '#')
        {
            while (c < end && *c != '\n')
            {
                c++;
            }
            continue;
        }
        break;
    }
    return c;
}

static const char *find_matching_brace(const char *open_brace, const char *end)
{
    int  depth = 1;
    char q     = 0;
    for (const char *c = open_brace + 1; c < end; c++)
    {
        if (q != 0)
        {
            if (*c == q)
            {
                q = 0;
            }
        }
        else if (*c == '"' || *c == '\'')
        {
            q = *c;
        }
        else if (*c == '#')
        {
            while (c + 1 < end && c[1] != '\n')
            {
                c++;
            }
        }
        else if (*c == '{')
        {
            depth++;
        }
        else if (*c == '}')
        {
            if (--depth == 0)
            {
                return c;
            }
        }
    }
    return NULL;
}

static int split_block(cli_calc_prog *p,
                       const char    *text,
                       size_t         text_len,
                       char          *err,
                       size_t         errsz);

static int emit_if_begin(cli_calc_prog *p,
                         const char    *cond_start,
                         size_t         cond_len,
                         char          *err,
                         size_t         errsz)
{
    while (cond_len > 0 && isspace((unsigned char) *cond_start))
    {
        cond_start++;
        cond_len--;
    }
    while (cond_len > 0 && isspace((unsigned char) cond_start[cond_len - 1]))
    {
        cond_len--;
    }
    if (cond_len == 0)
    {
        snprintf(err, errsz, "empty condition in 'if { ... }'");
        return -1;
    }
    if (p->n == PROG_MAXSTMT)
    {
        snprintf(err, errsz, "more than %d statements", PROG_MAXSTMT);
        return -1;
    }
    int        idx = p->n++;
    prog_stmt *st  = &p->st[idx];
    st->kind       = STMT_IF_BEGIN;
    st->block_id   = ++p->next_block_id;
    char   gname[32];
    size_t glen = cond_len + sizeof(gname) + 8;
    snprintf(gname, sizeof(gname), "__g%d", idx);
    st->guard = malloc(glen);
    st->text  = strdup("if");
    if (st->guard == NULL || st->text == NULL)
    {
        return -1;
    }
    snprintf(st->guard, glen, "%s = (%.*s)", gname, (int) cond_len, cond_start);
    st->guard_vid = create_variable_ID(gname, 0.0);
    return (st->guard_vid < 0) ? -1 : idx;
}

static const char *find_elif_open_brace(const char *cstart, const char *end)
{
    const char *ob     = cstart;
    int         pdepth = 0;
    char        q      = 0;
    for (; ob < end; ob++)
    {
        if (q != 0)
        {
            q = (*ob == q) ? 0 : q;
        }
        else if (*ob == '"' || *ob == '\'')
        {
            q = *ob;
        }
        else if (*ob == '(' || *ob == '[')
        {
            pdepth++;
        }
        else if ((*ob == ')' || *ob == ']') && pdepth > 0)
        {
            pdepth--;
        }
        else if (*ob == '{' && pdepth == 0)
        {
            return ob;
        }
    }
    return NULL;
}

static int emit_else_jump(cli_calc_prog *p,
                          int           *else_jumps,
                          int           *nej,
                          int            if_idx,
                          const char    *tag,
                          char          *err,
                          size_t         errsz)
{
    if (p->n == PROG_MAXSTMT || *nej >= 32)
    {
        snprintf(err, errsz, "more than %d statements", PROG_MAXSTMT);
        return -1;
    }
    int ej_idx             = p->n++;
    p->st[ej_idx].kind     = STMT_ELSE_JUMP;
    p->st[ej_idx].text     = strdup(tag);
    else_jumps[(*nej)++]   = ej_idx;
    p->st[if_idx].jump_idx = p->n;
    return 0;
}

static int parse_if_brace_chain(cli_calc_prog *p,
                                const char    *cond_start,
                                const char    *open_brace,
                                const char    *end,
                                const char   **out_next,
                                char          *err,
                                size_t         errsz)
{
    int else_jumps[32];
    int nej    = 0;
    int if_idx = emit_if_begin(p, cond_start, (size_t) (open_brace - cond_start), err, errsz);
    if (if_idx < 0)
    {
        return -1;
    }
    const char *close_brace = find_matching_brace(open_brace, end);
    if (close_brace == NULL)
    {
        snprintf(err, errsz, "unclosed '{' in if block");
        return -1;
    }
    p->next_block_id++;
    if (split_block(p, open_brace + 1, (size_t) (close_brace - (open_brace + 1)), err, errsz) != 0)
    {
        return -1;
    }
    const char *cur = close_brace + 1;
    while (cur < end)
    {
        const char *kw = skip_ws_comments(cur, end);
        if ((size_t) (end - kw) >= 4 && strncmp(kw, "elif", 4) == 0 &&
            (isspace((unsigned char) kw[4]) || kw[4] == '('))
        {
            const char *cstart = kw + 4;
            const char *ob     = find_elif_open_brace(cstart, end);
            if (ob == NULL)
            {
                snprintf(err, errsz, "expected '{' after elif condition");
                return -1;
            }
            if (emit_else_jump(p, else_jumps, &nej, if_idx, "elif", err, errsz) != 0)
            {
                return -1;
            }
            if_idx = emit_if_begin(p, cstart, (size_t) (ob - cstart), err, errsz);
            if (if_idx < 0)
            {
                return -1;
            }
            close_brace = find_matching_brace(ob, end);
            if (close_brace == NULL)
            {
                snprintf(err, errsz, "unclosed '{' in elif block");
                return -1;
            }
            p->next_block_id++;
            if (split_block(p, ob + 1, (size_t) (close_brace - (ob + 1)), err, errsz) != 0)
            {
                return -1;
            }
            cur = close_brace + 1;
            continue;
        }
        if ((size_t) (end - kw) >= 4 && strncmp(kw, "else", 4) == 0 &&
            (isspace((unsigned char) kw[4]) || kw[4] == '{'))
        {
            const char *ob = skip_ws_comments(kw + 4, end);
            if (ob >= end || *ob != '{')
            {
                snprintf(err, errsz, "expected '{' after else");
                return -1;
            }
            if (emit_else_jump(p, else_jumps, &nej, if_idx, "else", err, errsz) != 0)
            {
                return -1;
            }
            if_idx      = -1;
            close_brace = find_matching_brace(ob, end);
            if (close_brace == NULL)
            {
                snprintf(err, errsz, "unclosed '{' in else block");
                return -1;
            }
            p->next_block_id++;
            if (split_block(p, ob + 1, (size_t) (close_brace - (ob + 1)), err, errsz) != 0)
            {
                return -1;
            }
            cur = close_brace + 1;
        }
        break;
    } // while (cur < end)
    if (if_idx >= 0)
    {
        p->st[if_idx].jump_idx = p->n;
    }
    for (int k = 0; k < nej; k++)
    {
        p->st[else_jumps[k]].jump_idx = p->n;
    }
    p->next_block_id++;
    *out_next = cur;
    return 0;
}

static int parse_for_range_colons(const char  *h,
                                  const char  *open_brace,
                                  const char **out_colon1,
                                  const char **out_colon2,
                                  char        *err,
                                  size_t       errsz)
{
    const char *colon1 = NULL;
    const char *colon2 = NULL;
    int         pdepth = 0;
    for (const char *c = h; c < open_brace; c++)
    {
        if (*c == '(' || *c == '[')
        {
            pdepth++;
        }
        else if ((*c == ')' || *c == ']') && pdepth > 0)
        {
            pdepth--;
        }
        else if (*c == ':' && pdepth == 0)
        {
            if (colon1 == NULL)
            {
                colon1 = c;
            }
            else if (colon2 == NULL)
            {
                colon2 = c;
            }
            else
            {
                snprintf(err, errsz, "too many ':' in for range");
                return -1;
            }
        }
    }
    if (colon1 == NULL)
    {
        snprintf(err, errsz, "missing ':' in for range");
        return -1;
    }
    *out_colon1 = colon1;
    *out_colon2 = colon2;
    return 0;
}

static int parse_for_brace_block(cli_calc_prog *p,
                                 const char    *hdr_start,
                                 const char    *open_brace,
                                 const char    *end,
                                 const char   **out_next,
                                 char          *err,
                                 size_t         errsz)
{
    const char *h = hdr_start;
    while (h < open_brace && isspace((unsigned char) *h))
    {
        h++;
    }
    const char *vstart = h;
    while (h < open_brace && (isalnum((unsigned char) *h) || *h == '_'))
    {
        h++;
    }
    size_t vlen = (size_t) (h - vstart);
    while (h < open_brace && isspace((unsigned char) *h))
    {
        h++;
    }
    if (vlen == 0 || vlen >= STRINGMAXLEN_IMAGE_NAME || h >= open_brace || *h != '=')
    {
        snprintf(err, errsz, "invalid for header (expected: for var = start:stop[:step] { ... })");
        return -1;
    }
    char vname[STRINGMAXLEN_IMAGE_NAME];
    memcpy(vname, vstart, vlen);
    vname[vlen] = '\0';
    h++;
    const char *colon1 = NULL;
    const char *colon2 = NULL;
    if (parse_for_range_colons(h, open_brace, &colon1, &colon2, err, errsz) != 0)
    {
        return -1;
    }
    const char *close_brace = find_matching_brace(open_brace, end);
    if (close_brace == NULL)
    {
        snprintf(err, errsz, "unclosed '{' in for block");
        return -1;
    }
    size_t      start_len = (size_t) (colon1 - h);
    const char *stop_s    = colon1 + 1;
    size_t      stop_len  = (size_t) ((colon2 != NULL ? colon2 : open_brace) - stop_s);
    const char *step_s    = (colon2 != NULL) ? (colon2 + 1) : "1";
    size_t      step_len  = (colon2 != NULL) ? (size_t) (open_brace - step_s) : 1;

    char init_buf[512];
    snprintf(init_buf, sizeof(init_buf), "%s = (%.*s)", vname, (int) start_len, h);
    if (add_stmt(p, init_buf, strlen(init_buf), err, errsz) != 0)
    {
        return -1;
    }
    if (p->n == PROG_MAXSTMT)
    {
        snprintf(err, errsz, "more than %d statements", PROG_MAXSTMT);
        return -1;
    }
    int        for_idx = p->n++;
    prog_stmt *fst     = &p->st[for_idx];
    fst->kind          = STMT_FOR_BEGIN;
    fst->block_id      = ++p->next_block_id;
    fst->text          = strdup("for");
    snprintf(fst->post, sizeof(fst->post), "%s", vname);
    char stop_vname[32];
    char step_vname[32];
    snprintf(stop_vname, sizeof(stop_vname), "__fstop%d", for_idx);
    snprintf(step_vname, sizeof(step_vname), "__fstep%d", for_idx);
    size_t g_len   = stop_len + sizeof(stop_vname) + 8;
    size_t step_bl = step_len + sizeof(step_vname) + 8;
    fst->guard     = malloc(g_len);
    fst->for_step  = malloc(step_bl);
    if (fst->guard == NULL || fst->for_step == NULL || fst->text == NULL)
    {
        return -1;
    }
    snprintf(fst->guard, g_len, "%s = (%.*s)", stop_vname, (int) stop_len, stop_s);
    snprintf(fst->for_step, step_bl, "%s = (%.*s)", step_vname, (int) step_len, step_s);
    fst->guard_vid    = create_variable_ID(stop_vname, 0.0);
    fst->for_step_vid = create_variable_ID(step_vname, 1.0);

    p->next_block_id++;
    if (split_block(p, open_brace + 1, (size_t) (close_brace - (open_brace + 1)), err, errsz) != 0)
    {
        return -1;
    }
    if (p->n == PROG_MAXSTMT)
    {
        snprintf(err, errsz, "more than %d statements", PROG_MAXSTMT);
        return -1;
    }
    int        end_idx = p->n++;
    prog_stmt *est     = &p->st[end_idx];
    est->kind          = STMT_FOR_END;
    est->block_id      = ++p->next_block_id;
    est->jump_idx      = for_idx;
    est->text          = strdup("endfor");
    snprintf(est->post, sizeof(est->post), "%s", vname);
    p->st[for_idx].jump_idx = p->n;
    p->next_block_id++;
    *out_next = close_brace + 1;
    return 0;
}

static const char *find_header_brace(const char *scan, const char *end, int stop_on_colon)
{
    int  pdepth = 0;
    char q      = 0;
    for (; scan < end; scan++)
    {
        if (q != 0)
        {
            q = (*scan == q) ? 0 : q;
        }
        else if (*scan == '"' || *scan == '\'')
        {
            q = *scan;
        }
        else if (*scan == '(' || *scan == '[')
        {
            pdepth++;
        }
        else if ((*scan == ')' || *scan == ']') && pdepth > 0)
        {
            pdepth--;
        }
        else if (pdepth == 0 && *scan == '{')
        {
            return scan;
        }
        else if (pdepth == 0 &&
                 ((stop_on_colon && *scan == ':') || *scan == ';' || *scan == '\n' || *scan == '#'))
        {
            break;
        }
    }
    return NULL;
}

static int split_block(cli_calc_prog *p, const char *text, size_t text_len, char *err, size_t errsz)
{
    const char *end = text + text_len;
    const char *c   = text;

    while (c < end)
    {
        c = skip_ws_comments(c, end);
        if (c >= end)
        {
            break;
        }
        if ((size_t) (end - c) >= 3 && strncmp(c, "if", 2) == 0 &&
            (isspace((unsigned char) c[2]) || c[2] == '('))
        {
            const char *ob = find_header_brace(c + 2, end, 1);
            if (ob != NULL)
            {
                if (parse_if_brace_chain(p, c + 2, ob, end, &c, err, errsz) != 0)
                {
                    return -1;
                }
                continue;
            }
        }
        else if ((size_t) (end - c) >= 4 && strncmp(c, "for", 3) == 0 &&
                 isspace((unsigned char) c[3]))
        {
            const char *ob = find_header_brace(c + 3, end, 0);
            if (ob != NULL)
            {
                if (parse_for_brace_block(p, c + 3, ob, end, &c, err, errsz) != 0)
                {
                    return -1;
                }
                continue;
            }
        }

        const char *st    = c;
        int         depth = 0;
        char        q     = 0;
        for (; c < end; c++)
        {
            if (q != 0)
            {
                q = (*c == q) ? 0 : q;
            }
            else if (*c == '"' || *c == '\'')
            {
                q = *c;
            }
            else if (*c == '(' || *c == '[')
            {
                depth++;
            }
            else if ((*c == ')' || *c == ']') && depth > 0)
            {
                depth--;
            }
            else if (*c == '#' || (depth == 0 && (*c == ';' || *c == '\n')))
            {
                break;
            }
        }
        if (add_stmt(p, st, (size_t) (c - st), err, errsz) != 0)
        {
            return -1;
        }
        if (c < end && *c == '#')
        {
            while (c < end && *c != '\n')
            {
                c++;
            }
        }
        if (c < end)
        {
            c++;
        }
    }
    return 0;
}

/**
 * cli_calc_prog_split - Split program source @text into individual statements and control blocks
 */
int cli_calc_prog_split(cli_calc_prog *p, const char *text, char *err, size_t errsz)
{
    return split_block(p, text, strlen(text), err, errsz);
}

/** text_reads_ident - 1 if @text names @ident, directly or through the body of a `def` it calls. */
static int text_reads_ident(const char *text, const char *ident, int depth)
{
    if (cli_calc_prog_text_mentions_ident(text, ident))
    {
        return 1;
    }
    if (text == NULL || depth >= CALC_UDEF_MAX)
    {
        return 0;
    }
    for (int u = 0; u < CALC_UDEF_MAX; u++)
    {
        const cli_calc_udef_t *d = cli_calc_udef_get(u);
        if (d == NULL)
        {
            break;
        }
        if (cli_calc_prog_text_mentions_ident(text, d->name) &&
            text_reads_ident(d->body, ident, depth + 1))
        {
            return 1;
        }
    }
    return 0;
}

/** stmt_reads_ident - 1 if statement @st (its text, guard, step or kept programs) reads image @ident. */
static int stmt_reads_ident(const prog_stmt *st, const char *ident)
{
    const char *text = st->text;
    if (text != NULL && strcmp(st->dst_name, ident) == 0 && strchr(text, '=') != NULL)
    {
        text = strchr(text, '=') +
               1; /* the statement assigns @ident: only its right-hand side reads */
    }
    return text_reads_ident(text, ident, 0) || text_reads_ident(st->guard, ident, 0) ||
           (st->kind == STMT_FOR_BEGIN && text_reads_ident(st->for_step, ident, 0)) ||
           cli_calc_kept_reads_image(st->kept, ident) ||
           cli_calc_kept_reads_image(st->guard_kept, ident) ||
           cli_calc_kept_reads_image(st->for_step_kept, ident);
}

/**
 * cli_calc_prog_merge_basic_blocks - Merge adjacent kept elementwise statements in the same block
 *
 * A `local` destination is written to memory unless no other statement reads it: a statement
 * already merged into an earlier pass (`STMT_NOP`) still counts, as does a read through a `def`.
 */
void cli_calc_prog_merge_basic_blocks(cli_calc_prog *p)
{
    int prev = -1;
    for (int i = 0; i < p->n; i++)
    {
        prog_stmt *st = &p->st[i];
        if (st->kind == STMT_NOP)
        {
            continue;
        }
        if (st->kind != STMT_NORMAL || st->guard != NULL || st->kept == NULL)
        {
            prev = -1;
            continue;
        }
        if (prev >= 0 && p->st[prev].block_id == st->block_id)
        {
            int elide_dst1 = 0;
            if (p->st[prev].is_local && p->st[prev].dst_name[0] != '\0')
            {
                elide_dst1 = 1;
                for (int k = 0; k < p->n; k++)
                {
                    if (k != prev && k != i && stmt_reads_ident(&p->st[k], p->st[prev].dst_name))
                    {
                        elide_dst1 = 0;
                        break;
                    }
                }
            }
            if (cli_calc_fuse_try_merge_kept(p->st[prev].kept, st->kept, elide_dst1))
            {
                st->merged_into      = prev + 1;
                p->st[prev].is_local = st->is_local;
                snprintf(p->st[prev].dst_name, sizeof(p->st[prev].dst_name), "%s", st->dst_name);
                cli_calc_kept_free(st->kept);
                st->kept = NULL;
                st->kind = STMT_NOP;
                continue;
            }
        }
        prev = i;
    }
}

/**
 * cli_calc_prog_assigns - Query whether a bound variable is assigned by the program
 */
int cli_calc_prog_assigns(const cli_calc_prog *p, int i)
{
    return (i >= 0 && i < p->nvars) ? p->assigned[i] : 0;
}

/**
 * cli_calc_prog_set_processinfo - Attach a PROCESSINFO handle for output stream updates
 */
void cli_calc_prog_set_processinfo(cli_calc_prog *p, void *processinfo)
{
    if (p != NULL)
    {
        p->processinfo = processinfo;
    }
}

/**
 * cli_calc_prog_free - Release all memory associated with a compiled calculator program
 */
void cli_calc_prog_free(cli_calc_prog *p)
{
    if (p == NULL)
    {
        return;
    }
    for (int i = 0; i < p->n; i++)
    {
        free(p->st[i].text);
        cli_calc_kept_free(p->st[i].kept);
        free(p->st[i].guard);
        cli_calc_kept_free(p->st[i].guard_kept);
        free(p->st[i].for_step);
        cli_calc_kept_free(p->st[i].for_step_kept);
    }
    free(p->vars);
    free(p->vid);
    free(p->assigned);
    free(p);
}
