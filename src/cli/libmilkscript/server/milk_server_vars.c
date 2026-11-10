// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_vars.c
 * @brief Lossless bash <-> dcvar variable synchronization.
 *
 * Purpose:     calculator text run by the session sees the shell variables it names, and the
 *              outputs a command declares reach the shell, without losing digits on the way.
 * Does:        stores the shell variables a request carries (a number in `dcvar[]`, text in
 *              `cli_vars[]`; never over an image of that name), keeping a variable's exact value
 *              when the text is what milk printed for it; takes the shell's MILK_DPDIGITS for the
 *              request; fingerprints a command's declared outputs (CLICMDDATA.outvars) before it
 *              runs, and sends those it changed back as SETVAR; tells whether a line of milk-cli's
 *              calc mode can run in milk-cli's process, its names known and synced since the
 *              last bash command.
 * Used by:     milk_server_eval.c (CALC and TEST), milk_server_dispatch.c (CALL);
 *              milk_bash_coproc.c and milk_bash_exec.c (calc mode). Declared in
 *              milk_server_internal.h and milk_server.h.
 * Uses:        CLIcore.h and CLIcore_script.h (dcvar[], cli_var_set()), COREMOD_memory,
 *              front/mc_ast.h and mc_func.h (the calculator's tokens and names), milk_proto.h.
 * See also:    milk_client_conn.c (the shell variables a request carries);
 *              docs/guide/calculator.md, "Calculator (`@calc`, `@test`, `@pipe`)".
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "front/mc_ast.h"
#include "mc_func.h"
#include "plan/mc_plan.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"
#include "milk_proto.h"
#include "milk_server_internal.h"

#define DCVAR_CACHE_MAX 4096
static uint32_t g_bash_epoch = 1;
static uint32_t g_var_sync_epoch[DCVAR_CACHE_MAX];

/**
 * milk_server_reset_bash_epoch - Reset the bash variable synchronization epoch
 *
 * Called when a fresh bash co-process is spawned so `g_bash_epoch` restarts at 1
 * (indicating no user bash commands have executed yet) and all per-variable sync
 * stamps are cleared.
 *
 * Return: void
 */
void milk_server_reset_bash_epoch(void)
{
    g_bash_epoch = 1;
    memset(g_var_sync_epoch, 0, sizeof(g_var_sync_epoch));
}

/**
 * milk_server_bump_bash_epoch - Advance the bash variable synchronization epoch
 *
 * Invoked whenever a bash command executes in the co-process so subsequent Calc
 * mode expressions know that shell variables may have been created or modified.
 *
 * Return: void
 */
void milk_server_bump_bash_epoch(void)
{
    if (++g_bash_epoch <= 1)
    {
        memset(g_var_sync_epoch, 0, sizeof(g_var_sync_epoch));
        g_bash_epoch = 2;
    }
}

/** var_text() - the text milk prints for the value of dcvar entry @vid (to bash, with the shell's
 * digits) into @buf. */
static const char *var_text(long vid, char *buf, size_t sz)
{
    const VARIABLE *v = &dcvar[vid];
    if (v->type == 1)
    {
        snprintf(buf, sz, "%ld", v->value.l);
    }
    else if (v->type == 0)
    {
        snprintf(buf, sz, "%.*g", cli_float_digits, v->value.f);
    }
    else
    {
        snprintf(buf, sz, "%s", v->value.s);
    }
    return buf;
}

/**
 * @brief Unpack NUL-delimited bash scalar variables from @buf and update `dcvar`.
 *
 * The shell's MILK_DPDIGITS sets the digits this request prints with (15 when absent): digits
 * are per shell.
 *
 * @param buf       Payload buffer
 * @param len       Total payload length in bytes
 * @param nvars     Number of (name, value) pairs encoded at `*offset_io`
 * @param offset_io In/out byte offset into @buf
 */
void milk_server_sync_bash_vars(const uint8_t *buf,
                                uint32_t       len,
                                uint32_t       nvars,
                                uint32_t      *offset_io,
                                int            is_repl)
{
    uint32_t off     = *offset_io;
    cli_float_digits = 15;
    for (uint32_t i = 0; i < nvars && off < len; i++)
    {
        const char *vname = (const char *) (buf + off);
        size_t      nlen  = strnlen(vname, len - off);
        off += (uint32_t) (nlen + 1);
        if (off > len)
        {
            break;
        }
        const char *vval = (const char *) (buf + off);
        size_t      vlen = strnlen(vval, len - off);
        off += (uint32_t) (vlen + 1);

        if (strcmp(vname, "MILK_DPDIGITS") == 0)
        {
            long nd          = strtol(vval, NULL, 10);
            cli_float_digits = (nd >= 1 && nd <= 17) ? (int) nd : 15;
            continue;
        }
        if (vname[0] == '\0' || mc_is_reserved_const(vname)) /* pi, nan, inf */
        {
            continue;
        }
        /* Do not overwrite an existing image with a scalar variable */
        if (image_ID(vname, dcimg, dcnimg) != -1)
        {
            continue;
        }

        long vid = variable_ID(vname);
        char txt[256];
        if (is_repl && vid >= 0 && vid < DCVAR_CACHE_MAX && g_var_sync_epoch[vid] == g_bash_epoch)
        {
            /* Already synced or modified in Calc REPL mode during the current bash epoch */
            continue;
        }
        if (vid >= 0 && strcmp(var_text(vid, txt, sizeof(txt)), vval) == 0)
        {
            /* What milk prints for its value (a command's output, ...): keep the exact value */
            if (vid < DCVAR_CACHE_MAX)
            {
                g_var_sync_epoch[vid] = g_bash_epoch;
            }
            continue;
        }

        /* One store per name: a number in dcvar, text in the string table */
        const char *vp = vval;
        while (*vp == ' ' || *vp == '\t')
        {
            vp++;
        }
        if (*vp == '+' || *vp == '-')
        {
            vp++;
        }
        int is_num_start =
            isdigit((unsigned char) *vp) || (*vp == '.' && isdigit((unsigned char) vp[1]));
        int   is_hex_num = (vp[0] == '0' && (vp[1] == 'x' || vp[1] == 'X'));
        char *endl       = NULL;
        char *endd       = NULL;
        errno            = 0;
        long lval        = strtol(vval, &endl, 10);
        int  lok         = (vval[0] != '\0' && *endl == '\0' && errno == 0);
        errno            = 0;
        double dval      = strtod(vval, &endd);
        int dok = (vval[0] != '\0' && is_num_start && !is_hex_num && *endd == '\0' && errno == 0 &&
                   !isnan(dval) && !isinf(dval));
        if (lok || dok)
        {
            cli_var_set(vname, NULL);
            vid = lok ? create_variable_long_ID(vname, lval) : create_variable_ID(vname, dval);
            if (vid >= 0 && vid < DCVAR_CACHE_MAX)
            {
                g_var_sync_epoch[vid] = g_bash_epoch;
            }
        }
        else
        {
            if (vid >= 0 && !mc_var_bound(vid)) /* a unit's bound variable keeps its place */
            {
                delete_variable_ID(vname);
            }
            cli_var_set(vname, vval);
        }
    }
    *offset_io = off;
}

/* Declared outputs the commands of the current request changed, sent back by milk_server_outvars_send() */
static char g_outq[MILK_OUTVARS_MAX][80];
static int  g_noutq = 0;

/** next_outvar() - copy the next name of space-separated list *@p into @name; 0 at the end. */
static int next_outvar(const char **p, char *name, size_t sz)
{
    size_t n = 0;

    *p += strspn(*p, " ");
    while (**p != '\0' && **p != ' ')
    {
        if (n + 1 < sz)
        {
            name[n++] = **p;
        }
        (*p)++;
    }
    name[n] = '\0';
    return n > 0;
}

/** fnv1a() - continue a 64-bit FNV-1a hash over @n bytes. */
static uint64_t fnv1a(uint64_t h, const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++)
    {
        h = (h ^ b[i]) * 1099511628211ULL;
    }
    return h;
}

/** outvar_hash() - fingerprint of variable @name's type and value; 0 if it does not exist. */
static uint64_t outvar_hash(const char *name)
{
    long vid = variable_ID(name);
    if (vid < 0)
    {
        return 0;
    }
    const VARIABLE *v = &dcvar[vid];
    uint64_t        h = fnv1a(14695981039346656037ULL, &v->type, sizeof(v->type));
    if (v->type == 0)
    {
        h = fnv1a(h, &v->value.f, sizeof(v->value.f));
    }
    else if (v->type == 1)
    {
        h = fnv1a(h, &v->value.l, sizeof(v->value.l));
    }
    else
    {
        h = fnv1a(h, v->value.s, strnlen(v->value.s, sizeof(v->value.s)));
    }
    return h | 1;
}

/**
 * milk_server_outvars_snap - Fingerprint a command's declared output variables before execution
 * @outvars: Space-separated list of output variable names (`CLICMDDATA.outvars`)
 * @h:       Output array (up to `MILK_OUTVARS_MAX` elements) receiving 64-bit hashes
 *
 * Computes a pre-execution hash of each variable listed in @outvars so
 * `milk_server_outvars_diff` can detect which variables were created or modified
 * by the command.
 *
 * Return: Number of output variable hashes recorded in @h
 */
int milk_server_outvars_snap(const char *outvars, uint64_t *h)
{
    char name[80];
    int  n = 0;

    while (n < MILK_OUTVARS_MAX && next_outvar(&outvars, name, sizeof(name)))
    {
        h[n++] = outvar_hash(name);
    }
    return n;
}

/**
 * milk_server_outvars_diff - Queue declared output variables whose values changed during a command
 * @outvars: Space-separated list of output variable names (`CLICMDDATA.outvars`)
 * @h:       Pre-execution hash array populated by `milk_server_outvars_snap`
 * @n:       Number of entries in @h
 *
 * Compares each variable's current hash against @h and appends modified variable
 * names to `g_outq` (deduplicated across multi-command batches) for transmission
 * back to the client shell.
 *
 * Return: void
 */
void milk_server_outvars_diff(const char *outvars, const uint64_t *h, int n)
{
    char name[80];

    for (int i = 0; i < n && next_outvar(&outvars, name, sizeof(name)); i++)
    {
        int queued = (h[i] == outvar_hash(name)); /* unchanged: nothing to send */
        for (int j = 0; j < g_noutq && !queued; j++)
        {
            queued = (strcmp(g_outq[j], name) == 0);
        }
        if (!queued && g_noutq < MILK_OUTVARS_MAX)
        {
            snprintf(g_outq[g_noutq++], sizeof(g_outq[0]), "%s", name);
        }
    }
}

/** send_setvar() - send one SETVAR frame (`name\0value\0`), whatever the value length. */
static void send_setvar(int client_fd, const char *vname, const char *val)
{
    size_t   n1  = strlen(vname) + 1;
    size_t   n2  = strlen(val) + 1;
    uint8_t *pkt = (uint8_t *) malloc(n1 + n2);
    if (pkt == NULL)
    {
        return;
    }
    memcpy(pkt, vname, n1);
    memcpy(pkt + n1, val, n2);
    milk_proto_send(client_fd, MILK_MSG_SETVAR, 0, pkt, (uint32_t) (n1 + n2));
    free(pkt);
}

/**
 * milk_server_outvars_send - Transmit queued output variable updates to the client and reset queue
 * @client_fd: Connected client socket file descriptor
 *
 * Formats each variable in `g_outq` (var_text(): when bash sends that text back, the exact value
 * is kept), sends a `MILK_MSG_SETVAR` frame to @client_fd, and clears `g_noutq`.
 *
 * Return: void
 */
void milk_server_outvars_send(int client_fd)
{
    for (int j = 0; j < g_noutq && client_fd >= 0; j++)
    {
        long vid = variable_ID(g_outq[j]);
        char valstr[256];
        if (vid >= 0)
        {
            send_setvar(client_fd, g_outq[j], var_text(vid, valstr, sizeof(valstr)));
        }
    }
    g_noutq = 0;
}

/**
 * milk_server_mark_expr_vars_synced - Stamp all `dcvar` identifiers in @expr with `g_bash_epoch`
 * @expr: Calculator expression string that was just evaluated
 *
 * Marks each name of @expr (its calculator tokens) that is a variable in `dcvar[]`
 * as synchronized for the current `g_bash_epoch`, enabling subsequent Calc mode lines to
 * evaluate in-process.
 *
 * Return: void
 */
void milk_server_mark_expr_vars_synced(const char *expr)
{
    mc_tok_list_t tl = { 0 };
    int           nt = (expr != NULL && mc_lex(expr, &tl) == 0) ? tl.ntok : 0;
    for (int t = 0; t < nt; t++)
    {
        const char *id = tl.toks[t].sval;
        if (tl.toks[t].type == MC_TOK_IDENT && strchr(id, '.') == NULL)
        {
            long vid = variable_ID(id);
            if (vid >= 0 && vid < DCVAR_CACHE_MAX)
            {
                g_var_sync_epoch[vid] = g_bash_epoch;
            }
        }
    }
    mc_tok_list_free(&tl);
}

/**
 * milk_server_can_exec_calc_inproc - Check if @expr can run in-process without a bash round-trip
 * @expr: Calculator REPL expression to inspect
 *
 * Verifies that @expr lexes as calculator text with no shell expansions (`$`, backticks,
 * backslashes, `@{`) and that every name it references is either a function call, a constant,
 * an entity property path (`s.im.xsize`, `f.fps.param`, ...), an existing image in
 * `dcimg[]`, a variable in `dcvar[]` already synchronized at
 * `g_bash_epoch`, or a new LHS variable when no user bash commands have run yet.
 *
 * Return: 1 if @expr can be evaluated directly in `milk-cli`, 0 if bash sync is required.
 */
int milk_server_can_exec_calc_inproc(const char *expr)
{
    char          lhs_defs[16][64];
    int           nlhs       = 0;
    const char   *pending    = NULL; /* the target of the statement's `name =` */
    int           depth      = 0;
    int           stmt_start = 1;
    mc_tok_list_t tl         = { 0 };
    int           ok         = (expr != NULL && expr[0] != '\0' && strpbrk(expr, "\\$`") == NULL &&
                                mc_lex(expr, &tl) == 0);

    for (int t = 0; ok && t < tl.ntok; t++) /* the lexer adds EOF tokens: t + 1 is one */
    {
        const mc_tok_t *k  = &tl.toks[t];
        const mc_tok_t *nx = &tl.toks[t + 1];
        if ((k->type == MC_TOK_NEWLINE || k->type == MC_TOK_SEMI) && depth == 0)
        {
            if (pending != NULL && nlhs < 16)
            {
                snprintf(lhs_defs[nlhs++], sizeof(lhs_defs[0]), "%s", pending);
            }
            pending    = NULL;
            stmt_start = 1;
            continue;
        }
        depth += (k->type == MC_TOK_LPAREN || k->type == MC_TOK_LBRACK);
        depth -= (depth > 0 && (k->type == MC_TOK_RPAREN || k->type == MC_TOK_RBRACK));
        ok = !(k->type == MC_TOK_MATMUL && nx->type == MC_TOK_LBRACE);  /* @{...} */
        if (ok && k->type == MC_TOK_IDENT && nx->type != MC_TOK_LPAREN) /* not a function call */
        {
            const char *id    = k->sval;
            const char *dot   = strchr(id, '.');
            int         ndots = 0;
            for (const char *d = dot; d != NULL; d = strchr(d + 1, '.'))
            {
                ndots++;
            }
            if (dot != NULL) /* an entity path (s.cam.cnt0) */
            {
                ok = (ndots >= 2 && milk_is_entity_ns_prefix(id, (size_t) (dot - id)));
            }
            else
            {
                long vid = variable_ID(id);
                ok = mc_is_reserved_const(id) || image_ID(id, dcimg, dcnimg) != -1 ||
                     (vid >= 0 && vid < DCVAR_CACHE_MAX && g_var_sync_epoch[vid] == g_bash_epoch);
                for (int j = 0; j < nlhs && !ok; j++)
                {
                    ok = (strcmp(lhs_defs[j], id) == 0);
                }
                if (!ok && stmt_start && depth == 0 && nx->type == MC_TOK_ASSIGN && strlen(id) < 64)
                {
                    pending = id;
                    ok      = 1;
                }
            }
        }
        stmt_start = 0;
    }
    mc_tok_list_free(&tl);
    return ok;
}
