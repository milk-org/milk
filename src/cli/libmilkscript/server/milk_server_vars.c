// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_vars.c
 * @brief Lossless bash <-> dcvar variable synchronization and inline entity expansion.
 *
 * Role & Goal:
 *   1. Synchronizes scalar variables between the calling bash environment and
 *      milk's internal variable table (`data.core.variable[]` and `cli_vars[]`).
 *      Maintains a string cache (`g_dcvar_cached_str`) of the formatted representation
 *      last sent to bash so that 64-bit IEEE-754 floating-point values are not
 *      degraded by decimal string round-tripping unless bash actually modified them.
 *   2. Expands inline `@s.<stream>.<prop>`, `@f.<fps>.<param>`, `@p.<proc>.<field>`,
 *      and `@q.<seq>.<field>` references inside arithmetic (`@=`) and boolean (`@test`)
 *      expressions prior to evaluation by `cli_calc_eval_bash()`.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"
#include "milk_proto.h"
#include "milk_server_internal.h"

#define DCVAR_CACHE_MAX 4096
static char g_dcvar_cached_str[DCVAR_CACHE_MAX][128];

/**
 * @brief Return 1 if @name is a read-only built-in mathematical/physical constant.
 *
 * @param name Variable name
 * @return 1 if built-in constant, 0 otherwise
 */
static int is_builtin_const_var(const char *name)
{
    if (name == NULL || name[0] != '_')
    {
        return 0;
    }
    return (strcmp(name, "_PI") == 0 || strcmp(name, "_e") == 0 || strcmp(name, "_gamma") == 0 ||
            strcmp(name, "_c") == 0 || strcmp(name, "_h") == 0 || strcmp(name, "_k") == 0 ||
            strcmp(name, "_pc") == 0 || strcmp(name, "_ly") == 0 || strcmp(name, "_AU") == 0);
}

/**
 * @brief Snapshot initial built-in `dcvar` float entries into the string cache.
 */
void milk_server_init_dcvar_cache(void)
{
    if (data.core.variable == NULL)
    {
        return;
    }
    for (long i = 0; i < data.core.NB_MAX_VARIABLE && i < DCVAR_CACHE_MAX; i++)
    {
        if (data.core.variable[i].used && data.core.variable[i].type == 0)
        {
            snprintf(g_dcvar_cached_str[i], sizeof(g_dcvar_cached_str[i]), "%.*g", cli_float_digits,
                     data.core.variable[i].value.f);
        }
    }
}

/**
 * @brief Unpack NUL-delimited bash scalar variables from @buf and update `dcvar`.
 *
 * @param buf       Payload buffer
 * @param len       Total payload length in bytes
 * @param nvars     Number of (name, value) pairs encoded at `*offset_io`
 * @param offset_io In/out byte offset into @buf
 */
void milk_server_sync_bash_vars(const uint8_t *buf,
                                uint32_t       len,
                                uint32_t       nvars,
                                uint32_t      *offset_io)
{
    uint32_t off = *offset_io;
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

        if (vname[0] == '\0' || is_builtin_const_var(vname))
        {
            continue;
        }
        /* Do not overwrite an existing image with a scalar variable */
        if (image_ID(vname, data.core.image, data.core.NB_MAX_IMAGE) != -1)
        {
            continue;
        }

        long vid = variable_ID(vname);
        if (vid >= 0 && vid < DCVAR_CACHE_MAX && g_dcvar_cached_str[vid][0] != '\0' &&
            strcmp(g_dcvar_cached_str[vid], vval) == 0)
        {
            /* Unchanged in bash since milk last wrote it: preserve exact 64-bit value */
            continue;
        }

        char *endp = NULL;
        long  lval = strtol(vval, &endp, 10);
        if (vval[0] != '\0' && endp != vval && *endp == '\0')
        {
            vid = create_variable_long_ID(vname, lval);
            if (vid >= 0 && vid < DCVAR_CACHE_MAX)
            {
                snprintf(g_dcvar_cached_str[vid], sizeof(g_dcvar_cached_str[vid]), "%s", vval);
            }
        }
        else
        {
            double dval = strtod(vval, &endp);
            if (vval[0] != '\0' && endp != vval && *endp == '\0')
            {
                vid = create_variable_ID(vname, dval);
                if (vid >= 0 && vid < DCVAR_CACHE_MAX)
                {
                    snprintf(g_dcvar_cached_str[vid], sizeof(g_dcvar_cached_str[vid]), "%s", vval);
                }
            }
        }
        cli_var_set(vname, vval);
    }
    *offset_io = off;
}

/* Change detection for SETVAR: one mark per variable slot, written only for slots in use. A slot
 * whose mark has an old epoch was unused at snapshot time. */
typedef struct
{
    uint64_t h;     /* hash of name, type and value */
    uint32_t epoch; /* snapshot that recorded @h */
} var_mark_t;

static var_mark_t  g_cli_mark[CLI_MAX_VARS];
static var_mark_t *g_dc_mark    = NULL;
static long        g_dc_mark_n  = 0;
static uint32_t    g_mark_epoch = 0;

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

static uint64_t dcvar_hash(const VARIABLE *v)
{
    uint64_t h = fnv1a(14695981039346656037ULL, v->name, strlen(v->name) + 1);
    h          = fnv1a(h, &v->type, sizeof(v->type));
    if (v->type == 0)
    {
        return fnv1a(h, &v->value.f, sizeof(v->value.f));
    }
    if (v->type == 1)
    {
        return fnv1a(h, &v->value.l, sizeof(v->value.l));
    }
    return fnv1a(h, v->value.s, strnlen(v->value.s, sizeof(v->value.s)));
}

static uint64_t clivar_hash(const CLI_VAR *v)
{
    uint64_t h = fnv1a(14695981039346656037ULL, v->name, strlen(v->name) + 1);
    return fnv1a(h, v->val, strlen(v->val));
}

/**
 * @brief Record the current `dcvar` and `cli_vars` values, to report changes after a command.
 */
void milk_server_snapshot_dcvar(void)
{
    long nmax = (data.core.variable != NULL && data.core.NB_MAX_VARIABLE > 0)
                    ? data.core.NB_MAX_VARIABLE
                    : 0;
    if (nmax > g_dc_mark_n)
    {
        var_mark_t *m = (var_mark_t *) realloc(g_dc_mark, (size_t) nmax * sizeof(var_mark_t));
        if (m == NULL)
        {
            nmax = g_dc_mark_n;
        }
        else
        {
            memset(m + g_dc_mark_n, 0, (size_t) (nmax - g_dc_mark_n) * sizeof(var_mark_t));
            g_dc_mark   = m;
            g_dc_mark_n = nmax;
        }
    }
    if (++g_mark_epoch == 0) /* wrapped: forget every mark */
    {
        memset(g_dc_mark, 0, (size_t) g_dc_mark_n * sizeof(var_mark_t));
        memset(g_cli_mark, 0, sizeof(g_cli_mark));
        g_mark_epoch = 1;
    }
    for (long i = 0; i < nmax; i++)
    {
        if (data.core.variable[i].used)
        {
            g_dc_mark[i].h     = dcvar_hash(&data.core.variable[i]);
            g_dc_mark[i].epoch = g_mark_epoch;
        }
    }
    for (int j = 0; j < CLI_MAX_VARS; j++)
    {
        if (cli_vars[j].used)
        {
            g_cli_mark[j].h     = clivar_hash(&cli_vars[j]);
            g_cli_mark[j].epoch = g_mark_epoch;
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
 * @brief Send a `MILK_MSG_SETVAR` frame for every variable changed since milk_server_snapshot_dcvar().
 *
 * @param client_fd Connected client socket fd
 */
void milk_server_send_modified_dcvars(int client_fd)
{
    long nmax = data.core.NB_MAX_VARIABLE;
    if (data.core.variable != NULL)
    {
        for (long i = 0; i < nmax; i++)
        {
            if (!data.core.variable[i].used)
            {
                continue;
            }
            const char *vname = data.core.variable[i].name;
            if (vname[0] == '\0' || is_builtin_const_var(vname))
            {
                continue;
            }

            int changed = (i >= g_dc_mark_n || g_dc_mark[i].epoch != g_mark_epoch ||
                           g_dc_mark[i].h != dcvar_hash(&data.core.variable[i]));
            if (!changed)
            {
                continue;
            }

            char valstr[256];
            if (data.core.variable[i].type == 1)
            {
                snprintf(valstr, sizeof(valstr), "%ld", data.core.variable[i].value.l);
            }
            else if (data.core.variable[i].type == 0)
            {
                snprintf(valstr, sizeof(valstr), "%.*g", cli_float_digits,
                         data.core.variable[i].value.f);
            }
            else
            {
                snprintf(valstr, sizeof(valstr), "%s", data.core.variable[i].value.s);
            }

            if (i < DCVAR_CACHE_MAX)
            {
                snprintf(g_dcvar_cached_str[i], sizeof(g_dcvar_cached_str[i]), "%s", valstr);
            }

            send_setvar(client_fd, vname, valstr);
        }
    }

    /* Also send string variables assigned via cli_var_set (e.g. _su = "hello") */
    for (int j = 0; j < CLI_MAX_VARS; j++)
    {
        if (!cli_vars[j].used)
        {
            continue;
        }
        const char *vname = cli_vars[j].name;
        if (vname[0] == '\0' || is_builtin_const_var(vname) || strncmp(vname, "_tmpstr", 7) == 0 ||
            strncmp(vname, "_tmpcalc", 8) == 0 || variable_ID(vname) != -1)
        {
            continue;
        }
        int changed =
            (g_cli_mark[j].epoch != g_mark_epoch || g_cli_mark[j].h != clivar_hash(&cli_vars[j]));
        if (!changed)
        {
            continue;
        }
        send_setvar(client_fd, vname, cli_vars[j].val);
    }
}

/**
 * @brief Check whether a namespace prefix of length @nslen is a valid entity namespace.
 *
 * @param s     Pointer to start of namespace token
 * @param nslen Length of namespace token before the first `.`
 * @return 1 if valid entity namespace (`s`, `stream`, `f`, `fps`, `p`, `proc`, `procinfo`, `q`, `seq`), 0 otherwise
 */
int milk_server_is_entity_ns_prefix(const char *s, size_t nslen)
{
    return milk_is_entity_ns_prefix(s, nslen);
}

/**
 * @brief Expand inline entity references (`@s.im.xsize`, `f.fps.param`, etc.) inside an expression.
 *
 * Numeric values are substituted bare; string values are enclosed in double quotes.
 * Also runs `cli_expand_fpsvar()` for legacy `${fps...}` / `${stream...}` tokens.
 *
 * @param in    Input expression string
 * @param out   Output buffer for expanded expression
 * @param outsz Size of @out in bytes
 */
void milk_server_expand_entities_in_expr(const char *in, char *out, size_t outsz)
{
    if (outsz == 0)
    {
        return;
    }
    const char *src   = (in != NULL) ? in : "";
    size_t      ip    = 0;
    size_t      op    = 0;
    size_t      len   = strlen(src);
    int         in_sq = 0;
    int         in_dq = 0;

    while (ip < len && op + 1 < outsz)
    {
        char c = src[ip];
        if (c == '\'' && !in_dq)
        {
            in_sq     = !in_sq;
            out[op++] = src[ip++];
            continue;
        }
        if (c == '"' && !in_sq)
        {
            in_dq     = !in_dq;
            out[op++] = src[ip++];
            continue;
        }
        if (in_sq || in_dq)
        {
            out[op++] = src[ip++];
            continue;
        }

        int has_at = (c == '@' && ip + 1 < len &&
                      (isalpha((unsigned char) src[ip + 1]) || src[ip + 1] == '_'));
        if ((has_at || isalpha((unsigned char) c) || c == '_') &&
            (ip == 0 || (!isalnum((unsigned char) src[ip - 1]) && src[ip - 1] != '_' &&
                         src[ip - 1] != '.' && src[ip - 1] != '@')))
        {
            size_t tstart    = has_at ? (ip + 1) : ip;
            size_t end       = tstart;
            int    ndots     = 0;
            int    has_colon = 0;
            if (has_at && end + 1 < len && (src[end] == 'S' || src[end] == 'L') &&
                src[end + 1] == ':')
            {
                has_colon = 1;
                end += 2;
            }
            while (end < len && (isalnum((unsigned char) src[end]) || src[end] == '_' ||
                                 src[end] == '.' || src[end] == '-'))
            {
                if (src[end] == '.')
                {
                    ndots++;
                }
                end++;
            }
            while (end > tstart && src[end - 1] == '.')
            {
                end--;
                ndots--;
            }
            size_t tlen = end - tstart;
            if (((has_at && ndots >= 1 && (has_colon || end >= len || src[end] != ':')) ||
                 ndots >= 2) &&
                tlen < 256)
            {
                char tok[256];
                memcpy(tok, src + tstart, tlen);
                tok[tlen]         = '\0';
                const char *dot1  = strchr(tok, '.');
                size_t      nslen = (dot1 != NULL) ? (size_t) (dot1 - tok) : 0;
                if (has_at || milk_server_is_entity_ns_prefix(tok, nslen))
                {
                    char val[512];
                    val[0] = '\0';
                    if (milk_shm_prop_get(tok, val, sizeof(val)) == 0 && val[0] != '\0')
                    {
                        char  *endp = NULL;
                        double dv   = strtod(val, &endp);
                        (void) dv;
                        if (endp != val && *endp == '\0')
                        {
                            size_t vl = strlen(val);
                            if (op + vl < outsz)
                            {
                                memcpy(out + op, val, vl);
                                op += vl;
                            }
                        }
                        else
                        {
                            int n = snprintf(out + op, outsz - op, "\"%s\"", val);
                            if (n > 0 && op + (size_t) n < outsz)
                            {
                                op += (size_t) n;
                            }
                        }
                        ip = end;
                        continue;
                    }
                }
            }
        }
        out[op++] = src[ip++];
    } // while (ip < len && op + 1 < outsz)

    out[op] = '\0';
}
