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

/**
 * @brief Capture a snapshot of all current `dcvar` and `cli_vars` entries.
 *
 * @param out_nmax Receives `data.core.NB_MAX_VARIABLE` at snapshot time
 * @return Newly allocated snapshot array (caller must `free()`), or NULL
 */
dcvar_snap_t *milk_server_snapshot_dcvar(long *out_nmax)
{
    long nmax = data.core.NB_MAX_VARIABLE;
    if (nmax < 0)
    {
        nmax = 0;
    }
    long          total = nmax + CLI_MAX_VARS;
    dcvar_snap_t *snap  = (dcvar_snap_t *) calloc((size_t) total, sizeof(dcvar_snap_t));
    if (snap == NULL)
    {
        *out_nmax = 0;
        return NULL;
    }
    if (data.core.variable != NULL)
    {
        for (long i = 0; i < nmax; i++)
        {
            snap[i].used = data.core.variable[i].used;
            if (snap[i].used)
            {
                snap[i].type = data.core.variable[i].type;
                snprintf(snap[i].name, sizeof(snap[i].name), "%s", data.core.variable[i].name);
                if (snap[i].type == 0)
                {
                    snap[i].f = data.core.variable[i].value.f;
                }
                else if (snap[i].type == 1)
                {
                    snap[i].l = data.core.variable[i].value.l;
                }
                else
                {
                    snprintf(snap[i].s, sizeof(snap[i].s), "%s", data.core.variable[i].value.s);
                }
            }
        }
    }
    for (int j = 0; j < CLI_MAX_VARS; j++)
    {
        long idx       = nmax + j;
        snap[idx].used = cli_vars[j].used;
        if (cli_vars[j].used)
        {
            snap[idx].type = 2;
            snprintf(snap[idx].name, sizeof(snap[idx].name), "%s", cli_vars[j].name);
            snprintf(snap[idx].s, sizeof(snap[idx].s), "%s", cli_vars[j].val);
        }
    }
    *out_nmax = nmax;
    return snap;
}

/**
 * @brief Compare current `dcvar` and `cli_vars` against @snap and emit `MILK_MSG_SETVAR` frames.
 *
 * @param client_fd Connected client socket fd
 * @param snap      Snapshot taken before command execution
 * @param snap_nmax `NB_MAX_VARIABLE` recorded in @snap
 */
void milk_server_send_modified_dcvars(int client_fd, const dcvar_snap_t *snap, long snap_nmax)
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

            int changed = 0;
            if (snap == NULL || i >= snap_nmax || !snap[i].used ||
                strcmp(snap[i].name, vname) != 0 || snap[i].type != data.core.variable[i].type)
            {
                changed = 1;
            }
            else if (data.core.variable[i].type == 0 && snap[i].f != data.core.variable[i].value.f)
            {
                changed = 1;
            }
            else if (data.core.variable[i].type == 1 && snap[i].l != data.core.variable[i].value.l)
            {
                changed = 1;
            }
            else if (data.core.variable[i].type == 2 &&
                     strcmp(snap[i].s, data.core.variable[i].value.s) != 0)
            {
                changed = 1;
            }

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

            char   pkt[512];
            size_t n1 = strlen(vname) + 1;
            size_t n2 = strlen(valstr) + 1;
            if (n1 + n2 <= sizeof(pkt))
            {
                memcpy(pkt, vname, n1);
                memcpy(pkt + n1, valstr, n2);
                milk_proto_send(client_fd, MILK_MSG_SETVAR, 0, pkt, (uint32_t) (n1 + n2));
            }
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
        int changed = 0;
        if (snap == NULL)
        {
            changed = 1;
        }
        else
        {
            long idx = snap_nmax + j;
            if (!snap[idx].used || strcmp(snap[idx].name, vname) != 0 ||
                strcmp(snap[idx].s, cli_vars[j].val) != 0)
            {
                changed = 1;
            }
        }
        if (!changed)
        {
            continue;
        }
        char   pkt[600];
        size_t n1 = strlen(vname) + 1;
        size_t n2 = strlen(cli_vars[j].val) + 1;
        if (n1 + n2 <= sizeof(pkt))
        {
            memcpy(pkt, vname, n1);
            memcpy(pkt + n1, cli_vars[j].val, n2);
            milk_proto_send(client_fd, MILK_MSG_SETVAR, 0, pkt, (uint32_t) (n1 + n2));
        }
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
    return ((nslen == 1 && (s[0] == 's' || s[0] == 'f' || s[0] == 'p' || s[0] == 'q')) ||
            (nslen == 6 && strncmp(s, "stream", 6) == 0) ||
            (nslen == 3 && (strncmp(s, "fps", 3) == 0 || strncmp(s, "seq", 3) == 0)) ||
            (nslen == 4 && strncmp(s, "proc", 4) == 0) ||
            (nslen == 8 && strncmp(s, "procinfo", 8) == 0));
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
            size_t tstart = has_at ? (ip + 1) : ip;
            size_t end    = tstart;
            int    ndots  = 0;
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
            if (((has_at && ndots >= 1 && (end >= len || src[end] != ':')) || ndots >= 2) &&
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
    cli_expand_fpsvar(out, (int) outsz);
}
