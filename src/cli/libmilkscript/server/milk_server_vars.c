// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_vars.c
 * @brief Lossless bash <-> dcvar variable synchronization and inline entity expansion.
 *
 * Role & Goal:
 *   1. Moves scalar variables between the calling bash environment and milk's variable table
 *      (`data.core.variable[]` and `cli_vars[]`): the bash variables a calc expression names come in
 *      with the request; a command's declared outputs (CLICMDDATA.outvars) that it changed go back.
 *      A string cache (`g_dcvar_cached_str`) of the text last sent to bash keeps 64-bit values exact
 *      when bash sends that text back unchanged.
 *   2. Expands the entity references (`s.<stream>.<prop>`, `f.<fps>.<param>`, `p.<proc>.<field>`,
 *      `q.<seq>.<field>`) inside calc and test expressions before `cli_calc_eval_bash()` runs them.
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
        cli_var_set_text(vname, vval); /* a bash value is data, never a calc expression */
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
    const VARIABLE *v = &data.core.variable[vid];
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

void milk_server_outvars_send(int client_fd)
{
    for (int j = 0; j < g_noutq && client_fd >= 0; j++)
    {
        long vid = variable_ID(g_outq[j]);
        char valstr[256];
        if (vid < 0)
        {
            continue;
        }
        if (data.core.variable[vid].type == 1)
        {
            snprintf(valstr, sizeof(valstr), "%ld", data.core.variable[vid].value.l);
        }
        else if (data.core.variable[vid].type == 0)
        {
            snprintf(valstr, sizeof(valstr), "%.*g", cli_float_digits,
                     data.core.variable[vid].value.f);
        }
        else
        {
            snprintf(valstr, sizeof(valstr), "%s", data.core.variable[vid].value.s);
        }
        if (vid < DCVAR_CACHE_MAX)
        {
            snprintf(g_dcvar_cached_str[vid], sizeof(g_dcvar_cached_str[vid]), "%s", valstr);
        }
        send_setvar(client_fd, g_outq[j], valstr);
    }
    g_noutq = 0;
}

/**
 * @brief Expand the entity references (`s.im.xsize`, `f.fps.param`, ...) inside an expression.
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

        if ((isalpha((unsigned char) c) || c == '_') &&
            (ip == 0 || (!isalnum((unsigned char) src[ip - 1]) && src[ip - 1] != '_' &&
                         src[ip - 1] != '.' && src[ip - 1] != '@')))
        {
            size_t tstart = ip;
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
            if (ndots >= 2 && tlen < 256)
            {
                char tok[256];
                memcpy(tok, src + tstart, tlen);
                tok[tlen]         = '\0';
                const char *dot1  = strchr(tok, '.');
                size_t      nslen = (dot1 != NULL) ? (size_t) (dot1 - tok) : 0;
                if (milk_is_entity_ns_prefix(tok, nslen))
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
