// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_script_expand_fps.c
 * @brief Legacy `.milk` script `@`-token property expansion.
 *
 * Role & Goal:
 *   Scans command-line strings in legacy `.milk` scripts for unquoted
 *   `@namespace.name.prop` read tokens and `@namespace.name.prop=val` write
 *   assignments, delegating all shared-memory resolution to the unified
 *   `shm_prop/` subsystem (`milk_shm_prop_get()` and `milk_shm_prop_set()`).
 *
 * Public API (declared in CLIcore_script.h):
 *   cli_expand_fpsvar()
 */

#include <ctype.h>
#include <stddef.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_script_expand_internal.h"
#include "milk_shm_prop.h"

/**
 * @brief Parse and execute an inline `@token=value` assignment.
 *
 * Parses the RHS value string (respecting quotes, backslash escapes, and
 * parentheses), recursively expands `@`, `$`, and `$((...))` tokens inside
 * the value, and delegates the write to `milk_shm_prop_set()`. If the assignment
 * starts at the beginning of the command line, emits a no-op `:` command so
 * the CLI dispatcher treats the line as a valid statement.
 *
 * @param line   Full command-line buffer
 * @param i      Current parse position (immediately after '='), updated
 * @param token  Entity path token before '=' (without leading '@')
 * @param out    Output buffer
 * @param opos   Current write offset in @out (updated)
 * @param maxlen Size of @out in bytes
 */
static void expand_fpsvar_write(char       *line,
                                int        *i,
                                const char *token,
                                char       *out,
                                int        *opos,
                                int         maxlen)
{
    char valstr[512];
    int  vlen  = 0;
    int  depth = 0;
    int  in_sq = 0;
    int  in_dq = 0;
    int  esc   = 0;
    int  pos   = *i;

    while (line[pos] == ' ' || line[pos] == '\t')
    {
        pos++;
    }

    while (line[pos] != '\0' && vlen < 511)
    {
        char ch = line[pos];

        if (!esc)
        {
            if (ch == '\\')
            {
                esc            = 1;
                valstr[vlen++] = ch;
                pos++;
                continue;
            }
            if (!in_sq && !in_dq)
            {
                if (ch == '(')
                {
                    depth++;
                }
                else if (ch == ')' && depth > 0)
                {
                    depth--;
                }
                if (depth == 0 && (ch == ';' || ch == '\n'))
                {
                    break;
                }
            }
            if (!in_dq && ch == '\'')
            {
                in_sq = !in_sq;
            }
            else if (!in_sq && ch == '"')
            {
                in_dq = !in_dq;
            }
        }
        else
        {
            esc = 0;
        }
        valstr[vlen++] = ch;
        pos++;
    }

    while (vlen > 0 && (valstr[vlen - 1] == ' ' || valstr[vlen - 1] == '\t'))
    {
        vlen--;
    }
    valstr[vlen] = '\0';

    cli_expand_fpsvar(valstr, (int) sizeof(valstr));
    cli_expand_env(valstr, (int) sizeof(valstr));
    cli_expand_arith(valstr, (int) sizeof(valstr));

    (void) milk_shm_prop_set(token, valstr, NULL);

    if (*opos == 0)
    {
        if (*opos < maxlen - 1)
        {
            out[(*opos)++] = ':';
        }
        if (*opos < maxlen - 1)
        {
            out[(*opos)++] = ' ';
        }
    }
    *i = pos;
}

/**
 * @brief Collect an `@`-prefixed entity token from `line[*i]`.
 *
 * Absorbs `[A-Za-z0-9_.*-]` as well as embedded `$VAR` or `${VAR}` references,
 * then expands any embedded `$VAR` via `cli_expand_env()`.
 *
 * @param line   Input command line
 * @param i      Input index (immediately after '@'), updated past token
 * @param token  Output token buffer
 * @param tokmax Size of @token in bytes
 * @return Length of @token in bytes
 */
static int collect_at_token(const char *line, int *i, char *token, int tokmax)
{
    int tlen = 0;
    while (line[*i] != '\0' && tlen < tokmax - 2)
    {
        char tc = line[*i];
        if (isalnum((unsigned char) tc) || tc == '_' || tc == '.' || tc == '-' || tc == '*')
        {
            token[tlen++] = line[(*i)++];
        }
        else if (tc == '$')
        {
            token[tlen++] = line[(*i)++];
            if (line[*i] == '{')
            {
                token[tlen++] = line[(*i)++];
                while (line[*i] != '\0' && line[*i] != '}' && tlen < tokmax - 2)
                {
                    token[tlen++] = line[(*i)++];
                }
                if (line[*i] == '}' && tlen < tokmax - 1)
                {
                    token[tlen++] = line[(*i)++];
                }
            }
            else
            {
                while (line[*i] != '\0' && tlen < tokmax - 2 &&
                       (isalnum((unsigned char) line[*i]) || line[*i] == '_'))
                {
                    token[tlen++] = line[(*i)++];
                }
            }
        }
        else
        {
            break;
        }
    }
    token[tlen] = '\0';

    if (strchr(token, '$') != NULL)
    {
        cli_expand_env(token, tokmax);
        tlen = (int) strlen(token);
    }
    return tlen;
}

/**
 * @brief Expand `@namespace.name.prop` tokens in place across a command line.
 *
 * Expansion is suppressed inside single/double quotes and after backslash
 * escapes. Tokens without a dot (such as stream type modifiers `@TF32:name`)
 * are passed through unchanged.
 *
 * @param line   Command-line buffer (modified in place)
 * @param maxlen Buffer capacity in bytes
 */
void cli_expand_fpsvar(char *line, int maxlen)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos      = 0;
    int  i         = 0;
    int  in_single = 0;
    int  in_double = 0;
    int  out_esc   = 0;

    while (line[i] != '\0' && opos < maxlen - 1)
    {
        char c = line[i];

        if (out_esc)
        {
            out_esc     = 0;
            out[opos++] = c;
            i++;
            continue;
        }
        if (c == '\\' && !in_single)
        {
            out_esc     = 1;
            out[opos++] = c;
            i++;
            continue;
        }
        if (c == '\'' && !in_double)
        {
            in_single   = !in_single;
            out[opos++] = c;
            i++;
            continue;
        }
        if (c == '"' && !in_single)
        {
            in_double   = !in_double;
            out[opos++] = c;
            i++;
            continue;
        }

        if (c == '@' && !in_single && !in_double)
        {
            i++; /* skip '@' */
            char token[512];
            int  tlen = collect_at_token(line, &i, token, (int) sizeof(token));

            /* Write path: @token=val */
            if (line[i] == '=' && strchr(token, '.') != NULL)
            {
                i++;
                expand_fpsvar_write(line, &i, token, out, &opos, maxlen);
                continue;
            }

            /* No dot: pass '@token' through verbatim (e.g. @TF32:im) */
            if (strchr(token, '.') == NULL)
            {
                if (opos < maxlen - 1)
                {
                    out[opos++] = '@';
                }
                int clen = tlen;
                if (opos + clen > maxlen - 1)
                {
                    clen = maxlen - 1 - opos;
                }
                memcpy(out + opos, token, (size_t) clen);
                opos += clen;
                continue;
            }

            char vstr[512];
            if (milk_shm_prop_get(token, vstr, sizeof(vstr)) == 0)
            {
                int vlen  = (int) strlen(vstr);
                int avail = maxlen - 1 - opos;
                int clen  = (vlen < avail) ? vlen : avail;
                memcpy(out + opos, vstr, (size_t) clen);
                opos += clen;
            }
        }
        else
        {
            out[opos++] = line[i++];
        }
    } // while (line[i] != '\0')

    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}
