// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_script_expand_env.c
 *
 * @brief Environment and CLI variable expansion.
 *
 * Implements cli_expand_env(), which expands
 * $VAR and ${VAR} references in a command-line
 * buffer:
 *
 *   $VAR          — plain variable
 *   ${VAR}        — braced variable
 *   ${#VAR}       — string length
 *   ${VAR:-def}   — default value if unset
 *   ${VAR:=def}   — assign and use default
 *   ${VAR:?msg}   — error if unset
 *   ${VAR:+alt}   — alternate if set
 *   ${VAR:off:len} — substring extraction
 *   ${arr[@]}     — all array/assoc elements
 *   ${arr[i]}     — indexed array element
 *   ${@fps.param} — FPS property inline access
 *
 * CLI variables are looked up first; environment
 * variables are the fallback.
 *
 * $((  and $(  sequences are passed through
 * unchanged so that cli_expand_arith() and the
 * command-substitution pipeline can handle them.
 *
 * Public API (declared in CLIcore_script.h):
 *   cli_expand_env()
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"


/* ============================================================
 *  emit_str_local — append a C string to output buffer
 * ============================================================ */

/**
 * emit_str_local - append a C string to the output buffer
 * @out:    Destination buffer
 * @opos:   Current write position (updated)
 * @maxlen: Buffer size
 * @s:      NUL-terminated source string
 *
 * Writes characters until NUL or the buffer is full.
 */
static void emit_str_local(char *out, int *opos, int maxlen, const char *s)
{
    while (*s != '\0' && *opos < maxlen - 1)
    {
        out[(*opos)++] = *s++;
    }
}


/* ============================================================
 *  expand_env_array_all — expand ${arr[@]}
 * ============================================================ */

/**
 * expand_env_array_all - collect all elements of an array
 *      or associative array into a space-separated string.
 * @varname:     Name of the array variable
 * @is_length:   If non-zero, set *out_val to element count
 * @out_buf:     Caller buffer (STRINGMAXLEN_CLICMDLINE)
 * @out_val:     Set to out_buf on success, NULL if not found
 *
 * Searches associative arrays first, then indexed arrays.
 * When @is_length is set, writes the element count into
 * @out_buf instead of the concatenated values.
 */
static void expand_env_array_all(const char  *varname,
                                 int          is_length,
                                 char        *out_buf,
                                 const char **out_val)
{
    *out_val        = NULL;
    int elems_count = 0;
    out_buf[0]      = '\0';

    /* Search associative arrays first */
    for (int a = 0; a < CLI_MAX_ASSOC; a++)
    {
        if (!cli_assoc[a].used)
        {
            continue;
        }
        if (strcmp(cli_assoc[a].name, varname) != 0)
        {
            continue;
        }

        elems_count = cli_assoc[a].nelem;
        if (!is_length)
        {
            for (int e = 0; e < cli_assoc[a].nelem; e++)
            {
                if (e > 0)
                {
                    strncat(out_buf, " ", STRINGMAXLEN_CLICMDLINE - strlen(out_buf) - 1);
                }
                strncat(out_buf, cli_assoc[a].vals[e],
                        STRINGMAXLEN_CLICMDLINE - strlen(out_buf) - 1);
            }
        }
        *out_val = out_buf;
        break;
    }

    /* Fallback: indexed arrays */
    if (*out_val == NULL)
    {
        for (int a = 0; a < CLI_MAX_ARRAYS; a++)
        {
            if (!cli_arrays[a].used)
            {
                continue;
            }
            if (strcmp(cli_arrays[a].name, varname) != 0)
            {
                continue;
            }

            elems_count = cli_arrays[a].nelem;
            if (!is_length)
            {
                for (int e = 0; e < cli_arrays[a].nelem; e++)
                {
                    if (e > 0)
                    {
                        strncat(out_buf, " ", STRINGMAXLEN_CLICMDLINE - strlen(out_buf) - 1);
                    }
                    strncat(out_buf, cli_arrays[a].elem[e],
                            STRINGMAXLEN_CLICMDLINE - strlen(out_buf) - 1);
                }
            }
            *out_val = out_buf;
            break;
        }
    }

    /* When is_length, overwrite buf with the element count.
     * elems_count is 0 when no array was found, giving "0". */
    if (is_length)
    {
        char cnt_buf[32];
        snprintf(cnt_buf, sizeof(cnt_buf), "%d", elems_count);
        strncpy(out_buf, cnt_buf, STRINGMAXLEN_CLICMDLINE - 1);
        out_buf[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    }

    /* Always point out_val at out_buf: empty string when the
     * array was not found, "0" for is_length on a missing array,
     * preserving shell-like semantics. */
    *out_val = out_buf;
}


/* ============================================================
 *  expand_env_array_index — expand ${arr[n]} / ${assoc[key]}
 * ============================================================ */

/**
 * expand_env_array_index - look up a single element by index
 *      or key from a CLI array or associative array.
 * @varname:   Name of the array variable
 * @idx_val:   Index string (numeric for arrays, key for assoc)
 * @out_val:   Set to the element value, or NULL if not found
 *
 * Searches associative arrays first (by key), then indexed
 * arrays (by numeric index).
 */
static void expand_env_array_index(const char *varname, const char *idx_val, const char **out_val)
{
    *out_val = NULL;

    /* Search associative arrays by key */
    for (int a = 0; a < CLI_MAX_ASSOC; a++)
    {
        if (!cli_assoc[a].used)
        {
            continue;
        }
        if (strcmp(cli_assoc[a].name, varname) != 0)
        {
            continue;
        }

        for (int e = 0; e < cli_assoc[a].nelem; e++)
        {
            if (strcmp(cli_assoc[a].keys[e], idx_val) == 0)
            {
                *out_val = cli_assoc[a].vals[e];
                break;
            }
        }
        break; /* found the array, stop outer search */
    }

    if (*out_val != NULL)
    {
        return;
    }

    /* Fallback: indexed arrays by numeric index */
    int num_idx = atoi(idx_val);
    for (int a = 0; a < CLI_MAX_ARRAYS; a++)
    {
        if (!cli_arrays[a].used)
        {
            continue;
        }
        if (strcmp(cli_arrays[a].name, varname) != 0)
        {
            continue;
        }

        if (num_idx >= 0 && num_idx < cli_arrays[a].nelem)
        {
            *out_val = cli_arrays[a].elem[num_idx];
        }
        break;
    }
}


/* ============================================================
 *  apply_modifier — apply :- := :? :+ :off:len modifiers
 * ============================================================ */

/**
 * apply_modifier - apply a ${VAR:op:arg} modifier to *val.
 * @varname:      Variable name (used by := to store defaults)
 * @mod_op:       Two-char operator string, e.g. ":-", ":="
 * @mod_arg:      Argument after the operator
 * @val_buf:      Caller-supplied scratch buffer
 * @val_buf_size: Size of @val_buf in bytes
 * @val:          Pointer to current value; updated in place
 *
 * Supported operators:
 *   :-  use mod_arg if val is NULL or empty
 *   :=  same as :- but also stores into the variable
 *   :?  print error if val is NULL or empty
 *   :+  use mod_arg if val is set, empty otherwise
 *   :   substring ${VAR:off} or ${VAR:off:len}
 */
static void apply_modifier(const char  *varname,
                           const char  *mod_op,
                           char        *mod_arg,
                           char        *val_buf,
                           int          val_buf_size,
                           const char **val)
{
    char op = mod_op[1]; /* '-', '=', '?', '+', or '\0' */

    if (op == '-')
    {
        if (*val == NULL || (*val)[0] == '\0')
        {
            *val = mod_arg;
        }
        return;
    }

    if (op == '=')
    {
        if (*val == NULL || (*val)[0] == '\0')
        {
            *val = mod_arg;
            cli_var_set(varname, *val);
        }
        return;
    }

    if (op == '?')
    {
        if (*val == NULL || (*val)[0] == '\0')
        {
            printf("CLI expand error: %s: %s\n", varname, mod_arg);
            *val = "";
        }
        return;
    }

    if (op == '+')
    {
        *val = (*val != NULL && (*val)[0] != '\0') ? mod_arg : "";
        return;
    }

    /* Plain ':' — substring ${VAR:off} or ${VAR:off:len} */
    if (*val == NULL)
    {
        return;
    }

    int offset = 0;
    int length = 255;

    char *colon = strchr(mod_arg, ':');
    if (colon != NULL)
    {
        *colon         = '\0';
        const char *lv = cli_var_lookup(mod_arg);
        offset         = atoi(lv ? lv : mod_arg);
        const char *rv = cli_var_lookup(colon + 1);
        length         = atoi(rv ? rv : colon + 1);
    }
    else
    {
        const char *lv = cli_var_lookup(mod_arg);
        offset         = atoi(lv ? lv : mod_arg);
    }

    int vlen = (int) strlen(*val);

    if (offset < 0)
    {
        offset = vlen + offset;
    }
    if (offset < 0)
    {
        offset = 0;
    }
    if (offset > vlen)
    {
        offset = vlen;
    }
    if (length < 0)
    {
        length = vlen - offset + length;
    }
    if (length < 0)
    {
        length = 0;
    }
    if (offset + length > vlen)
    {
        length = vlen - offset;
    }
    /* Clamp to val_buf capacity to prevent overflow */
    if (length > val_buf_size - 1)
    {
        length = val_buf_size - 1;
    }

    strncpy(val_buf, *val + offset, (size_t) length);
    val_buf[length] = '\0';
    *val            = val_buf;
}


/* ============================================================
 *  cli_expand_env — public entry point
 * ============================================================ */

/**
 * @brief Expand $VAR / ${VAR} references in place
 *
 * Performs standard shell-style variable expansion,
 * including array indexing, substring, and default-
 * value modifiers. CLI variables take precedence
 * over environment variables.
 *
 * $((  and $(  prefixes are passed through so that
 * arithmetic and command-substitution pipelines can
 * handle them downstream.
 *
 * @param line    Buffer to expand in-place
 * @param maxlen  Buffer size
 */
void cli_expand_env(char *line, int maxlen)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i    = 0;

    while (line[i] != '\0' && opos < maxlen - 1)
    {
        /* ---- Escaped dollar: pass through unchanged ---- */
        if (line[i] == '\\' && line[i + 1] == '$')
        {
            if (opos < maxlen - 1)
            {
                out[opos++] = line[i++];
            }
            if (opos < maxlen - 1)
            {
                out[opos++] = line[i++];
            }
            continue;
        }

        /* ---- Non-dollar: copy verbatim ---- */
        if (line[i] != '$')
        {
            out[opos++] = line[i++];
            continue;
        }

        /* ---- Dollar sign: begin expansion ---- */

        /* Skip $((  — arithmetic, handled downstream */
        if (line[i + 1] == '(' && line[i + 2] == '(')
        {
            out[opos++] = line[i++];
            out[opos++] = line[i++];
            out[opos++] = line[i++];
            continue;
        }
        /* Skip $(  — command substitution, handled downstream */
        if (line[i + 1] == '(')
        {
            out[opos++] = line[i++];
            out[opos++] = line[i++];
            continue;
        }

        i++; /* consume the '$' */

        int has_brace = (line[i] == '{');
        if (has_brace)
        {
            i++;
        }

        int is_length = (has_brace && line[i] == '#');
        if (is_length)
        {
            i++;
        }

        /* ${@fps.param} — FPS inline expansion */
        if (has_brace && line[i] == '@')
        {
            i++; /* skip @ */
            char atbuf[512];
            int  alen     = 0;
            atbuf[alen++] = '@';
            while (line[i] != '\0' && line[i] != '}' && alen < 511)
            {
                atbuf[alen++] = line[i++];
            }
            atbuf[alen] = '\0';
            if (line[i] == '}')
            {
                i++;
            }
            cli_expand_fpsvar(atbuf, (int) sizeof(atbuf));
            int rlen  = (int) strlen(atbuf);
            int avail = maxlen - 1 - opos;
            int clen  = rlen < avail ? rlen : avail;
            memcpy(out + opos, atbuf, (size_t) clen);
            opos += clen;
            continue;
        }

        /* Collect variable name.
         * Accepted: alnum, '_', '?', '.'
         * Special: '#' is accepted ONLY as a standalone
         * one-character name (for $#, the argument count).
         * It must be the first char of the name and we
         * break immediately after so it is never treated
         * as part of a longer identifier. */
        char varname[256];
        int  vlen = 0;

        /* $# special case: treat '#' as a solo variable name */
        if (!is_length && !has_brace && line[i] == '#')
        {
            varname[vlen++] = '#';
            i++;
        }
        else
        {
            while (line[i] != '\0' && vlen < 255)
            {
                char c = line[i];
                if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                      c == '_' || c == '?' || c == '.'))
                {
                    break;
                }
                varname[vlen++] = line[i++];
                if (c == '?')
                {
                    break;
                }
            }
        }
        varname[vlen] = '\0';

        /* Optional array index: ${arr[n]} */
        char index_str[256];
        int  has_index = 0;
        if (has_brace && line[i] == '[')
        {
            i++;
            int ilen = 0;
            while (line[i] != '\0' && line[i] != ']' && ilen < 255)
            {
                index_str[ilen++] = line[i++];
            }
            if (line[i] == ']')
            {
                i++;
            }
            index_str[ilen] = '\0';
            has_index       = 1;
        }

        /* Optional modifier: ${VAR:-…} etc. */
        char mod_op[3]    = { 0 };
        char mod_arg[256] = { 0 };
        if (has_brace && line[i] == ':')
        {
            i++;
            if (line[i] == '-' || line[i] == '=' || line[i] == '?' || line[i] == '+')
            {
                mod_op[0] = ':';
                mod_op[1] = line[i++];
            }
            else
            {
                mod_op[0] = ':';
            }
            int mlen = 0;
            while (line[i] != '\0' && line[i] != '}' && mlen < 255)
            {
                mod_arg[mlen++] = line[i++];
            }
            mod_arg[mlen] = '\0';
        }

        /* Consume closing brace or emit passthrough */
        if (has_brace)
        {
            if (line[i] == '}')
            {
                i++;
            }
            else
            {
                /* Complex expression — pass through */
                out[opos++] = '$';
                out[opos++] = '{';
                if (is_length)
                {
                    out[opos++] = '#';
                }
                for (int k = 0; k < vlen; k++)
                {
                    if (opos < maxlen - 1)
                    {
                        out[opos++] = varname[k];
                    }
                }
                continue;
            }
        }

        /* ---- Value lookup ---- */
        const char *val = NULL;
        char        all_buf[STRINGMAXLEN_CLICMDLINE];
        all_buf[0] = '\0';

        if (has_index)
        {
            const char *idx_val = cli_var_lookup(index_str);
            if (idx_val == NULL)
            {
                idx_val = index_str;
            }

            if (strcmp(idx_val, "@") == 0)
            {
                expand_env_array_all(varname, is_length, all_buf, &val);
                is_length = 0; /* already handled */
            }
            else
            {
                expand_env_array_index(varname, idx_val, &val);
            }
        }
        else
        {
            val = cli_var_lookup(varname);
        }

        /* ---- Apply modifier if present ---- */
        char val_buf[256];
        val_buf[0] = '\0';
        if (mod_op[0] != '\0')
        {
            apply_modifier(varname, mod_op, mod_arg, val_buf, (int) sizeof(val_buf), &val);
        }

        /* ---- Emit result ---- */
        if (is_length)
        {
            int  len = val ? (int) strlen(val) : 0;
            char numstr[32];
            snprintf(numstr, sizeof(numstr), "%d", len);
            emit_str_local(out, &opos, maxlen, numstr);
        }
        else if (val != NULL)
        {
            emit_str_local(out, &opos, maxlen, val);
        }
        else
        {
            /* Variable not found — restore literal */
            out[opos++] = '$';
            if (has_brace)
            {
                out[opos++] = '{';
            }
            for (int k = 0; k < vlen; k++)
            {
                if (opos < maxlen - 1)
                {
                    out[opos++] = varname[k];
                }
            }
            if (has_brace && opos < maxlen - 1)
            {
                out[opos++] = '}';
            }
        }
    }

    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}
