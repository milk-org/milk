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
 *  Helper
 * ============================================================
 */

/**
 * emit_str_local - append a C string to the output buffer
 * @out:    Destination buffer
 * @opos:   Current write position (updated)
 * @maxlen: Buffer size
 * @s:      NUL-terminated source string
 *
 * Writes characters until NUL or the buffer is full.
 */
static void emit_str_local(
    char *out,
    int  *opos,
    int   maxlen,
    const char *s
)
{
    while(*s != '\0' && *opos < maxlen - 1)
    {
        out[(*opos)++] = *s++;
    }
}


/* ============================================================
 *  cli_expand_env
 * ============================================================
 */

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
void cli_expand_env(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0'
          && opos < maxlen - 1)
    {
        if(line[i] == '$')
        {
            /* Skip $((  — let arith handle it */
            if(line[i + 1] == '('
               && line[i + 2] == '(')
            {
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                continue;
            }
            /* Skip $(  — command substitution */
            if(line[i + 1] == '(')
            {
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                continue;
            }

            i++;
            int has_brace = 0;
            if(line[i] == '{')
            {
                has_brace = 1;
                i++;
            }

            int is_length = 0;
            if(has_brace && line[i] == '#')
            {
                is_length = 1;
                i++;
            }

            /* ${@fps.param} — FPS inline */
            if(has_brace && line[i] == '@')
            {
                i++; /* skip @ */
                char atbuf[512];
                int  alen = 0;
                atbuf[alen++] = '@';
                while(line[i] != '\0'
                      && line[i] != '}'
                      && alen < 511)
                {
                    atbuf[alen++] = line[i++];
                }
                atbuf[alen] = '\0';
                if(line[i] == '}')
                {
                    i++;
                }
                cli_expand_fpsvar(
                    atbuf,
                    (int) sizeof(atbuf));
                int rlen = (int) strlen(atbuf);
                int avail = maxlen - 1 - opos;
                int clen = rlen < avail
                           ? rlen : avail;
                memcpy(out + opos, atbuf,
                       (size_t) clen);
                opos += clen;
                continue;
            }

            char varname[256];
            int  vlen = 0;

            while(line[i] != '\0'
                  && vlen < 255)
            {
                char c = line[i];
                if(!((c >= 'A' && c <= 'Z')
                     || (c >= 'a' && c <= 'z')
                     || (c >= '0' && c <= '9')
                     || c == '_'
                     || c == '?'
                     || c == '.'))
                {
                    break;
                }
                varname[vlen++] = line[i++];
                if(c == '?')
                {
                    break;
                }
            }
            varname[vlen] = '\0';

            char index_str[256];
            int  has_index = 0;
            if(has_brace && line[i] == '[')
            {
                i++;
                int ilen = 0;
                while(line[i] != '\0'
                      && line[i] != ']'
                      && ilen < 255)
                {
                    index_str[ilen++] =
                        line[i++];
                }
                if(line[i] == ']')
                {
                    i++;
                }
                index_str[ilen] = '\0';
                has_index = 1;
            }

            char mod_op[3]  = {0};
            char mod_arg[256] = {0};
            if(has_brace && line[i] == ':')
            {
                i++;
                if(line[i] == '-'
                   || line[i] == '='
                   || line[i] == '?'
                   || line[i] == '+')
                {
                    mod_op[0] = ':';
                    mod_op[1] = line[i++];
                }
                else
                {
                    mod_op[0] = ':';
                }
                int mlen = 0;
                while(line[i] != '\0'
                      && line[i] != '}'
                      && mlen < 255)
                {
                    mod_arg[mlen++] = line[i++];
                }
                mod_arg[mlen] = '\0';
            }

            if(has_brace)
            {
                if(line[i] == '}')
                {
                    i++;
                }
                else
                {
                    /* Complex: pass through */
                    out[opos++] = '$';
                    out[opos++] = '{';
                    if(is_length)
                    {
                        out[opos++] = '#';
                    }
                    for(int k = 0;
                        k < vlen; k++)
                    {
                        if(opos < maxlen - 1)
                        {
                            out[opos++] =
                                varname[k];
                        }
                    }
                    continue;
                }
            }

            const char *val = NULL;
            char all_elems[
                STRINGMAXLEN_CLICMDLINE];
            all_elems[0] = '\0';
            int elems_count = 0;

            if(has_index)
            {
                const char *idx_val =
                    cli_var_lookup(index_str);
                if(idx_val == NULL)
                {
                    idx_val = index_str;
                }

                if(strcmp(idx_val, "@") == 0)
                {
                    int is_found = 0;
                    for(int a = 0;
                        a < CLI_MAX_ASSOC; a++)
                    {
                        if(cli_assoc[a].used
                           && strcmp(
                               cli_assoc[a].name,
                               varname) == 0)
                        {
                            elems_count =
                                cli_assoc[a]
                                    .nelem;
                            if(!is_length)
                            {
                                for(int e = 0;
                                    e < cli_assoc[a].nelem;
                                    e++)
                                {
                                    if(e > 0)
                                    {
                                        strncat(
                                            all_elems,
                                            " ",
                                            sizeof(all_elems)
                                            - strlen(all_elems)
                                            - 1);
                                    }
                                    strncat(
                                        all_elems,
                                        cli_assoc[a]
                                            .vals[e],
                                        sizeof(all_elems)
                                        - strlen(all_elems)
                                        - 1);
                                }
                            }
                            is_found = 1;
                            break;
                        }
                    }
                    if(!is_found)
                    {
                        for(int a = 0;
                            a < CLI_MAX_ARRAYS;
                            a++)
                        {
                            if(cli_arrays[a].used
                               && strcmp(
                                   cli_arrays[a]
                                       .name,
                                   varname) == 0)
                            {
                                elems_count =
                                    cli_arrays[a]
                                        .nelem;
                                if(!is_length)
                                {
                                    for(int e = 0;
                                        e < cli_arrays[a].nelem;
                                        e++)
                                    {
                                        if(e > 0)
                                        {
                                            strncat(
                                                all_elems,
                                                " ",
                                                sizeof(all_elems)
                                                - strlen(all_elems)
                                                - 1);
                                        }
                                        strncat(
                                            all_elems,
                                            cli_arrays[a]
                                                .elem[e],
                                            sizeof(all_elems)
                                            - strlen(all_elems)
                                            - 1);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    if(is_length)
                    {
                        char cnt_buf[32];
                        snprintf(cnt_buf,
                                 sizeof(cnt_buf),
                                 "%d",
                                 elems_count);
                        strncpy(all_elems,
                                cnt_buf,
                                sizeof(all_elems)
                                - 1);
                        is_length = 0;
                    }
                    val = all_elems;
                }
                else
                {
                    int is_found = 0;
                    for(int a = 0;
                        a < CLI_MAX_ASSOC; a++)
                    {
                        if(cli_assoc[a].used
                           && strcmp(
                               cli_assoc[a].name,
                               varname) == 0)
                        {
                            for(int e = 0;
                                e < cli_assoc[a]
                                    .nelem;
                                e++)
                            {
                                if(strcmp(
                                    cli_assoc[a]
                                        .keys[e],
                                    idx_val)
                                   == 0)
                                {
                                    val =
                                        cli_assoc[a]
                                            .vals[e];
                                    is_found = 1;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                    if(!is_found)
                    {
                        int num_idx =
                            atoi(idx_val);
                        for(int a = 0;
                            a < CLI_MAX_ARRAYS;
                            a++)
                        {
                            if(cli_arrays[a].used
                               && strcmp(
                                   cli_arrays[a]
                                       .name,
                                   varname) == 0)
                            {
                                if(num_idx >= 0
                                   && num_idx
                                      < cli_arrays[a]
                                            .nelem)
                                {
                                    val =
                                        cli_arrays[a]
                                            .elem[num_idx];
                                }
                                break;
                            }
                        }
                    }
                }
            }
            else
            {
                val = cli_var_lookup(varname);
            }

            char val_buf[256];
            val_buf[0] = '\0';

            if(mod_op[0] != '\0')
            {
                if(mod_op[1] == '-')
                {
                    if(val == NULL
                       || val[0] == '\0')
                    {
                        val = mod_arg;
                    }
                }
                else if(mod_op[1] == '=')
                {
                    if(val == NULL
                       || val[0] == '\0')
                    {
                        val = mod_arg;
                        cli_var_set(varname,
                                    val);
                    }
                }
                else if(mod_op[1] == '?')
                {
                    if(val == NULL
                       || val[0] == '\0')
                    {
                        printf(
                            "CLI expand "
                            "error: %s: %s\n",
                            varname, mod_arg);
                        val = "";
                    }
                }
                else if(mod_op[1] == '+')
                {
                    if(val != NULL
                       && val[0] != '\0')
                    {
                        val = mod_arg;
                    }
                    else
                    {
                        val = "";
                    }
                }
                else if(mod_op[0] == ':')
                {
                    int offset = 0;
                    int length = 255;
                    char *colon =
                        strchr(mod_arg, ':');
                    if(colon != NULL)
                    {
                        *colon = '\0';
                        const char *lval =
                            cli_var_lookup(
                                mod_arg);
                        offset = atoi(
                            lval ? lval
                                 : mod_arg);
                        const char *rval =
                            cli_var_lookup(
                                colon + 1);
                        length = atoi(
                            rval ? rval
                                 : colon + 1);
                    }
                    else
                    {
                        const char *lval =
                            cli_var_lookup(
                                mod_arg);
                        offset = atoi(
                            lval ? lval
                                 : mod_arg);
                    }

                    if(val != NULL)
                    {
                        int v1 =
                            (int) strlen(val);
                        if(offset < 0)
                        {
                            offset = v1 + offset;
                        }
                        if(offset < 0)
                        {
                            offset = 0;
                        }
                        if(offset > v1)
                        {
                            offset = v1;
                        }

                        if(length < 0)
                        {
                            length = v1
                                     - offset
                                     + length;
                        }
                        if(length < 0)
                        {
                            length = 0;
                        }
                        if(offset + length > v1)
                        {
                            length = v1 - offset;
                        }

                        strncpy(val_buf,
                                val + offset,
                                (size_t) length);
                        val_buf[length] = '\0';
                        val = val_buf;
                    }
                }
            }

            if(is_length)
            {
                int len = val
                          ? (int) strlen(val)
                          : 0;
                char numstr[32];
                snprintf(numstr, sizeof(numstr),
                         "%d", len);
                emit_str_local(out, &opos,
                               maxlen, numstr);
            }
            else if(val != NULL)
            {
                emit_str_local(out, &opos,
                               maxlen, val);
            }
            else
            {
                /* Variable not found */
                out[opos++] = '$';
                if(has_brace)
                {
                    out[opos++] = '{';
                }
                for(int k = 0; k < vlen; k++)
                {
                    if(opos < maxlen - 1)
                    {
                        out[opos++] = varname[k];
                    }
                }
                if(has_brace
                   && opos < maxlen - 1)
                {
                    out[opos++] = '}';
                }
            }
        }
        else if(line[i] == '\\'
                && line[i + 1] == '$')
        {
            /* Escaped $: pass both chars */
            out[opos++] = line[i++];
            out[opos++] = line[i++];
        }
        else
        {
            out[opos++] = line[i++];
        }
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}
