#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#ifdef USE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
#endif
#include "CLIcore.h"
#include "CLIcore_UI.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"
#include <glob.h>
#include <sys/wait.h>
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"



/*
 * ============================================================
 *  Configurable Prompt — setprompt command
 * ============================================================
 *
 * Format tokens:
 *   %h = hostname
 *   %u = username
 *   %d = cwd basename
 *   %t = HH:MM:SS
 *   %n = CLI process name (data.processname)
 */

/** prompt_format stored in data struct is TBD;
 *  for now use a file-scope buffer. */
char cli_prompt_format[200] = "";

/**
 * @brief Build prompt string from format tokens
 */
void cli_build_prompt(
    const char *fmt,
    char       *out,
    int         maxlen
)
{
    int pos = 0;
    for(int i = 0; fmt[i] != '\0'
            && pos < maxlen - 1; i++)
    {
        if(fmt[i] == '%' && fmt[i + 1] != '\0')
        {
            i++;
            switch(fmt[i])
            {
            case 'h':
            {
                char hn[64];
                gethostname(hn, sizeof(hn));
                pos += snprintf(out + pos,
                    (size_t)(maxlen - pos),
                    "%s", hn);
                break;
            }
            case 'u':
            {
                const char *u = getenv("USER");
                pos += snprintf(out + pos,
                    (size_t)(maxlen - pos),
                    "%s", u ? u : "?");
                break;
            }
            case 'd':
            {
                char cwd[256];
                if(getcwd(cwd, sizeof(cwd)))
                {
                    char *base = strrchr(cwd,
                                         '/');
                    pos += snprintf(out + pos,
                        (size_t)(maxlen - pos),
                        "%s",
                        base ? base + 1 : cwd);
                }
                break;
            }
            case 't':
            {
                time_t now = time(NULL);
                struct tm *tm = localtime(&now);
                pos += (int) strftime(
                    out + pos,
                    (size_t)(maxlen - pos),
                    "%H:%M:%S", tm);
                break;
            }
            case 'n':
                pos += snprintf(out + pos,
                    (size_t)(maxlen - pos),
                    "%s", data.processname);
                break;
            default:
                if(pos < maxlen - 2)
                {
                    out[pos++] = '%';
                    out[pos++] = fmt[i];
                }
                break;
            }
        }
        else
        {
            out[pos++] = fmt[i];
        }
    }
    out[pos] = '\0';
}

errno_t cli_setprompt(void)
{
    if(data.cmdNBarg < 2)
    {
        if(cli_prompt_format[0] != '\0')
        {
            printf("Current prompt format: "
                   "'%s'\n",
                   cli_prompt_format);
        }
        else
        {
            printf("Using default prompt\n");
        }
        printf("Tokens: %%h=host %%u=user "
               "%%d=dir %%t=time %%n=name\n");
        return RETURN_SUCCESS;
    }
    strncpy(cli_prompt_format,
            data.cmdargtoken[1].val.string,
            sizeof(cli_prompt_format) - 1);
    cli_prompt_format[
        sizeof(cli_prompt_format) - 1] = '\0';
    printf("Prompt set to: '%s'\n",
           cli_prompt_format);
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Brace Expansion
 * ============================================================
 *
 * Expand {N..M} into space-separated integers,
 * and {N..M..S} with step S.
 */

/**
 * @brief Expand {N..M} and {N..M..S} brace ranges
 *
 * Replaces tokens like {1..5} with "1 2 3 4 5"
 * and {0..10..2} with "0 2 4 6 8 10".
 */
void emit_str(
    char       *out,
    int        *opos,
    int         maxlen,
    const char *s
);
void cli_expand_braces(
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
        if(line[i] == '{')
        {
            /* Try {N..M} or {N..M..S} */
            char *endp = NULL;
            long sv =
                strtol(line + i + 1,
                       &endp, 10);
            if(endp != NULL
               && endp[0] == '.'
               && endp[1] == '.')
            {
                char *endp2 = NULL;
                long ev =
                    strtol(endp + 2,
                           &endp2, 10);
                long step = 1;
                if(endp2 != NULL
                   && endp2[0] == '.'
                   && endp2[1] == '.')
                {
                    char *endp3 = NULL;
                    step =
                        strtol(endp2 + 2,
                               &endp3,
                               10);
                    endp2 = endp3;
                }
                if(endp2 != NULL
                   && *endp2 == '}'
                   && step != 0)
                {
                    int first = 1;
                    if(sv <= ev)
                    {
                        if(step < 0)
                        {
                            step = -step;
                        }
                        for(long v = sv;
                            v <= ev;
                            v += step)
                        {
                            char nb[32];
                            snprintf(
                                nb,
                                sizeof(nb),
                                "%s%ld",
                                first
                                ? "" : " ",
                                v);
                            first = 0;
                            emit_str(
                                out, &opos,
                                maxlen, nb);
                        }
                    }
                    else
                    {
                        if(step > 0)
                        {
                            step = -step;
                        }
                        for(long v = sv;
                            v >= ev;
                            v += step)
                        {
                            char nb[32];
                            snprintf(
                                nb,
                                sizeof(nb),
                                "%s%ld",
                                first
                                ? "" : " ",
                                v);
                            first = 0;
                            emit_str(
                                out, &opos,
                                maxlen, nb);
                        }
                    }
                    i = (int)(endp2
                              - line) + 1;
                    continue;
                }
            }
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


/**
 * @brief Expand tilde (~) to $HOME
 *
 * Replaces ~ or ~/path at start of tokens
 * with the HOME environment variable value.
 */
void cli_expand_tilde(
    char *line,
    int   maxlen
)
{
    const char *home = getenv("HOME");
    if(home == NULL)
    {
        return;
    }
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;
    int  at_tok_start = 1;
    int  in_sq = 0;
    int  in_dq = 0;

    while(line[i] != '\0'
          && opos < maxlen - 1)
    {
        char c = line[i];
        if(c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
            out[opos++] = line[i++];
            at_tok_start = 0;
            continue;
        }
        if(c == '"' && !in_sq)
        {
            in_dq = !in_dq;
            out[opos++] = line[i++];
            at_tok_start = 0;
            continue;
        }
        if(c == ' ' || c == '\t')
        {
            out[opos++] = line[i++];
            at_tok_start = 1;
            continue;
        }
        if(at_tok_start
           && !in_sq && !in_dq
           && c == '~'
           && (line[i + 1] == '/'
               || line[i + 1] == ' '
               || line[i + 1] == '\t'
               || line[i + 1] == '\0'))
        {
            /* Replace ~ with $HOME */
            const char *hp = home;
            while(*hp != '\0'
                  && opos < maxlen - 1)
            {
                out[opos++] = *hp++;
            }
            i++; /* skip ~ */
            at_tok_start = 0;
            continue;
        }
        out[opos++] = line[i++];
        at_tok_start = 0;
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}

/**
 * @brief Expand filename globs (* and ?)
 *
 * Tokens containing * or ? that are not inside
 * quotes are expanded using POSIX glob().
 * Example: *.fits → file1.fits file2.fits
 */
void cli_expand_globs(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;
    int  in_sq = 0;
    int  in_dq = 0;

    while(line[i] != '\0'
          && opos < maxlen - 1)
    {
        char c = line[i];
        if(c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
            out[opos++] = line[i++];
            continue;
        }
        if(c == '"' && !in_sq)
        {
            in_dq = !in_dq;
            out[opos++] = line[i++];
            continue;
        }
        if(in_sq || in_dq)
        {
            out[opos++] = line[i++];
            continue;
        }
        if(c == ' ' || c == '\t')
        {
            out[opos++] = line[i++];
            continue;
        }
        /* Extract token */
        int tstart = i;
        int has_glob = 0;
        while(line[i] != '\0'
              && line[i] != ' '
              && line[i] != '\t')
        {
            if(line[i] == '*'
               || line[i] == '?')
            {
                has_glob = 1;
            }
            i++;
        }
        int tlen = i - tstart;
        if(!has_glob || tlen <= 0)
        {
            for(int j = tstart;
                j < i
                && opos < maxlen - 1;
                j++)
            {
                out[opos++] = line[j];
            }
            continue;
        }
        /* Run glob */
        char pat[512];
        int plen = tlen;
        if(plen >= 512)
        {
            plen = 511;
        }
        memcpy(pat, line + tstart,
               (size_t) plen);
        pat[plen] = '\0';

        glob_t gl;
        int gret = glob(pat,
                        GLOB_NOCHECK,
                        NULL, &gl);
        if(gret == 0
           && gl.gl_pathc > 0)
        {
            for(size_t g = 0;
                g < gl.gl_pathc; g++)
            {
                if(g > 0
                   && opos < maxlen - 1)
                {
                    out[opos++] = ' ';
                }
                const char *gp =
                    gl.gl_pathv[g];
                while(*gp != '\0'
                      && opos
                      < maxlen - 1)
                {
                    out[opos++] = *gp++;
                }
            }
            globfree(&gl);
        }
        else
        {
            if(gret == 0)
            {
                globfree(&gl);
            }
            for(int j = tstart;
                j < tstart + tlen
                && opos < maxlen - 1;
                j++)
            {
                out[opos++] = line[j];
            }
        }
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}


/*
 * ============================================================
 *  Command Substitution
 * ============================================================
 *
 * Replace $(cmd) and `cmd` in the command line with
 * the standard output of the command execution.
 */

void cli_expand_cmdsub(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0' && opos < maxlen - 1)
    {
        int is_dollar_paren =
            (line[i] == '$'
             && line[i + 1] == '('
             && line[i + 2] != '(');
        int is_backtick = (line[i] == '`');

        if(is_dollar_paren || is_backtick)
        {
            char cmd[512];
            int clen = 0;
            
            if (is_dollar_paren)
            {
                i += 2; /* Skip $( */
                while(line[i] != '\0' && line[i] != ')' && clen < 511)
                {
                    cmd[clen++] = line[i++];
                }
                if (line[i] == ')') i++; /* Skip ) */
            }
            else /* is_backtick */
            {
                i++; /* Skip ` */
                while(line[i] != '\0' && line[i] != '`' && clen < 511)
                {
                    cmd[clen++] = line[i++];
                }
                if (line[i] == '`') i++; /* Skip ` */
            }
            cmd[clen] = '\0';

            /* Execute command and read output */
            if (clen > 0)
            {
                FILE *fp = popen(cmd, "r");
                if (fp != NULL)
                {
                    char buf[1024];
                    size_t read_bytes = fread(buf, 1, sizeof(buf) - 1, fp);
                    buf[read_bytes] = '\0';
                    pclose(fp);

                    /* Strip trailing newlines */
                    while(read_bytes > 0 && (buf[read_bytes - 1] == '\n' || buf[read_bytes - 1] == '\r'))
                    {
                        buf[--read_bytes] = '\0';
                    }

                    /* Copy to output */
                    int vallen = (int) read_bytes;
                    int avail = maxlen - 1 - opos;
                    int copylen = vallen < avail ? vallen : avail;
                    memcpy(out + opos, buf, (size_t) copylen);
                    opos += copylen;
                }
            }
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


/*
 * ============================================================
 *  Environment Variable Expansion
 * ============================================================
 *
 * Replace $VAR and ${VAR} with variable values.
 * Supports string operations, arrays, and special
 * forms inside ${...}.
 */

/**
 * @brief Emit string into output buffer
 */
void emit_str(
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

/**
 * @brief Handle ${...} braced expansion
 *
 * Supports:
 *   ${var}          plain lookup
 *   ${#var}         string length
 *   ${var:-default} default if unset
 *   ${var:=default} assign if unset
 *   ${var:+alt}     alt if set
 *   ${var:?error}   error if unset
 *   ${var:off:len}  substring
 *   ${var%%pat}     strip longest suffix
 *   ${var##pat}     strip longest prefix
 *   ${var%pat}      strip shortest suffix
 *   ${var#pat}      strip shortest prefix
 *   ${var/find/rep} replace first
 *   ${var//find/rep} replace all
 *   ${arr[N]}       array element
 *   ${arr[@]}       all array elements
 *   ${#arr[@]}      array element count
 */
void expand_braced(
    char *out,
    int  *opos,
    int   maxlen,
    const char *inner
)
{
    /* ${#...} — length or array count */
    if(inner[0] == '#')
    {
        const char *nm = inner + 1;
        /* ${#arr[@]} */
        const char *br = strchr(nm, '[');
        if(br != NULL)
        {
            char aname[CLI_VAR_NAMELEN];
            int alen = (int)(br - nm);
            if(alen >= CLI_VAR_NAMELEN)
            {
                alen = CLI_VAR_NAMELEN - 1;
            }
            memcpy(aname, nm,
                   (size_t) alen);
            aname[alen] = '\0';
            for(int k = 0;
                k < CLI_MAX_ARRAYS; k++)
            {
                if(cli_arrays[k].used
                   && strcmp(
                       cli_arrays[k].name,
                       aname) == 0)
                {
                    char nb[32];
                    snprintf(
                        nb, sizeof(nb),
                        "%d",
                        cli_arrays[k]
                        .nelem);
                    emit_str(out, opos,
                             maxlen, nb);
                    return;
                }
            }
            emit_str(out, opos,
                     maxlen, "0");
            return;
        }
        /* ${#var} — string length */
        const char *val =
            cli_var_lookup(nm);
        char lb[32];
        snprintf(lb, sizeof(lb), "%d",
                 val ? (int) strlen(val)
                 : 0);
        emit_str(out, opos, maxlen, lb);
        return;
    }

    /* ${!var} — indirect expansion */
    if(inner[0] == '!')
    {
        const char *iname =
            inner + 1;
        const char *iref =
            cli_var_lookup(iname);
        if(iref != NULL)
        {
            const char *ival =
                cli_var_lookup(iref);
            if(ival != NULL)
            {
                emit_str(out, opos,
                         maxlen,
                         ival);
            }
        }
        return;
    }

    /* ${arr[N]} or ${arr[@]} or
     * ${assoc[key]} */
    {
        const char *br =
            strchr(inner, '[');
        if(br != NULL)
        {
            char aname[CLI_VAR_NAMELEN];
            int alen = (int)(br - inner);
            if(alen >= CLI_VAR_NAMELEN)
            {
                alen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(aname, inner,
                   (size_t) alen);
            aname[alen] = '\0';
            const char *idx_s = br + 1;
            if(idx_s[0] == '@')
            {
                for(int k = 0;
                    k < CLI_MAX_ARRAYS;
                    k++)
                {
                    if(cli_arrays[k].used
                       && strcmp(
                           cli_arrays[k]
                           .name,
                           aname) == 0)
                    {
                        for(int e = 0;
                            e
                            < cli_arrays[k]
                            .nelem;
                            e++)
                        {
                            if(e > 0)
                            {
                                emit_str(
                                    out,
                                    opos,
                                    maxlen,
                                    " ");
                            }
                            emit_str(
                                out,
                                opos,
                                maxlen,
                                cli_arrays
                                [k]
                                .elem[e]);
                        }
                        return;
                    }
                }
                return;
            }
            int idx = (int) strtol(
                idx_s, NULL, 0);
            for(int k = 0;
                k < CLI_MAX_ARRAYS;
                k++)
            {
                if(cli_arrays[k].used
                   && strcmp(
                       cli_arrays[k]
                       .name,
                       aname) == 0)
                {
                    if(idx >= 0
                       && idx
                       < cli_arrays[k]
                       .nelem)
                    {
                        emit_str(
                            out, opos,
                            maxlen,
                            cli_arrays[k]
                            .elem[idx]);
                    }
                    return;
                }
            }
            return;
        }
    }

    /* ${assoc[key]} — associative
     * array lookup */
    {
        const char *br =
            strchr(inner, '[');
        if(br != NULL)
        {
            const char *brend =
                strchr(br, ']');
            if(brend != NULL)
            {
                char aname[
                    CLI_VAR_NAMELEN];
                int nl =
                    (int)(br
                          - inner);
                if(nl
                   >= CLI_VAR_NAMELEN)
                {
                    nl =
                        CLI_VAR_NAMELEN
                        - 1;
                }
                memcpy(aname,
                       inner,
                       (size_t) nl);
                aname[nl] = '\0';
                char key[
                    CLI_VAR_NAMELEN];
                int kl =
                    (int)(brend
                          - br - 1);
                if(kl
                   >= CLI_VAR_NAMELEN)
                {
                    kl =
                        CLI_VAR_NAMELEN
                        - 1;
                }
                memcpy(key, br + 1,
                       (size_t) kl);
                key[kl] = '\0';
                for(int k = 0;
                    k
                    < CLI_MAX_ASSOC;
                    k++)
                {
                    if(cli_assoc[k]
                        .used
                       && strcmp(
                           cli_assoc[
                               k]
                           .name,
                           aname)
                       == 0)
                    {
                        for(int e = 0;
                            e
                            < cli_assoc[
                                k]
                            .nelem;
                            e++)
                        {
                            if(strcmp(
                                cli_assoc[
                                    k]
                                .keys[e],
                                key)
                               == 0)
                            {
                                emit_str(
                                    out,
                                    opos,
                                    maxlen,
                                    cli_assoc[
                                        k]
                                    .vals[
                                        e]);
                                return;
                            }
                        }
                        return;
                    }
                }
            }
        }
    }

    /* ${var//find/rep} — replace all */
    {
        const char *ds =
            strstr(inner, "//");
        if(ds != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen = (int)(ds - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *find =
                ds + 2;
            const char *sl2 =
                strchr(find, '/');
            char fp[256] = "";
            char rp[256] = "";
            if(sl2 != NULL)
            {
                int fl2 =
                    (int)(sl2 - find);
                if(fl2 > 255)
                {
                    fl2 = 255;
                }
                memcpy(fp, find,
                       (size_t) fl2);
                fp[fl2] = '\0';
                strncpy(rp,
                        sl2 + 1, 255);
            }
            else
            {
                strncpy(fp, find, 255);
            }
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL
               && fp[0] != '\0')
            {
                int fplen =
                    (int) strlen(fp);
                int rplen =
                    (int) strlen(rp);
                const char *s = val;
                while(*s != '\0'
                      && *opos
                      < maxlen - 1)
                {
                    if(strncmp(s, fp,
                               (size_t)
                               fplen)
                       == 0)
                    {
                        emit_str(
                            out, opos,
                            maxlen, rp);
                        s += fplen;
                    }
                    else
                    {
                        out[(*opos)++] =
                            *s++;
                    }
                }
            }
            else if(val != NULL)
            {
                emit_str(out, opos,
                         maxlen, val);
            }
            return;
        }
    }

    /* ${var/find/rep} — replace first */
    {
        const char *sl =
            strchr(inner, '/');
        if(sl != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen = (int)(sl - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *find = sl + 1;
            const char *sl2 =
                strchr(find, '/');
            char fp[256] = "";
            char rp[256] = "";
            if(sl2 != NULL)
            {
                int fl2 =
                    (int)(sl2 - find);
                if(fl2 > 255)
                {
                    fl2 = 255;
                }
                memcpy(fp, find,
                       (size_t) fl2);
                fp[fl2] = '\0';
                strncpy(rp,
                        sl2 + 1, 255);
            }
            else
            {
                strncpy(fp, find, 255);
            }
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL
               && fp[0] != '\0')
            {
                const char *m =
                    strstr(val, fp);
                if(m != NULL)
                {
                    int pre =
                        (int)(m - val);
                    int avail =
                        maxlen - 1
                        - *opos;
                    if(pre > avail)
                    {
                        pre = avail;
                    }
                    memcpy(
                        out + *opos,
                        val,
                        (size_t) pre);
                    *opos += pre;
                    emit_str(
                        out, opos,
                        maxlen, rp);
                    emit_str(
                        out, opos,
                        maxlen,
                        m + strlen(fp));
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var%%pattern} — strip suffix */
    {
        const char *pp =
            strstr(inner, "%%");
        if(pp != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(pp - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *pat = pp + 2;
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                const char *m =
                    strstr(val, pat);
                if(m != NULL)
                {
                    int clen =
                        (int)(m - val);
                    int avail =
                        maxlen - 1
                        - *opos;
                    if(clen > avail)
                    {
                        clen = avail;
                    }
                    memcpy(
                        out + *opos,
                        val,
                        (size_t) clen);
                    *opos += clen;
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var##pattern} — strip prefix */
    {
        const char *pp =
            strstr(inner, "##");
        if(pp != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(pp - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *pat = pp + 2;
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                const char *m =
                    strstr(val, pat);
                if(m != NULL)
                {
                    emit_str(
                        out, opos,
                        maxlen,
                        m + strlen(pat));
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var%pat} — strip shortest suffix */
    {
        const char *pp =
            strchr(inner, '%');
        if(pp != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(pp - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *pat = pp + 1;
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                int vl =
                    (int) strlen(val);
                int pl =
                    (int) strlen(pat);
                /* Find last occurrence */
                const char *last =
                    NULL;
                const char *s = val;
                while((s = strstr(
                           s, pat))
                      != NULL)
                {
                    last = s;
                    s++;
                }
                if(last != NULL
                   && (last + pl)
                   == (val + vl))
                {
                    int clen =
                        (int)(last
                              - val);
                    int avail =
                        maxlen - 1
                        - *opos;
                    if(clen > avail)
                    {
                        clen = avail;
                    }
                    memcpy(
                        out + *opos,
                        val,
                        (size_t) clen);
                    *opos += clen;
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var^^} uppercase / ${var,,} lowercase
     * ${var^}  first char upper
     * ${var,}  first char lower */
    {
        /* Find ^ or , in inner */
        const char *cp =
            strchr(inner, '^');
        const char *cl =
            strchr(inner, ',');
        /* Pick the earlier one */
        const char *op = NULL;
        if(cp != NULL && cl != NULL)
        {
            op = (cp < cl) ? cp : cl;
        }
        else if(cp != NULL)
        {
            op = cp;
        }
        else if(cl != NULL)
        {
            op = cl;
        }
        if(op != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(op - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                char tmp[
                    CLI_VAR_VALLEN];
                strncpy(tmp, val,
                        CLI_VAR_VALLEN
                        - 1);
                tmp[CLI_VAR_VALLEN
                    - 1] = '\0';
                if(op[0] == '^'
                   && op[1] == '^')
                {
                    /* Uppercase all */
                    for(int k = 0;
                        tmp[k]
                        != '\0'; k++)
                    {
                        tmp[k] =
                            (char)
                            toupper(
                                (unsigned
                                 char)
                                tmp[k]);
                    }
                }
                else if(op[0] == '^')
                {
                    /* First char */
                    if(tmp[0] != '\0')
                    {
                        tmp[0] =
                            (char)
                            toupper(
                                (unsigned
                                 char)
                                tmp[0]);
                    }
                }
                else if(op[0] == ','
                        && op[1]
                        == ',')
                {
                    /* Lowercase all */
                    for(int k = 0;
                        tmp[k]
                        != '\0'; k++)
                    {
                        tmp[k] =
                            (char)
                            tolower(
                                (unsigned
                                 char)
                                tmp[k]);
                    }
                }
                else if(op[0] == ',')
                {
                    /* First char */
                    if(tmp[0] != '\0')
                    {
                        tmp[0] =
                            (char)
                            tolower(
                                (unsigned
                                 char)
                                tmp[0]);
                    }
                }
                emit_str(out, opos,
                         maxlen,
                         tmp);
            }
            return;
        }
    }

    /* ${var#pat} — strip shortest prefix */
    {
        const char *pp =
            strchr(inner, '#');
        if(pp != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(pp - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *pat = pp + 1;
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                int pl =
                    (int) strlen(pat);
                if(strncmp(val, pat,
                           (size_t) pl)
                   == 0)
                {
                    emit_str(
                        out, opos,
                        maxlen,
                        val + pl);
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var:-default} ${var:=default}
     * ${var:+alt} ${var:?error}
     * ${var:offset:length} */
    {
        const char *col =
            strchr(inner, ':');
        if(col != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(col - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            char op = col[1];

            /* ${var:-default} */
            if(op == '-')
            {
                const char *val =
                    cli_var_lookup(vn);
                if(val != NULL
                   && val[0] != '\0')
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen,
                        col + 2);
                }
                return;
            }
            /* ${var:=default} */
            if(op == '=')
            {
                const char *val =
                    cli_var_lookup(vn);
                if(val != NULL
                   && val[0] != '\0')
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
                else
                {
                    cli_var_set(
                        vn, col + 2);
                    emit_str(
                        out, opos,
                        maxlen,
                        col + 2);
                }
                return;
            }
            /* ${var:+alt} */
            if(op == '+')
            {
                const char *val =
                    cli_var_lookup(vn);
                if(val != NULL
                   && val[0] != '\0')
                {
                    emit_str(
                        out, opos,
                        maxlen,
                        col + 2);
                }
                return;
            }
            /* ${var:?error} */
            if(op == '?')
            {
                const char *val =
                    cli_var_lookup(vn);
                if(val == NULL
                   || val[0] == '\0')
                {
                    fprintf(stderr,
                            "%s: %s\n",
                            vn,
                            col + 2);
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
                return;
            }

            /* ${var:offset:length} */
            int offset = (int) strtol(
                col + 1, NULL, 0);
            int slen = -1;
            const char *c2 =
                strchr(col + 1, ':');
            if(c2 != NULL)
            {
                slen = (int) strtol(
                    c2 + 1, NULL, 0);
            }
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                int vl =
                    (int) strlen(val);
                if(offset < 0)
                {
                    offset =
                        vl + offset;
                }
                if(offset < 0)
                {
                    offset = 0;
                }
                if(offset >= vl)
                {
                    return;
                }
                int rem =
                    vl - offset;
                if(slen < 0
                   || slen > rem)
                {
                    slen = rem;
                }
                int avail =
                    maxlen - 1
                    - *opos;
                if(slen > avail)
                {
                    slen = avail;
                }
                memcpy(
                    out + *opos,
                    val + offset,
                    (size_t) slen);
                *opos += slen;
            }
            return;
        }
    }

    /* Plain ${var} */
    const char *val =
        cli_var_lookup(inner);
    if(val != NULL)
    {
        emit_str(out, opos, maxlen,
                 val);
    }
}

/**
 * @brief Expand $VAR and ${VAR} in place
 *
 * Handles string ops, arrays, and special
 * forms inside ${...}.
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
        if(line[i] == '`')
        {
            i++; /* skip ` */
            char cmdsub[STRINGMAXLEN_CLICMDLINE];
            int  clen = 0;
            while(line[i] != '\0' && line[i] != '`' && clen < STRINGMAXLEN_CLICMDLINE - 1)
            {
                cmdsub[clen++] = line[i++];
            }
            if(line[i] == '`')
            {
                i++;
            }
            cmdsub[clen] = '\0';

            FILE *fp = popen(cmdsub, "r");
            if(fp)
            {
                char   resbuf[4096];
                size_t bytes_read = fread(resbuf, 1, sizeof(resbuf) - 1, fp);
                resbuf[bytes_read] = '\0';

                while(bytes_read > 0 &&
                      (resbuf[bytes_read - 1] == '\n' || resbuf[bytes_read - 1] == '\r'))
                {
                    resbuf[--bytes_read] = '\0';
                }

                for(size_t k = 0; k < bytes_read; k++)
                {
                    if(resbuf[k] == '\n' || resbuf[k] == '\r')
                    {
                        resbuf[k] = ' ';
                    }
                }

                for(size_t k = 0; k < bytes_read && opos < maxlen - 1; k++)
                {
                    out[opos++] = resbuf[k];
                }
                pclose(fp);
            }
            continue;
        }

        if(line[i] == '$')
        {
            /* Skip $(( — arithmetic */
            if(line[i + 1] == '('
               && line[i + 2] == '(')
            {
                out[opos++] = line[i++];
                continue;
            }
            /* Handle $( — command subst */
            if(line[i + 1] == '(')
            {
                i += 2; /* skip $( */
                char cmdsub[STRINGMAXLEN_CLICMDLINE];
                int  clen = 0;
                int  cdepth = 1;
                while(line[i] != '\0' && clen < STRINGMAXLEN_CLICMDLINE - 1)
                {
                    if(line[i] == '(')
                    {
                        cdepth++;
                    }
                    else if(line[i] == ')')
                    {
                        cdepth--;
                        if(cdepth == 0)
                        {
                            i++;
                            break;
                        }
                    }
                    cmdsub[clen++] = line[i++];
                }
                cmdsub[clen] = '\0';

                FILE *fp = popen(cmdsub, "r");
                if(fp)
                {
                    char   resbuf[4096];
                    size_t bytes_read = fread(resbuf, 1, sizeof(resbuf) - 1, fp);
                    resbuf[bytes_read] = '\0';

                    /* Trim trailing newlines */
                    while(bytes_read > 0 &&
                          (resbuf[bytes_read - 1] == '\n' || resbuf[bytes_read - 1] == '\r'))
                    {
                        resbuf[--bytes_read] = '\0';
                    }

                    /* Replace internal newlines with space */
                    for(size_t k = 0; k < bytes_read; k++)
                    {
                        if(resbuf[k] == '\n' || resbuf[k] == '\r')
                        {
                            resbuf[k] = ' ';
                        }
                    }

                    for(size_t k = 0; k < bytes_read && opos < maxlen - 1; k++)
                    {
                        out[opos++] = resbuf[k];
                    }
                    pclose(fp);
                }
                continue;
            }
            i++;
            if(line[i] == '{')
            {
                i++;
                char inner[512];
                int ilen = 0;
                int depth = 1;
                while(line[i] != '\0'
                      && ilen < 511)
                {
                    if(line[i] == '{')
                    {
                        depth++;
                    }
                    if(line[i] == '}')
                    {
                        depth--;
                        if(depth == 0)
                        {
                            i++;
                            break;
                        }
                    }
                    inner[ilen++] = line[i++];
                }
                inner[ilen] = '\0';
                expand_braced(out, &opos,
                              maxlen, inner);
            }
            else
            {
                /* $VAR — simple unbraced */
                char varname[256];
                int  vlen = 0;
                while(line[i] != '\0'
                      && vlen < 255)
                {
                    char c = line[i];
                    if(!((c >= 'A'
                          && c <= 'Z')
                         || (c >= 'a'
                             && c <= 'z')
                         || (c >= '0'
                             && c <= '9')
                         || c == '_'
                         || c == '?'))
                    {
                        break;
                    }
                    varname[vlen++] =
                        line[i++];
                    if(c == '?')
                    {
                        break;
                    }
                }
                varname[vlen] = '\0';
                const char *val =
                    cli_var_lookup(varname);
                if(val != NULL)
                {
                    emit_str(out, &opos,
                             maxlen, val);
                }
            }
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