#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif
#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
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
void cli_build_prompt(const char *fmt, char *out, int maxlen)
{
    int pos = 0;
    for (int i = 0; fmt[i] != '\0' && pos < maxlen - 1; i++)
    {
        if (fmt[i] == '%' && fmt[i + 1] != '\0')
        {
            i++;
            switch (fmt[i])
            {
            case 'h':
            {
                char hn[64];
                gethostname(hn, sizeof(hn));
                pos += snprintf(out + pos, (size_t) (maxlen - pos), "%s", hn);
                break;
            }
            case 'u':
            {
                const char *u = getenv("USER");
                pos += snprintf(out + pos, (size_t) (maxlen - pos), "%s", u ? u : "?");
                break;
            }
            case 'd':
            {
                char cwd[256];
                if (getcwd(cwd, sizeof(cwd)))
                {
                    char *base = strrchr(cwd, '/');
                    pos +=
                        snprintf(out + pos, (size_t) (maxlen - pos), "%s", base ? base + 1 : cwd);
                }
                break;
            }
            case 't':
            {
                time_t     now = time(NULL);
                struct tm *tm  = localtime(&now);
                pos += (int) strftime(out + pos, (size_t) (maxlen - pos), "%H:%M:%S", tm);
                break;
            }
            case 'n':
                pos += snprintf(out + pos, (size_t) (maxlen - pos), "%s", data.processname);
                break;
            default:
                if (pos < maxlen - 2)
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

/**
 * @brief Update the CLI prompt string.
 *
 * Reflects current directory, session name, and
 * script nesting level.
 */
errno_t cli_setprompt(void)
{
    if (data.cmdNBarg < 2)
    {
        if (cli_prompt_format[0] != '\0')
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
    strncpy(cli_prompt_format, data.cmdargtoken[1].val.string, sizeof(cli_prompt_format) - 1);
    cli_prompt_format[sizeof(cli_prompt_format) - 1] = '\0';
    printf("Prompt set to: '%s'\n", cli_prompt_format);
    return RETURN_SUCCESS;
}


/**
 * @brief Expand {N..M} and {N..M..S} brace ranges
 *
 * Replaces tokens like {1..5} with "1 2 3 4 5"
 * and {0..10..2} with "0 2 4 6 8 10".
 */
void emit_str(char *out, int *opos, int maxlen, const char *s);
/**
 * @brief Expand brace expressions in a command line.
 *
 * Supports {a,b,c} expansion like bash.
 */
void cli_expand_braces(char *line, int maxlen)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i    = 0;

    while (line[i] != '\0' && opos < maxlen - 1)
    {
        if (line[i] == '{')
        {
            /* Try {N..M} or {N..M..S} */
            char *endp = NULL;
            long  sv   = strtol(line + i + 1, &endp, 10);
            if (endp != NULL && endp[0] == '.' && endp[1] == '.')
            {
                char *endp2 = NULL;
                long  ev    = strtol(endp + 2, &endp2, 10);
                long  step  = 1;
                if (endp2 != NULL && endp2[0] == '.' && endp2[1] == '.')
                {
                    char *endp3 = NULL;
                    step        = strtol(endp2 + 2, &endp3, 10);
                    endp2       = endp3;
                }
                if (endp2 != NULL && *endp2 == '}' && step != 0)
                {
                    int first = 1;
                    if (sv <= ev)
                    {
                        if (step < 0)
                        {
                            step = -step;
                        }
                        for (long v = sv; v <= ev; v += step)
                        {
                            char nb[32];
                            snprintf(nb, sizeof(nb), "%s%ld", first ? "" : " ", v);
                            first = 0;
                            emit_str(out, &opos, maxlen, nb);
                        }
                    }
                    else
                    {
                        if (step > 0)
                        {
                            step = -step;
                        }
                        for (long v = sv; v >= ev; v += step)
                        {
                            char nb[32];
                            snprintf(nb, sizeof(nb), "%s%ld", first ? "" : " ", v);
                            first = 0;
                            emit_str(out, &opos, maxlen, nb);
                        }
                    }
                    i = (int) (endp2 - line) + 1;
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
 * @brief Emit string into output buffer
 */
void emit_str(char *out, int *opos, int maxlen, const char *s)
{
    while (*s != '\0' && *opos < maxlen - 1)
    {
        out[(*opos)++] = *s++;
    }
}

// cli_expand_env moved to CLIcore_script_expand.c
