/**
 * @file CLIcore_script_case.c
 *
 * @brief Case/esac block and user-defined functions
 *
 * Implements:
 *
 * - **function name { body }**: User-defined functions
 *   stored in a function table. When called, parameters
 *   are bound to $1, $2, etc. and local variable scoping
 *   is pushed/popped.
 *
 * - **case/esac**: Pattern-matching dispatch similar to
 *   bash case statements. Supports glob patterns with
 *   fnmatch() and |-delimited alternatives.
 *
 * ## Function storage
 *
 * Functions are stored in cli_funcs[] with their body
 * lines pre-captured. cli_func_call() pushes a local
 * scope, binds positional parameters, and executes
 * the function body. cli_return_flag terminates
 * execution early from within a function.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fnmatch.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"

/**
 * @brief Define a new script function.
 *
 * Registers a function body for later invocation
 * by name.
 */
void cli_func_define(const char *name, char body[][STRINGMAXLEN_CLICMDLINE], int nbody)
{
    /* Update existing */
    for (int i = 0; i < CLI_MAX_FUNCS; i++)
    {
        if (cli_funcs[i].used && strcmp(cli_funcs[i].name, name) == 0)
        {
            cli_funcs[i].nbody = nbody;
            for (int j = 0; j < nbody; j++)
            {
                strncpy(cli_funcs[i].body[j], body[j], STRINGMAXLEN_CLICMDLINE - 1);
            }
            return;
        }
    }
    /* Find empty slot */
    for (int i = 0; i < CLI_MAX_FUNCS; i++)
    {
        if (!cli_funcs[i].used)
        {
            strncpy(cli_funcs[i].name, name, CLI_FUNC_NAMELEN - 1);
            cli_funcs[i].name[CLI_FUNC_NAMELEN - 1] = '\0';
            cli_funcs[i].nbody                      = nbody;
            cli_funcs[i].used                       = 1;
            for (int j = 0; j < nbody; j++)
            {
                strncpy(cli_funcs[i].body[j], body[j], STRINGMAXLEN_CLICMDLINE - 1);
            }
            return;
        }
    }
    printf("Error: function table full "
           "(max %d)\n",
           CLI_MAX_FUNCS);
}


/* ============================================================
 *  Block Intercept — main entry point
 * ============================================================
 *
 * Called from CLI_execute_line() before any other
 * processing. Returns 1 if the line was consumed
 * (buffered or block completed).
 */

/**
 * @brief Intercept line for flow control
 *
 * @param line  The raw command line
 * @return 1 if consumed, 0 if not
 */
/* ============================================================
 *  Case/esac Evaluator
 * ============================================================
 *
 * Syntax:
 *   case <word> in
 *     pattern1) cmd1 ;;
 *     pat2|pat3) cmd2 ;;
 *     *) default ;;
 *   esac
 */
void cli_exec_block_case(char (*lines)[STRINGMAXLEN_CLICMDLINE], int nlines)
{
    /* Line 0 = "case <word> in" */
    const char *hdr = strip_ws(lines[0]);
    hdr += 4; /* skip "case" */
    hdr = strip_ws(hdr);
    char word[256];
    {
        int wi = 0;
        while (*hdr != '\0' && *hdr != ' ' && *hdr != '\t' && wi < 255)
        {
            word[wi++] = *hdr++;
        }
        word[wi] = '\0';
    }
    /* Expand word */
    cli_expand_env(word, 256);

    /* Scan patterns: "pat) body ;;" */
    for (int i = 1; i < nlines; i++)
    {
        const char *lp = strip_ws(lines[i]);
        /* Find closing ')' */
        const char *cp = strchr(lp, ')');
        if (cp == NULL)
        {
            continue;
        }
        /* Extract pattern(s) */
        char pat[256];
        int  plen = (int) (cp - lp);
        if (plen >= 256)
        {
            plen = 255;
        }
        memcpy(pat, lp, (size_t) plen);
        pat[plen] = '\0';

        /* Check match (supports pat1|pat2
         * and * wildcard) */
        int matched = 0;
        {
            char ptmp[256];
            strncpy(ptmp, pat, sizeof(ptmp) - 1);
            ptmp[sizeof(ptmp) - 1] = '\0';
            char *psave            = NULL;
            char *pp               = strtok_r(ptmp, "|", &psave);
            while (pp != NULL)
            {
                /* strip ws */
                while (*pp == ' ' || *pp == '\t')
                {
                    pp++;
                }
                if (strcmp(pp, "*") == 0 || strcmp(pp, word) == 0)
                {
                    matched = 1;
                    break;
                }
                pp = strtok_r(NULL, "|", &psave);
            }
        }
        if (!matched)
        {
            continue;
        }

        /* Collect body lines until ;; */
        const char *body_start = cp + 1;
        while (*body_start == ' ' || *body_start == '\t')
        {
            body_start++;
        }
        /* If body is on same line */
        if (*body_start != '\0')
        {
            /* Strip ;; from end */
            char cmdline[STRINGMAXLEN_CLICMDLINE];
            strncpy(cmdline, body_start, STRINGMAXLEN_CLICMDLINE - 1);
            cmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
            {
                int cl = (int) strlen(cmdline);
                while (cl > 1 && cmdline[cl - 1] == ';' && cmdline[cl - 2] == ';')
                {
                    cmdline[cl - 2] = '\0';
                    cl -= 2;
                }
                /* Trim trailing ws */
                while (cl > 0 && (cmdline[cl - 1] == ' ' || cmdline[cl - 1] == '\t'))
                {
                    cmdline[--cl] = '\0';
                }
            }
            if (strlen(cmdline) > 0)
            {
                strncpy(data.CLIcmdline, cmdline, STRINGMAXLEN_CLICMDLINE - 1);
                CLI_execute_line();
            }
        }
        else
        {
            /* Multi-line body */
            for (int j = i + 1; j < nlines; j++)
            {
                const char *bl = strip_ws(lines[j]);
                if (strcmp(bl, ";;") == 0)
                {
                    break;
                }
                /* Strip trailing ;; */
                char cmd2[STRINGMAXLEN_CLICMDLINE];
                strncpy(cmd2, bl, STRINGMAXLEN_CLICMDLINE - 1);
                cmd2[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
                {
                    int c2l        = (int) strlen(cmd2);
                    int ends_dsemi = 0;
                    while (c2l > 1 && cmd2[c2l - 1] == ';' && cmd2[c2l - 2] == ';')
                    {
                        cmd2[c2l - 2] = '\0';
                        c2l -= 2;
                        ends_dsemi = 1;
                    }
                    while (c2l > 0 && (cmd2[c2l - 1] == ' ' || cmd2[c2l - 1] == '\t'))
                    {
                        cmd2[--c2l] = '\0';
                    }
                    if (strlen(cmd2) > 0)
                    {
                        strncpy(data.CLIcmdline, cmd2, STRINGMAXLEN_CLICMDLINE - 1);
                        CLI_execute_line();
                    }
                    if (ends_dsemi)
                    {
                        break;
                    }
                }
            }
        }
        return; /* first match only */
    }
}
