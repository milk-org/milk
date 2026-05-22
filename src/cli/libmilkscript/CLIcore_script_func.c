#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"


/**
 * @brief Execute lines through CLI_execute_line
 */
void cli_exec_lines(char lines[][STRINGMAXLEN_CLICMDLINE], int nlines)
{
    for (int i = 0; i < nlines; i++)
    {
        if (cli_break_flag || cli_continue_flag || cli_return_flag)
        {
            break;
        }

        /* Copy to cmdline and execute */
        strncpy(data.CLIcmdline, lines[i], STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        CLI_execute_line();
    }
}

/**
 * @brief Find a user-defined function by name
 */
CLI_FUNC *cli_func_find(const char *name)
{
    for (int i = 0; i < CLI_MAX_FUNCS; i++)
    {
        if (cli_funcs[i].used && strcmp(cli_funcs[i].name, name) == 0)
        {
            return &cli_funcs[i];
        }
    }
    return NULL;
}


/**
 * @brief Try to call a user-defined function
 *
 * Syntax: funcname arg1 arg2 ...
 * Inside the function body, $1..$9 are args.
 *
 * @return 1 if matched, 0 if not
 */
int cli_try_func_call(const char *line)
{
    const char *p = strip_ws(line);

    /* Extract first word (function name) */
    char fname[CLI_FUNC_NAMELEN];
    {
        int fn = 0;
        while (*p != '\0' && *p != ' ' && *p != '\t' && fn < CLI_FUNC_NAMELEN - 1)
        {
            fname[fn++] = *p++;
        }
        fname[fn] = '\0';
    }

    CLI_FUNC *func = cli_func_find(fname);
    if (func == NULL)
    {
        return 0;
    }

    /* Parse arguments */
    p = strip_ws(p);
    char *args[CLI_FUNC_MAXARGS];
    char  argbuf[CLI_FUNC_MAXARGS][CLI_VAR_VALLEN];
    int   nargs = 0;

    while (*p != '\0' && nargs < CLI_FUNC_MAXARGS)
    {
        int ai = 0;
        while (*p != '\0' && *p != ' ' && *p != '\t' && ai < CLI_VAR_VALLEN - 1)
        {
            argbuf[nargs][ai++] = *p++;
        }
        argbuf[nargs][ai] = '\0';
        args[nargs]       = argbuf[nargs];
        nargs++;
        p = strip_ws(p);
    }

    /* Save old $1..$9, set new ones */
    char old_args[CLI_FUNC_MAXARGS][CLI_VAR_VALLEN];
    int  old_used[CLI_FUNC_MAXARGS];
    for (int i = 0; i < CLI_FUNC_MAXARGS; i++)
    {
        char aname[4];
        snprintf(aname, sizeof(aname), "%d", i + 1);
        const char *ov = cli_var_get(aname);
        old_used[i]    = (ov != NULL) ? 1 : 0;
        if (ov != NULL)
        {
            strncpy(old_args[i], ov, CLI_VAR_VALLEN - 1);
            old_args[i][CLI_VAR_VALLEN - 1] = '\0';
        }
        if (i < nargs)
        {
            cli_var_set(aname, args[i]);
        }
        else
        {
            cli_var_unset(aname);
        }
    }

    /* Push local variable scope */
    if (cli_local_depth < CLI_MAX_LOCAL_DEPTH - 1)
    {
        cli_local_depth++;
        cli_local_shadow_count[cli_local_depth] = 0;
    }

    /* Execute body lines */
    cli_return_flag = 0;
    cli_exec_lines(func->body, func->nbody);
    cli_return_flag = 0;

    /* Restore old $1..$9 */
    for (int i = 0; i < CLI_FUNC_MAXARGS; i++)
    {
        char aname[4];
        snprintf(aname, sizeof(aname), "%d", i + 1);
        if (old_used[i])
        {
            cli_var_set(aname, old_args[i]);
        }
        else
        {
            cli_var_unset(aname);
        }
    }

    /* Restore variables shadowed by 'local' */
    if (cli_local_depth > 0)
    {
        int scount = cli_local_shadow_count[cli_local_depth];
        for (int i = 0; i < scount; i++)
        {
            CLI_LOCAL_SHADOW *sh = &cli_local_shadows[cli_local_depth][i];
            if (sh->was_used)
            {
                cli_var_set(sh->name, sh->val);
            }
            else
            {
                cli_var_unset(sh->name);
            }
        }
        cli_local_depth--;
    }

    return 1;
}
