#include <stddef.h>
extern int cli_find_in_path(const char *cmd, char *outpath, size_t outsize);
extern int processinfo_procdirname(char *procdirname);
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "CLIcore.h"
#include "CLIcore_script.h"
#include "milkscript.h"
#include "CLIcore_utils.h"
#include "CLIcore_memory.h"
#include "CLIcore_modules.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_checkargs.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>

extern int cli_block_level;
extern int cli_break_flag;
extern int CLI_trap_enable;
extern int cli_cmd_delay_us;

/**
 * @brief Handler: while loop control flow.
 */
int cli_intercept_cmd_while(const char *p)
{
    if (starts_with(p, "while ") || starts_with(p, "while\t"))
    {
        if (cli_block_level >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk = &cli_block_stack[cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type   = CLI_BLOCK_WHILE;
        blk->active = 1;
        strncpy(blk->lines[0], p, STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }
    return 0;
}

/**
 * @brief Handler: until loop control flow.
 */
int cli_intercept_cmd_until(const char *p)
{
    if (starts_with(p, "until ") || starts_with(p, "until\t"))
    {
        if (cli_block_level >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk = &cli_block_stack[cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type   = CLI_BLOCK_UNTIL;
        blk->active = 1;
        strncpy(blk->lines[0], p, STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }
    return 0;
}

/**
 * @brief Handler: for loop control flow.
 */
int cli_intercept_cmd_for(const char *p)
{
    if (starts_with(p, "for ") || starts_with(p, "for\t"))
    {
        if (cli_block_level >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk = &cli_block_stack[cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type   = CLI_BLOCK_FOR;
        blk->active = 1;
        strncpy(blk->lines[0], p, STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_select(const char *p)
{
    if (starts_with(p, "select ") || starts_with(p, "select\t"))
    {
        if (cli_block_level >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk = &cli_block_stack[cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type   = CLI_BLOCK_SELECT;
        blk->active = 1;
        strncpy(blk->lines[0], p, STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_function(const char *p)
{
    if (starts_with(p, "function ") || starts_with(p, "function\t"))
    {
        if (cli_block_level >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk = &cli_block_stack[cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type   = CLI_BLOCK_FUNC;
        blk->active = 1;
        strncpy(blk->lines[0], p, STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_case(const char *p)
{
    if (starts_with(p, "case ") || starts_with(p, "case\t"))
    {
        if (cli_block_level >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk = &cli_block_stack[cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type   = CLI_BLOCK_CASE;
        blk->active = 1;
        strncpy(blk->lines[0], p, STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_true(const char *p)
{
    if (strcmp(p, "true") == 0)
    {
        cli_last_retval = 0;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_false(const char *p)
{
    if (strcmp(p, "false") == 0)
    {
        cli_last_retval = 1;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_math_eval(const char *p)
{
    if (starts_with(p, "((") && strlen(p) >= 5)
    {
        int plen = (int) strlen(p);
        if (p[plen - 1] == ')' && p[plen - 2] == ')')
        {
            char aexpr[STRINGMAXLEN_CLICMDLINE];
            int  elen = plen - 4;
            if (elen >= STRINGMAXLEN_CLICMDLINE)
            {
                elen = STRINGMAXLEN_CLICMDLINE - 1;
            }
            memcpy(aexpr, p + 2, (size_t) elen);
            aexpr[elen] = '\0';
            /* Wrap in $(( )) and
             * expand */
            char wrap[STRINGMAXLEN_CLICMDLINE + 64];
            snprintf(wrap, sizeof(wrap), "$((%s))", aexpr);
            cli_expand_env(wrap, STRINGMAXLEN_CLICMDLINE);
            long val        = strtol(wrap, NULL, 10);
            cli_last_retval = (val != 0) ? 0 : 1;
            return 1;
        }
    }
    return 0;
}

int cli_intercept_cmd_alias(const char *p)
{
    if (starts_with(p, "alias ") || starts_with(p, "alias\t") || strcmp(p, "alias") == 0)
    {
        p += 5;
        p = strip_ws(p);
        if (*p == '\0')
        {
            /* List all aliases */
            for (int k = 0; k < data.NBalias; k++)
            {
                printf("alias %s="
                       "'%s'\n",
                       data.alias[k].name, data.alias[k].cmd);
            }
        }
        else
        {
            /* alias name='cmd' or
             * alias name=cmd */
            char *eq = strchr(p, '=');
            if (eq != NULL)
            {
                char aname[CLI_ALIAS_NAMELEN];
                int  nl = (int) (eq - p);
                if (nl >= CLI_ALIAS_NAMELEN)
                {
                    nl = CLI_ALIAS_NAMELEN - 1;
                }
                memcpy(aname, p, (size_t) nl);
                aname[nl]      = '\0';
                const char *av = eq + 1;
                /* Strip quotes */
                int avl = (int) strlen(av);
                if (avl >= 2 && ((av[0] == '\'' && av[avl - 1] == '\'') ||
                                 (av[0] == '"' && av[avl - 1] == '"')))
                {
                    av++;
                    avl -= 2;
                }
                /* Update existing? */
                int slot = -1;
                for (int k = 0; k < data.NBalias; k++)
                {
                    if (strcmp(data.alias[k].name, aname) == 0)
                    {
                        slot = k;
                        break;
                    }
                }
                if (slot < 0 && data.NBalias < CLI_MAX_ALIASES)
                {
                    slot = data.NBalias++;
                }
                if (slot >= 0)
                {
                    strncpy(data.alias[slot].name, aname, CLI_ALIAS_NAMELEN - 1);
                    data.alias[slot].name[CLI_ALIAS_NAMELEN - 1] = '\0';
                    int cl = avl < CLI_ALIAS_CMDLEN - 1 ? avl : CLI_ALIAS_CMDLEN - 1;
                    memcpy(data.alias[slot].cmd, av, (size_t) cl);
                    data.alias[slot].cmd[cl] = '\0';
                }
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_unalias(const char *p)
{
    if (starts_with(p, "unalias ") || starts_with(p, "unalias\t"))
    {
        p += 7;
        p = strip_ws(p);
        for (int k = 0; k < data.NBalias; k++)
        {
            if (strcmp(data.alias[k].name, p) == 0)
            {
                /* Shift remaining */
                for (int j = k; j < data.NBalias - 1; j++)
                {
                    data.alias[j] = data.alias[j + 1];
                }
                data.NBalias--;
                break;
            }
        }
        return 1;
    }
    return 0;
}
