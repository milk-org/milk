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
 *  Command Alias Subsystem
 * ============================================================
 *
 * Aliases are stored in data.alias[] and persisted
 * to ~/.milk_aliases (one "name=command" per line).
 */

/** @brief Path to alias persistence file */
const char *CLI_alias_file(void)
{
    static char path[1024] = { 0 };
    if (path[0] == '\0')
    {
        const char *home = getenv("HOME");
        if (home)
        {
            snprintf(path, sizeof(path), "%s/.milk_aliases", home);
        }
        else
        {
            snprintf(path, sizeof(path), ".milk_aliases");
        }
    }
    return path;
}

/**
 * @brief Load aliases from ~/.milk_aliases
 */
void cli_alias_load(void)
{
    data.NBalias = 0;
    FILE *fp     = fopen(CLI_alias_file(), "r");
    if (fp == NULL)
    {
        return;
    }

    char line[CLI_ALIAS_NAMELEN + CLI_ALIAS_CMDLEN + 4];
    while (fgets(line, (int) sizeof(line), fp) != NULL)
    {
        if (data.NBalias >= CLI_MAX_ALIASES)
        {
            break;
        }
        /* Strip trailing newline */
        line[strcspn(line, "\n")] = '\0';
        /* Skip empty / comment lines */
        if (line[0] == '\0' || line[0] == '#')
        {
            continue;
        }
        /* Split on first '=' */
        char *eq = strchr(line, '=');
        if (eq == NULL)
        {
            continue;
        }
        *eq = '\0';
        strncpy(data.alias[data.NBalias].name, line, CLI_ALIAS_NAMELEN - 1);
        data.alias[data.NBalias].name[CLI_ALIAS_NAMELEN - 1] = '\0';
        strncpy(data.alias[data.NBalias].cmd, eq + 1, CLI_ALIAS_CMDLEN - 1);
        data.alias[data.NBalias].cmd[CLI_ALIAS_CMDLEN - 1] = '\0';
        data.NBalias++;
    }
    fclose(fp);
}

/**
 * @brief Save aliases to ~/.milk_aliases
 */
void cli_alias_save(void)
{
    FILE *fp = fopen(CLI_alias_file(), "w");
    if (fp == NULL)
    {
        printf("ERROR: cannot write %s\n", CLI_alias_file());
        return;
    }
    for (int i = 0; i < data.NBalias; i++)
    {
        fprintf(fp, "%s=%s\n", data.alias[i].name, data.alias[i].cmd);
    }
    fclose(fp);
}

/**
 * @brief CLI handler: alias <name> <command...>
 *
 * Creates or updates a command alias.
 */
errno_t cli_alias_add(void)
{
    if (data.cmdNBarg < 3)
    {
        printf("Usage: alias <name> <command...>\n");
        return RETURN_FAILURE;
    }

    const char *name = data.cmdargtoken[1].val.string;

    /* Build command from args 2..N */
    char cmd[CLI_ALIAS_CMDLEN];
    cmd[0] = '\0';
    for (long a = 2; a < data.cmdNBarg; a++)
    {
        if (a > 2)
        {
            strncat(cmd, " ", CLI_ALIAS_CMDLEN - strlen(cmd) - 1);
        }
        strncat(cmd, data.cmdargtoken[a].val.string, CLI_ALIAS_CMDLEN - strlen(cmd) - 1);
    }

    /* Check if alias already exists — update */
    for (int i = 0; i < data.NBalias; i++)
    {
        if (strcmp(data.alias[i].name, name) == 0)
        {
            strncpy(data.alias[i].cmd, cmd, CLI_ALIAS_CMDLEN - 1);
            data.alias[i].cmd[CLI_ALIAS_CMDLEN - 1] = '\0';
            cli_alias_save();
            printf("Alias updated: %s = %s\n", name, cmd);
            return RETURN_SUCCESS;
        }
    }

    /* Add new alias */
    if (data.NBalias >= CLI_MAX_ALIASES)
    {
        printf("ERROR: alias table full (%d)\n", CLI_MAX_ALIASES);
        return RETURN_FAILURE;
    }

    strncpy(data.alias[data.NBalias].name, name, CLI_ALIAS_NAMELEN - 1);
    data.alias[data.NBalias].name[CLI_ALIAS_NAMELEN - 1] = '\0';
    strncpy(data.alias[data.NBalias].cmd, cmd, CLI_ALIAS_CMDLEN - 1);
    data.alias[data.NBalias].cmd[CLI_ALIAS_CMDLEN - 1] = '\0';
    data.NBalias++;

    cli_alias_save();
    printf("Alias created: %s = %s\n", name, cmd);

    return RETURN_SUCCESS;
}

/**
 * @brief CLI handler: unalias <name>
 */
errno_t cli_alias_remove(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: unalias <name>\n");
        return RETURN_FAILURE;
    }

    const char *name = data.cmdargtoken[1].val.string;

    for (int i = 0; i < data.NBalias; i++)
    {
        if (strcmp(data.alias[i].name, name) == 0)
        {
            /* Shift remaining entries */
            for (int j = i; j < data.NBalias - 1; j++)
            {
                data.alias[j] = data.alias[j + 1];
            }
            data.NBalias--;
            cli_alias_save();
            printf("Alias removed: %s\n", name);
            return RETURN_SUCCESS;
        }
    }

    printf("Alias '%s' not found\n", name);
    return RETURN_FAILURE;
}

/**
 * @brief CLI handler: aliases
 */
errno_t cli_alias_list(void)
{
    if (data.NBalias == 0)
    {
        printf("No aliases defined.\n");
        return RETURN_SUCCESS;
    }
    printf("--- Aliases (%d) ---\n", data.NBalias);
    for (int i = 0; i < data.NBalias; i++)
    {
        printf("  %-16s = %s\n", data.alias[i].name, data.alias[i].cmd);
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Expand alias in data.CLIcmdline
 *
 * If the first word matches an alias name,
 * substitute it with the alias command, keeping
 * any trailing arguments.
 */
void cli_alias_expand(void)
{
    if (data.NBalias == 0)
    {
        return;
    }

    /* Extract first word */
    char        firstword[CLI_ALIAS_NAMELEN];
    int         fwlen = 0;
    const char *p     = data.CLIcmdline;

    /* Skip leading whitespace */
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }
    while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\n')
    {
        if (fwlen < CLI_ALIAS_NAMELEN - 1)
        {
            firstword[fwlen++] = *p;
        }
        p++;
    }
    firstword[fwlen] = '\0';

    if (fwlen == 0)
    {
        return;
    }

    /* Search aliases */
    for (int i = 0; i < data.NBalias; i++)
    {
        if (strcmp(data.alias[i].name, firstword) == 0)
        {
            /* Build expanded line */
            char expanded[STRINGMAXLEN_CLICMDLINE + CLI_ALIAS_CMDLEN];
            snprintf(expanded, sizeof(expanded), "%s%s", data.alias[i].cmd,
                     p); /* p points to rest */
            strncpy(data.CLIcmdline, expanded, STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
            return;
        }
    }
}
