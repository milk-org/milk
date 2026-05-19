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
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"

struct bookmark_entry
{
    char name[BOOKMARK_NAMELEN];
    char cmd[BOOKMARK_CMDLEN];
};

struct bookmark_entry bookmarks[BOOKMARK_MAX];
#include "CLIcore/cli_calc_parser.h"
#include <glob.h>
#include <sys/wait.h>
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

int bookmark_count = 0;

/**
 * @brief Load bookmarks from ~/.milk_bookmarks
 */
void cli_bookmark_load(void)
{
    char path[STRINGMAXLEN_FULLFILENAME];
    snprintf(path, STRINGMAXLEN_FULLFILENAME, "%s/.milk_bookmarks", getenv("HOME"));
    FILE *fp = fopen(path, "r");
    if(fp == NULL)
    {
        return;
    }
    bookmark_count = 0;
    char line[1024];
    while(fgets(line, (int) sizeof(line), fp)
            && bookmark_count < BOOKMARK_MAX)
    {
        size_t len = strlen(line);
        if(len > 0 && line[len - 1] == '\n')
        {
            line[len - 1] = '\0';
        }
        char *tab = strchr(line, '\t');
        if(tab == NULL)
        {
            continue;
        }
        *tab = '\0';
        strncpy(bookmarks[bookmark_count].name, line, BOOKMARK_NAMELEN - 1);
        bookmarks[bookmark_count].name[BOOKMARK_NAMELEN - 1] = '\0';
        strncpy(bookmarks[bookmark_count].cmd, tab + 1, BOOKMARK_CMDLEN - 1);
        bookmarks[bookmark_count].cmd[BOOKMARK_CMDLEN - 1] = '\0';
        bookmark_count++;
    }
    fclose(fp);
}

/**
 * @brief Save bookmarks to ~/.milk_bookmarks
 */
void cli_bookmark_save(void)
{
    char path[STRINGMAXLEN_FULLFILENAME];
    snprintf(path, STRINGMAXLEN_FULLFILENAME, "%s/.milk_bookmarks", getenv("HOME"));
    FILE *fp = fopen(path, "w");
    if(fp == NULL)
    {
        return;
    }
    for(int i = 0; i < bookmark_count; i++)
    {
        fprintf(fp, "%s\t%s\n", bookmarks[i].name, bookmarks[i].cmd);
    }
    fclose(fp);
}

/**
 * @brief Handle bookmark create/goto commands.
 *
 * Manages named bookmarks for directory navigation.
 */
errno_t cli_bookmark(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage:\n"
               "  bookmark save <name> "
               "\"cmd1 ; cmd2\"\n"
               "  bookmark run  <name>\n" "  bookmark list\n" "  bookmark rm   <name>\n");
        return RETURN_SUCCESS;
    }
    const char *action = data.cmdargtoken[1].val.string;

    if(strcmp(action, "list") == 0)
    {
        if(bookmark_count == 0)
        {
            printf("No bookmarks saved\n");
        }
        for(int i = 0; i < bookmark_count; i++)
        {
            printf("  \033[1m%-16s\033[0m %s\n", bookmarks[i].name, bookmarks[i].cmd);
        }
        return RETURN_SUCCESS;
    }

    if(strcmp(action, "save") == 0)
    {
        if(data.cmdNBarg < 4)
        {
            printf("Usage: bookmark save " "<name> \"cmd\"\n");
            return RETURN_FAILURE;
        }
        if(bookmark_count >= BOOKMARK_MAX)
        {
            printf("Bookmark limit reached\n");
            return RETURN_FAILURE;
        }
        strncpy(
            bookmarks[bookmark_count].name, data.cmdargtoken[2].val.string, BOOKMARK_NAMELEN - 1);
        bookmarks[bookmark_count].name[BOOKMARK_NAMELEN - 1] = '\0';
        /* Join remaining args as command */
        {
            char cmd[BOOKMARK_CMDLEN] = "";
            for(long a = 3;
                    a < data.cmdNBarg; a++)
            {
                if(a > 3)
                {
                    strncat(cmd, " ", BOOKMARK_CMDLEN - strlen(cmd) - 1);
                }
                strncat(cmd, data.cmdargtoken[a] .val.string, BOOKMARK_CMDLEN - strlen(cmd) - 1);
            }
            strncpy(bookmarks[bookmark_count].cmd, cmd, BOOKMARK_CMDLEN - 1);
            bookmarks[bookmark_count].cmd[BOOKMARK_CMDLEN - 1] = '\0';
        }
        bookmark_count++;
        cli_bookmark_save();
        printf("Bookmark '%s' saved\n", data.cmdargtoken[2].val.string);
        return RETURN_SUCCESS;
    }

    if(strcmp(action, "run") == 0)
    {
        if(data.cmdNBarg < 3)
        {
            printf("Usage: bookmark run " "<name>\n");
            return RETURN_FAILURE;
        }
        const char *name = data.cmdargtoken[2].val.string;
        for(int i = 0; i < bookmark_count; i++)
        {
            if(strcmp(bookmarks[i].name,
                      name) == 0)
            {
                strncpy(data.CLIcmdline, bookmarks[i].cmd, STRINGMAXLEN_CLICMDLINE - 1);
                data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
                return CLI_execute_line();
            }
        }
        printf("Bookmark '%s' not found\n", name);
        return RETURN_FAILURE;
    }

    if(strcmp(action, "rm") == 0)
    {
        if(data.cmdNBarg < 3)
        {
            printf("Usage: bookmark rm " "<name>\n");
            return RETURN_FAILURE;
        }
        const char *name = data.cmdargtoken[2].val.string;
        for(int i = 0; i < bookmark_count; i++)
        {
            if(strcmp(bookmarks[i].name,
                      name) == 0)
            {
                for(int j = i;
                        j < bookmark_count - 1;
                        j++)
                {
                    bookmarks[j] = bookmarks[j + 1];
                }
                bookmark_count--;
                cli_bookmark_save();
                printf("Bookmark '%s' " "removed\n", name);
                return RETURN_SUCCESS;
            }
        }
        printf("Bookmark '%s' not found\n", name);
        return RETURN_FAILURE;
    }

    printf("Unknown bookmark action '%s'\n", action);
    return RETURN_FAILURE;
}
