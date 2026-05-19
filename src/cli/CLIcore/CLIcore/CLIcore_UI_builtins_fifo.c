#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
#include "CLIcore.h"


/**
 * @file    CLIcore_UI_builtins_fifo.c
 * @brief   FIFO management and stream/FPS listing
 *
 * Contains:
 *  - cli_fifo()        — dynamic FIFO management
 *  - cli_list_streams()— list shared-memory streams
 *  - cli_list_fps()    — list FPS instances
 *
 * @see CLIcore_UI_builtins.c for other builtins.
 */


/* Forward declarations for FIFO helpers
 * defined in CLIcore.c */
extern int  cli_fifo_open(const char *path);
extern void cli_fifo_close(void);

/**
 * @brief Dynamic FIFO management command
 *
 * Sub-commands:
 *   fifo create [path] — create+open FIFO
 *   fifo open <path>   — connect to FIFO
 *   fifo close         — close+remove FIFO
 *   fifo on            — enable FIFO input
 *   fifo off           — disable FIFO input
 *   fifo status        — print FIFO state
 *   fifo (no args)     — same as status
 *
 * @return RETURN_SUCCESS or RETURN_FAILURE
 */
errno_t cli_fifo(void)
{
    const char *sub = NULL;

    if(data.cmdNBarg >= 2)
    {
        sub = data.cmdargtoken[1] .val.string;
    }

    /* No sub-command → status */
    if(sub == NULL
       || strcmp(sub, "status") == 0)
    {
        printf("FIFO status:\n");
        if(data.fifoON == 1)
        {
            printf("  state : \033[32m" "ON\033[0m\n");
            printf("  path  : %s\n", data.fifoname);
            printf("  fd    : %d\n", data.fifofd);
        }
        else if(data.fifofd >= 0)
        {
            printf("  state : \033[33m" "PAUSED\033[0m " "(fifo open, input off)\n");
            printf("  path  : %s\n", data.fifoname);
            printf("  fd    : %d\n", data.fifofd);
        }
        else
        {
            printf("  state : \033[2m" "OFF\033[0m " "(no fifo)\n");
        }
        return RETURN_SUCCESS;
    }

    /* fifo create [path] */
    if(strcmp(sub, "create") == 0)
    {
        const char *path = NULL;
        if(data.cmdNBarg >= 3)
        {
            path = data.cmdargtoken[2] .val.string;
        }
        if(cli_fifo_open(path) != 0)
        {
            return RETURN_FAILURE;
        }
        return RETURN_SUCCESS;
    }

    /* fifo open <path> */
    if(strcmp(sub, "open") == 0)
    {
        if(data.cmdNBarg < 3)
        {
            printf("Usage: fifo open " "<path>\n");
            return RETURN_FAILURE;
        }
        /* Check that the FIFO exists */
        const char *path = data.cmdargtoken[2] .val.string;
        struct stat sb;
        if(stat(path, &sb) != 0
           || !S_ISFIFO(sb.st_mode))
        {
            printf("\033[31mfifo open:" " '%s' is not a " "FIFO\033[0m\n", path);
            return RETURN_FAILURE;
        }
        if(cli_fifo_open(path) != 0)
        {
            return RETURN_FAILURE;
        }
        return RETURN_SUCCESS;
    }

    /* fifo close */
    if(strcmp(sub, "close") == 0)
    {
        if(data.fifofd < 0)
        {
            printf("fifo: no fifo open\n");
            return RETURN_SUCCESS;
        }
        printf("\033[36m[fifo]\033[0m " "closing: %s\n", data.fifoname);
        cli_fifo_close();
        return RETURN_SUCCESS;
    }

    /* fifo on */
    if(strcmp(sub, "on") == 0)
    {
        if(data.fifofd < 0)
        {
            printf("fifo: no fifo open " "(use 'fifo create' " "first)\n");
            return RETURN_FAILURE;
        }
        data.fifoON = 1;
        printf("\033[36m[fifo]\033[0m " "input enabled\n");
        return RETURN_SUCCESS;
    }

    /* fifo off */
    if(strcmp(sub, "off") == 0)
    {
        data.fifoON = 0;
        printf("\033[36m[fifo]\033[0m " "input disabled\n");
        return RETURN_SUCCESS;
    }

    /* Unknown sub-command */
    printf(
        "fifo: unknown sub-command "
        "'%s'\n" "Usage: fifo " "[create|open|close|on|off" "|status]\n", sub);
    return RETURN_FAILURE;
}

/*
 * ============================================================
 *  List Streams / FPS Command
 * ============================================================
 */

/**
 * @brief CLI handler: list-streams
 *
 * Prints all available ImageStreamIO streams
 * as a space-separated list.
 */
errno_t cli_list_streams(void)
{
    DIR *dir;
    struct dirent *ent;
    int first = 1;

    if((dir = opendir(dcshmdir)) != NULL)
    {
        while((ent = readdir(dir)) != NULL)
        {
            char *ext = strstr(ent->d_name, ".im.shm");
            if(ext != NULL
               && strcmp(ext, ".im.shm") == 0)
            {
                int namelen = ext - ent->d_name;
                if(!first)
                {
                    printf(" ");
                }
                printf("%.*s", namelen, ent->d_name);
                first = 0;
            }
        }
        closedir(dir);
    }
    printf("\n");
    return RETURN_SUCCESS;
}

/**
 * @brief CLI handler: list-fps
 *
 * Prints all available FPS instances as a
 * space-separated list.
 */
errno_t cli_list_fps(void)
{
    DIR *dir;
    struct dirent *ent;
    int first = 1;

    if((dir = opendir(dcshmdir)) != NULL)
    {
        while((ent = readdir(dir)) != NULL)
        {
            if(strncmp(ent->d_name,
                       "fps.", 4) == 0)
            {
                char *ext = strstr(ent->d_name, ".datadir");
                if(ext != NULL
                   && strcmp(ext, ".datadir")
                      == 0)
                {
                    int namelen = ext - (ent->d_name + 4);
                    if(!first)
                    {
                        printf(" ");
                    }
                    printf("%.*s", namelen, ent->d_name + 4);
                    first = 0;
                }
            }
        }
        closedir(dir);
    }
    printf("\n");
    return RETURN_SUCCESS;
}
