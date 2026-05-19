#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#ifdef USE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
#endif
#include "CLIcore.h"

/**
 * @file    CLIcore_UI_history.c
 * @brief   Core history management — paths, load,
 *          save, structured log init and logging.
 *
 * @see CLIcore_UI_history_display.c for display cmds.
 * @see CLIcore_UI_history_cmds.c for expansion, etc.
 */


/**
 * @brief Return path to the readline history
 *        file (~/.milk_history).
 *
 * Lazy-initialised: builds the path once and
 * caches it in a static buffer.
 */
const char *CLI_history_file(void)
{
    static char path[1024] = {0};
    if(path[0] == '\0')
    {
        const char *home = getenv("HOME");
        if(home)
        {
            snprintf(path, sizeof(path),
                     "%s/.milk_history", home);
        }
        else
        {
            snprintf(path, sizeof(path),
                     ".milk_history");
        }
    }
    return path;
}


/*
 * ============================================================
 *  Persistent History (~/.milk_history)
 * ============================================================
 */

#define MILK_HISTORY_MAXLINES 1000

/**
 * @brief Load persistent history from
 *        ~/.milk_history into readline.
 */
void cli_history_load(void)
{
#ifdef USE_READLINE
    char hpath[STRINGMAXLEN_FULLFILENAME];
    snprintf(hpath,
             STRINGMAXLEN_FULLFILENAME,
             "%s/.milk_history",
             getenv("HOME"));
    read_history(hpath);
#endif
}

/**
 * @brief Save readline history to
 *        ~/.milk_history, truncating to
 *        MILK_HISTORY_MAXLINES.
 */
void cli_history_save(void)
{
#ifdef USE_READLINE
    char hpath[STRINGMAXLEN_FULLFILENAME];
    snprintf(hpath,
             STRINGMAXLEN_FULLFILENAME,
             "%s/.milk_history",
             getenv("HOME"));
    write_history(hpath);
    history_truncate_file(hpath,
                          MILK_HISTORY_MAXLINES);
#endif
}


/*
 * ============================================================
 *  Structured History Log (~/.milk_history_log)
 *
 *  Append-only TSV file that records every
 *  command with session metadata:
 *    <timestamp>\t<session_id>\t<tty>\t<cmd>
 * ============================================================
 */

/**
 * @brief Path to structured history log file
 */
const char *CLI_history_log_file(void)
{
    static char path[1024] = {0};
    if(path[0] == '\0')
    {
        const char *home = getenv("HOME");
        if(home)
        {
            snprintf(path, sizeof(path),
                     "%s/.milk_history_log",
                     home);
        }
        else
        {
            snprintf(path, sizeof(path),
                     ".milk_history_log");
        }
    }
    return path;
}

/**
 * @brief Initialize session identity
 *
 * Records processname, PID, terminal,
 * and start time into data struct.
 * Called once at startup.
 */
void cli_history_log_init(void)
{
    pid_t pid = getpid();
    snprintf(data.session_id,
             sizeof(data.session_id),
             "%s-%d",
             data.processname, (int) pid);

    {
        const char *tty = ttyname(STDIN_FILENO);
        if(tty)
        {
            strncpy(data.session_tty, tty,
                    sizeof(data.session_tty) - 1);
            data.session_tty[
                sizeof(data.session_tty) - 1]
                = '\0';
        }
        else
        {
            strncpy(data.session_tty, "?",
                    sizeof(data.session_tty) - 1);
        }
    }

    clock_gettime(CLOCK_REALTIME,
                  &data.session_start);
}

/**
 * @brief Internal: log a typed entry to the
 *        structured history file.
 *
 * Appends a 5-field TSV line:
 *   <timestamp>\t<session>\t<tty>\t<type>\t<text>
 *
 * Type codes:
 *   P = prompt (raw user input)
 *   C = command (resolved milk command)
 *   S = shell bypass
 *
 * @param text   The entry text to log.
 * @param type   Single-character type code.
 */
static void cli_history_log_entry(
    const char *text,
    char       type)
{
    if(text == NULL || text[0] == '\0')
    {
        return;
    }

    /* Do not log history-display commands —
     * they would pollute the log they read. */
    {
        static const char *const skip[] =
        {
            "ghistory",
            "lhistory",
            NULL
        };
        for(int k = 0; skip[k] != NULL; k++)
        {
            size_t n = strlen(skip[k]);
            if(strncmp(text, skip[k], n) == 0
               && (text[n] == '\0'
                   || text[n] == ' '
                   || text[n] == '\t'))
            {
                return;
            }
        }
    }

    FILE *fp = fopen(
        CLI_history_log_file(), "a");
    if(fp == NULL)
    {
        return;
    }

    {
        time_t now = time(NULL);
        char tbuf[64];
        strftime(tbuf, sizeof(tbuf),
                 "%Y-%m-%dT%H:%M:%S",
                 localtime(&now));
        fprintf(fp, "%s\t%s\t%s\t%c\t%s\n",
                tbuf,
                data.session_id,
                data.session_tty,
                type,
                text);
    }
    fclose(fp);
}

/**
 * @brief Log a resolved milk command (type C).
 */
void cli_history_log_cmd(
    const char *cmd
)
{
    cli_history_log_entry(cmd, 'C');
}

/**
 * @brief Log raw prompt input (type P).
 *
 * Called from rl_cb_linehandler() and the
 * fgets fallback to record exactly what the
 * user typed before alias/history expansion.
 */
void cli_history_log_prompt(
    const char *prompt
)
{
    cli_history_log_entry(prompt, 'P');
}

/**
 * @brief Log a shell bypass entry (type S).
 *
 * Called when a command is delegated directly
 * to bash via system().
 */
void cli_history_log_shell(
    const char *cmd
)
{
    cli_history_log_entry(cmd, 'S');
}
