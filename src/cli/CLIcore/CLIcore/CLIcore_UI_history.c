// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    CLIcore_UI_history.c
 * @brief   Core history management — paths, load,
 *          save, structured log init and logging.
 *
 * @see CLIcore_UI_history_display.c for display cmds.
 * @see CLIcore_UI_history_cmds.c for expansion, etc.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif
#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "milk_bash_engine.h"
#include "milk_server.h"

/**
 * @brief Return path to the readline history
 *        file (~/.milk_history).
 *
 * Lazy-initialised: builds the path once and
 * caches it in a static buffer.
 */
const char *CLI_history_file(void)
{
    static char path[1024] = { 0 };
    if (path[0] == '\0')
    {
        const char *home = getenv("HOME");
        if (home)
        {
            snprintf(path, sizeof(path), "%s/.milk_history", home);
        }
        else
        {
            snprintf(path, sizeof(path), ".milk_history");
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
 *        ~/.milk_history into readline (made if missing: lines are appended to it as they run).
 */
void cli_history_load(void)
{
#ifdef USE_READLINE
    int fd = open(CLI_history_file(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd >= 0)
    {
        close(fd);
    }
    read_history(CLI_history_file());
#endif
}

/**
 * @brief At exit: cap ~/.milk_history to MILK_HISTORY_MAXLINES. It is never rewritten from this
 *        session's list, which would drop the lines other sessions appended meanwhile (D39).
 */
void cli_history_save(void)
{
#ifdef USE_READLINE
    history_truncate_file(CLI_history_file(), MILK_HISTORY_MAXLINES);
#endif
}

const char *cli_history_session(void)
{
    static char own[16];
    const char *s = milk_bash_engine_session(); // the session milk-cli's bash uses
    if (s[0] == '\0')
    {
        s = milk_server_session_name(); // this process's own
    }
    if (s[0] == '\0') // milk-cli's own, before it listens: named after its pid
    {
        snprintf(own, sizeof(own), "%d", (int) getpid());
        s = own;
    }
    return s;
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
    static char path[1024] = { 0 };
    if (path[0] == '\0')
    {
        const char *home = getenv("HOME");
        if (home)
        {
            snprintf(path, sizeof(path), "%s/.milk_history_log", home);
        }
        else
        {
            snprintf(path, sizeof(path), ".milk_history_log");
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
    snprintf(data.session_id, sizeof(data.session_id), "%s-%d", data.processname, (int) pid);

    {
        const char *tty = ttyname(STDIN_FILENO);
        if (tty)
        {
            strncpy(data.session_tty, tty, sizeof(data.session_tty) - 1);
            data.session_tty[sizeof(data.session_tty) - 1] = '\0';
        }
        else
        {
            strncpy(data.session_tty, "?", sizeof(data.session_tty) - 1);
        }
    }

    clock_gettime(CLOCK_REALTIME, &data.session_start);
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
static FILE *s_log_fp = NULL;

/** log_entry_nf() - cli_history_log_entry() without the flush: a batch flushes once. */
static void log_entry_nf(const char *text, char type)
{
    if (text == NULL || text[0] == '\0')
    {
        return;
    }

    /* Do not log history-display commands —
     * they would pollute the log they read. */
    {
        static const char *const skip[] = { "ghistory", "lhistory", NULL };
        for (int k = 0; skip[k] != NULL; k++)
        {
            size_t n = strlen(skip[k]);
            if (strncmp(text, skip[k], n) == 0 &&
                (text[n] == '\0' || text[n] == ' ' || text[n] == '\t'))
            {
                return;
            }
        }
    }

    static time_t s_last_sec = 0;
    static char   s_tbuf[64] = "";

    if (s_log_fp == NULL)
    {
        s_log_fp = fopen(CLI_history_log_file(), "ae");
        if (s_log_fp == NULL)
        {
            return;
        }
    }

    {
        time_t now = time(NULL);
        if (now != s_last_sec)
        {
            s_last_sec = now;
            strftime(s_tbuf, sizeof(s_tbuf), "%Y-%m-%dT%H:%M:%S", localtime(&now));
        }
        fprintf(s_log_fp, "%s\t%s\t%s\t%c\t%s\n", s_tbuf, cli_history_session(), data.session_tty,
                type, text);
    }
}

static void cli_history_log_entry(const char *text, char type)
{
    log_entry_nf(text, type);
    if (s_log_fp != NULL)
    {
        fflush(s_log_fp);
    }
}

void cli_history_log_prompt_lines(const char *text)
{
    char  sbuf[256];
    char *heap = NULL;
    for (const char *p = text; p != NULL && *p != '\0';)
    {
        const char *nl  = strchr(p, '\n');
        size_t      len = (nl != NULL) ? (size_t) (nl - p) : strlen(p);
        char       *line;
        if (len < sizeof(sbuf))
        {
            memcpy(sbuf, p, len);
            sbuf[len] = '\0';
            line      = sbuf;
        }
        else
        {
            free(heap);
            heap = strndup(p, len);
            line = heap;
        }
        if (line != NULL)
        {
            log_entry_nf(line, 'P');
            cli_session_log_cmd(line);
        }
        p = (nl != NULL) ? nl + 1 : p + len;
    }
    free(heap);
    if (s_log_fp != NULL)
    {
        fflush(s_log_fp);
    }
}

/**
 * @brief Log a resolved milk command (type C).
 */
void cli_history_log_cmd(const char *cmd)
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
void cli_history_log_prompt(const char *prompt)
{
    cli_history_log_entry(prompt, 'P');
    cli_session_log_cmd(prompt);
}

/**
 * @brief Log a shell bypass entry (type S).
 *
 * Called when a command is delegated directly
 * to bash via system().
 */
void cli_history_log_shell(const char *cmd)
{
    cli_history_log_entry(cmd, 'S');
}
