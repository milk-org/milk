// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
#include "milk_bash_engine.h"
#include "cli_calc_parser.h"
#include <glob.h>
#include <sys/wait.h>

#include "timeutils.h"
#include "treesitter/cli_treesitter.h"

/** cli_milkrc_load() - source ~/.milkrc in the bash engine, if it exists (interactive startup only). */
void cli_milkrc_load(void)
{
    const char *home = getenv("HOME");
    char        rcpath[STRINGMAXLEN_FULLFILENAME];

    if (home == NULL)
    {
        return;
    }
    snprintf(rcpath, sizeof(rcpath), "%s/.milkrc", home);
    if (access(rcpath, R_OK) != 0)
    {
        return;
    }
    milk_bash_engine_exec_file(rcpath, NULL);
}

/**
 * @brief Display per-command invocation counts.
 *
 * Shows the top-20 most frequently called CLI
 * commands during this session, sorted by call
 * count in descending order.
 */
errno_t cli_cmdstats(void)
{
    typedef struct
    {
        const char *key;
        uint32_t    count;
    } CmdStatEntry;

    CmdStatEntry entries[DATA_NB_MAX_COMMAND];
    int          nused = 0;
    for (uint32_t i = 0; i < data.NBcmd; i++)
    {
        if (data.cmd[i].callcount > 0)
        {
            entries[nused].key   = data.cmd[i].key;
            entries[nused].count = data.cmd[i].callcount;
            nused++;
        }
    }
    if (nused == 0)
    {
        printf("No commands executed yet.\n");
        return RETURN_SUCCESS;
    }
    for (int i = 1; i < nused; i++)
    {
        CmdStatEntry tmp = entries[i];
        int          j   = i - 1;
        while (j >= 0 && entries[j].count < tmp.count)
        {
            entries[j + 1] = entries[j];
            j--;
        }
        entries[j + 1] = tmp;
    }
    int show = nused < 20 ? nused : 20;
    printf("\n\033[1mCommand usage "
           "(top %d):\033[0m\n",
           show);
    printf("  %-30s  %s\n", "COMMAND", "CALLS");
    printf("  %-30s  %s\n", "------------------------------", "-----");
    for (int i = 0; i < show; i++)
    {
        printf("  %-30s  %u\n", entries[i].key, entries[i].count);
    }
    printf("\n");
    return RETURN_SUCCESS;
}


/**
 * @brief Toggle command execution timing display.
 *
 * Usage: milk.timing [on|off]
 * With no args, toggles the current state.
 */
errno_t cli_timing_toggle(void)
{
    if (data.cmdNBarg >= 2)
    {
        const char *arg = data.cmdargtoken[1].val.string;
        if (strcmp(arg, "on") == 0 || strcmp(arg, "1") == 0)
        {
            data.print_cmd_timing = 1;
            printf("Command execution timing ON\n");
        }
        else if (strcmp(arg, "off") == 0 || strcmp(arg, "0") == 0)
        {
            data.print_cmd_timing = 0;
            printf("Command execution timing OFF\n");
        }
        else
        {
            printf("Usage: milk.timing [on|off]\n");
        }
    }
    else
    {
        data.print_cmd_timing = !data.print_cmd_timing;
        printf("Command execution timing %s\n", data.print_cmd_timing ? "ON" : "OFF");
    }
    return RETURN_SUCCESS;
}

#ifdef USE_READLINE
/**
 * @brief Toggle readline syntax highlighting.
 *
 * Usage: synhl [on|off]
 * With no args, toggles the current state.
 */
errno_t cli_syntax_highlight_toggle(void)
{
    if (data.cmdNBarg >= 2)
    {
        const char *arg = data.cmdargtoken[1].val.string;
        if (strcmp(arg, "on") == 0)
        {
            data.syntax_highlight = 2; // Default to full TS
            printf("Syntax highlighting ON (level 2)\n");
        }
        else if (strcmp(arg, "off") == 0)
        {
            data.syntax_highlight = 0;
            printf("Syntax highlighting OFF\n");
        }
        else if (arg[0] >= '0' && arg[0] <= '2' && arg[1] == '\0')
        {
            data.syntax_highlight = arg[0] - '0';
            printf("Syntax highlighting set to level %d\n", data.syntax_highlight);
        }
        else
        {
            printf("Usage: @config synhl [on|off|0|1|2]\n");
        }
    }
    else
    {
        data.syntax_highlight = (data.syntax_highlight + 1) % 3;
        printf("Syntax highlighting level %d\n", data.syntax_highlight);
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Toggle real-time inline syntax diagnostics
 *
 * Controls whether real-time syntax error and continuation
 * diagnostics are displayed in the bottom hint area and highlighted
 * in the input line.
 * With an argument of "on" or "1", enables diagnostics.
 * With an argument of "off" or "0", disables diagnostics.
 * With no args, toggles the current state.
 *
 * @return RETURN_SUCCESS
 */
errno_t cli_syntax_diag_toggle(void)
{
    if (data.cmdNBarg >= 2)
    {
        const char *arg = data.cmdargtoken[1].val.string;
        if (strcmp(arg, "on") == 0 || strcmp(arg, "1") == 0)
        {
            data.syntax_diagnostics = 1;
            printf("Syntax diagnostics ON\n");
        }
        else if (strcmp(arg, "off") == 0 || strcmp(arg, "0") == 0)
        {
            data.syntax_diagnostics = 0;
            printf("Syntax diagnostics OFF\n");
        }
        else
        {
            printf("Usage: @config syndiag [on|off]\n");
        }
    }
    else
    {
        data.syntax_diagnostics = !data.syntax_diagnostics;
        printf("Syntax diagnostics %s\n", data.syntax_diagnostics ? "ON" : "OFF");
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Toggle or set multi-line auto-indentation.
 *
 * Controls automatic indentation for multi-line continuation lines
 * based on block nesting depth.
 * Usage: cliindent [on|off|<spaces>]
 *
 * @return RETURN_SUCCESS
 */
errno_t cli_auto_indent_toggle(void)
{
    if (data.cmdNBarg >= 2)
    {
        const char *arg = data.cmdargtoken[1].val.string;
        if (strcmp(arg, "on") == 0 || strcmp(arg, "1") == 0)
        {
            data.auto_indent = 4;
            printf("Auto-indentation ON (4 spaces)\n");
        }
        else if (strcmp(arg, "off") == 0 || strcmp(arg, "0") == 0)
        {
            data.auto_indent = 0;
            printf("Auto-indentation OFF\n");
        }
        else
        {
            int n = atoi(arg);
            if (n >= 1 && n <= 16)
            {
                data.auto_indent = n;
                printf("Auto-indentation set to %d spaces\n", n);
            }
            else
            {
                printf("Usage: @config indent [on|off|<spaces>]\n");
            }
        }
    }
    else
    {
        if (data.auto_indent > 0)
        {
            printf("Auto-indentation is ON (%d spaces)\n", data.auto_indent);
        }
        else
        {
            printf("Auto-indentation is OFF\n");
        }
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Toggle structural delimiter & block keyword matching (showmatch).
 *
 * Controls whether matching parentheses, brackets, braces, expansions, and
 * block keywords (if/fi, do/done, case/esac) are highlighted at cursor.
 * Usage: showmatch [on|off]
 *
 * @return RETURN_SUCCESS
 */
errno_t cli_showmatch_toggle(void)
{
    if (data.cmdNBarg >= 2)
    {
        const char *arg = data.cmdargtoken[1].val.string;
        if (strcmp(arg, "on") == 0 || strcmp(arg, "1") == 0)
        {
            data.show_match = 1;
            printf("Showmatch ON\n");
        }
        else if (strcmp(arg, "off") == 0 || strcmp(arg, "0") == 0)
        {
            data.show_match = 0;
            printf("Showmatch OFF\n");
        }
        else
        {
            printf("Usage: @config showmatch [on|off]\n");
        }
    }
    else
    {
        data.show_match = !data.show_match;
        printf("Showmatch %s\n", data.show_match ? "ON" : "OFF");
    }
    return RETURN_SUCCESS;
}
#endif


/* read_file() - the contents of @fname (NUL-terminated, to free), or NULL */
static char *read_file(const char *fname)
{
    FILE *f = fopen(fname, "rb");
    char *buf;
    long  sz;

    if (f == NULL)
    {
        fprintf(stderr, "milk-cli: cannot open '%s'\n", fname);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (sz >= 0) ? (char *) malloc((size_t) sz + 1) : NULL;
    if (buf != NULL)
    {
        buf[fread(buf, 1, (size_t) sz, f)] = '\0';
    }
    fclose(f);
    return buf;
}

int cli_format_file(const char *fname, int in_place, int width)
{
    char *buf       = read_file(fname);
    char *formatted = (buf != NULL) ? cli_ts_format_code(buf, width) : NULL;
    FILE *out;

    free(buf);
    if (formatted == NULL)
    {
        return 1;
    }
    out = in_place ? fopen(fname, "wb") : stdout;
    if (out == NULL)
    {
        fprintf(stderr, "milk-cli: cannot write '%s'\n", fname);
        free(formatted);
        return 1;
    }
    fputs(formatted, out);
    if (in_place)
    {
        fclose(out);
    }
    free(formatted);
    return 0;
}

int cli_outline_file(const char *fname)
{
    char *buf = read_file(fname);

    if (buf == NULL)
    {
        return 1;
    }
    int count = cli_ts_print_block_folds(buf, fname, stdout);
    if (count == 0)
    {
        printf("No multi-line blocks in '%s'.\n", fname);
    }
    else
    {
        printf("Total blocks: %d\n", count);
    }
    free(buf);
    return 0;
}


FILE           *session_log_fp = NULL;
struct timespec session_log_t0;

/**
 * @brief Start/stop session command logging.
 *
 * Logs every command with a wall-clock timestamp
 * and elapsed time to a file.
 *
 * Usage: sessionlog [on|off|<filename>]
 * Default log path: ~/.milk_session.log
 */
errno_t cli_sessionlog(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: @config sessionlog "
               "[on|off|<filename>]\n");
        printf("Status: %s\n", session_log_fp ? "ON" : "OFF");
        return RETURN_SUCCESS;
    }
    const char *arg = data.cmdargtoken[1].val.string;

    if (strcmp(arg, "off") == 0)
    {
        if (session_log_fp)
        {
            fclose(session_log_fp);
            session_log_fp = NULL;
            printf("Session logging stopped\n");
        }
        return RETURN_SUCCESS;
    }

    /* Close previous if open */
    if (session_log_fp)
    {
        fclose(session_log_fp);
        session_log_fp = NULL;
    }

    char logpath[STRINGMAXLEN_FULLFILENAME];
    if (strcmp(arg, "on") == 0)
    {
        snprintf(logpath, STRINGMAXLEN_FULLFILENAME, "%s/.milk_session.log", getenv("HOME"));
    }
    else
    {
        strncpy(logpath, arg, STRINGMAXLEN_FULLFILENAME - 1);
        logpath[STRINGMAXLEN_FULLFILENAME - 1] = '\0';
    }

    session_log_fp = fopen(logpath, "a");
    if (session_log_fp == NULL)
    {
        printf("Cannot open '%s'\n", logpath);
        return RETURN_FAILURE;
    }
    clock_gettime(CLOCK_MONOTONIC, &session_log_t0);
    printf("Session logging to '%s'\n", logpath);

    /* Write session start marker */
    {
        time_t now = time(NULL);
        char   tbuf[64];
        strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", localtime(&now));
        fprintf(session_log_fp, "# Session started %s\n", tbuf);
        fflush(session_log_fp);
    }
    return RETURN_SUCCESS;
}
