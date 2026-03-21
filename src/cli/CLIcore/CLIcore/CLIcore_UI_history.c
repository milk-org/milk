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
#include "CLIcore_UI.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"
#include <glob.h>
#include <sys/wait.h>
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"


const char *CLI_history_file(void)
{
    static char path[1024] = {0};
    if(path[0] == '\0')
    {
        const char *home = getenv("HOME");
        if(home)
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
 * @brief Log command to structured history file
 *
 * Appends a TSV line:
 *   <ISO-timestamp>\t<session_id>\t<tty>\t<cmd>
 *
 * Called alongside add_history() in the main
 * command dispatch loop.
 */
void cli_history_log_cmd(
    const char *cmd
)
{
    if(cmd == NULL || cmd[0] == '\0')
    {
        return;
    }

    FILE *fp = fopen(CLI_history_log_file(), "a");
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
        fprintf(fp, "%s\t%s\t%s\t%s\n",
                tbuf,
                data.session_id,
                data.session_tty,
                cmd);
    }
    fclose(fp);
}


/**
 * @brief Helper: read history log and display
 *
 * @param filter_session  if non-NULL, only show
 *        entries matching this session_id
 * @param max_entries     show at most this many
 *        entries (0 = all)
 */
void history_log_display(
    const char *filter_session,
    int         max_entries
)
{
    FILE *fp = fopen(CLI_history_log_file(), "r");
    if(fp == NULL)
    {
        printf("No history log found (%s)\n",
               CLI_history_log_file());
        return;
    }

    /* First pass: count matching lines */
    char line[2048];
    int total = 0;

    /* Store lines in a simple dynamic array */
    int cap = 1024;
    char **lines = (char **) malloc(
        (size_t) cap * sizeof(char *));
    if(lines == NULL)
    {
        fclose(fp);
        printf("Memory allocation error\n");
        return;
    }

    while(fgets(line, (int) sizeof(line), fp))
    {
        /* Remove trailing newline */
        {
            size_t len = strlen(line);
            if(len > 0
               && line[len - 1] == '\n')
            {
                line[len - 1] = '\0';
            }
        }

        /* Parse: timestamp\tsession\ttty\tcmd */
        if(filter_session != NULL)
        {
            char *tab1 = strchr(line, '\t');
            if(tab1 == NULL)
            {
                continue;
            }
            char *tab2 = strchr(tab1 + 1, '\t');
            if(tab2 == NULL)
            {
                continue;
            }
            int slen = (int)(tab2 - tab1 - 1);
            if(slen
               != (int) strlen(filter_session)
               || strncmp(tab1 + 1,
                          filter_session,
                          (size_t) slen) != 0)
            {
                continue;
            }
        }

        /* Store line */
        if(total >= cap)
        {
            cap *= 2;
            char **tmp = (char **) realloc(
                lines,
                (size_t) cap * sizeof(char *));
            if(tmp == NULL)
            {
                break;
            }
            lines = tmp;
        }
        lines[total] = strdup(line);
        total++;
    }
    fclose(fp);

    if(total == 0)
    {
        if(filter_session)
        {
            printf("No history for session"
                   " '%s'\n", filter_session);
        }
        else
        {
            printf("No history entries\n");
        }
        free(lines);
        return;
    }

    /* Determine start index */
    int start = 0;
    if(max_entries > 0
       && max_entries < total)
    {
        start = total - max_entries;
    }

    /* Print header */
    if(filter_session != NULL)
    {
        printf("\033[1;36m %-19s  %s\033[0m\n",
               "Time", "Command");
    }
    else
    {
        printf("\033[1;36m %-24s %-19s  "
               "%s\033[0m\n",
               "Session",
               "Time", "Command");
    }

    /* Print entries */
    for(int i = start; i < total; i++)
    {
        /* Parse fields */
        char *ts = lines[i];
        char *tab1 = strchr(ts, '\t');
        if(tab1 == NULL)
        {
            continue;
        }
        *tab1 = '\0';
        char *sid = tab1 + 1;
        char *tab2 = strchr(sid, '\t');
        if(tab2 == NULL)
        {
            continue;
        }
        *tab2 = '\0';
        char *tty = tab2 + 1;
        char *tab3 = strchr(tty, '\t');
        if(tab3 == NULL)
        {
            continue;
        }
        *tab3 = '\0';
        char *cmd = tab3 + 1;

        /* Shorten tty: /dev/pts/3 → /pts/3 */
        const char *tty_short = tty;
        if(strncmp(tty, "/dev", 4) == 0)
        {
            tty_short = tty + 4;
        }

        if(filter_session != NULL)
        {
            /* local: no session column */
            printf(" %-19s  %s\n", ts, cmd);
        }
        else
        {
            /* global: show session + tty */
            char sess_col[40];
            snprintf(sess_col,
                     sizeof(sess_col),
                     "%s %s", sid, tty_short);
            printf(" %-24s %-19s  %s\n",
                   sess_col, ts, cmd);
        }
    }

    /* Summary line */
    int shown = total - start;
    if(filter_session)
    {
        printf("(%d entr%s, session %s)\n",
               shown,
               shown == 1 ? "y" : "ies",
               filter_session);
    }
    else
    {
        printf("(%d entr%s)\n",
               shown,
               shown == 1 ? "y" : "ies");
    }

    /* Free */
    for(int i = 0; i < total; i++)
    {
        free(lines[i]);
    }
    free(lines);
}

/**
 * @brief Global history command
 *
 * Usage:
 *   ghistory         — last 20 entries
 *   ghistory N       — last N entries
 *   ghistory -s SID  — filter by session ID
 */
errno_t cli_ghistory(void)
{
    int n = 20;
    const char *filter = NULL;

    if(data.cmdNBarg >= 2)
    {
        const char *arg1 =
            data.cmdargtoken[1].val.string;
        if(strcmp(arg1, "-s") == 0)
        {
            if(data.cmdNBarg >= 3)
            {
                filter =
                    data.cmdargtoken[2]
                        .val.string;
            }
            else
            {
                printf("Usage: ghistory"
                       " -s <session_id>\n");
                return RETURN_FAILURE;
            }
            n = 0; /* show all for session */
        }
        else
        {
            n = atoi(arg1);
            if(n <= 0)
            {
                n = 20;
            }
        }
    }

    history_log_display(filter, n);
    return RETURN_SUCCESS;
}

/**
 * @brief Local (session) history command
 *
 * Usage:
 *   lhistory         — all entries, this session
 *   lhistory N       — last N entries
 */
errno_t cli_lhistory(void)
{
    int n = 0; /* default: show all */

    if(data.cmdNBarg >= 2)
    {
        n = atoi(
            data.cmdargtoken[1].val.string);
        if(n <= 0)
        {
            n = 0;
        }
    }

    history_log_display(data.session_id, n);
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  History Expansion (!! and !$)
 * ============================================================
 *
 * Called at the very start of CLI_execute_line(),
 * before alias expansion.
 *
 * !!    → replace with last executed command
 * !$    → replace with last argument of previous cmd
 * !<prefix> → last command starting with <prefix>
 */

void cli_history_expand(void)
{
#ifdef USE_READLINE
    char *line = data.CLIcmdline;

    /* Quick check: must start with '!' */
    if(line[0] != '!')
    {
        return;
    }

    /* !! — replay last command */
    if(line[1] == '!')
    {
        HIST_ENTRY *prev = history_get(
            history_length);
        if(prev != NULL)
        {
            char suffix[STRINGMAXLEN_CLICMDLINE];
            suffix[0] = '\0';
            if(line[2] != '\0')
            {
                strncpy(suffix, line + 2,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                suffix[
                    STRINGMAXLEN_CLICMDLINE - 1]
                    = '\0';
            }
            snprintf(data.CLIcmdline,
                     STRINGMAXLEN_CLICMDLINE,
                     "%s%s",
                     prev->line, suffix);
            printf(">> %s\n", data.CLIcmdline);
        }
        return;
    }

    /* !$ — last argument of previous command */
    if(line[1] == '$')
    {
        if(data.last_argument[0] != '\0')
        {
            char rest[STRINGMAXLEN_CLICMDLINE];
            strncpy(rest, line + 2,
                    STRINGMAXLEN_CLICMDLINE - 1);
            rest[
                STRINGMAXLEN_CLICMDLINE - 1]
                = '\0';
            snprintf(data.CLIcmdline,
                     STRINGMAXLEN_CLICMDLINE,
                     "%s%s",
                     data.last_argument, rest);
            printf(">> %s\n", data.CLIcmdline);
        }
        return;
    }

    /* !<prefix> — last command starting with it */
    {
        const char *prefix = line + 1;
        size_t plen = strlen(prefix);
        /* Trim trailing spaces from prefix */
        while(plen > 0
                && prefix[plen - 1] == ' ')
        {
            plen--;
        }
        if(plen == 0)
        {
            return;
        }
        HIST_ENTRY **hist = history_list();
        if(hist == NULL)
        {
            return;
        }
        int hlen = history_length;
        for(int i = hlen - 1; i >= 0; i--)
        {
            if(strncmp(hist[i]->line,
                       prefix, plen) == 0)
            {
                strncpy(data.CLIcmdline,
                        hist[i]->line,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                data.CLIcmdline[
                    STRINGMAXLEN_CLICMDLINE - 1]
                    = '\0';
                printf(">> %s\n",
                       data.CLIcmdline);
                return;
            }
        }
        printf("!%.*s: event not found\n",
               (int) plen, prefix);
        data.CLIcmdline[0] = '\0';
    }
#endif
}


/**
 * @brief Save last argument after command execution
 */
void cli_save_last_argument(void)
{
    if(data.cmdNBarg > 1)
    {
        long last = data.cmdNBarg - 1;
        strncpy(data.last_argument,
                data.cmdargtoken[last].val.string,
                sizeof(data.last_argument) - 1);
        data.last_argument[
            sizeof(data.last_argument) - 1]
            = '\0';
    }
}


/*
 * ============================================================
 *  Save History — export readline history
 * ============================================================
 */

/**
 * @brief Write readline command history to a file.
 *
 * Usage: savehistory <filename>
 */
errno_t cli_savehistory(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: savehistory "
               "<filename>\n");
        return RETURN_FAILURE;
    }
    const char *fname =
        data.cmdargtoken[1].val.string;

#ifdef USE_READLINE
    if(write_history(fname) != 0)
    {
        printf("savehistory: failed to write "
               "'%s'\n", fname);
        return RETURN_FAILURE;
    }
    printf("History saved to '%s'\n", fname);
    return RETURN_SUCCESS;
#else
    printf("savehistory: readline not "
           "available\n");
    (void) fname;
    return RETURN_FAILURE;
#endif
}


/*
 * ============================================================
 *  History <N> Command
 * ============================================================
 */

errno_t cli_history_show(void)
{
#ifdef USE_READLINE
    int n = 20;  /* default */
    if(data.cmdNBarg >= 2)
    {
        n = atoi(
            data.cmdargtoken[1].val.string);
        if(n <= 0)
        {
            n = 20;
        }
    }

    HIST_ENTRY **hlist = history_list();
    if(hlist == NULL)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    int total = history_length;
    int start = total - n;
    if(start < 0)
    {
        start = 0;
    }
    for(int i = start; i < total; i++)
    {
        printf(" %4d  %s\n",
               i + 1, hlist[i]->line);
    }
#else
    printf("Readline not available\n");
#endif
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Fuzzy History Search (searchhist)
 * ============================================================
 *
 * Search history for entries containing a
 * substring. Shows all matches with index.
 */

errno_t cli_searchhist(void)
{
#ifdef USE_READLINE
    if(data.cmdNBarg < 2)
    {
        printf("Usage: searchhist <pattern>\n");
        return RETURN_SUCCESS;
    }
    const char *pattern =
        data.cmdargtoken[1].val.string;

    HIST_ENTRY **hlist = history_list();
    if(hlist == NULL)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    int total = history_length;
    int found = 0;
    for(int i = 0; i < total; i++)
    {
        if(strcasestr(hlist[i]->line,
                      pattern) != NULL)
        {
            /* Highlight matching substring */
            const char *pos =
                strcasestr(hlist[i]->line,
                           pattern);
            int pre = (int)(pos
                            - hlist[i]->line);
            int plen = (int) strlen(pattern);
            printf(" %4d  %.*s"
                   "\033[1;33m%.*s\033[0m"
                   "%s\n",
                   i + 1,
                   pre, hlist[i]->line,
                   plen, pos,
                   pos + plen);
            found++;
        }
    }
    if(found == 0)
    {
        printf("No history entries match"
               " '%s'\n", pattern);
    }
    else
    {
        printf("(%d match%s)\n",
               found,
               found == 1 ? "" : "es");
    }
#else
    printf("Readline not available\n");
#endif
    return RETURN_SUCCESS;
}