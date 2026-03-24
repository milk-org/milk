#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <fnmatch.h>
#include <ctype.h>
#ifdef USE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
#endif
#include "CLIcore.h"
#include "CLIcore_UI.h"
#include <glob.h>
#include <sys/wait.h>
#include "timeutils.h"


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
    char        type
)
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


/**
 * Filter and display options for history_log_display().
 */
typedef struct
{
    const char *filter_session; /**< NULL = all       */
    int         max_entries;    /**< 0 = unlimited    */
    const char *glob_cmd;       /**< NULL = no glob   */
    time_t      time_after;     /**< 0 = no lower     */
    time_t      time_before;    /**< 0 = no upper     */
    int         highlight_self; /**< 1 = highlight    */
    char        type_filter;    /**< \0=all P/C/S     */
} HistDisplayOpts;


/**
 * @brief Parse a time argument string into a time_t value.
 *
 * Accepts:
 *   today                  midnight today
 *   Nm / Nh / Nd           N minutes / hours / days ago
 *   YYYY-MM-DD             midnight on that date
 *   YYYY-MM-DDTHH:MM:SS    exact timestamp
 *
 * @return 0 on success, -1 on parse error.
 */
static int parse_time_arg(
    const char *s,
    time_t     *out
)
{
    if(s == NULL || out == NULL)
    {
        return -1;
    }
    time_t now = time(NULL);

    if(strcmp(s, "today") == 0)
    {
        struct tm *t = localtime(&now);
        t->tm_hour = 0;
        t->tm_min  = 0;
        t->tm_sec  = 0;
        *out = mktime(t);
        return 0;
    }

    /* Relative: Nm, Nh, Nd */
    {
        size_t slen = strlen(s);
        if(slen >= 2 && isdigit((unsigned char) s[0]))
        {
            char unit = s[slen - 1];
            int  val  = atoi(s);
            if(val > 0)
            {
                if(unit == 'm')
                {
                    *out = now - (time_t) val * 60;
                    return 0;
                }
                if(unit == 'h')
                {
                    *out = now - (time_t) val * 3600;
                    return 0;
                }
                if(unit == 'd')
                {
                    *out = now - (time_t) val * 86400;
                    return 0;
                }
            }
        }
    }

    /* ISO formats */
    {
        struct tm tm0;
        memset(&tm0, 0, sizeof(tm0));
        if(strptime(s, "%Y-%m-%dT%H:%M:%S", &tm0)
           || strptime(s, "%Y-%m-%d", &tm0))
        {
            tm0.tm_isdst = -1;
            *out = mktime(&tm0);
            return 0;
        }
    }

    return -1;
}



/**
 * @brief Split data.CLIcmdline into argv tokens.
 *
 * The calc parser mangles flags like -n, --since
 * (treats '-' as arithmetic minus), so parse
 * data.CLIcmdline directly instead.
 *
 * @param[out] argc_out  Number of tokens.
 * @param[out] argv_out  Heap-allocated (argc+1)
 *             array of heap-allocated strings.
 *             Caller must free with cmdline_free.
 */
static void cmdline_split(
    int    *argc_out,
    char ***argv_out
)
{
    char buf[STRINGMAXLEN_CLICMDLINE];
    strncpy(buf, data.CLIcmdline,
            sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

#define HIST_ARGV_MAX 64
    char *tokens[HIST_ARGV_MAX];
    int   ntok    = 0;

    const char *p = buf;
    while(*p != '\0' && ntok < HIST_ARGV_MAX - 1)
    {
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        if(*p == '\0')
        {
            break;
        }
        char tmp[STRINGMAXLEN_CLICMDLINE];
        int  j = 0;

        if(*p == '"')
        {
            p++;
            while(*p != '"' && *p != '\0'
                  && j < (int) sizeof(tmp) - 1)
            {
                tmp[j++] = *p++;
            }
            if(*p == '"')
            {
                p++;
            }
        }
        else if(*p == '\'')
        {
            p++;
            while(*p != '\'' && *p != '\0'
                  && j < (int) sizeof(tmp) - 1)
            {
                tmp[j++] = *p++;
            }
            if(*p == '\'')
            {
                p++;
            }
        }
        else
        {
            while(*p != ' ' && *p != '\t'
                  && *p != '\0'
                  && j < (int) sizeof(tmp) - 1)
            {
                tmp[j++] = *p++;
            }
        }
        tmp[j]       = '\0';
        tokens[ntok] = strdup(tmp);
        ntok++;
    }

    *argc_out = ntok;
    *argv_out = (char **) malloc(
        (size_t)(ntok + 1) * sizeof(char *));
    for(int i = 0; i < ntok; i++)
    {
        (*argv_out)[i] = tokens[i];
    }
    (*argv_out)[ntok] = NULL;
}

/**
 * @brief Free the argv array from cmdline_split().
 */
static void cmdline_free(int argc, char **argv)
{
    for(int i = 0; i < argc; i++)
    {
        free(argv[i]);
    }
    free(argv);
}


/**
 * @brief Read history log and display entries,
 *        applying all filters from opts.
 *
 * Entries matching the current session are
 * highlighted in bold green when opts->highlight_self
 * is set.
 *
 * @param opts  Pointer to filter/display options.
 */
void history_log_display(
    const HistDisplayOpts *opts
)
{
    FILE *fp = fopen(CLI_history_log_file(), "r");
    if(fp == NULL)
    {
        printf("No history log found (%s)\n",
               CLI_history_log_file());
        return;
    }

    char line[2048];
    int  cap    = 1024;

    char **lines   = (char **) malloc(
        (size_t) cap * sizeof(char *));
    int  *is_self  = (int *)   malloc(
        (size_t) cap * sizeof(int));
    char *types    = (char *)  malloc(
        (size_t) cap * sizeof(char));

    if(lines == NULL || is_self == NULL
       || types == NULL)
    {
        if(lines)   free(lines);
        if(is_self) free(is_self);
        if(types)   free(types);
        fclose(fp);
        printf("Memory allocation error\n");
        return;
    }

    int total = 0;

    while(fgets(line, (int) sizeof(line), fp))
    {
        /* Remove trailing newline */
        {
            size_t len = strlen(line);
            if(len > 0 && line[len - 1] == '\n')
            {
                line[len - 1] = '\0';
            }
        }

        /* Parse: ts\tsid\ttty\t[type\t]text
         * New format (5 fields):
         *   ts\tsid\ttty\ttype\ttext
         * Old format (4 fields, compat):
         *   ts\tsid\ttty\ttext  (type='C')
         */
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
        char *tab3 = strchr(tab2 + 1, '\t');
        if(tab3 == NULL)
        {
            continue;
        }

        char *sid_start = tab1 + 1;
        int   sid_len   = (int)(tab2 - tab1 - 1);

        /* Detect new 5-field vs old 4-field */
        char  entry_type = 'C';
        char *text_start;
        {
            char *tab4 = strchr(
                tab3 + 1, '\t');
            if(tab4 != NULL
               && (tab4 - tab3) == 2)
            {
                /* 5-field: type is single
                 * char between tab3..tab4 */
                entry_type = *(tab3 + 1);
                text_start = tab4 + 1;
            }
            else
            {
                /* Old 4-field format */
                text_start = tab3 + 1;
            }
        }

        /* Type filter */
        if(opts->type_filter != '\0'
           && entry_type != opts->type_filter)
        {
            continue;
        }

        /* Session filter */
        if(opts->filter_session != NULL)
        {
            int fslen =
                (int) strlen(
                    opts->filter_session);
            if(sid_len != fslen
               || strncmp(sid_start,
                          opts->filter_session,
                          (size_t) sid_len)
               != 0)
            {
                continue;
            }
        }

        /* Glob filter on text */
        if(opts->glob_cmd != NULL)
        {
            if(fnmatch(opts->glob_cmd,
                       text_start,
                       FNM_CASEFOLD) != 0)
            {
                continue;
            }
        }

        /* Time filter (parse ts field) */
        if(opts->time_after || opts->time_before)
        {
            char     ts_buf[32];
            int      tslen =
                (int)(tab1 - line);
            if(tslen
               >= (int) sizeof(ts_buf))
            {
                tslen =
                    (int) sizeof(ts_buf) - 1;
            }
            memcpy(ts_buf, line,
                   (size_t) tslen);
            ts_buf[tslen] = '\0';

            struct tm tm0;
            memset(&tm0, 0, sizeof(tm0));
            if(strptime(ts_buf,
                        "%Y-%m-%dT%H:%M:%S",
                        &tm0))
            {
                tm0.tm_isdst = -1;
                time_t entry_t = mktime(&tm0);
                if(opts->time_after
                   && entry_t
                   < opts->time_after)
                {
                    continue;
                }
                if(opts->time_before
                   && entry_t
                   > opts->time_before)
                {
                    continue;
                }
            }
        }

        /* Is this the current session? */
        int self =
            (sid_len
             == (int) strlen(data.session_id)
             && strncmp(sid_start,
                        data.session_id,
                        (size_t) sid_len)
             == 0);

        /* Store */
        if(total >= cap)
        {
            cap *= 2;
            char **tmp1 = (char **) realloc(
                lines,
                (size_t) cap * sizeof(char *));
            int  *tmp2  = (int *) realloc(
                is_self,
                (size_t) cap * sizeof(int));
            char *tmp3  = (char *) realloc(
                types,
                (size_t) cap * sizeof(char));
            if(tmp1 == NULL || tmp2 == NULL
               || tmp3 == NULL)
            {
                break;
            }
            lines   = tmp1;
            is_self = tmp2;
            types   = tmp3;
        }
        lines[total]   = strdup(line);
        is_self[total] = self;
        types[total]   = entry_type;
        total++;
    }
    fclose(fp);

    if(total == 0)
    {
        printf("No matching history entries\n");
        free(lines);
        free(is_self);
        free(types);
        return;
    }

    /* Determine start index */
    int start = 0;
    if(opts->max_entries > 0
       && opts->max_entries < total)
    {
        start = total - opts->max_entries;
    }

    /* Header */
    int show_sess =
        (opts->filter_session == NULL);
    if(show_sess)
    {
        printf("\033[1;36m %-24s %-19s"
               "  T  %s\033[0m\n",
               "Session", "Time", "Entry");
    }
    else
    {
        printf("\033[1;36m %-19s"
               "  T  %s\033[0m\n",
               "Time", "Entry");
    }

    /* Print entries */
    int shown = 0;
    for(int i = start; i < total; i++)
    {
        /* Re-parse for display */
        char buf[2048];
        strncpy(buf, lines[i],
                sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';

        char *ts   = buf;
        char *p1   = strchr(ts,  '\t');
        if(p1 == NULL) continue;
        *p1 = '\0';
        char *sid  = p1 + 1;
        char *p2   = strchr(sid, '\t');
        if(p2 == NULL) continue;
        *p2 = '\0';
        char *tty  = p2 + 1;
        char *p3   = strchr(tty, '\t');
        if(p3 == NULL) continue;
        *p3 = '\0';

        /* Detect 5-field vs 4-field */
        char  etype = types[i];
        char *etext;
        {
            char *p4 = strchr(p3 + 1, '\t');
            if(p4 != NULL
               && (p4 - (p3 + 1)) == 1)
            {
                etext = p4 + 1;
            }
            else
            {
                etext = p3 + 1;
            }
        }

        /* Shorten tty: /dev/pts/3 → /pts/3 */
        const char *tty_s = tty;
        if(strncmp(tty, "/dev", 4) == 0)
        {
            tty_s = tty + 4;
        }

        /* Color by entry type and session */
        const char *col_on  = "";
        const char *col_off = "";
        if(etype == 'P')
        {
            col_on  = "\033[1;36m";
            col_off = "\033[0m";
        }
        else if(etype == 'S')
        {
            col_on  = "\033[2;33m";
            col_off = "\033[0m";
        }
        else if(opts->highlight_self
                && is_self[i])
        {
            col_on  = "\033[1;32m";
            col_off = "\033[0m";
        }

        if(show_sess)
        {
            char sess_col[40];
            snprintf(sess_col,
                     sizeof(sess_col),
                     "%s %s", sid, tty_s);
            printf("%s %-24s %-19s"
                   "  %c  %s%s\n",
                   col_on, sess_col, ts,
                   etype, etext, col_off);
        }
        else
        {
            printf("%s %-19s"
                   "  %c  %s%s\n",
                   col_on, ts, etype,
                   etext, col_off);
        }
        shown++;
    }

    printf("(%d entr%s)\n",
           shown, shown == 1 ? "y" : "ies");

    for(int i = 0; i < total; i++)
    {
        free(lines[i]);
    }
    free(lines);
    free(is_self);
    free(types);
}


/**
 * @brief Global history command (ghistory)
 *
 * Usage:
 *   ghistory [N]              last N entries (default 20)
 *   ghistory -n N             last N entries
 *   ghistory -s SID           filter by session ID
 *   ghistory -t TYPE          filter: prompt/cmd/shell
 *   ghistory -g PATTERN       glob-filter on text
 *   ghistory --since TS       entries after TS
 *   ghistory --until TS       entries before TS
 *   ghistory --today          entries from today
 *
 * TS formats: today  Nm  Nh  Nd  YYYY-MM-DD
 *             YYYY-MM-DDTHH:MM:SS
 *
 * Current-session entries are highlighted in
 * bold green.
 */
errno_t cli_ghistory(void)
{
    HistDisplayOpts opts;
    memset(&opts, 0, sizeof(opts));
    opts.max_entries    = 20;
    opts.highlight_self = 1;

    int    argc = 0;
    char **argv = NULL;
    cmdline_split(&argc, &argv);

    /* argv[0] is the command name, skip it */
    for(int i = 1; i < argc; i++)
    {
        const char *a = argv[i];

        if(strcmp(a, "-n") == 0
           && i + 1 < argc)
        {
            i++;
            opts.max_entries = atoi(argv[i]);
            if(opts.max_entries < 0)
            {
                opts.max_entries = 0;
            }
        }
        else if(strcmp(a, "-s") == 0
                && i + 1 < argc)
        {
            i++;
            opts.filter_session = argv[i];
            opts.max_entries    = 0;
        }
        else if(strcmp(a, "-t") == 0
                && i + 1 < argc)
        {
            i++;
            const char *tv = argv[i];
            if(strcmp(tv, "prompt") == 0
               || strcmp(tv, "p") == 0)
            {
                opts.type_filter = 'P';
            }
            else if(strcmp(tv, "cmd") == 0
                    || strcmp(tv, "c") == 0
                    || strcmp(tv,
                             "command") == 0)
            {
                opts.type_filter = 'C';
            }
            else if(strcmp(tv, "shell") == 0
                    || strcmp(tv, "s") == 0)
            {
                opts.type_filter = 'S';
            }
            else
            {
                printf("ghistory: unknown "
                       "type '%s'\n"
                       "  valid: prompt "
                       "cmd shell\n", tv);
                cmdline_free(argc, argv);
                return RETURN_FAILURE;
            }
        }
        else if(strcmp(a, "-g") == 0
                && i + 1 < argc)
        {
            i++;
            opts.glob_cmd = argv[i];
        }
        else if(strcmp(a, "--since") == 0
                && i + 1 < argc)
        {
            i++;
            if(parse_time_arg(argv[i],
                              &opts.time_after)
               != 0)
            {
                printf("ghistory: bad time"
                       " '%s'\n", argv[i]);
                cmdline_free(argc, argv);
                return RETURN_FAILURE;
            }
            opts.max_entries = 0;
        }
        else if(strcmp(a, "--until") == 0
                && i + 1 < argc)
        {
            i++;
            if(parse_time_arg(argv[i],
                              &opts.time_before)
               != 0)
            {
                printf("ghistory: bad time"
                       " '%s'\n", argv[i]);
                cmdline_free(argc, argv);
                return RETURN_FAILURE;
            }
            opts.max_entries = 0;
        }
        else if(strcmp(a, "--today") == 0)
        {
            parse_time_arg("today",
                           &opts.time_after);
            opts.max_entries = 0;
        }
        else if(isdigit((unsigned char) a[0]))
        {
            int n = atoi(a);
            if(n > 0)
            {
                opts.max_entries = n;
            }
        }
        else
        {
            printf("ghistory: unknown option"
                   " '%s'\n", a);
            printf(
                "Usage: ghistory [N]\n"
                "  -n N       last N entries\n"
                "  -s SID     filter session\n"
                "  -t TYPE    prompt/cmd/"
                "shell\n"
                "  -g PAT     glob on text\n"
                "  --since T  after time\n"
                "  --until T  before time\n"
                "  --today    today only\n"
                "T: today Nm Nh Nd "
                "YYYY-MM-DD "
                "YYYY-MM-DDTHH:MM:SS\n");
            cmdline_free(argc, argv);
            return RETURN_FAILURE;
        }
    }

    history_log_display(&opts);
    cmdline_free(argc, argv);
    return RETURN_SUCCESS;
}


/**
 * @brief Local (session) history command (lhistory)
 *
 * Usage:
 *   lhistory [N]              all or last N entries
 *   lhistory -n N             last N entries
 *   lhistory -t TYPE          filter: prompt/cmd/shell
 *   lhistory -g PATTERN       glob-filter on text
 *   lhistory --since TS       entries after TS
 *   lhistory --until TS       entries before TS
 *   lhistory --today          entries from today
 */
errno_t cli_lhistory(void)
{
    HistDisplayOpts opts;
    memset(&opts, 0, sizeof(opts));
    opts.filter_session = data.session_id;
    opts.max_entries    = 0; /* default: all */

    int    argc = 0;
    char **argv = NULL;
    cmdline_split(&argc, &argv);

    for(int i = 1; i < argc; i++)
    {
        const char *a = argv[i];

        if(strcmp(a, "-n") == 0
           && i + 1 < argc)
        {
            i++;
            opts.max_entries = atoi(argv[i]);
            if(opts.max_entries < 0)
            {
                opts.max_entries = 0;
            }
        }
        else if(strcmp(a, "-t") == 0
                && i + 1 < argc)
        {
            i++;
            const char *tv = argv[i];
            if(strcmp(tv, "prompt") == 0
               || strcmp(tv, "p") == 0)
            {
                opts.type_filter = 'P';
            }
            else if(strcmp(tv, "cmd") == 0
                    || strcmp(tv, "c") == 0
                    || strcmp(tv,
                             "command") == 0)
            {
                opts.type_filter = 'C';
            }
            else if(strcmp(tv, "shell") == 0
                    || strcmp(tv, "s") == 0)
            {
                opts.type_filter = 'S';
            }
            else
            {
                printf("lhistory: unknown "
                       "type '%s'\n"
                       "  valid: prompt "
                       "cmd shell\n", tv);
                cmdline_free(argc, argv);
                return RETURN_FAILURE;
            }
        }
        else if(strcmp(a, "-g") == 0
                && i + 1 < argc)
        {
            i++;
            opts.glob_cmd = argv[i];
        }
        else if(strcmp(a, "--since") == 0
                && i + 1 < argc)
        {
            i++;
            if(parse_time_arg(argv[i],
                              &opts.time_after)
               != 0)
            {
                printf("lhistory: bad time"
                       " '%s'\n", argv[i]);
                cmdline_free(argc, argv);
                return RETURN_FAILURE;
            }
        }
        else if(strcmp(a, "--until") == 0
                && i + 1 < argc)
        {
            i++;
            if(parse_time_arg(argv[i],
                              &opts.time_before)
               != 0)
            {
                printf("lhistory: bad time"
                       " '%s'\n", argv[i]);
                cmdline_free(argc, argv);
                return RETURN_FAILURE;
            }
        }
        else if(strcmp(a, "--today") == 0)
        {
            parse_time_arg("today",
                           &opts.time_after);
        }
        else if(isdigit((unsigned char) a[0]))
        {
            int n = atoi(a);
            if(n > 0)
            {
                opts.max_entries = n;
            }
        }
        else
        {
            printf("lhistory: unknown option"
                   " '%s'\n", a);
            printf(
                "Usage: lhistory [N]\n"
                "  -n N       last N entries\n"
                "  -t TYPE    prompt/cmd/"
                "shell\n"
                "  -g PAT     glob on text\n"
                "  --since T  after time\n"
                "  --until T  before time\n"
                "  --today    today only\n");
            cmdline_free(argc, argv);
            return RETURN_FAILURE;
        }
    }

    history_log_display(&opts);
    cmdline_free(argc, argv);
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

/**
 * @brief Expand history references in the
 *        current command line.
 *
 * Supports:
 *   !!        — replay last command
 *   !$        — last argument of previous cmd
 *   !<prefix> — last command starting with prefix
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
            char expanded[STRINGMAXLEN_CLICMDLINE * 2];
            snprintf(expanded,
                     sizeof(expanded),
                     "%s%s",
                     prev->line, suffix);
            strncpy(data.CLIcmdline, expanded, STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
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
            char expanded[STRINGMAXLEN_CLICMDLINE * 2];
            snprintf(expanded,
                     sizeof(expanded),
                     "%s%s",
                     data.last_argument, rest);
            strncpy(data.CLIcmdline, expanded, STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
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

/**
 * @brief Show recent readline history entries.
 *
 * Usage: history [N]  (default 20)
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

/**
 * @brief Case-insensitive substring search
 *        through readline history.
 *
 * Usage: searchhist <pattern>
 * Highlights matches in bold yellow.
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