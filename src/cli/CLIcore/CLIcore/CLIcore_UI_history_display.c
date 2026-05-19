#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <fnmatch.h>
#include "CLIcore.h"

/**
 * @file    CLIcore_UI_history_display.c
 * @brief   History log display and query commands.
 *
 * Implements ghistory and lhistory commands that
 * read the structured history log and display
 * filtered results.
 *
 * @see CLIcore_UI_history.c for core log management.
 */

/* Forward declaration — defined in
 * CLIcore_UI_history.c */
const char *CLI_history_log_file(void);


/**
 * Filter and display options for
 * history_log_display().
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
 * @brief Parse a time argument string into a
 *        time_t value.
 *
 * Accepts:
 *   today                  midnight today
 *   Nm / Nh / Nd           N min/hours/days ago
 *   YYYY-MM-DD             midnight on that date
 *   YYYY-MM-DDTHH:MM:SS    exact timestamp
 *
 * @return 0 on success, -1 on parse error.
 */
static int parse_time_arg(
    const char *s,
    time_t     *out)
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
        if(slen >= 2
           && isdigit((unsigned char) s[0]))
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
        if(strptime(s,
                    "%Y-%m-%dT%H:%M:%S", &tm0)
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
    int *argc_out,
    char ***argv_out)
{
    char buf[STRINGMAXLEN_CLICMDLINE];
    strncpy(buf, data.CLIcmdline, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

#define HIST_ARGV_MAX 64
    char *tokens[HIST_ARGV_MAX];
    int   ntok    = 0;

    const char *p = buf;
    while(*p != '\0'
          && ntok < HIST_ARGV_MAX - 1)
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
    *argv_out = (char **) malloc((size_t)(ntok + 1) * sizeof(char *));
    for(int i = 0; i < ntok; i++)
    {
        (*argv_out)[i] = tokens[i];
    }
    (*argv_out)[ntok] = NULL;
}

/**
 * @brief Free the argv array from cmdline_split().
 */
static void cmdline_free(
    int  argc,
    char **argv)
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
 * highlighted in bold green when
 * opts->highlight_self is set.
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
        printf("No history log found (%s)\n", CLI_history_log_file());
        return;
    }

    char line[2048];
    int  cap    = 1024;

    char **lines   = (char **) malloc((size_t) cap * sizeof(char *));
    int  *is_self  = (int *)   malloc((size_t) cap * sizeof(int));
    char *types    = (char *)  malloc((size_t) cap * sizeof(char));

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

    while(fgets(line,
                (int) sizeof(line), fp))
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

        /* Parse: ts\tsid\ttty\t[type\t]text */
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

        /* Detect 5-field vs 4-field */
        char  entry_type = 'C';
        char *text_start;
        {
            char *tab4 = strchr(tab3 + 1, '\t');
            if(tab4 != NULL
               && (tab4 - tab3) == 2)
            {
                entry_type = *(tab3 + 1);
                text_start = tab4 + 1;
            }
            else
            {
                text_start = tab3 + 1;
            }
        }

        /* Type filter */
        if(opts->type_filter != '\0'
           && entry_type
              != opts->type_filter)
        {
            continue;
        }

        /* Session filter */
        if(opts->filter_session != NULL)
        {
            int fslen = (int) strlen(opts->filter_session);
            if(sid_len != fslen
               || strncmp(
                      sid_start,
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
        if(opts->time_after
           || opts->time_before)
        {
            char     ts_buf[32];
            int      tslen = (int)(tab1 - line);
            if(tslen
               >= (int) sizeof(ts_buf))
            {
                tslen = (int) sizeof(ts_buf) - 1;
            }
            memcpy(ts_buf, line, (size_t) tslen);
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
             == (int) strlen(
                    data.session_id)
             && strncmp(sid_start, data.session_id, (size_t) sid_len) == 0);

        /* Store */
        if(total >= cap)
        {
            cap *= 2;
            char **tmp1 = (char **) realloc(lines, (size_t) cap * sizeof(char *));
            int  *tmp2  = (int *) realloc(is_self, (size_t) cap * sizeof(int));
            char *tmp3  = (char *) realloc(types, (size_t) cap * sizeof(char));
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
    int show_sess = (opts->filter_session == NULL);
    if(show_sess)
    {
        printf("\033[1;36m %-24s %-19s" "  T  %s\033[0m\n", "Session", "Time", "Entry");
    }
    else
    {
        printf("\033[1;36m %-19s" "  T  %s\033[0m\n", "Time", "Entry");
    }

    /* Print entries */
    int shown = 0;
    for(int i = start; i < total; i++)
    {
        /* Re-parse for display */
        char buf[2048];
        strncpy(buf, lines[i], sizeof(buf) - 1);
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

        /* Shorten tty */
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
            snprintf(sess_col, sizeof(sess_col), "%s %s", sid, tty_s);
            printf("%s %-24s %-19s" "  %c  %s%s\n", col_on, sess_col, ts, etype, etext, col_off);
        }
        else
        {
            printf("%s %-19s" "  %c  %s%s\n", col_on, ts, etype, etext, col_off);
        }
        shown++;
    }

    printf("(%d entr%s)\n", shown, shown == 1 ? "y" : "ies");

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
 *   ghistory [N]        last N entries (default 20)
 *   ghistory -n N       last N entries
 *   ghistory -s SID     filter by session ID
 *   ghistory -t TYPE    filter: prompt/cmd/shell
 *   ghistory -g PAT     glob-filter on text
 *   ghistory --since TS entries after TS
 *   ghistory --until TS entries before TS
 *   ghistory --today    entries from today
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
                printf("ghistory: unknown " "type '%s'\n" "  valid: prompt " "cmd shell\n", tv);
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
                printf("ghistory: bad time" " '%s'\n", argv[i]);
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
                printf("ghistory: bad time" " '%s'\n", argv[i]);
                cmdline_free(argc, argv);
                return RETURN_FAILURE;
            }
            opts.max_entries = 0;
        }
        else if(strcmp(a, "--today") == 0)
        {
            parse_time_arg("today", &opts.time_after);
            opts.max_entries = 0;
        }
        else if(isdigit(
                    (unsigned char) a[0]))
        {
            int n = atoi(a);
            if(n > 0)
            {
                opts.max_entries = n;
            }
        }
        else
        {
            printf("ghistory: unknown " "option '%s'\n", a);
            printf(
                "Usage: ghistory [N]\n"
                "  -n N       last N\n"
                "  -s SID     session\n"
                "  -t TYPE    prompt/cmd/"
                "shell\n"
                "  -g PAT     glob\n"
                "  --since T  after time\n"
                "  --until T  before time\n"
                "  --today    today only\n"
                "T: today Nm Nh Nd " "YYYY-MM-DD " "YYYY-MM-DDTHH:MM:SS\n");
            cmdline_free(argc, argv);
            return RETURN_FAILURE;
        }
    }

    history_log_display(&opts);
    cmdline_free(argc, argv);
    return RETURN_SUCCESS;
}


/**
 * @brief Local (session) history (lhistory)
 *
 * Usage:
 *   lhistory [N]        all or last N entries
 *   lhistory -n N       last N entries
 *   lhistory -t TYPE    filter: prompt/cmd/shell
 *   lhistory -g PAT     glob-filter on text
 *   lhistory --since TS entries after TS
 *   lhistory --until TS entries before TS
 *   lhistory --today    entries from today
 */
errno_t cli_lhistory(void)
{
    HistDisplayOpts opts;
    memset(&opts, 0, sizeof(opts));
    opts.filter_session = data.session_id;
    opts.max_entries    = 0;

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
                printf("lhistory: unknown " "type '%s'\n" "  valid: prompt " "cmd shell\n", tv);
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
                printf("lhistory: bad time" " '%s'\n", argv[i]);
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
                printf("lhistory: bad time" " '%s'\n", argv[i]);
                cmdline_free(argc, argv);
                return RETURN_FAILURE;
            }
        }
        else if(strcmp(a, "--today") == 0)
        {
            parse_time_arg("today", &opts.time_after);
        }
        else if(isdigit(
                    (unsigned char) a[0]))
        {
            int n = atoi(a);
            if(n > 0)
            {
                opts.max_entries = n;
            }
        }
        else
        {
            printf("lhistory: unknown " "option '%s'\n", a);
            printf(
                "Usage: lhistory [N]\n"
                "  -n N       last N\n"
                "  -t TYPE    prompt/cmd/"
                "shell\n"
                "  -g PAT     glob\n"
                "  --since T  after time\n"
                "  --until T  before time\n" "  --today    today only\n");
            cmdline_free(argc, argv);
            return RETURN_FAILURE;
        }
    }

    history_log_display(&opts);
    cmdline_free(argc, argv);
    return RETURN_SUCCESS;
}
