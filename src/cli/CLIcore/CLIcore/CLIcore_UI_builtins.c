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

#include "timeutils.h"


/**
 * @brief CLI handler: watch <interval_ms> <command>
 *
 * Repeats a command at a fixed interval with
 * in-place terminal refresh. Press any key to stop.
 */
errno_t cli_watch(void)
{
    if (data.cmdNBarg < 3)
    {
        printf("Usage: watch <interval_ms>"
               " <command...>\n");
        return RETURN_FAILURE;
    }

    long interval_ms = data.cmdargtoken[1].val.numl;
    if (interval_ms < 10)
    {
        interval_ms = 10;
    }

    /* Build command from remaining args */
    char watchcmd[STRINGMAXLEN_CLICMDLINE];
    watchcmd[0] = '\0';
    for (long a = 2; a < data.cmdNBarg; a++)
    {
        if (a > 2)
        {
            strncat(watchcmd, " ", STRINGMAXLEN_CLICMDLINE - strlen(watchcmd) - 1);
        }
        strncat(watchcmd, data.cmdargtoken[a].val.string,
                STRINGMAXLEN_CLICMDLINE - strlen(watchcmd) - 1);
    }

    /* Switch terminal to raw mode so we can
     * detect single keypresses without Enter.
     * Readline leaves the terminal in cooked
     * mode which buffers input. */
    struct termios orig_termios;
    struct termios raw_termios;
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw_termios = orig_termios;
    raw_termios.c_lflag &= ~((tcflag_t) ICANON | (tcflag_t) ECHO);
    raw_termios.c_cc[VMIN]  = 0;
    raw_termios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw_termios);

    /* Loop until keypress */
    for (;;)
    {
        /* Clear screen, move cursor to top */
        printf("\033[2J\033[H");

        /* Print header */
        {
            time_t     now = time(NULL);
            struct tm *tm  = localtime(&now);
            printf("Every %ldms: %s   "
                   "%02d:%02d:%02d"
                   "  (press any key to stop)\n\n",
                   interval_ms, watchcmd, tm->tm_hour, tm->tm_min, tm->tm_sec);
        }

        /* Execute the command */
        strncpy(data.CLIcmdline, watchcmd, STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        CLI_execute_line();

        fflush(stdout);

        /* Sleep in small increments, checking
         * for keypress */
        {
            long slept = 0;
            long step  = 50000; /* 50 ms */
            while (slept < interval_ms * 1000)
            {
                struct timeval tv;
                fd_set         fds;
                tv.tv_sec  = 0;
                tv.tv_usec = step;
                FD_ZERO(&fds);
                FD_SET(STDIN_FILENO, &fds);
                int r = select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
                if (r > 0)
                {
                    /* Consume the keypress */
                    char discard;
                    if (read(STDIN_FILENO, &discard, 1) > 0)
                    {
                        /* ignore value */
                    }
                    goto watch_done;
                }
                slept += step;
            }
        }
    }

watch_done:
    /* Restore original terminal settings */
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
    printf("\nwatch stopped.\n");

    return RETURN_SUCCESS;
}


/**
 * @brief Source ~/.milkrc on startup.
 *
 * Reads the user's milkrc file line-by-line and
 * executes each non-blank, non-comment line
 * through CLI_execute_line(). Silently skips
 * processing if the file does not exist.
 */
void cli_milkrc_load(void)
{
    char rcpath[STRINGMAXLEN_FULLFILENAME];
    snprintf(rcpath, STRINGMAXLEN_FULLFILENAME, "%s/.milkrc", getenv("HOME"));

    FILE *fp = fopen(rcpath, "r");
    if (fp == NULL)
    {
        return;
    }

    char line[STRINGMAXLEN_CLICMDLINE];
    while (fgets(line, STRINGMAXLEN_CLICMDLINE, fp) != NULL)
    {
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n')
        {
            line[len - 1] = '\0';
        }
        const char *p = line;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '\0' || *p == '#')
        {
            continue;
        }
        strncpy(data.CLIcmdline, line, STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        CLI_execute_line();
    }
    fclose(fp);
}


/**
 * @brief Measure and report wall-clock execution
 *        time of a command.
 *
 * Usage: time <command...>
 */
errno_t cli_time(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: time <command...>\n");
        return RETURN_FAILURE;
    }
    char timecmd[STRINGMAXLEN_CLICMDLINE];
    timecmd[0] = '\0';
    for (long a = 1; a < data.cmdNBarg; a++)
    {
        if (a > 1)
        {
            strncat(timecmd, " ", STRINGMAXLEN_CLICMDLINE - strlen(timecmd) - 1);
        }
        strncat(timecmd, data.cmdargtoken[a].val.string,
                STRINGMAXLEN_CLICMDLINE - strlen(timecmd) - 1);
    }
    struct timespec t0;
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    strncpy(data.CLIcmdline, timecmd, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    CLI_execute_line();

    clock_gettime(CLOCK_MONOTONIC, &t1);
    {
        double elapsed =
            (double) (t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double) (t1.tv_nsec - t0.tv_nsec);
        printf("\n\033[33mElapsed: %.6f s\033[0m\n", elapsed);
    }
    return RETURN_SUCCESS;
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
 * Usage: cli.timing [on|off]
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
            printf("Usage: cli.timing [on|off]\n");
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
            printf("Usage: synhl [on|off|0|1|2]\n");
        }
    }
    else
    {
        data.syntax_highlight = (data.syntax_highlight + 1) % 3;
        printf("Syntax highlighting level %d\n", data.syntax_highlight);
    }
    return RETURN_SUCCESS;
}
#endif


/**
 * @brief Execute a milk script file (source cmd).
 *
 * Opens the file and executes each line through
 * CLI_execute_line(). Maintains a source file
 * location stack for error reporting.
 *
 * Usage: source <filename>
 */
errno_t cli_source(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: source <filename>\n");
        return RETURN_FAILURE;
    }
    const char *fname = data.cmdargtoken[1].val.string;
    FILE       *fp    = fopen(fname, "r");
    if (fp == NULL)
    {
        printf("source: cannot open '%s'\n", fname);
        return RETURN_FAILURE;
    }

    /* Push source location */
    int src_slot = -1;
    if (cli_src_depth < CLI_SRC_STACK_DEPTH)
    {
        src_slot = cli_src_depth;
        strncpy(cli_src_stack[src_slot].file, fname, sizeof(cli_src_stack[0].file) - 1);
        cli_src_stack[src_slot].file[sizeof(cli_src_stack[0].file) - 1] = '\0';
        cli_src_stack[src_slot].line                                    = 0;
        cli_src_depth++;
    }

    char line[STRINGMAXLEN_CLICMDLINE];
    int  lineno = 0;
    while (fgets(line, STRINGMAXLEN_CLICMDLINE, fp) != NULL)
    {
        lineno++;
        if (src_slot >= 0)
        {
            cli_src_stack[src_slot].line = lineno;
        }
        {
            size_t len = strlen(line);
            if (len > 0 && line[len - 1] == '\n')
            {
                line[len - 1] = '\0';
            }
        }
        const char *p = line;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '\0' || *p == '#')
        {
            continue;
        }
        strncpy(data.CLIcmdline, line, STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        if (data.echo_input)
        {
            printf("\033[32m[echo]\033[0m \u2190 \"%s\"\n", data.CLIcmdline);
        }
        errno_t ret = CLI_execute_line();
        if (ret != RETURN_SUCCESS)
        {
            printf("\033[31m[source:%s:%d] "
                   "error\033[0m\n",
                   fname, lineno);
            cli_print_source_trace();
        }
    }
    fclose(fp);

    /* Pop source location */
    if (src_slot >= 0 && cli_src_depth > 0)
    {
        cli_src_depth--;
    }

    return RETURN_SUCCESS;
}


/**
 * @brief Write all CLI variables and user functions
 *        to a file that can be sourced later.
 *
 * Usage: savescript <filename>
 */
errno_t cli_savescript(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: savescript <filename>\n");
        return RETURN_FAILURE;
    }
    const char *fname = data.cmdargtoken[1].val.string;
    FILE       *fp    = fopen(fname, "w");
    if (fp == NULL)
    {
        printf("savescript: cannot open "
               "'%s' for writing\n",
               fname);
        return RETURN_FAILURE;
    }

    fprintf(fp, "# milk-cli script\n");
    fprintf(fp, "# saved by savescript command\n\n");

    /* Export variables */
    int nv = 0;
    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        if (cli_vars[i].used)
        {
            fprintf(fp, "%s=%s\n", cli_vars[i].name, cli_vars[i].val);
            nv++;
        }
    }
    if (nv > 0)
    {
        fprintf(fp, "\n");
    }

    /* Export user-defined functions */
    int nf = 0;
    for (int i = 0; i < CLI_MAX_FUNCS; i++)
    {
        if (cli_funcs[i].used)
        {
            fprintf(fp, "function %s {\n", cli_funcs[i].name);
            for (int j = 0; j < cli_funcs[i].nbody; j++)
            {
                fprintf(fp, "%s\n", cli_funcs[i].body[j]);
            }
            fprintf(fp, "}\n\n");
            nf++;
        }
    }

    fclose(fp);
    printf("Saved %d variables, %d functions "
           "to '%s'\n",
           nv, nf, fname);
    return RETURN_SUCCESS;
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
        printf("Usage: sessionlog "
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

/**
 * @brief Log a command to session log if active
 */
void cli_session_log_cmd(const char *cmd)
{
    if (session_log_fp == NULL)
    {
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed_ms = (double) (now.tv_sec - session_log_t0.tv_sec) * 1000.0 +
                        (double) (now.tv_nsec - session_log_t0.tv_nsec) / 1.0e6;
    {
        time_t t = time(NULL);
        char   tbuf[64];
        strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", localtime(&t));
        fprintf(session_log_fp, "[%s] [%10.1f ms] %s\n", tbuf, elapsed_ms, cmd);
        fflush(session_log_fp);
    }
}

/**
 * @brief Change the current working directory.
 *
 * Usage: cd [dir]  (defaults to $HOME)
 */
errno_t cli_cd(void)
{
    const char *dir = getenv("HOME");
    if (data.cmdNBarg >= 2)
    {
        dir = data.cmdargtoken[1].val.string;
    }
    if (dir != NULL)
    {
        if (chdir(dir) != 0)
        {
            printf("cd: %s: %s\n", dir, strerror(errno));
            return RETURN_FAILURE;
        }
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Print the current working directory.
 */
errno_t cli_pwd(void)
{
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        printf("%s\n", cwd);
        return RETURN_SUCCESS;
    }
    else
    {
        PRINT_ERROR("pwd: %s", strerror(errno));
        return RETURN_FAILURE;
    }
}
