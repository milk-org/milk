/**
 * @file CLIcore_UI_execute.c
 *
 * @brief Core CLI command dispatcher.
 *
 * Contains CLI_execute_line(), the central command-
 * processing loop. Helper functions are in sub-modules:
 *
 *  CLIcore_UI_execute_debug.c
 *      write_tracedebugfile, CLI_execute_string
 *  CLIcore_UI_execute_preproc.c
 *      Text transforms: logical ops, pipe splitting,
 *      semicolons, dot-source rewrite
 *  CLIcore_UI_execute_redir.c
 *      I/O redirection, subshell, background,
 *      external command runner
 */

#include <stdio.h>
#include <ctype.h>
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
#include "CLIcore/cli_calc_parser.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_UI_execute_internal.h"

#include <errno.h>
#include <fnmatch.h>
#include <glob.h>
#include <strings.h>
#include <spawn.h>
#include <sys/wait.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

#include "ImageStreamIO.h"
#include "fps_connect.h"
#include "fps_paramvalue.h"

#define CLICOMPLETIONMODE_COMMANDS 0
#define CLICOMPLETIONMODE_IMAGES   1
#define CLICOMPLETIONMODE_CMDARGS  2
#define CLICOMPLETIONMODE_FILES    3

#define CLICOMPLETIONMODE_FPSPARAMS 4

// COLORRESET removed to prevent redefinition with fps.h
#define COLORRED       "\001\033[31m\002" /* Red */
#define COLORHBOLDCYAN "\001\e[0;96m\002" /* High Intensity Bold Cyan */
#define COLORDIMYELLOW "\033[2;33m" /* Dim Yellow (no RL wrap) */
#include <wordexp.h>
#define COLORRST       "\033[0m"    /* Reset (no RL wrap) */
#define RL_COLORRESET  "\001\033[0m\002"

extern void yy_scan_string(const char *);
extern int  yylex_destroy(void);

/**
 * @brief Central command dispatcher — the main entry
 *        point for every line the CLI processes.
 *
 * Performs all stages of command processing in order:
 *  1. History expansion (!! and !$)
 *  2. Script flow-control interception (if/while/for)
 *  3. Semicolon splitting and command chaining
 *  4. Variable expansion ($VAR, ${VAR}, $())
 *  5. FPS variable expansion (@fps.param)
 *  6. Arithmetic expansion ($((...)))
 *  7. Variable assignment (VAR=val)
 *  8. Alias expansion
 *  9. Command lookup in the registered command table
 * 10. Math expression detection (via calc tokenizer)
 * 11. Fallback to /bin/sh for unrecognized commands
 *
 * Each processed command is logged to the session log
 * and timing statistics are collected if enabled.
 *
 * @return RETURN_SUCCESS on success, or an errno_t
 *         error code
 */
errno_t CLI_execute_line()
{
    DEBUG_TRACE_FSTART();

    char            *cmdargstring __attribute__((unused));
    int strmaxlen   = 200;
    char             str[strmaxlen];
    FILE            *fp;
    time_t           t;
    struct tm       *uttime;
    struct timespec *thetime =
        (struct timespec *) malloc(sizeof(struct timespec));
    char calctmpimname[STRINGMAXLEN_IMGNAME];

    /* Expand history (!! and !$) first */
    cli_history_expand();
    if(data.CLIcmdline[0] == '\0')
    {
        free(thetime);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    /* Poll engine event traps */
    cli_engine_traps_poll();

    /* Expand aliases before anything else */
    cli_alias_expand();

    /* Split at top-level && or || */
    {
        errno_t ret;
        if (cli_split_logical_op(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Rewrite stream |> pipeline */
    {
        errno_t ret;
        if (cli_rewrite_stream_pipe(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Split at milk-to-milk pipe */
    {
        errno_t ret;
        if (cli_split_pipe(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Dot-sourcing: ". file" → "source file" */
    cli_rewrite_dot_source();

    /* Flow control: if/while/for/function
     * and user-defined function calls.
     * Must run BEFORE expansion so block
     * accumulator stores raw lines with
     * $VAR unexpanded. */
    if(cli_script_intercept(data.CLIcmdline))
    {
        data.CMDexecuted = 1;
        free(thetime);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    /* Expand @fpsname.param tokens */
    cli_expand_fpsvar(data.CLIcmdline,
                      STRINGMAXLEN_CLICMDLINE);

    /* Expand milk variables ($VAR, $cam.xsize).
     * Leaves other expansions $(...), $((...)) for wordexp. */
    cli_expand_env(data.CLIcmdline,
                   STRINGMAXLEN_CLICMDLINE);

    /* Expand brace ranges {N..M} {N..M..S} (wordexp does not support) */
    cli_expand_braces(data.CLIcmdline,
                      STRINGMAXLEN_CLICMDLINE);

    /* Log command to session log if active */
    cli_session_log_cmd(data.CLIcmdline);

    /* set -x: trace output */
    if(cli_flag_xtrace)
    {
        fprintf(stderr, "+ %s\n",
                data.CLIcmdline);
    }

    /* Output redirection: > or >> */
    {
        errno_t ret;
        if (cli_handle_output_redir(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Here-string: cmd <<< "text" */
    {
        errno_t ret;
        if (cli_handle_herestring_early(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Background: cmd & */
    {
        errno_t ret;
        if (cli_handle_background(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Subshell: (cmd1; cmd2) */
    {
        errno_t ret;
        if (cli_handle_subshell(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Here-string: cmd <<< "text" (late) */
    {
        errno_t ret;
        if (cli_handle_herestring_late(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Stderr redirect: 2>&1, 2>file */
    {
        errno_t ret;
        if (cli_handle_stderr_redir(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Input redirection: cmd < file */
    {
        errno_t ret;
        if (cli_handle_input_redir(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Command chaining: ; && || */
    {
        errno_t ret;
        if (cli_split_semicolon(&ret))
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return ret;
        }
    }

    /* Check for array assignment: arr=(a b c) */
    if(cli_try_array_assign(data.CLIcmdline))
    {
        data.CMDexecuted = 1;
        free(thetime);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    /* ---- Calc expression evaluation ---- */
    /* Try native mathematical evaluation for
     * expressions like: crop1 = wfs + 1.0
     * This works in all modes (interactive,
     * -c flag, FIFO input).
     * Must run BEFORE cli_try_var_assign()
     * because var_assign intercepts space-
     * separated = signs and doesn't handle
     * image arithmetic or rename. */
    if (data.CLIcmdline[0] != '\0'
        && data.CLIcmdline[0] != '!')
    {
        char firstword[2048];
        if (sscanf(data.CLIcmdline,
                   " %2047s",
                   firstword) == 1)
        {
            int is_internal = 0;
            if (strcmp(firstword, "if") == 0
                || strcmp(firstword,
                         "elif") == 0
                || strcmp(firstword,
                         "else") == 0
                || strcmp(firstword,
                         "fi") == 0
                || strcmp(firstword,
                         "for") == 0
                || strcmp(firstword,
                         "while") == 0
                || strcmp(firstword,
                         "do") == 0
                || strcmp(firstword,
                         "done") == 0
                || strcmp(firstword,
                         ".") == 0
                || strcmp(firstword,
                         "source") == 0)
            {
                is_internal = 1;
            }
            if (!is_internal)
            {
                for (long i = 0;
                     i < (long) data.NBcmd;
                     i++)
                {
                    size_t cmdlen =
                        strlen(
                            data.cmd[i].key);
                    if (strncmp(
                            firstword,
                            data.cmd[i].key,
                            cmdlen) == 0
                        && (firstword[cmdlen]
                                == '\0'
                            || firstword[
                                   cmdlen]
                                   == ':'
                            || firstword[
                                   cmdlen]
                                   == ' '))
                    {
                        is_internal = 1;
                        break;
                    }
                }
            }
            if (!is_internal)
            {
                if (cli_calc_eval_line(
                        data.CLIcmdline))
                {
                    free(thetime);
                    return RETURN_SUCCESS;
                }
            }
        }
    }

    /* Check for variable assignment (VAR=val) */
    if(cli_try_var_assign(data.CLIcmdline))
    {
        data.CMDexecuted = 1;
        free(thetime);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    /* ---- Smart Shell Fallback Bypass ---- */
    /* If the command does not start with an
     * internal milk command, alias, or script
     * keyword, bypass manual pipe/redirect
     * parsing and instantly delegate the
     * full pipeline to bash. */
    if (data.CLIcmdline[0] != '\0'
        && data.CLIcmdline[0] != '!'
        && data.CLIloopON == 1)
    {
        char firstword[2048];
        if (sscanf(data.CLIcmdline,
                   " %2047s",
                   firstword) == 1)
        {
            int is_internal = 0;
            if (strcmp(firstword, "if") == 0
                || strcmp(firstword,
                         "elif") == 0
                || strcmp(firstword,
                         "else") == 0
                || strcmp(firstword,
                         "fi") == 0
                || strcmp(firstword,
                         "for") == 0
                || strcmp(firstword,
                         "while") == 0
                || strcmp(firstword,
                         "do") == 0
                || strcmp(firstword,
                         "done") == 0
                || strcmp(firstword,
                         ".") == 0
                || strcmp(firstword,
                         "source") == 0)
            {
                is_internal = 1;
            }
            if (!is_internal)
            {
                const char *eq =
                    strchr(firstword, '=');
                if (eq != NULL)
                {
                    is_internal = 1;
                }
            }

            if (!is_internal)
            {
                for (long i = 0;
                     i < (long) data.NBcmd;
                     i++)
                {
                    size_t cmdlen =
                        strlen(
                            data.cmd[i].key);
                    if (strncmp(
                            firstword,
                            data.cmd[i].key,
                            cmdlen) == 0
                        && (firstword[cmdlen]
                                == '\0'
                            || firstword[
                                   cmdlen]
                                   == ':'
                            || firstword[
                                   cmdlen]
                                   == ' '))
                    {
                        is_internal = 1;
                        break;
                    }
                }
            }
            if (!is_internal)
            {
                printf(
                    COLORDIMYELLOW
                    "[shell bypass] %s"
                    COLORRST "\n",
                    data.CLIcmdline);
                cli_history_log_shell(
                    data.CLIcmdline);
                cli_export_vars_to_env();
                cli_run_external(
                    data.CLIcmdline);
                free(thetime);
                return RETURN_SUCCESS;
            }
        }
    }

    /* ---- Pipe to shell ---- */
    FILE *pipe_fp = NULL;
    int   saved_stdout_fd = -1;
    {
        char *pipe_pos = NULL;
        {
            int depth = 0;
            int in_sq = 0;
            int in_dq = 0;
            for(int si = 0; data.CLIcmdline[si] != '\0'; si++)
            {
                char c = data.CLIcmdline[si];
                if(c == '\'' && !in_dq)
                {
                    in_sq = !in_sq;
                }
                else if(c == '"' && !in_sq)
                {
                    in_dq = !in_dq;
                }
                else if(!in_sq && !in_dq)
                {
                    if(c == '(')
                    {
                        depth++;
                    }
                    else if(c == ')' && depth > 0)
                    {
                        depth--;
                    }
                    else if(depth == 0 && c == '|')
                    {
                        pipe_pos = data.CLIcmdline + si;
                        break;
                    }
                }
            }
        }
        if(pipe_pos != NULL)
        {
            *pipe_pos = '\0';
            const char *rhs = pipe_pos + 1;
            while(*rhs == ' ' || *rhs == '\t')
            {
                rhs++;
            }
            if(*rhs != '\0')
            {
                printf(COLORDIMYELLOW
                       "[shell pipe] %s"
                       COLORRST "\n", rhs);
                cli_export_vars_to_env();
                pipe_fp = popen(rhs, "w");
                if(pipe_fp != NULL)
                {
                    saved_stdout_fd =
                        dup(STDOUT_FILENO);
                    dup2(fileno(pipe_fp),
                         STDOUT_FILENO);
                }
            }
        }
    }

    /* ---- Output redirect to file ---- */
    FILE *redir_fp = NULL;
    int   saved_stdout_redir = -1;
    if(pipe_fp == NULL)
    {
        char *redir_pos = NULL;
        {
            int depth = 0;
            int in_sq = 0;
            int in_dq = 0;
            for(int si = 0; data.CLIcmdline[si] != '\0'; si++)
            {
                char c = data.CLIcmdline[si];
                if(c == '\'' && !in_dq)
                {
                    in_sq = !in_sq;
                }
                else if(c == '"' && !in_sq)
                {
                    in_dq = !in_dq;
                }
                else if(!in_sq && !in_dq)
                {
                    if(c == '(')
                    {
                        depth++;
                    }
                    else if(c == ')' && depth > 0)
                    {
                        depth--;
                    }
                    else if(depth == 0 && c == '>')
                    {
                        redir_pos = data.CLIcmdline + si;
                        break;
                    }
                }
            }
        }
        if(redir_pos != NULL)
        {
            *redir_pos = '\0';
            const char *fname = redir_pos + 1;
            while(*fname == ' '
                    || *fname == '\t')
            {
                fname++;
            }
            if(*fname != '\0')
            {
                char fpath[500];
                strncpy(fpath, fname, 499);
                fpath[499] = '\0';
                {
                    size_t fl = strlen(fpath);
                    while(fl > 0
                            && (fpath[fl - 1]
                                == ' '
                                || fpath[fl - 1]
                                == '\t'
                                || fpath[fl - 1]
                                == '\n'))
                    {
                        fpath[--fl] = '\0';
                    }
                }
                redir_fp = fopen(fpath, "w");
                if(redir_fp != NULL)
                {
                    saved_stdout_redir =
                        dup(STDOUT_FILENO);
                    dup2(fileno(redir_fp),
                         STDOUT_FILENO);
                }
            }
        }
    }

    /* Log resolved command (type C) to
     * structured history.  add_history()
     * is now called in rl_cb_linehandler()
     * with the raw prompt text. */
    cli_history_log_cmd(data.CLIcmdline);

    //
    // If line starts with !, run as external
    // command via cli_run_external()
    //
    if(data.CLIcmdline[0] == '!')
    {
        data.CLIcmdline[0] = ' ';
        printf(COLORDIMYELLOW
               "[shell] %s" COLORRST "\n",
               data.CLIcmdline);
        cli_export_vars_to_env();
        if(cli_run_external(
               data.CLIcmdline) != 0)
        {
            PRINT_ERROR("shell command error");
            exit(4);
        }
        data.CMDexecuted = 1;
    }
    else if(data.CLIcmdline[0] == '#')
    {
        // do nothing... this is a comment
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "listim ", 7) == 0)
    {
        /* listim <pattern> — glob filter
         * Only intercept when pattern has
         * wildcard chars (* or ?).
         * Non-wildcard falls through to
         * normal registered command. */
        const char *pat =
            data.CLIcmdline + 7;
        while(*pat == ' ' || *pat == '\t')
        {
            pat++;
        }
        if(strchr(pat, '*') != NULL
           || strchr(pat, '?') != NULL)
        {
            /* Build glob pattern for
             * /dev/shm matching */
            char shmglob[512];
            snprintf(shmglob,
                     sizeof(shmglob),
                     "%s.im.shm", pat);
            DIR *dp = opendir("/dev/shm");
            if(dp != NULL)
            {
                struct dirent *de;
                int count = 0;
                while((de = readdir(dp))
                      != NULL)
                {
                    if(fnmatch(
                        shmglob,
                        de->d_name,
                        0) == 0)
                    {
                        /* Strip .im.shm
                         * suffix */
                        char nm[256];
                        strncpy(nm,
                                de->d_name,
                                sizeof(nm)
                                - 1);
                        nm[sizeof(nm) - 1]
                            = '\0';
                        char *sfx =
                            strstr(nm,
                                   ".im.shm");
                        if(sfx != NULL)
                        {
                            *sfx = '\0';
                        }
                        printf("  %s\n", nm);
                        count++;
                    }
                }
                closedir(dp);
                printf("%d stream(s) "
                       "matched\n",
                       count);
            }
            data.CMDexecuted = 1;
        }
        /* else: no wildcard, fall through
         * to normal listim command */
    }
    else if(strncmp(data.CLIcmdline,
                    "echo ", 5) == 0
            || strcmp(data.CLIcmdline,
                      "echo") == 0)
    {
        /* Handle echo before tokenization
         * to avoid image name resolution */
        const char *args =
            data.CLIcmdline + 4;
        while(*args == ' ')
        {
            args++;
        }
        int nl = 1;
        if(strncmp(args, "-n ", 3) == 0)
        {
            nl = 0;
            args += 3;
            while(*args == ' ')
            {
                args++;
            }
        }
        printf("%s", args);
        if(nl)
        {
            printf("\n");
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "printf ", 7) == 0)
    {
        /* Intercept printf before
         * tokenization so % and
         * backslash are preserved */
        const char *raw =
            data.CLIcmdline + 7;
        while(*raw == ' ')
        {
            raw++;
        }

        /* Tokenize manually: split on
         * spaces, respecting quotes */
        data.cmdNBarg = 1;
        strncpy(
            data.cmdargtoken[0].val.string,
            "printf",
            STRINGMAXLEN_CMDARGTOKEN_VAL - 1);

        const char *s = raw;
        while(*s != '\0'
              && data.cmdNBarg
                 < NB_ARG_MAX)
        {
            while(*s == ' ') s++;
            if(*s == '\0') break;
            int ai = 0;
            if(*s == '"')
            {
                s++;
                while(*s != '\0'
                      && *s != '"'
                      && ai
                         < STRINGMAXLEN_CMDARGTOKEN_VAL
                           - 1)
                {
                    data.cmdargtoken[
                        data.cmdNBarg]
                        .val.string[ai++]
                        = *s++;
                }
                if(*s == '"') s++;
            }
            else
            {
                while(*s != '\0'
                      && *s != ' '
                      && ai
                         < STRINGMAXLEN_CMDARGTOKEN_VAL
                           - 1)
                {
                    data.cmdargtoken[
                        data.cmdNBarg]
                        .val.string[ai++]
                        = *s++;
                }
            }
            data.cmdargtoken[
                data.cmdNBarg]
                .val.string[ai] = '\0';
            data.cmdNBarg++;
        }

        cli_cmd_printf();
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "export ", 7) == 0
            || strcmp(data.CLIcmdline,
                     "export") == 0)
    {
        /* Intercept export before
         * tokenization so = in
         * VAR=value is preserved */
        const char *raw =
            data.CLIcmdline + 6;
        while(*raw == ' ')
        {
            raw++;
        }

        data.cmdNBarg = 1;
        strncpy(
            data.cmdargtoken[0].val.string,
            "export",
            STRINGMAXLEN_CMDARGTOKEN_VAL - 1);

        if(*raw != '\0')
        {
            int ai = 0;
            while(*raw != '\0'
                  && *raw != ' '
                  && ai
                     < STRINGMAXLEN_CMDARGTOKEN_VAL
                       - 1)
            {
                data.cmdargtoken[1]
                    .val.string[ai++]
                    = *raw++;
            }
            data.cmdargtoken[1]
                .val.string[ai] = '\0';
            data.cmdNBarg = 2;
        }

        cli_cmd_export();
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "source ", 7) == 0)
    {
        /* Handle before tokenization so
         * file paths with dots are not
         * misinterpreted by the parser */
        const char *arg =
            data.CLIcmdline + 7;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: source "
                   "<filename>\n");
        }
        else
        {
            data.cmdNBarg = 2;
            strncpy(
                data.cmdargtoken[1]
                .val.string,
                arg,
                sizeof(
                    data.cmdargtoken[1]
                    .val.string) - 1);
            cli_source();
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "include_once ", 13) == 0)
    {
        /* include_once <file> — source only
         * if not already sourced. Uses a
         * static table of resolved paths. */
        static char sourced[128][PATH_MAX];
        static int nsourced = 0;

        const char *arg =
            data.CLIcmdline + 13;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: include_once "
                   "<filename>\n");
        }
        else
        {
            char rp[PATH_MAX];
            char *resolved =
                realpath(arg, rp);
            if(resolved == NULL)
            {
                printf("include_once: "
                       "%s: %s\n",
                       arg,
                       strerror(errno));
            }
            else
            {
                int found = 0;
                for(int k = 0;
                    k < nsourced; k++)
                {
                    if(strcmp(sourced[k],
                             rp) == 0)
                    {
                        found = 1;
                        break;
                    }
                }
                if(!found)
                {
                    if(nsourced < 128)
                    {
                        strncpy(
                            sourced[nsourced],
                            rp,
                            PATH_MAX - 1);
                        nsourced++;
                    }
                    data.cmdNBarg = 2;
                    strncpy(
                        data.cmdargtoken[1]
                        .val.string,
                        arg,
                        sizeof(
                            data.cmdargtoken[1]
                            .val.string) - 1);
                    cli_source();
                }
            }
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "savescript ", 11) == 0)
    {
        /* Handle before tokenization so
         * file paths with dots etc. are
         * not misinterpreted */
        const char *arg =
            data.CLIcmdline + 11;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: savescript "
                   "<filename>\n");
        }
        else
        {
            /* Temporarily set cmdNBarg and
             * token for cli_savescript() */
            data.cmdNBarg = 2;
            strncpy(
                data.cmdargtoken[1]
                .val.string,
                arg,
                sizeof(
                    data.cmdargtoken[1]
                    .val.string) - 1);
            cli_savescript();
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "savehistory ", 12) == 0)
    {
        const char *arg =
            data.CLIcmdline + 12;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: savehistory "
                   "<filename>\n");
        }
        else
        {
            data.cmdNBarg = 2;
            strncpy(
                data.cmdargtoken[1]
                .val.string,
                arg,
                sizeof(
                    data.cmdargtoken[1]
                    .val.string) - 1);
            cli_savehistory();
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "on_update ", 10) == 0 ||
            strcmp(data.CLIcmdline,
                   "on_update") == 0)
    {
        /* on_update [-l] [-n N] <stream> { cmd }
         * Wait for stream semaphore,
         * then execute cmd.
         * -l: loop forever
         * -n N: loop N times */
        const char *arg = data.CLIcmdline;
        if(strncmp(data.CLIcmdline, "on_update ", 10) == 0) arg += 10;
        else arg += 9;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }

        /* Parse flags */
        int loop_count = 1; /* default: once */
        while(*arg == '-')
        {
            if(strncmp(arg, "-l", 2) == 0
               && (arg[2] == ' '
                   || arg[2] == '\t'
                   || arg[2] == '\0'))
            {
                loop_count = -1;
                arg += 2;
            }
            else if(strncmp(arg, "-n", 2)
                    == 0)
            {
                arg += 2;
                while(*arg == ' '
                      || *arg == '\t')
                {
                    arg++;
                }
                char *endptr = NULL;
                long nval = strtol(arg, &endptr, 10);
                if(endptr == arg || nval <= 0)
                {
                    fprintf(stderr,
                            "Invalid value for -n option: '%s' (expected positive integer)\n",
                            arg);
                    /* Treat invalid/zero as a no-op for loop_count */
                }
                else
                {
                    loop_count = (int) nval;
                    arg = endptr;
                }
            }
            else
            {
                /* Unknown flag — stop parsing */
                break;
            }
            while(*arg == ' '
                  || *arg == '\t')
            {
                arg++;
            }
        }

        /* Parse stream name */
        char sname[200];
        {
            int si = 0;
            while(*arg != '\0'
                  && *arg != ' '
                  && *arg != '\t'
                  && si < 199)
            {
                sname[si++] = *arg++;
            }
            sname[si] = '\0';
        }
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        /* Skip optional { and } */
        if(*arg == '{')
        {
            arg++;
        }
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        /* Find end, strip } */
        char body[STRINGMAXLEN_CLICMDLINE];
        strncpy(body, arg,
                STRINGMAXLEN_CLICMDLINE - 1);
        body[
            STRINGMAXLEN_CLICMDLINE - 1]
            = '\0';
        {
            int blen = (int) strlen(body);
            while(blen > 0
                  && (body[blen - 1] == '}'
                      || body[blen - 1] == ' '
                      || body[blen - 1]
                      == '\t'))
            {
                blen--;
            }
            body[blen] = '\0';
        }
        if(sname[0] == '\0'
           || body[0] == '\0')
        {
            printf("Usage: on_update "
                   "[-l] [-n N] "
                   "<stream> "
                   "{ command }\n");
        }
        else
        {
            /* Connect to stream and
             * wait for semaphore */
            IMAGE img;
            if(ImageStreamIO_read_sharedmem_image_toIMAGE(
                   sname, &img)
               == IMAGESTREAMIO_SUCCESS)
            {
                int semidx =
                    ImageStreamIO_getsemwaitindex(
                        &img, 0);
                if(semidx >= 0)
                {
                    /* Create processinfo for
                     * loop mode */
                    PROCESSINFO *procinfo =
                        NULL;
                    int is_loop =
                        (loop_count != 1);
                    if(is_loop)
                    {
                        char pname[64];
                        snprintf(pname,
                            sizeof(pname),
                            "on_update_%s",
                            sname);
                        procinfo =
                            processinfo_shm_create(
                                pname,
                                PROCESSINFO_CTRLVAL_RUN);
                    }

                    int iter = 0;
                    int keep_going = 1;
                    while(keep_going
                          && !cli_break_flag)
                    {
                        /* Check procctl */
                        if(procinfo != NULL)
                        {
                            if(procinfo->CTRLval
                               == PROCESSINFO_CTRLVAL_EXIT)
                            {
                                break;
                            }
                            while(procinfo
                                ->CTRLval
                                == PROCESSINFO_CTRLVAL_PAUSE)
                            {
                                usleep(10000);
                                if(cli_break_flag)
                                {
                                    break;
                                }
                            }
                        }

                        ImageStreamIO_semwait(
                            &img, semidx);

                        /* Execute body */
                        strncpy(
                            data.CLIcmdline,
                            body,
                            STRINGMAXLEN_CLICMDLINE
                            - 1);
                        data.CLIcmdline[
                            STRINGMAXLEN_CLICMDLINE
                            - 1] = '\0';
                        CLI_execute_line();

                        iter++;
                        if(procinfo != NULL)
                        {
                            procinfo->loopcnt
                                = iter;
                        }
                        if(loop_count > 0
                           && iter
                           >= loop_count)
                        {
                            keep_going = 0;
                        }
                    }
                    cli_break_flag = 0;

                    if(procinfo != NULL)
                    {
                        processinfo_cleanExit(
                            procinfo);
                    }
                }
            }
            else
            {
                printf("on_update: "
                       "stream %s not "
                       "found\n", sname);
            }
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "on_fpschange ",
                    13) == 0)
    {
        /* on_fpschange [-l] [-n N]
         *   fpsname.param { cmd }
         * Poll FPS parameter, execute body
         * when it changes. */
        const char *arg =
            data.CLIcmdline + 13;
        while(*arg == ' '
              || *arg == '\t')
        {
            arg++;
        }

        /* Parse flags */
        int loop_count = 1;
        while(*arg == '-')
        {
            if(strncmp(arg, "-l", 2) == 0
               && (arg[2] == ' '
                   || arg[2] == '\t'
                   || arg[2] == '\0'))
            {
                loop_count = -1;
                arg += 2;
            }
            else if(strncmp(arg, "-n", 2)
                    == 0)
            {
                arg += 2;
                while(*arg == ' '
                      || *arg == '\t')
                {
                    arg++;
                }
                char *endptr = NULL;
                long nval = strtol(
                    arg, &endptr, 10);
                if(endptr == arg
                   || nval <= 0)
                {
                    fprintf(stderr,
                            "Invalid value for"
                            " -n option: '%s'"
                            " (expected positive"
                            " integer)\n",
                            arg);
                }
                else
                {
                    loop_count = (int) nval;
                    arg = endptr;
                }
            }
            else
            {
                /* Unknown flag — stop parsing */
                break;
            }
            while(*arg == ' '
                  || *arg == '\t')
            {
                arg++;
            }
        }

        /* Extract fpsname.param */
        char fparg[256];
        {
            int ai = 0;
            while(*arg != '\0'
                  && *arg != ' '
                  && *arg != '\t'
                  && ai < 255)
            {
                fparg[ai++] = *arg++;
            }
            fparg[ai] = '\0';
        }
        /* Split at dot */
        char *dot = strchr(fparg, '.');
        if(dot == NULL)
        {
            printf(
                "on_fpschange: "
                "use fpsname.param\n");
            cli_last_retval = 1;
            data.CMDexecuted = 1;
        }
        else
        {
            *dot = '\0';
            const char *fpsn = fparg;
            const char *parn = dot + 1;
            /* Extract body between { } */
            while(*arg == ' '
                  || *arg == '\t')
            {
                arg++;
            }
            char body[
                STRINGMAXLEN_CLICMDLINE];
            body[0] = '\0';
            if(*arg == '{')
            {
                arg++;
                while(*arg == ' '
                      || *arg == '\t')
                {
                    arg++;
                }
                int bi = 0;
                while(*arg != '\0'
                      && *arg != '}'
                      && bi
                      < STRINGMAXLEN_CLICMDLINE
                      - 1)
                {
                    body[bi++] = *arg++;
                }
                body[bi] = '\0';
                /* trim trailing spaces */
                while(bi > 0
                      && (body[bi - 1]
                          == ' '
                          || body[bi - 1]
                          == '\t'))
                {
                    body[--bi] = '\0';
                }
            }
            /* Connect to FPS */
            FUNCTION_PARAMETER_STRUCT fps;
            if(
                function_parameter_struct_connect(
                    fpsn, &fps,
                    FPSCONNECT_SIMPLE)
                != EXIT_SUCCESS)
            {
                printf(
                    "on_fpschange: "
                    "cannot connect "
                    "to fps '%s'\n",
                    fpsn);
                cli_last_retval = 1;
            }
            else
            {
                long pidx =
                    functionparameter_GetParamIndex(
                        &fps, parn);
                if(pidx < 0)
                {
                    printf(
                        "on_fpschange: "
                        "param '%s' not "
                        "found\n", parn);
                    cli_last_retval = 1;
                }
                else
                {
                    /* Create processinfo
                     * for loop mode */
                    PROCESSINFO *procinfo =
                        NULL;
                    int is_loop =
                        (loop_count != 1);
                    if(is_loop)
                    {
                        char pname[64];
                        snprintf(pname,
                            sizeof(pname),
                            "on_fpschg_%s",
                            fpsn);
                        procinfo =
                            processinfo_shm_create(
                                pname,
                                PROCESSINFO_CTRLVAL_RUN);
                    }

                    char prev[256];
                    functionparameter_GetParamValueString(
                        &fps.parray[pidx],
                        prev,
                        sizeof(prev));

                    int iter = 0;
                    int keep_going = 1;
                    while(keep_going
                          && !cli_break_flag)
                    {
                        /* Check procctl */
                        if(procinfo != NULL)
                        {
                            if(procinfo
                                ->CTRLval
                               == PROCESSINFO_CTRLVAL_EXIT)
                            {
                                break;
                            }
                            while(procinfo
                                ->CTRLval
                                == PROCESSINFO_CTRLVAL_PAUSE)
                            {
                                usleep(10000);
                                if(cli_break_flag)
                                {
                                    break;
                                }
                            }
                        }

                        /* Poll for change */
                        char cur[256];
                        for(;;)
                        {
                            usleep(100000);
                            if(cli_break_flag)
                            {
                                break;
                            }
                            if(procinfo
                               != NULL
                               && procinfo
                                ->CTRLval
                               != PROCESSINFO_CTRLVAL_RUN)
                            {
                                break;
                            }
                            functionparameter_GetParamValueString(
                                &fps
                                .parray[pidx],
                                cur,
                                sizeof(cur));
                            if(strcmp(cur,
                                     prev)
                               != 0)
                            {
                                break;
                            }
                        }
                        if(cli_break_flag)
                        {
                            break;
                        }
                        if(procinfo != NULL
                           && procinfo
                            ->CTRLval
                           != PROCESSINFO_CTRLVAL_RUN)
                        {
                            break;
                        }

                        /* Execute body */
                        strncpy(prev, cur,
                                sizeof(prev)
                                - 1);
                        prev[sizeof(prev) - 1] = '\0';
                        strncpy(
                            data.CLIcmdline,
                            body,
                            STRINGMAXLEN_CLICMDLINE
                            - 1);
                        data.CLIcmdline[
                            STRINGMAXLEN_CLICMDLINE
                            - 1] = '\0';
                        CLI_execute_line();

                        iter++;
                        if(procinfo != NULL)
                        {
                            procinfo
                                ->loopcnt
                                = iter;
                        }
                        if(loop_count > 0
                           && iter
                           >= loop_count)
                        {
                            keep_going = 0;
                        }
                    }
                    cli_break_flag = 0;

                    if(procinfo != NULL)
                    {
                        processinfo_cleanExit(
                            procinfo);
                    }
                }
                function_parameter_struct_disconnect(
                    &fps);
            }
            data.CMDexecuted = 1;
        }
    }
    else if(strncmp(data.CLIcmdline,
                    "sleep ", 6) == 0
            || strcmp(data.CLIcmdline,
                     "sleep") == 0)
    {
        /* sleep <seconds> — float-capable
         * delay. Handle before tokenization
         * because the parser would try to
         * interpret decimals. */
        const char *arg =
            data.CLIcmdline + 5;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: sleep "
                   "<seconds>\n");
        }
        else
        {
            double secs = strtod(arg, NULL);
            if(secs > 0.0)
            {
                usleep(
                    (useconds_t)
                    (secs * 1e6));
            }
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "printf ", 7) == 0)
    {
        /* printf "fmt" arg1 arg2 ...
         * Supports %d %f %s %% \n \t */
        const char *p =
            data.CLIcmdline + 7;
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        /* Extract format string */
        char fmt[512];
        int fi = 0;
        char delim = '"';
        if(*p == '"' || *p == '\'')
        {
            delim = *p++;
            while(*p != '\0'
                  && *p != delim
                  && fi < 511)
            {
                fmt[fi++] = *p++;
            }
            if(*p == delim)
            {
                p++;
            }
        }
        else
        {
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && fi < 511)
            {
                fmt[fi++] = *p++;
            }
        }
        fmt[fi] = '\0';
        /* Collect remaining args */
        char *args[16];
        int nargs = 0;
        while(*p != '\0' && nargs < 16)
        {
            while(*p == ' ' || *p == '\t')
            {
                p++;
            }
            if(*p == '\0')
            {
                break;
            }
            char abuf[256];
            int ai = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && ai < 255)
            {
                abuf[ai++] = *p++;
            }
            abuf[ai] = '\0';
            args[nargs] =
                strdup(abuf);
            nargs++;
        }
        /* Print with format */
        {
            int ai = 0;
            for(int k = 0; fmt[k] != '\0';
                k++)
            {
                if(fmt[k] == '\\'
                   && fmt[k + 1] != '\0')
                {
                    k++;
                    if(fmt[k] == 'n')
                    {
                        putchar('\n');
                    }
                    else if(fmt[k] == 't')
                    {
                        putchar('\t');
                    }
                    else if(fmt[k] == '\\')
                    {
                        putchar('\\');
                    }
                    else
                    {
                        putchar('\\');
                        putchar(fmt[k]);
                    }
                }
                else if(fmt[k] == '%'
                        && fmt[k + 1]
                        != '\0')
                {
                    k++;
                    if(fmt[k] == '%')
                    {
                        putchar('%');
                    }
                    else if(fmt[k] == 'd'
                            && ai < nargs)
                    {
                        printf("%ld",
                               strtol(
                                   args[ai++],
                                   NULL, 0));
                    }
                    else if(fmt[k] == 'f'
                            && ai < nargs)
                    {
                        printf("%f",
                               strtod(
                                   args[ai++],
                                   NULL));
                    }
                    else if(fmt[k] == 's'
                            && ai < nargs)
                    {
                        printf("%s",
                               args[ai++]);
                    }
                    else if(fmt[k] == '.'
                            && ai < nargs)
                    {
                        /* Handle %.Nf */
                        char pfmt[16];
                        int pfi = 0;
                        pfmt[pfi++] = '%';
                        pfmt[pfi++] = '.';
                        k++;
                        while(fmt[k] >= '0'
                              && fmt[k] <= '9'
                              && pfi < 14)
                        {
                            pfmt[pfi++] =
                                fmt[k++];
                        }
                        pfmt[pfi++] =
                            fmt[k]; /* f */
                        pfmt[pfi] = '\0';
                        printf(pfmt,
                               strtod(
                                   args[ai++],
                                   NULL));
                    }
                    else
                    {
                        putchar('%');
                        putchar(fmt[k]);
                    }
                }
                else
                {
                    putchar(fmt[k]);
                }
            }
        }
        fflush(stdout);
        for(int k = 0; k < nargs; k++)
        {
            free(args[k]);
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "read ", 5) == 0
            || strcmp(data.CLIcmdline,
                     "read") == 0)
    {
        /* read [-p "prompt"] [-t N]
         * [-a arr] varname
         * Read line from stdin */
        const char *p =
            data.CLIcmdline + 4;
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        /* Parse flags */
        int rd_timeout = -1;
        int rd_array = 0;
        char rd_prompt[256] = {'\0'};
        char rd_aname[CLI_VAR_NAMELEN]
            = {'\0'};
        while(p[0] == '-')
        {
            if(strncmp(p, "-p ", 3)
               == 0)
            {
                p += 3;
                while(*p == ' '
                      || *p == '\t')
                {
                    p++;
                }
                if(*p == '"'
                   || *p == '\'')
                {
                    char delim = *p++;
                    int pi = 0;
                    while(*p != '\0'
                          && *p
                          != delim
                          && pi < 254)
                    {
                        rd_prompt[pi++]
                            = *p++;
                    }
                    rd_prompt[pi] =
                        '\0';
                    if(*p == delim)
                    {
                        p++;
                    }
                }
                else
                {
                    int pi = 0;
                    while(*p != '\0'
                          && *p != ' '
                          && *p
                          != '\t'
                          && pi < 254)
                    {
                        rd_prompt[pi++]
                            = *p++;
                    }
                    rd_prompt[pi] =
                        '\0';
                }
            }
            else if(strncmp(
                        p, "-t ", 3)
                    == 0)
            {
                p += 3;
                while(*p == ' '
                      || *p == '\t')
                {
                    p++;
                }
                rd_timeout = (int)
                    strtol(p, NULL,
                           10);
                while(*p != '\0'
                      && *p != ' '
                      && *p != '\t')
                {
                    p++;
                }
            }
            else if(strncmp(
                        p, "-a ", 3)
                    == 0)
            {
                p += 3;
                while(*p == ' '
                      || *p == '\t')
                {
                    p++;
                }
                rd_array = 1;
                {
                    int ni = 0;
                    while(*p != '\0'
                          && *p != ' '
                          && *p
                          != '\t'
                          && ni
                          < CLI_VAR_NAMELEN
                          - 1)
                    {
                        rd_aname[ni++]
                            = *p++;
                    }
                    rd_aname[ni] =
                        '\0';
                }
            }
            else
            {
                /* Unknown flag */
                p++;
                while(*p != '\0'
                      && *p != ' '
                      && *p != '\t')
                {
                    p++;
                }
            }
            while(*p == ' '
                  || *p == '\t')
            {
                p++;
            }
        }
        /* Print prompt */
        if(rd_prompt[0] != '\0')
        {
            printf("%s", rd_prompt);
            fflush(stdout);
        }
        /* Timeout with select() */
        int rd_ok = 1;
        if(rd_timeout >= 0)
        {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(STDIN_FILENO,
                   &fds);
            struct timeval tv;
            tv.tv_sec = rd_timeout;
            tv.tv_usec = 0;
            int sr = select(
                STDIN_FILENO + 1,
                &fds, NULL, NULL,
                &tv);
            if(sr <= 0)
            {
                rd_ok = 0;
                cli_last_retval = 1;
            }
        }
        if(rd_ok)
        {
            char rbuf[1024];
            if(fgets(rbuf,
                     sizeof(rbuf),
                     stdin)
               != NULL)
            {
                /* Strip trailing
                 * newline */
                size_t rlen =
                    strlen(rbuf);
                while(rlen > 0
                      && (rbuf[
                              rlen - 1]
                          == '\n'
                          || rbuf[
                              rlen - 1]
                          == '\r'))
                {
                    rbuf[--rlen] =
                        '\0';
                }
                if(rd_array)
                {
                    /* Split into array
                     * elements */
                    for(int k = 0;
                        k
                        < CLI_MAX_ARRAYS;
                        k++)
                    {
                        if(!cli_arrays[
                            k].used)
                        {
                            cli_arrays[
                                k]
                                .used = 1;
                            strncpy(
                                cli_arrays[
                                    k]
                                .name,
                                rd_aname,
                                CLI_VAR_NAMELEN
                                - 1);
                            cli_arrays[
                                k]
                                .nelem
                                = 0;
                            char *tok
                                = strtok(
                                    rbuf,
                                    " \t");
                            while(tok
                                  != NULL
                                  && cli_arrays[
                                      k]
                                  .nelem
                                  < CLI_ARRAY_MAXELEM)
                            {
                                strncpy(
                                    cli_arrays[
                                        k]
                                    .elem[
                                        cli_arrays[
                                            k]
                                        .nelem],
                                    tok,
                                    CLI_VAR_VALLEN
                                    - 1);
                                cli_arrays[
                                    k]
                                    .nelem++;
                                tok
                                    = strtok(
                                        NULL,
                                        " \t");
                            }
                            break;
                        }
                    }
                }
                else if(*p != '\0')
                {
                    /* Scalar var */
                    char vname[
                        CLI_VAR_NAMELEN
                    ];
                    int vi = 0;
                    while(*p != '\0'
                          && *p != ' '
                          && *p
                          != '\t'
                          && vi
                          < CLI_VAR_NAMELEN
                          - 1)
                    {
                        vname[vi++] =
                            *p++;
                    }
                    vname[vi] = '\0';
                    cli_var_set(
                        vname, rbuf);
                }
                cli_last_retval = 0;
            }
            else
            {
                cli_last_retval = 1;
            }
        }
        data.CMDexecuted = 1;
    }
    else
    {
        // some initialization
        data.parseerror      = 0;
        data.calctmp_imindex = 0;
        for(int i = 0; i < NB_ARG_MAX; i++)
        {
            data.cmdargtoken[i].type          = CMDARGTOKEN_TYPE_UNSOLVED;
            data.cmdargtoken[i].val.string[0] = '\0';
        }

        // log command if CLIlogON active
        if(data.CLIlogON == 1)
        {
            t      = time(NULL);
            uttime = gmtime(&t);
            clock_gettime(CLOCK_MILK, thetime);

            snprintf(data.CLIlogname,
                     STRINGMAXLEN_FULLFILENAME,
                     "%s/logdir/%04d%02d%02d/%04d%02d%02d_CLI-%s.log",
                     getenv("HOME"),
                     1900 + uttime->tm_year,
                     1 + uttime->tm_mon,
                     uttime->tm_mday,
                     1900 + uttime->tm_year,
                     1 + uttime->tm_mon,
                     uttime->tm_mday,
                     data.processname);

            fp = fopen(data.CLIlogname, "a");
            if(fp == NULL)
            {
                printf("ERROR: cannot log into file %s\n", data.CLIlogname);
                EXECUTE_SYSTEM_COMMAND("mkdir -p %s/logdir/%04d%02d%02d\n",
                                       getenv("HOME"),
                                       1900 + uttime->tm_year,
                                       1 + uttime->tm_mon,
                                       uttime->tm_mday);
            }
            else
            {
                fprintf(fp,
                        "%04d/%02d/%02d %02d:%02d:%02d.%09ld %10s "
                        "%6ld %s\n",
                        1900 + uttime->tm_year,
                        1 + uttime->tm_mon,
                        uttime->tm_mday,
                        uttime->tm_hour,
                        uttime->tm_min,
                        uttime->tm_sec,
                        thetime->tv_nsec,
                        data.processname,
                        (long) getpid(),
                        data.CLIcmdline);
                fclose(fp);
            }
        }

        //
        data.cmdNBarg = 0;


        if(dcdebug > 0)
        {
        }

        // extract first word
        // Replaced internal tokenization with POSIX wordexp to handle nested quotes safely
        
        cli_export_vars_to_env(); // export variables prior to wordexp evaluation
        
        if (cli_check_unquoted_restricted_symbols(data.CLIcmdline) != 0)
        {
            printf(
                "\n%c[%d;%dm ERROR %c[%d;m Syntax error: flow process symbols must be quoted\n",
                (char) 27, 1, 31, (char) 27, 0);
            data.CMDexecuted = 1; // Prevent fallback to shell
            data.parseerror = 1;
            return RETURN_FAILURE;
        }

        wordexp_t p;
        int we_ret = wordexp(data.CLIcmdline, &p, WRDE_SHOWERR | WRDE_UNDEF);
        if(we_ret == 0)
        {
            for(size_t i = 0; i < p.we_wordc; i++)
            {
                if (data.cmdNBarg >= NB_ARG_MAX - 1) break;
                
                char *cmdargstring = p.we_wordv[i];
                
                if(data.cmdNBarg > 0
                   && data.cmdargtoken[0].type
                      == CMDARGTOKEN_TYPE_COMMAND
                   && (cmdargstring[0] == '-'
                       || cmdargstring[0] == '/'))
                {
                    strncpy(
                        data.cmdargtoken[data.cmdNBarg]
                            .val.string,
                        cmdargstring,
                        STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                    data.cmdargtoken[data.cmdNBarg]
                        .val.string[
                            STRINGMAXLEN_CMDARGTOKEN_VAL
                            - 1] = '\0';
                    data.cmdargtoken[data.cmdNBarg]
                        .type = CMDARGTOKEN_TYPE_RAWSTRING;
                }
                else
                {
                    strncpy(
                        data.cmdargtoken[data.cmdNBarg].val.string,
                        cmdargstring,
                        STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                    data.cmdargtoken[data.cmdNBarg]
                        .val.string[STRINGMAXLEN_CMDARGTOKEN_VAL - 1] = '\0';

                    snprintf(str, strmaxlen,
                             "%s\n", cmdargstring);
                    cli_parse(str);
                }
                data.cmdNBarg++;
            }
            wordfree(&p);
        }
        else
        {
            // Fallback if wordexp fails (e.g. WRDE_SYNTAX due to unmatched quotes)
            // It will trigger CMDARGTOKEN_TYPE_UNSOLVED which then correctly routes to bash transparently!
            strncpy(data.cmdargtoken[0].val.string, data.CLIcmdline, STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
            data.cmdargtoken[0].val.string[STRINGMAXLEN_CMDARGTOKEN_VAL - 1] = '\0';
            data.cmdargtoken[0].type = CMDARGTOKEN_TYPE_RAWSTRING;
            data.cmdNBarg = 1;
        }

        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_UNSOLVED;


        if(dcdebug > 0)
        {
            printf("DEBUG: %s %d: data.cmdNBarg = %ld\n", __func__, __LINE__,
                   data.cmdNBarg);
        }

        if(dcdebug > 1)
        {
            long i = 0;

            if(dcdebug > 0)
            {
                printf("DEBUG: %s %d: TOKEN %ld type : %d\n",
                       __func__, __LINE__,
                       i,
                       data.cmdargtoken[i].type);
            }

            while(data.cmdargtoken[i].type != 0)
            {

                printf("DEBUG: %s %d: TOKEN %ld/%ld   \"%s\"  type : %d\n",
                       __func__, __LINE__,
                       i,
                       data.cmdNBarg,
                       data.cmdargtoken[i].val.string,
                       data.cmdargtoken[i].type);
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_FLOAT) // double
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_FLOAT           : "
                        "%g\n",
                        data.cmdargtoken[i].val.numf);
                }
                if(data.cmdargtoken[i].type == CMDARGTOKEN_TYPE_LONG)  // long
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_LONG           : "
                        "%ld\n",
                        data.cmdargtoken[i].val.numl);
                }
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_STRING) // new variable/image
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_STRING        : "
                        "%s\n",
                        data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_EXISTINGIMAGE) // existing image
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_EXISTINGIMAGE : "
                        "%s\n",
                        data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_COMMAND) // command
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_COMMAND       : "
                        "%s\n",
                        data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_RAWSTRING) // unprocessed string
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_RAWSTRING    : "
                        "%s\n",
                        data.cmdargtoken[i].val.string);
                }

                i++;
            }
        }

        if(dcdebug > 0)
        {
            printf("DEBUG: %s %d: data.parseerror = %d\n",
                   __func__, __LINE__,
                   data.parseerror);
        }

        if(data.parseerror == 0)
        {
            if(data.cmdargtoken[0].type == CMDARGTOKEN_TYPE_COMMAND)
            {
                // Execute CLI command
                data.cmd[data.cmdindex]
                    .callcount++;

                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);

                data.CMDerrstatus =
                    data.cmd[data.cmdindex].fp();

                if(data.print_cmd_timing)
                {
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + 
                                        (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
                    printf("Execution time: %.3f ms\n", elapsed_ms);
                }

                cli_save_last_argument();

                if(data.CMDerrstatus != RETURN_SUCCESS)
                {
                    // CLI function returns error
                    // print function key name and error code
                    printf(
                        "\n%c[%d;%dm ERROR %c[%d;m CLI "
                        "function %s returns %d\n",
                        (char) 27,
                        1,
                        31,
                        (char) 27,
                        0,
                        data.cmd[data.cmdindex].key,
                        data.CMDerrstatus);

                    if(dcerrorexit == 1)
                    {
                        printf(
                            "%c[%d;%dm -> EXIT CLI "
                            "%c[%d;m\n",
                            (char) 27,
                            1,
                            31,
                            (char) 27,
                            0);
                        dcexitcode = data.CMDerrstatus;

#ifndef NDEBUG
                        // output trace debug
                        write_tracedebugfile();
#endif
                    }
                }

                data.CMDexecuted = 1;
            }
        }
        else
        {
            if(dcerrorexit == 1)
            {
                dcexitcode = 1;
            }
        }

        for(int i = 0; i < data.calctmp_imindex; i++)
        {
            CREATE_IMAGENAME(calctmpimname, "_tmpcalc%d", i);
            if(image_ID(calctmpimname, dcimg, dcnimg) != -1)
            {
                if(dcdebug == 1)
                {
                    printf("Deleting %s\n", calctmpimname);
                }
                delete_image_ID(calctmpimname, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }

        if(!((data.cmdargtoken[0].type == CMDARGTOKEN_TYPE_STRING) ||
                (data.cmdargtoken[0].type == CMDARGTOKEN_TYPE_RAWSTRING) ||
                (data.cmdargtoken[0].type == CMDARGTOKEN_TYPE_UNSOLVED)))
        {
            data.CMDexecuted = 1;
        }
    }

    if((data.CMDexecuted == 0) && (data.CLIloopON == 1))
    {
        /* Attempt transparent OS shell fallback.
         * cli_run_external() uses posix_spawnp()
         * for simple commands (no shell
         * metacharacters) and /bin/sh -c only
         * when required, avoiding the extra shell
         * layer that system() always spawns. */
        cli_export_vars_to_env();
        int sys_ret =
            cli_run_external(data.CLIcmdline);
        int os_not_found = (sys_ret == 127);

        if(!os_not_found)
        {
            /* OS processed it — print shell tag */
            printf(COLORDIMYELLOW
                   "[shell] %s" COLORRST "\n",
                   data.CLIcmdline);
            if(sys_ret != -1)
            {
                cli_last_retval = sys_ret;
            }
        }

        if(os_not_found)
        {
#ifdef USE_READLINE
            if(data.cmdNBarg > 0 && strlen(data.cmdargtoken[0].val.string) > 0)
            {
                const char *input_cmd = data.cmdargtoken[0].val.string;
                
                struct MatchNode {
                    int dist;
                    const char *cmd;
                } matches[3] = { {9999, NULL}, {9999, NULL}, {9999, NULL} };

                for(unsigned int i = 0; i < data.NBcmd; i++) {
                    int d = levenshtein_distance((const char*)input_cmd,
                        (const char*)data.cmd[i].key);
                    
                    if (d < matches[2].dist) {
                        matches[2].dist = d;
                        matches[2].cmd = data.cmd[i].key;
                        
                        if (matches[2].dist < matches[1].dist) {
                            struct MatchNode tmp = matches[1];
                            matches[1] = matches[2];
                            matches[2] = tmp;
                        }
                        if (matches[1].dist < matches[0].dist) {
                            struct MatchNode tmp = matches[0];
                            matches[0] = matches[1];
                            matches[1] = tmp;
                        }
                    }
                }

                if(matches[0].dist <= 4 && matches[0].cmd != NULL) {
                    printf(COLORRED "Command '%s' not found. " COLORRESET
                           "Did you mean:\n", input_cmd);
                    for (int m = 0; m < 3; m++) {
                        if (matches[m].cmd && matches[m].dist <= 4 && matches[m].dist < 9999) {
                            printf("  - " COLORHBOLDCYAN "%s" COLORRESET "\n", matches[m].cmd);
                        }
                    }
                } else {
                    printf(COLORRED "Command not found, or command with no effect\n" COLORRESET);
                }
            }
            else
#endif
            {
                printf(COLORRED
                       "Command not found, or command with no effect\n" COLORRESET);
            }
        }
    }

    /* Restore stdout if pipe was active */
    if(pipe_fp != NULL)
    {
        fflush(stdout);
        dup2(saved_stdout_fd, STDOUT_FILENO);
        close(saved_stdout_fd);
        pclose(pipe_fp);
    }
    /* Restore stdout if redirect was active */
    if(redir_fp != NULL)
    {
        fflush(stdout);
        dup2(saved_stdout_redir,
             STDOUT_FILENO);
        close(saved_stdout_redir);
        fclose(redir_fp);
    }

    free(thetime);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

