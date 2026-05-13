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

    /* Lines starting with # are comments —
     * skip before ANY expansion so $, !, @
     * tokens inside comments are never
     * processed. */
    {
        const char *p = data.CLIcmdline;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '#')
        {
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
        }
    }

    /* History expansion (!! and !$) has already been performed
     * by rl_cb_linehandler() before calling CLI_execute_line. */

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
        PRINT_ERROR("+ %s", data.CLIcmdline);
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
     * Must run BEFORE cli_try_var_assign().
     * Skip if this is a known internal keyword
     * or registered command, but NOT on the
     * basis of '=' alone — "a=b+1" is arithmetic. */
    if(data.CLIcmdline[0] != '\0'
       && data.CLIcmdline[0] != '!')
    {
        char firstword[2048];
        if(sscanf(data.CLIcmdline,
                  " %2047s",
                  firstword) == 1
           && !is_internal_cmd(firstword, 0))
        {
            if(cli_calc_eval_line(
                   data.CLIcmdline))
            {
                free(thetime);
                return RETURN_SUCCESS;
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
    /* If the first token is not a known internal
     * milk keyword or command, instantly delegate
     * the full line to the OS shell. */
    if(data.CLIcmdline[0] != '\0'
       && data.CLIcmdline[0] != '!'
       && data.CLIloopON == 1)
    {
        char firstword[2048];
        if(sscanf(data.CLIcmdline,
                  " %2047s",
                  firstword) == 1
           && !is_internal_cmd(firstword, 1))
        {
            if(dcquiet == 0)
            {
                printf(COLORDIMYELLOW
                       "[shell bypass] %s"
                       COLORRST "\n",
                       data.CLIcmdline);
            }
            cli_history_log_shell(
                data.CLIcmdline);
            cli_export_vars_to_env();
            cli_run_external(
                data.CLIcmdline);
            free(thetime);
            return RETURN_SUCCESS;
        }
    }

    /* ---- Pipe to shell ---- */
    FILE *pipe_fp = NULL;
    int   saved_stdout_fd = -1;
    cli_pipe_setup(&pipe_fp, &saved_stdout_fd);

    /* ---- Output redirect to file ---- */
    FILE *redir_fp = NULL;
    int   saved_stdout_redir = -1;
    if(pipe_fp == NULL)
    {
        cli_redir_setup(&redir_fp, &saved_stdout_redir);
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
    if(cli_handle_shell_builtins())
    {
        // already handled
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
                EXECUTE_SYSTEM_COMMAND_NOCHECK("mkdir -p %s/logdir/%04d%02d%02d\n",
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
         * Uses posix_spawnp() for simple commands
         * and /bin/sh -c only when needed. */
        cli_export_vars_to_env();
        int sys_ret =
            cli_run_external(data.CLIcmdline);
        int os_not_found = (sys_ret == 127);

        if(!os_not_found && sys_ret != -1)
        {
            printf(COLORDIMYELLOW
                   "[shell] %s" COLORRST "\n",
                   data.CLIcmdline);
            cli_last_retval = sys_ret;
        }

        if(os_not_found)
        {
            const char *bad_cmd =
                (data.cmdNBarg > 0)
                ? data.cmdargtoken[0].val.string
                : NULL;
            handle_did_you_mean(bad_cmd);
        }
    }

    /* Restore stdout if pipe or redirect was active */
    cli_pipe_teardown(pipe_fp, saved_stdout_fd);
    cli_redir_teardown(redir_fp, saved_stdout_redir);

    free(thetime);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

