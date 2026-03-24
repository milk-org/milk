/**
 * @file CLIcore_UI.c
 *
 * @brief User input (UI) functions
 *
 */

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
#include "CLIcore/cli_calc_parser.h"
#include "CLIcore_script.h"
#include "CLIcore_UI.h"

#include <fnmatch.h>
#include <glob.h>
#include <sys/wait.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

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
 * @brief Dump the circular code-trace buffer to a log
 *        file for post-mortem debugging.
 *
 * Writes every non-empty entry from the dctestptarr[]
 * ring buffer into a timestamped log file named
 * milk-codetracepoint.<PID>.log. Each entry includes
 * the source file, line number, function name, message,
 * and the full function-call stack at that tracepoint.
 *
 * This is called on abnormal exit (signal handler or
 * explicit user request) so the developer can inspect
 * the execution path that led to a crash.
 */
errno_t write_tracedebugfile()
{
    pid_t thisPID = getpid();

    char fname[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fname, "milk-codetracepoint.%05d.log", thisPID);

    printf("Writing output trace to file %s\n", fname);
    printf("dctestptinit = %d\n", dctestptinit);

    FILE *fp = fopen(fname, "w");
    if(fp != NULL)
    {
        for(uint64_t i = 0; i < CODETESTPOINTARRAY_NBCNT; i++)
        {
            long j = (i + dctestptcnt) % CODETESTPOINTARRAY_NBCNT;

            uint64_t index =
                dctestptarr[j].loopcnt * CODETESTPOINTARRAY_NBCNT + j;

            if(dctestptarr[j].line != 0)
            {
                char timestring[TIMESTRINGLEN];
                mkUTtimestring_nanosec(timestring, dctestptarr[j].time);

                // extract last word
                char str[STRINGMAXLEN_FULLFILENAME];
                strcpy(str, dctestptarr[j].file);
                char *lastword = strrchr(str, '/') + 1;

                fprintf(fp,
                        "T %6ld %s %-20s %6d %-20s  %s\n",
                        index,
                        timestring,
                        lastword,
                        dctestptarr[j].line,
                        dctestptarr[j].func,
                        dctestptarr[j].msg);
                fprintf(fp,
                        "       FTRACE %d ",
                        dctestptarr[j].funclevel);
                for(int level = 0; level < dctestptarr[j].funclevel;
                        level++)
                {
                    fprintf(fp,
                            " (%d) >> %ld:%s",
                            dctestptarr[j].linestack[level],
                            dctestptarr[j].fcntstack[level],
                            dctestptarr[j].funcstack[level]);
                }
                fprintf(fp, "\n\n");

                //printf("%s\n", p + 1);
            }
        }
        fclose(fp);
    }

    return RETURN_SUCCESS;
}



/**
 * @brief Execute a command string without clobbering
 *        the current command line.
 *
 * Saves the current data.CLIcmdline, replaces it with
 * @cmd, runs CLI_execute_line(), then restores the
 * original. This allows re-entrant command execution
 * (e.g. from within scripts, trap handlers, or
 * on_update callbacks) without losing the outer
 * command context.
 *
 * @param cmd  Null-terminated command string to execute
 * @return RETURN_SUCCESS on success, error code on
 *         failure
 */
errno_t CLI_execute_string(const char *cmd)
{
    char save_cmdline[STRINGMAXLEN_CLICMDLINE];
    strncpy(save_cmdline, data.CLIcmdline, STRINGMAXLEN_CLICMDLINE);
    strncpy(data.CLIcmdline, cmd, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    errno_t ret = CLI_execute_line();
    strncpy(data.CLIcmdline, save_cmdline, STRINGMAXLEN_CLICMDLINE);
    return ret;
}


/**
 * @brief Detect shell meta-characters outside quotes.
 *
 * Scans @cmdline character by character, tracking
 * single-quote, double-quote, and backslash-escape
 * state. Returns 1 if any restricted symbol
 * (; < > | [ ] ( ) & * $) appears outside quotes.
 *
 * The '=' character is only allowed in valid
 * assignment context (after an identifier).
 *
 * This gate decides whether the line must be
 * dispatched to /bin/sh (restricted symbols present)
 * or can be handled by the milk parser directly.
 *
 * @param cmdline  Command line to scan
 * @return 1 if restricted symbols found, 0 if clean
 */
static int cli_check_unquoted_restricted_symbols(
    const char *cmdline
)
{
    int in_squote = 0;
    int in_dquote = 0;
    int esc = 0;
    
    /* '[' and ']' removed to allow stream
     * slicing syntax (e.g. im[0:19,10:29]). */
    const char *restricted = ";<>|()*&$";
    
    int word_start = 1;
    int valid_assign_prefix = 0; 
    
    for (int i = 0; cmdline[i] != '\0'; i++) {
        char c = cmdline[i];
        
        if (esc) {
            esc = 0;
            word_start = 0;
            valid_assign_prefix = 0;
            continue;
        }
        
        if (c == '\\') {
            esc = 1;
            word_start = 0;
            valid_assign_prefix = 0;
            continue;
        }
        
        if (in_squote) {
            if (c == '\'') in_squote = 0;
            continue;
        }
        
        if (in_dquote) {
            if (c == '"') in_dquote = 0;
            continue;
        }
        
        if (c == '\'') {
            in_squote = 1;
            word_start = 0;
            valid_assign_prefix = 0;
            continue;
        }
        
        if (c == '"') {
            in_dquote = 1;
            word_start = 0;
            valid_assign_prefix = 0;
            continue;
        }
        
        if (isspace(c)) {
            word_start = 1;
            valid_assign_prefix = 0;
            continue;
        }
        
        if (strchr(restricted, c) != NULL) {
            return 1;
        }
        
        if (c == '=') {
            if (!valid_assign_prefix) {
                return 1;
            }
            valid_assign_prefix = 0;
            continue;
        }
        
        if (word_start) {
            if (isalpha(c) || c == '_') {
                valid_assign_prefix = 1;
            } else {
                valid_assign_prefix = 0;
            }
            word_start = 0;
        } else {
            if (!isalnum(c) && c != '_') {
                valid_assign_prefix = 0;
            }
        }
    }
    
    return 0;
}

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

    /* Expand aliases before anything else */
    cli_alias_expand();

    /* Logical operators: && and ||
     * Split line at top-level && / || and
     * execute segments conditionally.
     * Skip && / || inside quotes or $(). */
    {
        const char *src = data.CLIcmdline;
        int depth = 0;   /* () nesting */
        int in_sq = 0;   /* single quote */
        int in_dq = 0;   /* double quote */
        int found = 0;
        int split_pos = -1;
        int op_len = 0;  /* 2 for && or || */
        int op_is_and = 0;

        for(int si = 0; src[si] != '\0'; si++)
        {
            char c = src[si];
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
                else if(depth == 0
                        && c == '&'
                        && src[si + 1] == '&')
                {
                    found = 1;
                    split_pos = si;
                    op_len = 2;
                    op_is_and = 1;
                    break;
                }
                else if(depth == 0
                        && c == '|'
                        && src[si + 1] == '|')
                {
                    found = 1;
                    split_pos = si;
                    op_len = 2;
                    op_is_and = 0;
                    break;
                }
            }
        }
        if(found && split_pos >= 0)
        {
            char right[STRINGMAXLEN_CLICMDLINE];
            const char *rp =
                src + split_pos + op_len;
            while(*rp == ' '
                  || *rp == '\t')
            {
                rp++;
            }
            strncpy(right, rp, STRINGMAXLEN_CLICMDLINE - 1);
            right[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

            char left[STRINGMAXLEN_CLICMDLINE];
            strncpy(left, data.CLIcmdline,
                    (size_t) split_pos);
            left[split_pos] = '\0';
            strncpy(data.CLIcmdline, left,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE
                - 1] = '\0';

            errno_t lret = CLI_execute_line();
            int ok = (cli_last_retval == 0);
            /* Decide whether to run right */
            int run_right =
                (op_is_and && ok)
                || (!op_is_and && !ok);
            if(run_right)
            {
                strncpy(data.CLIcmdline, right,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                data.CLIcmdline[
                    STRINGMAXLEN_CLICMDLINE
                    - 1] = '\0';
                errno_t rret =
                    CLI_execute_line();
                free(thetime);
                DEBUG_TRACE_FEXIT();
                return rret;
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return lret;
        }
    }

    /* Stream pipeline: stream |> cmd args
     * Rewrites to: cmd stream args
     * Must be checked BEFORE the pipe
     * handler below. */
    {
        const char *src = data.CLIcmdline;
        int depth = 0;
        int in_sq = 0;
        int in_dq = 0;
        int gpipe = -1;

        for(int si = 0;
            src[si] != '\0'; si++)
        {
            char c = src[si];
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
                else if(c == ')'
                        && depth > 0)
                {
                    depth--;
                }
                else if(depth == 0
                        && c == '|'
                        && src[si + 1]
                        == '>')
                {
                    gpipe = si;
                    break;
                }
            }
        }
        if(gpipe >= 0)
        {
            char lhs[
                STRINGMAXLEN_CLICMDLINE];
            strncpy(lhs,
                    data.CLIcmdline,
                    (size_t) gpipe);
            lhs[gpipe] = '\0';
            /* Trim trailing spaces */
            {
                int e = gpipe - 1;
                while(e >= 0
                      && (lhs[e] == ' '
                          || lhs[e]
                          == '\t'))
                {
                    lhs[e--] = '\0';
                }
            }
            const char *rhs =
                data.CLIcmdline
                + gpipe + 2;
            while(*rhs == ' '
                  || *rhs == '\t')
            {
                rhs++;
            }
            /* rhs = "cmd args..." */
            const char *sp =
                strchr(rhs, ' ');
            char newcmd[
                STRINGMAXLEN_CLICMDLINE];
            if(sp != NULL)
            {
                snprintf(newcmd,
                         sizeof(newcmd),
                         "%.*s %s %s",
                         (int)(sp - rhs),
                         rhs, lhs,
                         sp + 1);
            }
            else
            {
                snprintf(newcmd,
                         sizeof(newcmd),
                         "%s %s",
                         rhs, lhs);
            }
            strncpy(
                data.CLIcmdline,
                newcmd,
                STRINGMAXLEN_CLICMDLINE
                - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE
                - 1] = '\0';
            errno_t lret =
                CLI_execute_line();
            data.CMDexecuted = 1;
            DEBUG_TRACE_FEXIT();
            return lret;
        }
    }

    /* Pipe: cmd1 | cmd2
     * Capture stdout of left command into a
     * temp file, then feed it as stdin to
     * the right command. Only matches single
     * '|', not '||'. */
    {
        const char *src = data.CLIcmdline;
        int depth = 0;
        int in_sq = 0;
        int in_dq = 0;
        int pipe_pos = -1;

        for(int si = 0; src[si] != '\0'; si++)
        {
            char c = src[si];
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
                else if(c == ')'
                        && depth > 0)
                {
                    depth--;
                }
                else if(depth == 0
                        && c == '|'
                        && src[si + 1]
                        != '|'
                        && src[si + 1]
                        != '>')
                {
                    pipe_pos = si;
                    break;
                }
            }
        }
        if(pipe_pos >= 0)
        {
            /* Split at pipe */
            char left[STRINGMAXLEN_CLICMDLINE];
            strncpy(left, data.CLIcmdline,
                    (size_t) pipe_pos);
            left[pipe_pos] = '\0';
            const char *rp =
                data.CLIcmdline
                + pipe_pos + 1;
            while(*rp == ' ' || *rp == '\t')
            {
                rp++;
            }
            char right[STRINGMAXLEN_CLICMDLINE];
            strncpy(right, rp,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
            right[STRINGMAXLEN_CLICMDLINE
                  - 1] = '\0';

            /* Check whether the right side is
             * a registered milk command.  If
             * not, fall through so the popen-
             * based handler can pipe to a shell
             * command (e.g. grep, sort, wc). */
            {
                char rword[200];
                int  rw = 0;
                const char *rr = right;
                while(*rr == ' '
                      || *rr == '\t')
                {
                    rr++;
                }
                while(*rr != '\0'
                      && *rr != ' '
                      && *rr != '\t'
                      && rw < (int)
                         sizeof(rword) - 1)
                {
                    rword[rw++] = *rr++;
                }
                rword[rw] = '\0';
                if(!cli_is_command(rword))
                {
                    goto pipe_fallthrough;
                }
            }

            /* Capture left stdout */
            FILE *tmpfp = tmpfile();
            if(tmpfp != NULL)
            {
                fflush(stdout);
                int saved_stdout =
                    dup(STDOUT_FILENO);
                dup2(fileno(tmpfp),
                     STDOUT_FILENO);

                strncpy(data.CLIcmdline,
                        left,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                CLI_execute_line();

                fflush(stdout);
                dup2(saved_stdout,
                     STDOUT_FILENO);
                close(saved_stdout);

                /* Feed to right stdin */
                rewind(tmpfp);
                int saved_stdin =
                    dup(STDIN_FILENO);
                dup2(fileno(tmpfp),
                     STDIN_FILENO);

                strncpy(data.CLIcmdline,
                        right,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                errno_t pret =
                    CLI_execute_line();

                dup2(saved_stdin,
                     STDIN_FILENO);
                close(saved_stdin);
                fclose(tmpfp);

                free(thetime);
                DEBUG_TRACE_FEXIT();
                return pret;
            }
        }
    }
pipe_fallthrough:
    (void) 0;

    /* Dot-sourcing: ". file" → "source file"
     * Must check before script intercept */
    {
        const char *p = data.CLIcmdline;
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        if(p[0] == '.' && p[1] == ' ')
        {
            char tmp[STRINGMAXLEN_CLICMDLINE];
            snprintf(tmp,
                     STRINGMAXLEN_CLICMDLINE,
                     "source %s", p + 2);
            strncpy(data.CLIcmdline, tmp,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE
                - 1] = '\0';
        }
    }

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

    /* Output redirection: cmd > file,
     * cmd >> file. Scan from end for
     * unquoted > or >> and redirect. */
    {
        int redir_mode = 0; /* 1=trunc 2=app */
        int redir_pos = -1;
        int in_sq2 = 0;
        int in_dq2 = 0;
        int depth2 = 0;
        const char *cl2 = data.CLIcmdline;
        for(int ri = 0;
            cl2[ri] != '\0'; ri++)
        {
            if(cl2[ri] == '\''
               && !in_dq2)
            {
                in_sq2 = !in_sq2;
            }
            else if(cl2[ri] == '"'
                    && !in_sq2)
            {
                in_dq2 = !in_dq2;
            }
            else if(!in_sq2
                    && !in_dq2)
            {
                if(cl2[ri] == '(')
                {
                    depth2++;
                }
                else if(cl2[ri] == ')'
                        && depth2 > 0)
                {
                    depth2--;
                }
                else if(depth2 == 0
                        && cl2[ri] == '>')
                {
                    if(cl2[ri + 1] == '>')
                    {
                        redir_mode = 2;
                        redir_pos = ri;
                    }
                    else
                    {
                        redir_mode = 1;
                        redir_pos = ri;
                    }
                }
            }
        }
        if(redir_pos >= 0)
        {
            /* Extract filename */
            int fstart = redir_pos
                         + ((redir_mode
                             == 2) ? 2 : 1);
            while(data.CLIcmdline[fstart]
                  == ' '
                  || data.CLIcmdline[fstart]
                  == '\t')
            {
                fstart++;
            }
            char rfile[512];
            int fi = 0;
            while(data.CLIcmdline[fstart]
                  != '\0'
                  && data.CLIcmdline[fstart]
                  != ' '
                  && data.CLIcmdline[fstart]
                  != '\t'
                  && fi < 511)
            {
                rfile[fi++] =
                    data.CLIcmdline[
                        fstart++];
            }
            rfile[fi] = '\0';

            /* Truncate cmd at redir */
            data.CLIcmdline[
                redir_pos] = '\0';
            {
                int cl3 = redir_pos - 1;
                while(cl3 >= 0
                      && (data.CLIcmdline[
                              cl3] == ' '
                          || data.CLIcmdline[
                              cl3]
                          == '\t'))
                {
                    data.CLIcmdline[
                        cl3--] = '\0';
                }
            }

            /* Open file and redirect */
            FILE *rfp = fopen(
                rfile,
                (redir_mode == 2)
                ? "a" : "w");
            if(rfp != NULL)
            {
                fflush(stdout);
                int sv_out =
                    dup(STDOUT_FILENO);
                dup2(fileno(rfp),
                     STDOUT_FILENO);

                errno_t rret =
                    CLI_execute_line();

                fflush(stdout);
                dup2(sv_out,
                     STDOUT_FILENO);
                close(sv_out);
                fclose(rfp);

                free(thetime);
                DEBUG_TRACE_FEXIT();
                return rret;
            }
        }
    }

    /* Here-string: cmd <<< "string"
     * Provides string as stdin to cmd */
    {
        char *hs = strstr(
            data.CLIcmdline, "<<<");
        if(hs != NULL)
        {
            /* Split at <<< */
            *hs = '\0';
            const char *hsval =
                hs + 3;
            while(*hsval == ' '
                  || *hsval == '\t')
            {
                hsval++;
            }
            /* Strip quotes */
            int hvlen =
                (int) strlen(hsval);
            char hvbuf[STRINGMAXLEN_CLICMDLINE];
            if(hvlen >= 2
               && ((hsval[0] == '"'
                    && hsval[
                        hvlen - 1]
                    == '"')
                   || (hsval[0] == '\''
                       && hsval[
                           hvlen - 1]
                       == '\'')))
            {
                memcpy(hvbuf,
                       hsval + 1,
                       (size_t)
                       (hvlen - 2));
                hvbuf[hvlen - 2] = '\0';
            }
            else
            {
                strncpy(
                    hvbuf, hsval,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
                hvbuf[
                    STRINGMAXLEN_CLICMDLINE
                    - 1] = '\0';
            }

            /* Write to tmpfile, feed
             * as stdin */
            FILE *hsfp = tmpfile();
            if(hsfp != NULL)
            {
                fprintf(hsfp, "%s\n",
                        hvbuf);
                rewind(hsfp);
                int sv_in =
                    dup(STDIN_FILENO);
                dup2(fileno(hsfp),
                     STDIN_FILENO);

                errno_t hsret =
                    CLI_execute_line();

                dup2(sv_in,
                     STDIN_FILENO);
                close(sv_in);
                fclose(hsfp);

                free(thetime);
                DEBUG_TRACE_FEXIT();
                return hsret;
            }
        }
    }

    /* Background: cmd & */
    {
        int ll =
            (int) strlen(
                data.CLIcmdline);
        /* Scan backward for & */
        int bi = ll - 1;
        while(bi >= 0
              && (data.CLIcmdline[bi]
                  == ' '
                  || data.CLIcmdline[
                      bi]
                  == '\t'))
        {
            bi--;
        }
        if(bi >= 0
           && data.CLIcmdline[bi]
           == '&'
           && (bi == 0
               || data.CLIcmdline[
                      bi - 1]
               != '&'))
        {
            /* Strip trailing & */
            data.CLIcmdline[bi] =
                '\0';
            pid_t cpid = fork();
            if(cpid == 0)
            {
                /* Child */
                CLI_execute_string(
                    data.CLIcmdline);
                _exit(0);
            }
            else if(cpid > 0)
            {
                printf("[bg] %d\n",
                       (int) cpid);
                char pb[32];
                snprintf(
                    pb, sizeof(pb),
                    "%d",
                    (int) cpid);
                cli_var_set("!", pb);
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return 0;
        }
    }

    /* Subshell: (cmd1; cmd2) */
    {
        const char *sp =
            data.CLIcmdline;
        int spl =
            (int) strlen(sp);
        if(spl >= 3
           && sp[0] == '('
           && sp[spl - 1] == ')')
        {
            char sbuf[
                STRINGMAXLEN_CLICMDLINE
            ];
            memcpy(sbuf, sp + 1,
                   (size_t)(spl - 2));
            sbuf[spl - 2] = '\0';
            pid_t spid = fork();
            if(spid == 0)
            {
                /* Execute in child */
                /* Split on ; */
                char *tok =
                    strtok(sbuf, ";");
                while(tok != NULL)
                {
                    const char *st =
                        tok;
                    while(*st == ' '
                          || *st
                          == '\t')
                    {
                        st++;
                    }
                    if(*st != '\0')
                    {
                        CLI_execute_string(
                            (char *) st
                        );
                    }
                    tok =
                        strtok(
                            NULL,
                            ";");
                }
                _exit(0);
            }
            else if(spid > 0)
            {
                int wst;
                waitpid(spid,
                        &wst, 0);
                cli_last_retval =
                    WEXITSTATUS(wst);
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return 0;
        }
    }

    /* Here-string: cmd <<< "text" */
    {
        const char *hs =
            strstr(data.CLIcmdline,
                   "<<<");
        if(hs != NULL)
        {
            /* Extract command before
             * <<< */
            char hcmd[
                STRINGMAXLEN_CLICMDLINE
            ];
            int hcl =
                (int)(hs
                      - data.CLIcmdline);
            if(hcl
               >= STRINGMAXLEN_CLICMDLINE)
            {
                hcl =
                    STRINGMAXLEN_CLICMDLINE
                    - 1;
            }
            memcpy(hcmd,
                   data.CLIcmdline,
                   (size_t) hcl);
            hcmd[hcl] = '\0';
            /* Trim trailing ws */
            while(hcl > 0
                  && (hcmd[hcl - 1]
                      == ' '
                      || hcmd[hcl - 1]
                      == '\t'))
            {
                hcmd[--hcl] = '\0';
            }
            /* Get the text */
            const char *tp = hs + 3;
            while(*tp == ' '
                  || *tp == '\t')
            {
                tp++;
            }
            /* Strip quotes */
            char htxt[1024];
            strncpy(htxt, tp,
                    sizeof(htxt) - 1);
            htxt[sizeof(htxt) - 1] =
                '\0';
            int htl =
                (int) strlen(htxt);
            if(htl >= 2
               && ((htxt[0] == '"'
                    && htxt[htl - 1]
                    == '"')
                   || (htxt[0] == '\''
                       && htxt[
                           htl - 1]
                       == '\'')))
            {
                htxt[htl - 1] = '\0';
                memmove(htxt,
                        htxt + 1,
                        (size_t)
                        (htl - 1));
            }
            /* Create pipe */
            int pfd[2];
            if(pipe(pfd) == 0)
            {
                /* Write text to pipe */
                ssize_t wr_ignore;
                wr_ignore =
                write(pfd[1], htxt,
                      strlen(htxt));
                wr_ignore =
                write(pfd[1], "\n", 1);
                (void) wr_ignore;
                close(pfd[1]);
                /* Redirect stdin */
                int sv =
                    dup(STDIN_FILENO);
                dup2(pfd[0],
                     STDIN_FILENO);
                close(pfd[0]);
                CLI_execute_string(hcmd);
                dup2(sv,
                     STDIN_FILENO);
                close(sv);
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return 0;
        }
    }

    /* Stderr redirect: 2>&1, 2>/dev/null,
     * 2>file */
    {
        const char *se =
            strstr(data.CLIcmdline,
                   "2>");
        if(se != NULL)
        {
            /* Extract cmd before 2> */
            char scmd[
                STRINGMAXLEN_CLICMDLINE
            ];
            int scl =
                (int)(se
                      - data.CLIcmdline);
            if(scl
               >= STRINGMAXLEN_CLICMDLINE)
            {
                scl =
                    STRINGMAXLEN_CLICMDLINE
                    - 1;
            }
            memcpy(scmd,
                   data.CLIcmdline,
                   (size_t) scl);
            scmd[scl] = '\0';
            while(scl > 0
                  && (scmd[scl - 1]
                      == ' '
                      || scmd[scl - 1]
                      == '\t'))
            {
                scmd[--scl] = '\0';
            }
            const char *target =
                se + 2;
            while(*target == ' '
                  || *target == '\t')
            {
                target++;
            }
            int sv_err =
                dup(STDERR_FILENO);
            if(strncmp(target, "&1",
                       2) == 0)
            {
                dup2(STDOUT_FILENO,
                     STDERR_FILENO);
            }
            else
            {
                /* Strip trailing ws */
                char fname[256];
                int fi = 0;
                while(target[fi]
                      != '\0'
                      && target[fi]
                      != ' '
                      && target[fi]
                      != '\t'
                      && fi < 254)
                {
                    fname[fi] =
                        target[fi];
                    fi++;
                }
                fname[fi] = '\0';
                FILE *ef =
                    fopen(fname, "w");
                if(ef != NULL)
                {
                    dup2(fileno(ef),
                         STDERR_FILENO);
                    fclose(ef);
                }
            }
            CLI_execute_string(scmd);
            dup2(sv_err,
                 STDERR_FILENO);
            close(sv_err);
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return 0;
        }
    }

    /* Input redirection: cmd < file
     * Scan for unquoted < that is not
     * << or <<<. */
    {
        const char *cl4 =
            data.CLIcmdline;
        int in_sq4 = 0, in_dq4 = 0;
        int depth4 = 0;
        int inr_pos = -1;
        for(int ri = 0;
            cl4[ri] != '\0'; ri++)
        {
            if(cl4[ri] == '\''
               && !in_dq4)
            {
                in_sq4 = !in_sq4;
            }
            else if(cl4[ri] == '"'
                    && !in_sq4)
            {
                in_dq4 = !in_dq4;
            }
            else if(!in_sq4
                    && !in_dq4)
            {
                if(cl4[ri] == '(')
                {
                    depth4++;
                }
                else if(cl4[ri] == ')'
                        && depth4 > 0)
                {
                    depth4--;
                }
                else if(depth4 == 0
                        && cl4[ri] == '<'
                        && cl4[ri + 1]
                        != '<')
                {
                    inr_pos = ri;
                    break;
                }
            }
        }
        if(inr_pos >= 0)
        {
            int fst = inr_pos + 1;
            while(data.CLIcmdline[fst]
                  == ' '
                  || data.CLIcmdline[fst]
                  == '\t')
            {
                fst++;
            }
            char infile[512];
            int ifi = 0;
            while(data.CLIcmdline[fst]
                  != '\0'
                  && data.CLIcmdline[fst]
                  != ' '
                  && data.CLIcmdline[fst]
                  != '\t'
                  && ifi < 511)
            {
                infile[ifi++] =
                    data.CLIcmdline[
                        fst++];
            }
            infile[ifi] = '\0';

            /* Truncate cmd at < */
            data.CLIcmdline[
                inr_pos] = '\0';
            {
                int cl5 = inr_pos - 1;
                while(cl5 >= 0
                      && (data.CLIcmdline[
                              cl5] == ' '
                          || data.CLIcmdline[
                              cl5]
                          == '\t'))
                {
                    data.CLIcmdline[
                        cl5--] = '\0';
                }
            }

            FILE *ifp =
                fopen(infile, "r");
            if(ifp != NULL)
            {
                int sv_in =
                    dup(STDIN_FILENO);
                dup2(fileno(ifp),
                     STDIN_FILENO);

                errno_t iret =
                    CLI_execute_line();

                dup2(sv_in,
                     STDIN_FILENO);
                close(sv_in);
                fclose(ifp);

                free(thetime);
                DEBUG_TRACE_FEXIT();
                return iret;
            }
        }
    }

    /*
     * ---- Command chaining: ; && || ----
     *
     * Scan for the first unquoted chaining
     * operator and split there.
     * Must run BEFORE variable assignment so
     * that "a=1;b=2" is split into two
     * commands rather than assigned as one.
     */
    {
        char fullline[STRINGMAXLEN_CLICMDLINE];
        strncpy(fullline, data.CLIcmdline,
                STRINGMAXLEN_CLICMDLINE - 1);
        fullline[
            STRINGMAXLEN_CLICMDLINE - 1]
            = '\0';

        /* Find first chaining operator */
        int  chain_type = 0;
        /* 1=;  2=&&  3=||  */
        int  chain_off = -1;
        int  chain_len = 0;
        for(int ci = 0; fullline[ci] != '\0';
                ci++)
        {
            /* Skip quoted strings */
            if(fullline[ci] == '"')
            {
                ci++;
                while(fullline[ci] != '\0'
                        && fullline[ci] != '"')
                {
                    ci++;
                }
                if(fullline[ci] == '\0')
                {
                    break;
                }
                continue;
            }
            if(fullline[ci] == ';')
            {
                chain_type = 1;
                chain_off = ci;
                chain_len = 1;
                break;
            }
            if(fullline[ci] == '&'
                    && fullline[ci + 1] == '&')
            {
                chain_type = 2;
                chain_off = ci;
                chain_len = 2;
                break;
            }
            if(fullline[ci] == '|'
                    && fullline[ci + 1] == '|')
            {
                chain_type = 3;
                chain_off = ci;
                chain_len = 2;
                break;
            }
        }

        if(chain_off >= 0)
        {
            /* Extract first part */
            fullline[chain_off] = '\0';
            strncpy(data.CLIcmdline, fullline,
                    STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE - 1]
                = '\0';

            /* Execute first part */
            errno_t ret1 = CLI_execute_line();

            /* Determine whether to run rest */
            int run_rest = 0;
            if(chain_type == 1)
            {
                run_rest = 1; /* ; always */
            }
            else if(chain_type == 2)
            {
                /* && only on success */
                run_rest =
                    (ret1 == RETURN_SUCCESS)
                    ? 1 : 0;
            }
            else if(chain_type == 3)
            {
                /* || only on failure */
                run_rest =
                    (ret1 != RETURN_SUCCESS)
                    ? 1 : 0;
            }

            if(run_rest)
            {
                const char *rest =
                    fullline + chain_off
                    + chain_len;
                while(*rest == ' '
                        || *rest == '\t')
                {
                    rest++;
                }
                if(*rest != '\0')
                {
                    strncpy(data.CLIcmdline,
                            rest,
                            STRINGMAXLEN_CLICMDLINE
                            - 1);
                    data.CLIcmdline[
                        STRINGMAXLEN_CLICMDLINE
                        - 1] = '\0';
                    CLI_execute_line();
                }
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
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
                system(data.CLIcmdline);
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
    // If line starts with !, use system()
    //
    if(data.CLIcmdline[0] == '!')
    {
        data.CLIcmdline[0] = ' ';
        printf(COLORDIMYELLOW
               "[shell] %s" COLORRST "\n",
               data.CLIcmdline);
        cli_export_vars_to_env();
        if(system(data.CLIcmdline) != 0)
        {
            PRINT_ERROR("system call error");
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
                    "on_update ", 10) == 0)
    {
        /* on_update <stream> { cmd }
         * Wait for stream semaphore,
         * then execute cmd. */
        const char *arg =
            data.CLIcmdline + 10;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
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
        /* on_fpschange fpsname.param { cmd }
         * Poll FPS parameter, execute body
         * when it changes. */
        const char *arg =
            data.CLIcmdline + 13;
        while(*arg == ' '
              || *arg == '\t')
        {
            arg++;
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
                    char prev[256];
                    functionparameter_GetParamValueString(
                        &fps.parray[pidx],
                        prev,
                        sizeof(prev));
                    /* Poll until change */
                    char cur[256];
                    for(;;)
                    {
                        usleep(100000);
                        functionparameter_GetParamValueString(
                            &fps.parray[pidx],
                            cur,
                            sizeof(cur));
                        if(strcmp(cur,
                                 prev)
                           != 0)
                        {
                            break;
                        }
                    }
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
        /* Attempt transparent OS shell fallback */
        cli_export_vars_to_env();
        int sys_ret = system(data.CLIcmdline);
        int os_not_found = 0;
        
        /* system() returns 127 << 8 if command not found by sh */
        if(sys_ret != -1 && ((sys_ret >> 8) & 0xff) == 127)
        {
            os_not_found = 1;
        }
        else
        {
            /* OS processed it — print shell tag */
            printf(COLORDIMYELLOW
                   "[shell] %s" COLORRST "\n",
                   data.CLIcmdline);
            if(sys_ret != -1)
            {
                cli_last_retval = (sys_ret >> 8) & 0xff;
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

