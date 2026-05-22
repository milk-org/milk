/**
 * @file CLIcore_UI_execute_debug.c
 *
 * @brief CLI execution debug utilities and entry points.
 *
 * Contains:
 *  - write_tracedebugfile() — post-mortem ring-buffer dump
 *  - CLI_execute_string()   — re-entrant execute wrapper
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "timeutils.h"


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
    if (fp != NULL)
    {
        for (uint64_t i = 0; i < CODETESTPOINTARRAY_NBCNT; i++)
        {
            long j = (i + dctestptcnt) % CODETESTPOINTARRAY_NBCNT;

            uint64_t index = dctestptarr[j].loopcnt * CODETESTPOINTARRAY_NBCNT + j;

            if (dctestptarr[j].line != 0)
            {
                char timestring[TIMESTRINGLEN];
                mkUTtimestring_nanosec(timestring, dctestptarr[j].time);

                /* Extract last word from path */
                char str[STRINGMAXLEN_FULLFILENAME];
                snprintf(str, sizeof(str), "%s", dctestptarr[j].file);
                char *slash    = strrchr(str, '/');
                char *lastword = (slash != NULL) ? (slash + 1) : str;

                fprintf(fp,
                        "T %6ld %s %-20s"
                        " %6d %-20s  %s\n",
                        index, timestring, lastword, dctestptarr[j].line, dctestptarr[j].func,
                        dctestptarr[j].msg);
                fprintf(fp, "       FTRACE %d ", dctestptarr[j].funclevel);
                for (int level = 0; level < dctestptarr[j].funclevel; level++)
                {
                    fprintf(fp, " (%d) >> %ld:%s", dctestptarr[j].linestack[level],
                            dctestptarr[j].fcntstack[level], dctestptarr[j].funcstack[level]);
                }
                fprintf(fp, "\n\n");
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
    strncpy(save_cmdline, data.CLIcmdline, STRINGMAXLEN_CLICMDLINE - 1);
    save_cmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    strncpy(data.CLIcmdline, cmd, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    errno_t ret                                  = CLI_execute_line();
    strncpy(data.CLIcmdline, save_cmdline, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    return ret;
}
