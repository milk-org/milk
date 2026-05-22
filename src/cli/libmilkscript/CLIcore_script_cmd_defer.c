/**
 * @file CLIcore_script_cmd_defer.c
 *
 * @brief Defer command and LIFO cleanup stack.
 *
 * Implements the `defer` builtin, which pushes a
 * command onto a LIFO stack that is executed in
 * reverse order when the script exits (via the
 * EXIT trap mechanism).
 *
 * Public API (declared in CLIcore_script.h):
 *   cli_cmd_defer()   — defer builtin command
 *   cli_defer_run()   — execute deferred commands
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"

/* ============================================================
 *  Defer stack
 * ============================================================
 */

#define CLI_DEFER_MAX 32

static char cli_defer_stack[CLI_DEFER_MAX][STRINGMAXLEN_CLICMDLINE];
static int  cli_defer_count = 0;

/**
 * @brief defer command — register cleanup command
 *
 * Pushes a command onto a LIFO stack that is
 * executed in reverse order when the script
 * exits (integrated with trap EXIT).
 *
 * The command is captured verbatim from
 * data.CLIcmdline after the "defer" keyword,
 * preserving original quoting and spacing.
 *
 * Usage: defer <command ...>
 */
errno_t cli_cmd_defer(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: defer <command>\n");
        return RETURN_FAILURE;
    }

    if (cli_defer_count >= CLI_DEFER_MAX)
    {
        printf("defer: stack full "
               "(max %d)\n",
               CLI_DEFER_MAX);
        return RETURN_FAILURE;
    }

    /* Capture the deferred command from the
     * original command line after "defer" so
     * that quoting/escaping are preserved. */
    char cmd[STRINGMAXLEN_CLICMDLINE];
    cmd[0] = '\0';

    const char *line = data.CLIcmdline;
    const char *p    = line;

    /* Skip leading whitespace */
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }

    /* Expect "defer" as the first token */
    const char keyword[] = "defer";
    size_t     klen      = sizeof(keyword) - 1;

    if (strncmp(p, keyword, klen) == 0)
    {
        p += klen;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
    }

    strncpy(cmd, p, sizeof(cmd) - 1);
    cmd[sizeof(cmd) - 1] = '\0';
    strncpy(cli_defer_stack[cli_defer_count], cmd, STRINGMAXLEN_CLICMDLINE - 1);
    cli_defer_stack[cli_defer_count][STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    cli_defer_count++;

    return RETURN_SUCCESS;
}

/**
 * @brief Execute deferred cleanup commands
 *
 * Called from cli_trap_run_exit() to run all
 * deferred commands in LIFO order. Guards against
 * re-entrance so deferred commands that themselves
 * call defer do not cause infinite recursion.
 *
 * New defers pushed during execution are picked up
 * because the while loop re-reads cli_defer_count
 * each iteration.
 */
void cli_defer_run(void)
{
    static int running = 0;

    if (running)
    {
        return;
    }
    running = 1;

    while (cli_defer_count > 0)
    {
        cli_defer_count--;
        char cmd[STRINGMAXLEN_CLICMDLINE];
        strncpy(cmd, cli_defer_stack[cli_defer_count], STRINGMAXLEN_CLICMDLINE - 1);
        cmd[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        CLI_execute_string(cmd);
    }

    running = 0;
}
