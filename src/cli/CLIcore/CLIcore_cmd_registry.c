// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_cmd_registry.c
 *
 * @brief Core CLI command registration table.
 *
 * Contains runCLI_cmd_init() and its static helper
 * callbacks. This function registers every built-in
 * command available in the milk interactive shell
 * using RegisterCLIcommand().
 *
 * Extracted from CLIcore.c to keep that file focused
 * on startup, REPL, and option processing.
 *
 * The helpers are declared static — they are only
 * referenced via function pointers passed to
 * RegisterCLIcommand(), so no header declarations
 * are needed.
 *
 * Internal entry point defined in this file and
 * forward-referenced from CLIcore.c:
 *   runCLI_cmd_init()
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "milk_session.h"
#include "CLIcore_script.h"
#include "CLIcore_help.h"
#include "CLIcore_modules.h"
#include "CLIcore_UI_execute.h"

/* Forward references to functions in CLIcore.c
 * that are registered as CLI command callbacks.
 * These are not in a public header because they
 * are only called via the command registry. */
extern errno_t function_parameter_structure_load__cli(void);
extern errno_t cli_fifo(void);

/* ============================================================
 *  Help topic callback wrappers
 * ============================================================
 */
/**
 * print_session_name - Print the CLI session name.
 *
 * Prints the process name set by the -n option.
 * If no name was given, prints "(none)".
 */
static errno_t print_session_name()
{
    printf("%s\n", data.processname);
    return RETURN_SUCCESS;
}

/**
 * runCLI_cmd_init - Register all built-in CLI core and UI commands
 *
 * Populates the root and `milk.*` command namespaces with built-in interactive
 * commands (session name, FPS loader, FIFO control, command timing, usage
 * statistics, syntax highlighting/diagnostics, auto-indentation, delimiter
 * matching, history utilities, and the interactive FPS parameter editor), and
 * binds the help TUI hook.
 *
 * Return: void
 */
void runCLI_cmd_init()
{
    // ensure that commands below belong to root/MAIN module
    data.moduleindex   = -1;
    milk_help_tui_hook = cli_fhelp; /* @help --tui */

    milk_module_begin(); /* milk.<name>: engine utilities of the milk-cli front-end */
    RegisterCLIcommand("name", __FILE__, print_session_name,
                       "print CLI session name (set by milk -n)", "no argument", "milk.name",
                       "print_session_name()");
    RegisterCLIcommand("fpsload", __FILE__, function_parameter_structure_load__cli,
                       "Load function parameter struct (FPS)", "<fpsname>",
                       "milk.fpsload imanalyze",
                       "long function_parameter_structure_load(char *fpsname)");
    RegisterCLIcommand("fifo", __FILE__, cli_fifo, "manage command FIFO input",
                       "[create|open|close|on|off"
                       "|status] [path]",
                       "milk.fifo create /tmp/myfifo", "cli_fifo()");
    RegisterCLIcommand("timing", __FILE__, cli_timing_toggle,
                       "toggle display of command execution timing", "[on|off]", "milk.timing on",
                       "cli_timing_toggle()");
    milk_module_end();

    RegisterCLIcommand("cmdstats", __FILE__, cli_cmdstats, "show command usage statistics",
                       "no argument", "cmdstats", "cli_cmdstats()");

#ifdef USE_READLINE
    RegisterCLIcommand("synhl", __FILE__, cli_syntax_highlight_toggle, "toggle syntax highlighting",
                       "[on|off]", "synhl off", "cli_syntax_highlight_toggle()");

    RegisterCLIcommand("syndiag", __FILE__, cli_syntax_diag_toggle,
                       "toggle real-time syntax diagnostics", "[on|off]", "syndiag off",
                       "cli_syntax_diag_toggle()");

    RegisterCLIcommand("cliindent", __FILE__, cli_auto_indent_toggle,
                       "toggle multi-line auto-indentation", "[on|off|<spaces>]", "cliindent 4",
                       "cli_auto_indent_toggle()");

    RegisterCLIcommand("showmatch", __FILE__, cli_showmatch_toggle,
                       "toggle structural delimiter and keyword matching", "[on|off]",
                       "showmatch off", "cli_showmatch_toggle()");
#endif

    RegisterCLIcommand("savehistory", __FILE__, cli_savehistory, "save command history to a file",
                       "<filename>", "savehistory cmds.milk", "cli_savehistory()");

    RegisterCLIcommand("setprompt", __FILE__, cli_setprompt, "set custom prompt format",
                       "[<format>]", "setprompt \"%u@%h %d > \"", "cli_setprompt()");

    RegisterCLIcommand("sessionlog", __FILE__, cli_sessionlog, "enable session command logging",
                       "[on|off|<filename>]", "sessionlog on", "cli_sessionlog()");

    RegisterCLIcommand("history", __FILE__, cli_history_show, "show recent command history",
                       "[<N>]", "history 50", "cli_history_show()");

    RegisterCLIcommand("searchhist", __FILE__, cli_searchhist, "search history for pattern",
                       "<pattern>", "searchhist listim", "cli_searchhist()");

    RegisterCLIcommand("fhist", __FILE__, cli_fhist, "interactive fuzzy search history", "",
                       "fhist", "cli_fhist()");

    RegisterCLIcommand("ghistory", __FILE__, cli_ghistory, "global history (all sessions)",
                       "[N] [-s <session_id>]", "ghistory 50", "cli_ghistory()");

    RegisterCLIcommand("lhistory", __FILE__, cli_lhistory, "local history (current session)", "[N]",
                       "lhistory", "cli_lhistory()");

    RegisterCLIcommand("fparam", __FILE__, cli_fparam, "interactive FPS parameter editor",
                       "<fpsname>", "fparam cnt2push", "cli_fparam()");

    //  init_modules();

    if (dcquiet == 0)
    {
        printf("        Loaded %ld modules, %u commands\n", data.NBmodule, data.NBcmd);
        printf("        \n");
    }
}
