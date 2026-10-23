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
 * using MILK_RUNNABLE descriptors and milk_register_runnable().
 *
 * Extracted from CLIcore.c to keep that file focused
 * on startup, REPL, and option processing.
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
#include "CLIcore_UI_execute.h"
#include "CLIcore_help.h"
#include "CLIcore_modules.h"
#include "CLIcore_script.h"
#include "milk_runnable.h"
#include "milk_session.h"

/* Forward references to functions in CLIcore.c
 * that are registered as CLI command callbacks.
 * These are not in a public header because they
 * are only called via the command registry. */
extern errno_t function_parameter_structure_load__cli(void);
extern errno_t cli_fifo(void);

/**
 * print_session_name - Print the CLI session name.
 *
 * Prints the process name set by the -n option.
 * If no name was given, prints "(none)".
 *
 * Return: RETURN_SUCCESS
 */
static errno_t print_session_name(void)
{
    printf("%s\n", data.processname);
    return RETURN_SUCCESS;
}

#define REGISTRY_CLI_CMD(tag, keystr, descstr, fn)                    \
    MILK_RUNNABLE_RT_DEFINE(tag);                                     \
    static const MILK_RUNNABLE unit_##tag = {                         \
        .kind     = MILK_RUNNABLE_CLI,                                \
        .app      = { .cmdkey = (keystr), .description = (descstr) }, \
        .srcfile  = __FILE__,                                         \
        .cli_func = (fn),                                             \
        .rt       = &tag##_rt,                                        \
    }

REGISTRY_CLI_CMD(name, "name", "print CLI session name (set by milk -n)", print_session_name);
REGISTRY_CLI_CMD(fpsload,
                 "fpsload",
                 "Load function parameter struct (FPS)",
                 function_parameter_structure_load__cli);
REGISTRY_CLI_CMD(fifo, "fifo", "manage command FIFO input", cli_fifo);
REGISTRY_CLI_CMD(timing, "timing", "toggle display of command execution timing", cli_timing_toggle);
REGISTRY_CLI_CMD(cmdstats, "cmdstats", "show command usage statistics", cli_cmdstats);

#ifdef USE_READLINE
REGISTRY_CLI_CMD(synhl, "synhl", "toggle syntax highlighting", cli_syntax_highlight_toggle);
REGISTRY_CLI_CMD(syndiag, "syndiag", "toggle real-time syntax diagnostics", cli_syntax_diag_toggle);
REGISTRY_CLI_CMD(cliindent,
                 "cliindent",
                 "toggle multi-line auto-indentation",
                 cli_auto_indent_toggle);
REGISTRY_CLI_CMD(showmatch,
                 "showmatch",
                 "toggle structural delimiter and keyword matching",
                 cli_showmatch_toggle);
#endif

REGISTRY_CLI_CMD(savehistory, "savehistory", "save command history to a file", cli_savehistory);
REGISTRY_CLI_CMD(setprompt, "setprompt", "set custom prompt format", cli_setprompt);
REGISTRY_CLI_CMD(sessionlog, "sessionlog", "enable session command logging", cli_sessionlog);
REGISTRY_CLI_CMD(history, "history", "show recent command history", cli_history_show);
REGISTRY_CLI_CMD(searchhist, "searchhist", "search history for pattern", cli_searchhist);
REGISTRY_CLI_CMD(fhist, "fhist", "interactive fuzzy search history", cli_fhist);
REGISTRY_CLI_CMD(ghistory, "ghistory", "global history (all sessions)", cli_ghistory);
REGISTRY_CLI_CMD(lhistory, "lhistory", "local history (current session)", cli_lhistory);
REGISTRY_CLI_CMD(fparam, "fparam", "interactive FPS parameter editor", cli_fparam);

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
void runCLI_cmd_init(void)
{
    // ensure that commands below belong to root/MAIN module
    data.moduleindex   = -1;
    milk_help_tui_hook = cli_fhelp; /* @help --tui */

    milk_module_begin(); /* milk.<name>: engine utilities of the milk-cli front-end */
    milk_register_runnable(&unit_name);
    milk_register_runnable(&unit_fpsload);
    milk_register_runnable(&unit_fifo);
    milk_register_runnable(&unit_timing);
    milk_module_end();

    milk_register_runnable(&unit_cmdstats);

#ifdef USE_READLINE
    milk_register_runnable(&unit_synhl);
    milk_register_runnable(&unit_syndiag);
    milk_register_runnable(&unit_cliindent);
    milk_register_runnable(&unit_showmatch);
#endif

    milk_register_runnable(&unit_savehistory);
    milk_register_runnable(&unit_setprompt);
    milk_register_runnable(&unit_sessionlog);
    milk_register_runnable(&unit_history);
    milk_register_runnable(&unit_searchhist);
    milk_register_runnable(&unit_fhist);
    milk_register_runnable(&unit_ghistory);
    milk_register_runnable(&unit_lhistory);
    milk_register_runnable(&unit_fparam);

    if (dcquiet == 0)
    {
        printf("        Loaded %ld modules, %u commands\n", data.NBmodule, data.NBcmd);
        printf("        \n");
    }
}
