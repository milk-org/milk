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
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_help.h"
#include "CLIcore_modules.h"
#include "CLIcore_UI_execute.h"

/* Forward references to functions in CLIcore.c
 * that are registered as CLI command callbacks.
 * These are not in a public header because they
 * are only called via the command registry. */
extern errno_t exitCLI(void);
extern errno_t set_processinfoON(void);
extern errno_t set_processinfoOFF(void);
extern errno_t set_default_precision_single(void);
extern errno_t set_default_precision_double(void);
extern errno_t milk_usleep__cli(void);
extern errno_t load_so__cli(void);
extern errno_t load_module__cli(void);
extern errno_t CLIcore__load_module_as__cli(void);
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

/* Thin wrappers for per-topic help CLI commands -------------------- */
static errno_t help_cmdopts_cmd(void)
{
    help_topic_cmdopts();
    return RETURN_SUCCESS;
}

/**
 * @brief CLI handler: display command syntax help.
 */
static errno_t help_syntax_cmd(void)
{
    help_topic_syntax();
    return RETURN_SUCCESS;
}

/**
 * @brief CLI handler: list all available commands.
 */
static errno_t help_commands_cmd(void)
{
    help_topic_commands();
    return RETURN_SUCCESS;
}

/**
 * @brief CLI handler: list all active variables.
 */
static errno_t help_variables_cmd(void)
{
    help_topic_variables();
    return RETURN_SUCCESS;
}

static errno_t help_flowcontrol_cmd(void)
{
    help_topic_flowcontrol();
    return RETURN_SUCCESS;
}

static errno_t help_scripting_cmd(void)
{
    help_topic_scripting();
    return RETURN_SUCCESS;
}

static errno_t help_milk_cmd(void)
{
    help_topic_milk();
    return RETURN_SUCCESS;
}

void runCLI_cmd_init()
{
    // ensure that commands below belong to root/MAIN module
    data.moduleindex = -1;

    RegisterCLIcommand("exit",
                       __FILE__,
                       exitCLI,
                       "exit program (same as quit command)",
                       "no argument",
                       "exit",
                       "exitCLI");

    RegisterCLIcommand("quit",
                       __FILE__,
                       exitCLI,
                       "exit program (same as exit command)",
                       "no argument",
                       "quit",
                       "exitCLI");

    RegisterCLIcommand("exitCLI",
                       __FILE__,
                       exitCLI,
                       "exit program (same as quit command)",
                       "no argument",
                       "exitCLI",
                       "exitCLI");

    RegisterCLIcommand("name",
                       __FILE__,
                       print_session_name,
                       "print CLI session name (set by milk -n)",
                       "no argument",
                       "name",
                       "print_session_name()");

    RegisterCLIcommand("help",
                       __FILE__,
                       help,
                       "show help",
                       "no argument",
                       "help",
                       "int help()");

    RegisterCLIcommand("fhelp",
                       __FILE__,
                       cli_fhelp,
                       "interactive fuzzy help search",
                       "no argument",
                       "fhelp",
                       "int cli_fhelp()");

    RegisterCLIcommand("?",
                       __FILE__,
                       help,
                       "show help",
                       "no argument",
                       "?",
                       "int help()");

    RegisterCLIcommand("helprl",
                       __FILE__,
                       help,
                       "show readline help",
                       "no argument",
                       "helprl",
                       "int help()");

    /* per-topic help commands --------------------------------------- */
    RegisterCLIcommand(
        "help-cmdopts",
        __FILE__,
        help_cmdopts_cmd,
        "help: command-line flags",
        "no argument",
        "help-cmdopts",
        "help_topic_cmdopts()");

    RegisterCLIcommand(
        "help-syntax",
        __FILE__,
        help_syntax_cmd,
        "help: syntax & interactive features",
        "no argument",
        "help-syntax",
        "help_topic_syntax()");

    RegisterCLIcommand(
        "help-commands",
        __FILE__,
        help_commands_cmd,
        "help: built-in CLI commands",
        "no argument",
        "help-commands",
        "help_topic_commands()");

    RegisterCLIcommand(
        "help-variables",
        __FILE__,
        help_variables_cmd,
        "help: variables, arrays and arithmetic",
        "no argument",
        "help-variables",
        "help_topic_variables()");

    RegisterCLIcommand(
        "help-flowcontrol",
        __FILE__,
        help_flowcontrol_cmd,
        "help: if/for/while/case/functions",
        "no argument",
        "help-flowcontrol",
        "help_topic_flowcontrol()");

    RegisterCLIcommand(
        "help-scripting",
        __FILE__,
        help_scripting_cmd,
        "help: script files, I/O, builtins, traps",
        "no argument",
        "help-scripting",
        "help_topic_scripting()");

    RegisterCLIcommand(
        "help-milk",
        __FILE__,
        help_milk_cmd,
        "help: milk streams, FPS, and milk-specific",
        "no argument",
        "help-milk",
        "help_topic_milk()");


    RegisterCLIcommand("cmd?",
                       __FILE__,
                       help_cmd,
                       "list/help command(s)",
                       "<command name>(optional)",
                       "cmd?",
                       "int help_cmd()");

    RegisterCLIcommand("cmdinfo?",
                       __FILE__,
                       cmdinfosearch,
                       "search for string/regex in command info",
                       "<search expression>",
                       "cmdinfo? image",
                       "int cmdinfosearch()");

    RegisterCLIcommand("m?",
                       __FILE__,
                       help_module,
                       "list/help module(s)",
                       "<module name>(optional)",
                       "m? COREMOD_memory",
                       "errno_t list_commands_module()");

    RegisterCLIcommand("soload",
                       __FILE__,
                       load_so__cli,
                       "load shared object",
                       "<shared object name>",
                       "soload mysharedobj.so",
                       "int load_sharedobj(char *libname)");

    RegisterCLIcommand("mload",
                       __FILE__,
                       load_module__cli,
                       "load module from shared object",
                       "<module name>",
                       "mload mymodule",
                       "errno_t load_module_shared(char *modulename)");

    RegisterCLIcommand("mloadas",
                       __FILE__,
                       CLIcore__load_module_as__cli,
                       "load module from shared object, use short name binding",
                       "<module name> <shortname>",
                       "mloadas mymodule mymod",
                       "errno_t load_module_shared(char *modulename)");

    RegisterCLIcommand("ci",
                       __FILE__,
                       printInfo,
                       "Print version, settings, info and exit",
                       "no argument",
                       "ci",
                       "int printInfo()");

    RegisterCLIcommand("dpsingle",
                       __FILE__,
                       set_default_precision_single,
                       "Set default precision to single",
                       "no argument",
                       "dpsingle",
                       "dcprecision = 0");

    RegisterCLIcommand("dpdouble",
                       __FILE__,
                       set_default_precision_double,
                       "Set default precision to double",
                       "no argument",
                       "dpdouble",
                       "dcprecision = 1");

    // process info

    RegisterCLIcommand("setprocinfoON",
                       __FILE__,
                       set_processinfoON,
                       "Set processes info ON",
                       "no argument",
                       "setprocinfoON",
                       "set_processinfoON()");

    RegisterCLIcommand("setprocinfoOFF",
                       __FILE__,
                       set_processinfoOFF,
                       "Set processes info OFF",
                       "no argument",
                       "setprocinfoOFF",
                       "set_processinfoOFF()");



    // FPS
    RegisterCLIcommand("fpsload",
                       __FILE__,
                       function_parameter_structure_load__cli,
                       "Load function parameter struct (FPS)",
                       "<fpsname>",
                       "fpsload imanalyze",
                       "long function_parameter_structure_load(char *fpsname)");



    RegisterCLIcommand("usleep",
                       __FILE__,
                       milk_usleep__cli,
                       "usleep",
                       "<us>",
                       "usleep 1000",
                       "usleep(long tus)");

    RegisterCLIcommand("cd",
                       __FILE__,
                       cli_cd,
                       "change current directory",
                       "<dir>",
                       "cd /tmp",
                       "cli_cd()");

    RegisterCLIcommand("pwd",
                       __FILE__,
                       cli_pwd,
                       "print current directory",
                       "no argument",
                       "pwd",
                       "cli_pwd()");

    RegisterCLIcommand(
        "fifo",
        __FILE__,
        cli_fifo,
        "manage command FIFO input",
        "[create|open|close|on|off"
        "|status] [path]",
        "fifo create /tmp/myfifo",
        "cli_fifo()");

    RegisterCLIcommand("alias",
                       __FILE__,
                       cli_alias_add,
                       "create/update command alias",
                       "<name> <command...>",
                       "alias ld mem.listim",
                       "cli_alias_add()");

    RegisterCLIcommand("unalias",
                       __FILE__,
                       cli_alias_remove,
                       "remove command alias",
                       "<name>",
                       "unalias ld",
                       "cli_alias_remove()");

    RegisterCLIcommand("aliases",
                       __FILE__,
                       cli_alias_list,
                       "list all command aliases",
                       "no argument",
                       "aliases",
                       "cli_alias_list()");

    RegisterCLIcommand("watch",
                       __FILE__,
                       cli_watch,
                       "repeat command at interval",
                       "<interval_ms> <command...>",
                       "watch 1000 mem.listim",
                       "cli_watch()");

    RegisterCLIcommand("list-streams",
                       __FILE__,
                       cli_list_streams,
                       "list available ImageStreamIO streams",
                       "no argument",
                       "list-streams",
                       "cli_list_streams()");

    RegisterCLIcommand("list-fps",
                       __FILE__,
                       cli_list_fps,
                       "list available FPS instances",
                       "no argument",
                       "list-fps",
                       "cli_list_fps()");


    RegisterCLIcommand("time",
                       __FILE__,
                       cli_time,
                       "measure command execution time",
                       "<command...>",
                       "time mem.listim",
                       "cli_time()");

    RegisterCLIcommand("cmdstats",
                       __FILE__,
                       cli_cmdstats,
                       "show command usage statistics",
                       "no argument",
                       "cmdstats",
                       "cli_cmdstats()");

    RegisterCLIcommand(
        "cli.timing",
        __FILE__,
        cli_timing_toggle,
        "toggle display of command execution timing",
        "[on|off]",
        "cli.timing on",
        "cli_timing_toggle()");

#ifdef USE_READLINE
    RegisterCLIcommand(
        "synhl",
        __FILE__,
        cli_syntax_highlight_toggle,
        "toggle syntax highlighting",
        "[on|off]",
        "synhl off",
        "cli_syntax_highlight_toggle()");
#endif

    RegisterCLIcommand("source",
                       __FILE__,
                       cli_source,
                       "execute a milk script file",
                       "<filename>",
                       "source myscript.milk",
                       "cli_source()");

    RegisterCLIcommand(
        "savescript",
        __FILE__,
        cli_savescript,
        "save variables and functions "
        "to a script file",
        "<filename>",
        "savescript state.milk",
        "cli_savescript()");

    RegisterCLIcommand(
        "savehistory",
        __FILE__,
        cli_savehistory,
        "save command history to a file",
        "<filename>",
        "savehistory cmds.milk",
        "cli_savehistory()");

    RegisterCLIcommand(
        "setprompt",
        __FILE__,
        cli_setprompt,
        "set custom prompt format",
        "[<format>]",
        "setprompt \"%u@%h %d > \"",
        "cli_setprompt()");

    RegisterCLIcommand(
        "bookmark",
        __FILE__,
        cli_bookmark,
        "manage command bookmarks",
        "save|run|list|rm <name> [cmd]",
        "bookmark save myjob \"cmd1 ; cmd2\"",
        "cli_bookmark()");

    RegisterCLIcommand(
        "sessionlog",
        __FILE__,
        cli_sessionlog,
        "enable session command logging",
        "[on|off|<filename>]",
        "sessionlog on",
        "cli_sessionlog()");

    RegisterCLIcommand(
        "history",
        __FILE__,
        cli_history_show,
        "show recent command history",
        "[<N>]",
        "history 50",
        "cli_history_show()");

    RegisterCLIcommand(
        "searchhist",
        __FILE__,
        cli_searchhist,
        "search history for pattern",
        "<pattern>",
        "searchhist listim",
        "cli_searchhist()");

    RegisterCLIcommand(
        "fhist",
        __FILE__,
        cli_fhist,
        "interactive fuzzy search history",
        "",
        "fhist",
        "cli_fhist()");

    RegisterCLIcommand(
        "ghistory",
        __FILE__,
        cli_ghistory,
        "global history (all sessions)",
        "[N] [-s <session_id>]",
        "ghistory 50",
        "cli_ghistory()");

    RegisterCLIcommand(
        "lhistory",
        __FILE__,
        cli_lhistory,
        "local history (current session)",
        "[N]",
        "lhistory",
        "cli_lhistory()");

    RegisterCLIcommand(
        "fparam",
        __FILE__,
        cli_fparam,
        "interactive FPS parameter editor",
        "<fpsname>",
        "fparam cnt2push",
        "cli_fparam()");

    RegisterCLIcommand(
        "echo",
        __FILE__,
        cli_cmd_echo,
        "print arguments",
        "[-n] <args...>",
        "echo hello world",
        "cli_cmd_echo()");

    RegisterCLIcommand(
        "unset",
        __FILE__,
        cli_cmd_unset,
        "remove a CLI variable",
        "<varname>",
        "unset myvar",
        "cli_cmd_unset()");

    RegisterCLIcommand(
        "vars",
        __FILE__,
        cli_cmd_vars,
        "list all CLI variables",
        "",
        "vars",
        "cli_cmd_vars()");

    RegisterCLIcommand(
        "fpsset",
        __FILE__,
        cli_cmd_fpsset,
        "set FPS parameter value",
        "<fpsname.param> <value>",
        "fpsset loopctrl.gain 0.3",
        "cli_cmd_fpsset()");

    RegisterCLIcommand(
        "read",
        __FILE__,
        cli_cmd_read,
        "read a line into a variable",
        "[-p \"prompt\"] <varname>",
        "read -p \"Enter: \" x",
        "cli_cmd_read()");

    RegisterCLIcommand(
        "export",
        __FILE__,
        cli_cmd_export,
        "push CLI variable to environ",
        "<varname>[=value]",
        "export MYVAR",
        "cli_cmd_export()");

    RegisterCLIcommand(
        "shift",
        __FILE__,
        cli_cmd_shift,
        "shift positional parameters left",
        "[N]",
        "shift",
        "cli_cmd_shift()");

    RegisterCLIcommand(
        "printf",
        __FILE__,
        cli_cmd_printf,
        "formatted output",
        "<format> [args...]",
        "printf \"%s=%d\\n\" name 42",
        "cli_cmd_printf()");

    RegisterCLIcommand(
        "fpslist",
        __FILE__,
        cli_cmd_fpslist,
        "list live FPS instances",
        "[--json] [pattern]",
        "fpslist --json dm*",
        "cli_cmd_fpslist()");

    RegisterCLIcommand(
        "fpsdump",
        __FILE__,
        cli_cmd_fpsdump,
        "dump FPS parameters as key=value",
        "[-t] [--json] <fpsname>",
        "fpsdump loopctrl",
        "cli_cmd_fpsdump()");

    RegisterCLIcommand(
        "streamlist",
        __FILE__,
        cli_cmd_streamlist,
        "list live SHM streams",
        "[-l] [--json] [pattern]",
        "streamlist -l dm*",
        "cli_cmd_streamlist()");

    RegisterCLIcommand(
        "proclist",
        __FILE__,
        cli_cmd_proclist,
        "list active processes",
        "[-l] [--json]",
        "proclist -l",
        "cli_cmd_proclist()");

    RegisterCLIcommand(
        "milkquery",
        __FILE__,
        cli_cmd_milkquery,
        "unified JSON system snapshot",
        "[--fps [pat]] [--streams [pat]]"
        " [--procs]",
        "milkquery --fps dm*",
        "cli_cmd_milkquery()");

    RegisterCLIcommand(
        "defer",
        __FILE__,
        cli_cmd_defer,
        "register cleanup command (LIFO)",
        "<command...>",
        "defer mem.rm _tmp 0",
        "cli_cmd_defer()");

    //  init_modules();

    if(dcquiet == 0)
    {
        printf("        Loaded %ld modules, %u commands\n",
               data.NBmodule,
               data.NBcmd);
        printf("        \n");
    }
}
