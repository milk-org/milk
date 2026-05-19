/**
 * @file CLIcore_help_topics.c
 *
 * @brief Help topic pages for the milk CLI
 *
 * Contains the content for each help topic page
 * accessible via "help <topic>". Topics cover:
 * - Command-line options
 * - Shell syntax reference
 * - Command categories
 * - Variable system
 * - Flow control constructs
 * - Scripting guide
 * - Milk-specific features
 *
 * Also provides the topic dispatcher, topic listing,
 * and structured output modes (JSON, porcelain) for
 * external tool integration.
 */

#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_help.h"

#define C_RST    "\033[0m"
#define C_TITLE  "\033[1;36m"
#define C_HDR    "\033[1;35m"
#define C_CMD    "\033[32m"
#define C_BOLD   "\033[1m"
#define C_NOTE   "\033[33m"
#define C_ERR    "\033[1;31m"

#ifndef COLORRESET
#define COLORRESET     C_RST
#endif
#ifndef COLORCMD
#define COLORCMD       C_CMD
#endif
#ifndef COLORINFO
#define COLORINFO      "\033[32m"
#endif

/**
 * @brief Print the high-level milk framework overview.
 *
 * Displays the three-pillar architecture (Streams,
 * FPS, Processinfo) with pointers to detailed guides.
 */
void print_milk_framework_help(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                    milk OVERVIEW\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf("The milk framework is built around four core pillars for\n");
    printf("high-performance, real-time data processing:\n");
    printf("\n");

    printf(C_HDR "1. ImageStreamIO (Streams)\n" C_RST);
    printf("Fast, low-latency shared-memory data streams designed to\n");
    printf("pass images and multi-dimensional arrays between distinct\n");
    printf("processes with zero-copy overhead.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-stream-help\n" C_RST);
    printf("\n");

    printf(C_HDR "2. Function Parameter Structure (FPS)\n" C_RST);
    printf("A shared memory architecture providing a unified namespace\n");
    printf("to manage configurations, parameters, and telemetry for\n");
    printf("applications seamlessly across the CLI and API.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-fps-help\n" C_RST);
    printf("\n");

    printf(C_HDR "3. Processinfo (procinfo API)\n" C_RST);
    printf("Advanced real-time execution management, CPU affinity,\n");
    printf("scheduling policies, and stream-based process triggering.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-procinfo-help\n" C_RST);
    printf("\n");

    printf(C_HDR "4. Sequencer (seq API)\n" C_RST);
    printf("Deterministic real-time scheduling and execution engine\n");
    printf("for hardware calibration loops and automated routines.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-seq-help\n" C_RST);
    printf("\n");

    printf(C_TITLE "--------------------------------------------------------\n" C_RST);
    printf(C_HDR "General Usage\n" C_RST);
    printf("To enter the interactive milk shell, simply type:\n");
    printf("  $ " C_CMD "milk-cli\n" C_RST);
    printf("\n");
    printf("From within the milk shell, you can list\n");
    printf("available commands to see all capabilities:\n");
    printf("For CLI specific help, run " C_CMD "milk-cli-help\n" C_RST);
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* Topic: cmdopts — command-line options                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for milk-cli command-line flags.
 */
void help_topic_cmdopts(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "          COMMAND LINE OPTIONS  (cmdopts)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_CMD "  -h, --help         " C_RST "print help index and exit\n");
    printf(C_CMD "  -v, --version      " C_RST "print version and exit\n");
    printf(C_CMD "  -i, --info         " C_RST "print version, settings, info\n");
    printf(C_CMD "  --verbose          " C_RST "be verbose\n");
    printf(C_CMD "  -d <level>         " C_RST "set debug level at startup\n");
    printf(C_CMD "  -o, --overwrite    " C_RST
           "overwrite existing FITS files " C_NOTE "(USE WITH CAUTION)\n" C_RST);
    printf(C_CMD "  -e, --errorexit    " C_RST "exit on error\n");
    printf(C_CMD "  -Z, --idle         " C_RST "only run when X is idle\n");
    printf(C_CMD "  -A, --autocomplete " C_RST
           "enable inline autocomplete " C_NOTE "(ON by default)\n" C_RST);
    printf(C_CMD "  --no-autocomplete  " C_RST "disable inline autocomplete\n");
    printf(C_CMD "  --no-history-suggest " C_RST "disable history suggestions\n");
    printf(C_CMD "  --no-arg-hints     " C_RST "disable argument hint line\n");
    printf(C_CMD "  --no-fuzzy         " C_RST "disable fuzzy/substring matching\n");
    printf(C_CMD "  -f, --fifoflag     " C_RST "enable default fifo input\n");
    printf(C_CMD "  -F <fifoname>      " C_RST "specify custom fifo name\n");
    printf(C_CMD "  -c, --command <cmd>" C_RST " execute single command and exit\n");
    printf(C_CMD "  -s <file>          " C_RST "execute startup script\n");
    printf(C_CMD "  -n <name>          " C_RST "specify process name\n");
    printf(C_CMD "  -p <priority>      " C_RST "set RT priority (0-99)\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: syntax — shell syntax and interaction                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for CLI syntax and interactive features.
 */
void help_topic_syntax(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "           SYNTAX & INTERACTION  (syntax)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Syntax Rules:\n" C_RST);
    printf("  Spaces separate arguments. Use " C_BOLD "#" C_RST " for comments.\n");
    printf("  Example: " C_CMD "command arg1 arg2 # comment\n" C_RST);
    printf("\n");
    printf(C_HDR "Tab Completion & UX Features:\n" C_RST);
    printf("  1st arg: Match commands, then images, " "then files.\n");
    printf("  Subsequent: Match images, then files.\n");
    printf("  " C_NOTE "History:" C_RST " Commands saved across sessions (Up/Down).\n");
    printf("  " C_NOTE "Autocorrection:" C_RST " Mistyped commands suggest closest match.\n");
    printf("  " C_NOTE "Fuzzy finding:" C_RST " " C_CMD "fhelp" C_RST " filters interactively.\n");
    printf("  " C_NOTE "Bash completion:" C_RST " Source scripts/milk-completion.sh.\n");
    printf("\n");
    printf(C_HDR "Piping Commands:\n" C_RST);
    printf("  Commands can be piped via stdin:\n");
    printf("  " C_CMD "echo -e \"a=1\\nb=2\\nc=a+b\" | milk-cli\n" C_RST);
    printf("  Use " C_BOLD "\\n" C_RST " to separate multiple commands.\n");
    printf("\n");
    printf(C_HDR "Shell Pass-through:\n" C_RST);
    printf("  Prefix OS commands with " C_CMD "!" C_RST " in interactive mode:\n");
    printf("  " C_CMD "!ls -la\n" C_RST);
    printf("  In script files the " C_CMD "!" C_RST " prefix is not required.\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: commands — built-in CLI commands                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for the most important built-in CLI commands.
 */
void help_topic_commands(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "           IMPORTANT COMMANDS  (commands)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Help & Discovery:\n" C_RST);
    printf(C_CMD "  help              " C_RST "Show help topic index\n");
    printf(C_CMD "  help-<topic>      " C_RST "Show help for a specific topic\n");
    printf(C_CMD "  cmd? [cmd]        " C_RST "Help for a specific command\n");
    printf(C_CMD "  m? [module]       " C_RST "List commands in a module\n");
    printf(C_CMD "  h? [string]       " C_RST "Search command descriptions\n");
    printf(C_CMD "  fhelp             " C_RST "Interactive fuzzy command search\n");
    printf(C_CMD "  fhist             " C_RST "Interactive fuzzy history search\n");
    printf("\n");
    printf(C_HDR "System Info:\n" C_RST);
    printf(C_CMD "  ci                " C_RST "System info and memory usage\n");
    printf(C_CMD "  mem.listim        " C_RST "List images in memory\n");
    printf("\n");
    printf(C_HDR "File I/O:\n" C_RST);
    printf(C_CMD "  iofits.loadfits   " C_RST
           "Load FITS file " C_NOTE "(requires CFITSIO)\n" C_RST);
    printf(C_CMD "  iofits.savefits   " C_RST
           "Save FITS file " C_NOTE "(requires CFITSIO)\n" C_RST);
    printf("\n");
    printf(C_HDR "Session Control:\n" C_RST);
    printf(C_CMD "  quit / exit       " C_RST "Exit the milk shell\n");
    printf(C_CMD "  !<syscommand>     " C_RST "Execute OS shell command\n");
    printf(C_CMD "  logon / logoff    " C_RST "Enable/disable session log\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: variables — variables, arrays, arithmetic, FPS access       */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for variables, arrays, and arithmetic.
 */
void help_topic_variables(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "       VARIABLES, ARRAYS & ARITHMETIC  (variables)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Basic Variables:\n" C_RST);
    printf("  " C_CMD "x=42" C_RST "              Set variable\n");
    printf("  " C_CMD "echo $x" C_RST "           Print variable\n");
    printf("  " C_CMD "echo ${x}" C_RST "         Braced form\n");
    printf("  " C_CMD "unset x" C_RST "           Remove variable\n");
    printf("  " C_CMD "vars" C_RST "              List all variables\n");
    printf("\n");
    printf(C_HDR "String Operations:\n" C_RST);
    printf("  " C_CMD "${#var}" C_RST "           String length\n");
    printf("  " C_CMD "${var:2:3}" C_RST "        Substring (offset:len)\n");
    printf("  " C_CMD "${var%%pat}" C_RST "        Strip longest suffix\n");
    printf("  " C_CMD "${var##pat}" C_RST "       Strip longest prefix\n");
    printf("  " C_CMD "${var/p/r}" C_RST "        Replace first match\n");
    printf("  " C_CMD "${var//p/r}" C_RST "       Replace all matches\n");
    printf("  " C_CMD "${var^^}" C_RST "          Uppercase all\n");
    printf("  " C_CMD "${var,,}" C_RST "          Lowercase all\n");
    printf("\n");
    printf(C_HDR "Parameter Defaults:\n" C_RST);
    printf("  " C_CMD "${v:-def}" C_RST "         default if unset\n");
    printf("  " C_CMD "${v:=def}" C_RST "         assign if unset\n");
    printf("  " C_CMD "${v:+alt}" C_RST "         alt value if set\n");
    printf("  " C_CMD "${v:?err}" C_RST "         error if unset\n");
    printf("\n");
    printf(C_HDR "Arrays:\n" C_RST);
    printf("  " C_CMD "arr=(a b c)" C_RST "     Create array\n");
    printf("  " C_CMD "${arr[0]}" C_RST "        Access element\n");
    printf("  " C_CMD "${arr[@]}" C_RST "        All elements\n");
    printf("  " C_CMD "${#arr[@]}" C_RST "       Array length\n");
    printf("  " C_CMD "declare -A m" C_RST "     Associative array\n");
    printf("  " C_CMD "${m[key]}" C_RST "        Associative lookup\n");
    printf("\n");
    printf(C_HDR "Arithmetic:\n" C_RST);
    printf("  " C_CMD "y=$((x + 5))" C_RST "  Integer +, -, *, /, %%\n");
    printf("  " C_CMD "((expr))" C_RST "        Arithmetic conditional\n");
    printf("\n");
    printf(C_HDR "FPS Parameter Access:\n" C_RST);
    printf("  " C_CMD "@fpsname.param" C_RST "    Read FPS parameter\n");
    printf("  " C_CMD "@seq.NAME.prop" C_RST "    Read Sequencer property\n");
    printf("  " C_CMD "fpsset fps p v" C_RST "    Write FPS parameter\n");
    printf("\n");
    printf(C_HDR "Milk Stream Metadata (@ Namespace):\n" C_RST);
    printf("  " C_CMD "@s.name.xsize" C_RST "      Stream width  (size[0])\n");
    printf("  " C_CMD "@s.name.ysize" C_RST "      Stream height (size[1])\n");
    printf("  " C_CMD "@s.name.zsize" C_RST "      Stream depth  (size[2])\n");
    printf("  " C_CMD "@s.name.naxis" C_RST "      Number of axes\n");
    printf("  " C_CMD "@s.name.type" C_RST "       Datatype code\n");
    printf("  " C_CMD "@s.name.typename" C_RST "   Datatype name\n");
    printf("  " C_CMD "@s.name.typeid" C_RST "     Datatype code (alias)\n");
    printf("  " C_CMD "@s.name.cnt0" C_RST "       Frame counter (total)\n");
    printf("  " C_CMD "@s.name.cnt1" C_RST "       Frame counter (recent)\n");
    printf("  " C_CMD "@s.name.sem" C_RST "        Semaphore count\n");
    printf("  " C_CMD "@s.name.pid" C_RST "        Creator PID\n");
    printf("  " C_CMD "@s.name.ownerPID" C_RST "   Owner PID\n");
    printf("  " C_CMD "@s.name.nelement" C_RST "   Total elements\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: flowcontrol — if, loops, case, functions                    */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for flow control constructs.
 */
void help_topic_flowcontrol(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "              FLOW CONTROL  (flowcontrol)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Conditionals:\n" C_RST);
    printf("  " C_CMD "if [ $x -gt 5 ]; then" C_RST "\n");
    printf("  " C_CMD "    echo big" C_RST "\n");
    printf("  " C_CMD "elif [$x -gt 2]; then" C_RST "  ← cascading branch\n");
    printf("  " C_CMD "else" C_RST "\n");
    printf("  " C_CMD "    echo small" C_RST "\n");
    printf("  " C_CMD "fi" C_RST "\n");
    printf("  Tests: "
           C_NOTE "-eq -ne -gt -ge -lt -le"
           C_RST " (numeric), " C_NOTE "= !=" C_RST " (string)\n");
    printf("  File tests: "
           C_NOTE "-f" C_RST " (file), "
           C_NOTE "-d" C_RST " (dir), " C_NOTE "-e" C_RST " (exists)\n");
    printf("  Negate: " C_CMD "[ ! expr ]" C_RST " logical NOT\n");
    printf("  Extended: " C_CMD "[[$s =~ ^[0-9]+$]]" C_RST " regex\n");
    printf("\n");
    printf(C_HDR "Loops:\n" C_RST);
    printf("  " C_CMD "while [$n -lt 10]; do" C_RST " ... " C_CMD "done" C_RST "\n");
    printf("  " C_CMD "for x in a b c; do" C_RST " ... " C_CMD "done" C_RST "\n");
    printf("  " C_CMD "for ((i=0; i<10; i++)); do"
           C_RST " ... " C_CMD "done" C_RST "  ← C-style\n");
    printf("  " C_CMD "break" C_RST " exits loop, " C_CMD "continue" C_RST " next iter\n");
    printf("  " C_CMD "break 2" C_RST "  / " C_CMD "continue 2" C_RST "  (nested)\n");
    printf("\n");
    printf(C_HDR "Case Statement:\n" C_RST);
    printf("  " C_CMD "case $var in" C_RST "\n");
    printf("  " C_CMD "  yes) echo ok ;;" C_RST "\n");
    printf("  " C_CMD "  a|b) echo ab ;;" C_RST "  ← alternation\n");
    printf("  " C_CMD "  *) echo default ;;" C_RST "\n");
    printf("  " C_CMD "esac" C_RST "\n");
    printf("\n");
    printf(C_HDR "Functions:\n" C_RST);
    printf("  " C_CMD "function myfunc {" C_RST " ... " C_CMD "}" C_RST "\n");
    printf("  " C_CMD "myfunc arg1 arg2" C_RST "  call with " C_NOTE "$1..$9" C_RST " in body\n");
    printf("  " C_CMD "return [val]" C_RST "      exit function, set $?\n");
    printf("  " C_CMD "local VAR=val" C_RST "     declare local variable\n");
    printf("\n");
    printf(C_HDR "Logical Operators:\n" C_RST);
    printf("  " C_CMD "cmd1 && cmd2" C_RST "  run cmd2 if cmd1 succeeds\n");
    printf("  " C_CMD "cmd1 || cmd2" C_RST "  run cmd2 if cmd1 fails\n");
    printf("\n");
    printf(C_HDR "Select Menu:\n" C_RST);
    printf("  " C_CMD "select x in a b c; do" C_RST "\n");
    printf("  " C_CMD "  echo $x" C_RST "\n");
    printf("  " C_CMD "done" C_RST "  interactive numbered menu\n");
    printf("\n");
    printf(C_HDR "Stream Event:\n" C_RST);
    printf("  " C_CMD "on_update <stream> { cmd }" C_RST "\n");
    printf("  Waits for stream update then runs cmd\n");
    printf("\n");
    printf(C_HDR "Unified Event Wait:\n" C_RST);
    printf("  " C_CMD "wait_any [-t T] " "S:s F:f.p=v P:n:STATE" C_RST "\n");
    printf("  Block until any event fires; " "returns index as $?\n");
    printf("  " C_NOTE "S:" C_RST
           "stream  " C_NOTE "F:" C_RST "fps.param  " C_NOTE "P:" C_RST "proc:state\n");
    printf("  Comparisons: " C_CMD "= != >= <=" C_RST " (e.g. F:dmcomb.gain>=0.5)\n");
    printf("\n");
    printf(C_HDR "Engine Event Traps:\n" C_RST);
    printf("  " C_CMD "trap 'cmd' " "STREAM:name" C_RST "  fire on stream update\n");
    printf("  " C_CMD "trap 'cmd' " "FPS:f.p=v" C_RST "    fire on FPS match\n");
    printf("  " C_CMD "trap 'cmd' " "PROC:n:STATE" C_RST "  fire on state\n");
    printf("  " C_CMD "trap '' " "STREAM:name" C_RST "    clear trap\n");
    printf("  " C_CMD "trap -l" C_RST "              list traps\n");
    printf("  Flags: "
           C_CMD "-i ms" C_RST
           " throttle interval (def 100ms)" "  " C_CMD "-n N" C_RST " fire count limit\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: scripting — script files, I/O, builtins, traps              */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for scripting features and built-ins.
 */
void help_topic_scripting(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "              SCRIPTING FEATURES  (scripting)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Script Files:\n" C_RST);
    printf(C_CMD "  source <file>      " C_RST "Execute a script file\n");
    printf(C_CMD "  . <file>           " C_RST "Same (dot-source)\n");
    printf(C_CMD "  include_once <f>   " C_RST "Source only once per session\n");
    printf(C_CMD "  savescript <file>  " C_RST "Save variables & functions\n");
    printf(C_CMD "  savehistory <file> " C_RST "Save command history\n");
    printf("  Startup: " C_CMD "milk-cli -s <file>" C_RST "  run on launch\n");
    printf("  Startup: " C_CMD "milk-script <file>" C_RST "  standalone runner\n");
    printf("  Auto-load: " C_CMD "~/.milkrc" C_RST " sourced at startup\n");
    printf("  Shebang:   " C_NOTE "#!/usr/bin/env milk-script" C_RST "\n");
    printf("\n");
    printf(C_HDR "Built-in Commands:\n" C_RST);
    printf(C_CMD "  echo <str>         " C_RST "Print a line\n");
    printf(C_CMD "  printf \"fmt\" a..   " C_RST "Formatted output (%%s %%d %%f)\n");
    printf(C_CMD "  sleep <sec>        " C_RST "Pause (float-capable)\n");
    printf(C_CMD "  read [-p p] var    " C_RST "Read line from stdin\n");
    printf(C_CMD "  read -t N var      " C_RST "Timed read (seconds)\n");
    printf(C_CMD "  read -a ARR        " C_RST "Read words into array\n");
    printf(C_CMD "  read -n N var      " C_RST "Read N chars (raw mode)\n");
    printf(C_CMD "  exit [N]           " C_RST "Exit with status N\n");
    printf(C_CMD "  shift [N]          " C_RST "Shift $1..$9 by N\n");
    printf(C_CMD "  true / false       " C_RST "Set $? to 0 / 1\n");
    printf("\n");
    printf(C_HDR "Pipes & Redirection:\n" C_RST);
    printf("  " C_CMD "cmd1 | cmd2" C_RST "     pipe stdout → stdin\n");
    printf("  " C_CMD "cmd > file" C_RST "      write to file\n");
    printf("  " C_CMD "cmd >> file" C_RST "     append to file\n");
    printf("  " C_CMD "cmd < file" C_RST "      stdin from file\n");
    printf("  " C_CMD "cmd <<< \"str\"" C_RST "  here-string\n");
    printf("  " C_CMD "cmd 2>&1" C_RST "        stderr to stdout\n");
    printf("  " C_CMD "cmd 2>/dev/null" C_RST " discard stderr\n");
    printf("\n");
    printf(C_HDR "Brace & Glob Expansion:\n" C_RST);
    printf("  " C_CMD "{1..5}" C_RST " → 1 2 3 4 5\n");
    printf("  " C_CMD "{0..10..2}" C_RST " → 0 2 4 6 8 10\n");
    printf("  " C_CMD "*.fits" C_RST " expands to matching files\n");
    printf("  " C_CMD "data_??.bin" C_RST " single-char wildcard\n");
    printf("\n");
    printf(C_HDR "Heredocs:\n" C_RST);
    printf("  " C_CMD "VAR=<<EOF" C_RST "\n");
    printf("  " C_CMD "  line 1" C_RST "\n");
    printf("  " C_CMD "EOF" C_RST " → multi-line variable\n");
    printf("\n");
    printf(C_HDR "Signal Traps:\n" C_RST);
    printf("  " C_CMD "trap 'cmd' EXIT INT" C_RST " handler\n");
    printf("  " C_CMD "trap 'rm /tmp/f' EXIT" C_RST " cleanup\n");
    printf("\n");
    printf(C_HDR "Shell Options:\n" C_RST);
    printf("  " C_CMD "set -e" C_RST "  exit on error\n");
    printf("  " C_CMD "set -x" C_RST "  trace commands\n");
    printf("  " C_CMD "set +e" C_RST "  / " C_CMD "set +x" C_RST "  disable above\n");
    printf("\n");
    printf(C_HDR "Environment & Read-only:\n" C_RST);
    printf("  " C_CMD "export VAR=val" C_RST "  env var\n");
    printf("  " C_CMD "readonly VAR=val" C_RST " immutable\n");
    printf("\n");
    printf(C_HDR "Aliases & Indirect Expansion:\n" C_RST);
    printf("  " C_CMD "alias n='cmd'" C_RST " create alias\n");
    printf("  " C_CMD "unalias n" C_RST "     remove alias\n");
    printf("  " C_CMD "${!var}" C_RST "       indirect expansion\n");
    printf("\n");
    printf(C_HDR "Miscellaneous:\n" C_RST);
    printf("  " C_CMD "getopts \"ab:\" opt" C_RST "  option parsing\n");
    printf("  " C_CMD "mapfile -t arr < file" C_RST "  lines → array\n");
    printf("  " C_CMD "cmd &" C_RST " background; " C_CMD "wait" C_RST " for bg jobs\n");
    printf("  " C_CMD "(cmd1; cmd2)" C_RST " subshell\n");
    printf("  " C_CMD "~/path" C_RST " → $HOME/path\n");
    printf("  " C_CMD "basename / dirname" C_RST "  path utilities\n");
    printf("  " C_CMD "pushd / popd / dirs" C_RST "  directory stack\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: milk — milk-specific runtime features                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for milk-specific CLI features.
 */
void help_topic_milk(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "          MILK-SPECIFIC FEATURES  (milk)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Stream Metadata @s.<name>.<prop> namespace:\n" C_RST);
    printf("  " C_CMD "@s.name.xsize" C_RST "      Stream width  (size[0])\n");
    printf("  " C_CMD "@s.name.ysize" C_RST "      Stream height (size[1])\n");
    printf("  " C_CMD "@s.name.zsize" C_RST "      Stream depth  (size[2])\n");
    printf("  " C_CMD "@s.name.naxis" C_RST "      Number of axes\n");
    printf("  " C_CMD "@s.name.type" C_RST "       Datatype code\n");
    printf("  " C_CMD "@s.name.typename" C_RST "   Datatype name\n");
    printf("  " C_CMD "@s.name.typeid" C_RST "     Datatype code (alias)\n");
    printf("  " C_CMD "@s.name.cnt0" C_RST "       Frame counter (total)\n");
    printf("  " C_CMD "@s.name.cnt1" C_RST "       Frame counter (recent)\n");
    printf("  " C_CMD "@s.name.sem" C_RST "        Semaphore count\n");
    printf("  " C_CMD "@s.name.pid" C_RST "        Creator PID\n");
    printf("  " C_CMD "@s.name.ownerPID" C_RST "   Owner PID\n");
    printf("  " C_CMD "@s.name.nelement" C_RST "   Total elements\n");
    printf("\n");
    printf(C_HDR "Waiting for Resources:\n" C_RST);
    printf("  " C_CMD "waitfor_stream s T" C_RST "  Block up to T sec for SHM stream\n");
    printf("  " C_CMD "waitfor_fps f T   " C_RST "  Block up to T sec for FPS\n");
    printf("  " C_CMD "on_update <name> { cmd }" C_RST "  Trigger on stream write\n");
    printf("  " C_CMD "wait_any [-t T] events" C_RST "  Multiplex S:/F:/P: events\n");
    printf("\n");
    printf(C_HDR "Introspection (JSON):\n" C_RST);
    printf("  " C_CMD "fpsdump --json <fps>" C_RST "  FPS params as JSON\n");
    printf("  " C_CMD "fpslist --json      " C_RST "  FPS instances as JSON\n");
    printf("  " C_CMD "streamlist --json   " C_RST "  Streams as JSON\n");
    printf("  " C_CMD "proclist --json     " C_RST "  Processes as JSON\n");
    printf("  " C_CMD "milkquery           " C_RST "  Unified system snapshot\n");
    printf("\n");
    printf(C_HDR "FPS Parameters:\n" C_RST);
    printf("  " C_CMD "@fpsname.param   " C_RST "Read FPS parameter value\n");
    printf("  " C_CMD "fpsset fps p v   " C_RST "Write FPS parameter\n");
    printf("  " C_CMD "fparam <fpsname> " C_RST "Interactive FPS parameter editor\n");
    printf("\n");
    printf(C_HDR "Stream Management:\n" C_RST);
    printf("  " C_CMD "milk-FITS2shm f.fits s " C_RST "Load FITS into SHM stream\n");
    printf("  " C_CMD "milk-shm2FITS s f.fits " C_RST "Save SHM stream to FITS\n");
    printf("  " C_CMD "milk-stream-help        " C_RST "Stream usage guide\n");
    printf("\n");
    printf(C_HDR "FPS Executables:\n" C_RST);
    printf("  " C_CMD "milk-fpsexec-list       " C_RST "List all fpsexec programs\n");
    printf("  " C_CMD "milk-fpsexec-<name> -h1 " C_RST "One-line description\n");
    printf("  " C_CMD "milk-fpsCTRL           " C_RST "TUI parameter controller\n");
    printf("  " C_CMD "milk-fps-help           " C_RST "FPS usage guide\n");
    printf("\n");
    printf(C_HDR "Process Monitoring:\n" C_RST);
    printf("  " C_CMD "milk-streamCTRL        " C_RST "TUI stream monitor\n");
    printf("  " C_CMD "milk-procCTRL          " C_RST "TUI process monitor\n");
    printf("  " C_CMD "milk-procinfo-help      " C_RST "Processinfo guide\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic dispatch by name                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Dispatch help output to the correct topic function.
 *
 * @param topic  Topic keyword string, or NULL / "" for index.
 * @return  0 on success, 1 if topic not found.
 */
int help_topic_dispatch(const char *topic)
{
    if (!topic || topic[0] == '\0')
    {
        return 1; /* caller should print the index */
    }

    if (strcmp(topic, "cmdopts") == 0)
    {
        help_topic_cmdopts();
    }
    else if (strcmp(topic, "syntax") == 0)
    {
        help_topic_syntax();
    }
    else if (strcmp(topic, "commands") == 0)
    {
        help_topic_commands();
    }
    else if (strcmp(topic, "variables") == 0)
    {
        help_topic_variables();
    }
    else if (strcmp(topic, "flowcontrol") == 0)
    {
        help_topic_flowcontrol();
    }
    else if (strcmp(topic, "scripting") == 0)
    {
        help_topic_scripting();
    }
    else if (strcmp(topic, "milk") == 0)
    {
        help_topic_milk();
    }
    else
    {
        return 1; /* unknown topic */
    }
    return 0;
}


/* ------------------------------------------------------------------ */
/* Main help index (replaces the old monolithic function)             */
/* ------------------------------------------------------------------ */

/**
 * @brief Print a compact help index listing all available topics.
 *
 * The full content for each topic is obtained via help-<topic> CLI
 * commands or  milk-cli-help <topic>  from the shell.
 */
/**
 * @brief Print only the available-topics list.
 *
 * Shown both in the full help index and as a concise
 * hint when an unknown topic is supplied.
 */
void print_help_topic_list(void)
{
    printf(C_HDR "Available help topics:\n" C_RST);
    printf("  " C_CMD "cmdopts     " C_RST "Command-line flags (-h, -s, -n\xe2\x80\xa6)\n");
    printf("  " C_CMD "syntax      " C_RST "Syntax, tab completion, piping\n");
    printf("  " C_CMD "commands    " C_RST "Built-in CLI commands (?, cmd?\xe2\x80\xa6)\n");
    printf("  " C_CMD "variables   " C_RST "Variables, arrays, arithmetic\n");
    printf("  " C_CMD "flowcontrol " C_RST "if/while/for/case/function\n");
    printf("  " C_CMD "scripting   " C_RST "Script files, I/O, builtins\n");
    printf("  " C_CMD "milk        " C_RST "Streams, FPS, milk-specific\n");
    printf("\n");
}

/**
 * @brief Emit help index as JSON for machine parsing.
 *
 * Outputs a JSON object with "topics" and
 * "quick_reference" arrays for integration with
 * external tooling.
 */
void print_milk_cli_help_json(void)
{
    printf("{\n");
    printf("  \"topics\": [\n");
    printf("    {\"name\": \"cmdopts\", \"description\": \"Command-line flags (-h, -s, -n...)\"},\n");
    printf("    {\"name\": \"syntax\", \"description\": \"Syntax, tab completion, piping\"},\n");
    printf("    {\"name\": \"commands\", \"description\": \"Built-in CLI commands (?, cmd?...)\"},\n");
    printf("    {\"name\": \"variables\", \"description\": \"Variables, arrays, arithmetic\"},\n");
    printf("    {\"name\": \"flowcontrol\", \"description\": \"if/while/for/case/function\"},\n");
    printf("    {\"name\": \"scripting\", \"description\": \"Script files, I/O, builtins\"},\n");
    printf("    {\"name\": \"milk\", \"description\": \"Streams, FPS, milk-specific\"}\n");
    printf("  ],\n");
    printf("  \"quick_reference\": [\n");
    printf("    {\"command\": \"cmd?\", \"args\": \"[name]\", \"description\": \"Help for a specific command\"},\n");
    printf("    {\"command\": \"m?\", \"args\": \"[module]\", \"description\": \"List commands in a module\"},\n");
    printf("    {\"command\": \"h?\", \"args\": \"[string]\", \"description\": \"Search command descriptions\"},\n");
    printf("    {\"command\": \"fhelp\", \"args\": \"\", \"description\": \"Interactive fuzzy command search\"},\n");
    printf("    {\"command\": \"quit / exit\", \"args\": \"\", \"description\": \"Exit the milk shell\"}\n");
    printf("  ]\n");
    printf("}\n");
}

/**
 * @brief Emit help index in tab-separated format.
 *
 * Outputs TYPE, NAME, ARGS, DESCRIPTION columns
 * for easy parsing by scripts and completion
 * generators.
 */
void print_milk_cli_help_porcelain(void)
{
    printf("TYPE\tNAME\tARGS\tDESCRIPTION\n");
    printf("TOPIC\tcmdopts\t\tCommand-line flags (-h, -s, -n...)\n");
    printf("TOPIC\tsyntax\t\tSyntax, tab completion, piping\n");
    printf("TOPIC\tcommands\t\tBuilt-in CLI commands (?, cmd?...)\n");
    printf("TOPIC\tvariables\t\tVariables, arrays, arithmetic\n");
    printf("TOPIC\tflowcontrol\t\tif/while/for/case/function\n");
    printf("TOPIC\tscripting\t\tScript files, I/O, builtins\n");
    printf("TOPIC\tmilk\t\tStreams, FPS, milk-specific\n");
    printf("QUICKREF\tcmd?\t[name]\tHelp for a specific command\n");
    printf("QUICKREF\tm?\t[module]\tList commands in a module\n");
    printf("QUICKREF\th?\t[string]\tSearch command descriptions\n");
    printf("QUICKREF\tfhelp\t\tInteractive fuzzy command search\n");
    printf("QUICKREF\tquit / exit\t\tExit the milk shell\n");
}

/**
 * @brief Print the main milk-cli help page.
 *
 * Two-section layout:
 *   1. Program-level description: NAME, SYNOPSIS,
 *      DESCRIPTION, and all command-line OPTIONS for
 *      the milk-cli binary itself.
 *   2. Internal-commands section: topics and quick
 *      reference for commands available *inside* the
 *      running milk-cli shell or a milk-script.
 *
 * Output format is selected via help_format_mode
 * (0 = human, 1 = JSON, 2 = porcelain).
 */
void print_milk_cli_help(void)
{
    if (help_format_mode == 1) {
        print_milk_cli_help_json();
        return;
    } else if (help_format_mode == 2) {
        print_milk_cli_help_porcelain();
        return;
    }

    /* ── Section 1: program-level description of milk-cli ───── */
    printf("\n");
    printf(C_TITLE "========================================\n" C_RST);
    printf(C_TITLE "         milk-cli \xe2\x80\x94 PROGRAM HELP\n" C_RST);
    printf(C_TITLE "========================================\n" C_RST);
    printf("\n");
    printf(C_BOLD "NAME\n" C_RST);
    printf("  milk-cli \xe2\x80\x94 interactive shell and scripting engine\n");
    printf("           for the milk real-time imaging framework\n");
    printf("\n");
    printf(C_BOLD "SYNOPSIS\n" C_RST);
    printf("  " C_CMD "milk-cli" C_RST "                      Start interactive shell\n");
    printf("  " C_CMD "milk-cli -c <cmd>" C_RST "          Execute one command and exit\n");
    printf("  " C_CMD "milk-cli -s <file>" C_RST
           "         Run startup script, then enter shell\n");
    printf("  " C_CMD "echo cmds | milk-cli" C_RST "       Read commands from stdin\n");
    printf("  " C_CMD "milk-script <file>" C_RST "         Run a script non-interactively\n");
    printf("\n");
    printf(C_BOLD "DESCRIPTION\n" C_RST);
    printf("  milk-cli is the interactive command-line interface to\n");
    printf("  the milk framework. It provides:\n");
    printf("    \xe2\x80\xa2 A REPL with tab-completion, history and fuzzy search\n");
    printf("    \xe2\x80\xa2 A scripting engine shared with milk-script\n");
    printf("    \xe2\x80\xa2 Access to all loaded milk/cacao module commands\n");
    printf("    \xe2\x80\xa2 Real-time stream, FPS, and process management\n");
    printf("\n");
    printf(C_BOLD "OPTIONS\n" C_RST);
    printf(C_CMD "  -h, --help           " C_RST "Print this help and exit\n");
    printf(C_CMD "  -v, --version        " C_RST "Print version and exit\n");
    printf(C_CMD "  -i, --info           " C_RST "Print version, paths, and build info\n");
    printf(C_CMD "  -c <cmd>             " C_RST "Execute single command and exit\n");
    printf(C_CMD "  -s <file>            " C_RST "Execute startup script on launch\n");
    printf(C_CMD "  -n <name>            " C_RST "Set process name\n");
    printf(C_CMD "  -p <priority>        " C_RST
           "Set real-time priority (0" "\xe2\x80\x93" "99)\n");
    printf(C_CMD "  -e, --errorexit      " C_RST "Exit on first error\n");
    printf(C_CMD "  -E, --echo-input     " C_RST "Echo each input line (colored)\n");
    printf(C_CMD "  -d <level>           " C_RST "Set debug level at startup\n");
    printf(C_CMD "  -o, --overwrite      " C_RST
           "Overwrite existing FITS files " C_NOTE "(caution)\n" C_RST);
    printf(C_CMD "  -f, --fifoflag       " C_RST "Enable default FIFO input\n");
    printf(C_CMD "  -F <fifoname>        " C_RST "Specify custom FIFO name\n");
    printf(C_CMD "  -m <monitor>         " C_RST "Set memory monitor stream\n");
    printf(C_CMD "  -Z, --idle           " C_RST "Only run when display is idle\n");
    printf(C_CMD "  -A, --autocomplete   " C_RST
           "Enable inline autocomplete " C_NOTE "(on by default)\n" C_RST);
    printf(C_CMD "  --no-autocomplete    " C_RST "Disable inline autocomplete\n");
    printf(C_CMD "  --no-history-suggest " C_RST "Disable history suggestions\n");
    printf(C_CMD "  --no-arg-hints       " C_RST "Disable argument hint line\n");
    printf(C_CMD "  --no-fuzzy           " C_RST "Disable fuzzy/substring matching\n");
    printf(C_CMD "  --verbose            " C_RST "Be verbose\n");
    printf("\n");

    /* ── Section 2: help available inside the running shell ─── */
    printf(C_TITLE "========================================\n" C_RST);
    printf(C_TITLE "  COMMANDS AVAILABLE INSIDE milk-cli\n" C_RST);
    printf(C_TITLE "========================================\n" C_RST);
    printf("\n");
    printf("The following topics describe commands and syntax\n");
    printf("available " C_BOLD "within" C_RST " the milk-cli shell or a milk-script:\n");
    printf("\n");
    print_help_topic_list();
    printf(C_NOTE "From the OS shell:\n" C_RST);
    printf("  $ " C_CMD "milk-cli-help <topic>\n" C_RST);
    printf(C_NOTE "From within milk-cli:\n" C_RST);
    printf("  > " C_CMD "help <topic>\n" C_RST);
    printf("\n");
    printf(C_HDR "Quick reference (inside milk-cli):\n" C_RST);
    printf("  " C_CMD "cmd? [name]   " C_RST "Help for a specific command\n");
    printf("  " C_CMD "m? [module]   " C_RST "List commands in a module\n");
    printf("  " C_CMD "h? [string]   " C_RST "Search command descriptions\n");
    printf("  " C_CMD "fhelp         " C_RST "Interactive fuzzy command search\n");
    printf("  " C_CMD "exitCLI       " C_RST "Exit the milk shell / script\n");
    printf("\n");
}
