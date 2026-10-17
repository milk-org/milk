// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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

#define C_RST "\033[0m"
#define C_TITLE "\033[1;36m"
#define C_HDR "\033[1;35m"
#define C_CMD "\033[32m"
#define C_BOLD "\033[1m"
#define C_NOTE "\033[33m"
#define C_ERR "\033[1;31m"

#ifndef COLORRESET
#    define COLORRESET C_RST
#endif
#ifndef COLORCMD
#    define COLORCMD C_CMD
#endif
#ifndef COLORINFO
#    define COLORINFO "\033[32m"
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
    printf("  " C_NOTE "Detailed guide:" C_RST " " C_CMD "@h streams" C_RST " or " C_CMD
           "milk-stream-help\n" C_RST);
    printf("\n");

    printf(C_HDR "2. Function Parameter Structure (FPS)\n" C_RST);
    printf("A shared memory architecture providing a unified namespace\n");
    printf("to manage configurations, parameters, and telemetry for\n");
    printf("applications seamlessly across the CLI and API.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " " C_CMD "@h fps" C_RST "     or " C_CMD
           "milk-fps-help\n" C_RST);
    printf("\n");

    printf(C_HDR "3. Processinfo (procinfo API)\n" C_RST);
    printf("Advanced real-time execution management, CPU affinity,\n");
    printf("scheduling policies, and stream-based process triggering.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " " C_CMD "@h procinfo" C_RST " or " C_CMD
           "milk-procinfo-help\n" C_RST);
    printf("\n");

    printf(C_HDR "4. Sequencer (seq API)\n" C_RST);
    printf("Deterministic real-time scheduling and execution engine\n");
    printf("for hardware calibration loops and automated routines.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " " C_CMD "@h seq" C_RST "     or " C_CMD
           "milk-seq-help\n" C_RST);
    printf("\n");

    printf(C_TITLE "--------------------------------------------------------\n" C_RST);
    printf(C_HDR "Interactive & Scripting Workflows (@-commands vs milk-*)\n" C_RST);
    printf("  " C_CMD "source milk.bash" C_RST "   Enable zero-fork " C_CMD "@..." C_RST
           " commands in your current Bash shell.\n");
    printf("                     Use " C_CMD "@h" C_RST ", " C_CMD "@hb" C_RST ", " C_CMD
           "@hc <cmd>" C_RST ", " C_CMD "@get" C_RST ", " C_CMD "@set" C_RST ", and " C_CMD
           "@<mod>.<cmd>" C_RST ".\n");
    printf("  " C_CMD "milk-cli" C_RST "           Launch the standalone interactive milk REPL.\n");
    printf("  " C_CMD "milk-*" C_RST "             Standalone OS executables: TUI monitors (" C_CMD
           "milk-CTRL" C_RST ",\n");
    printf("                     " C_CMD "milk-fpsCTRL" C_RST ", " C_CMD "milk-streamCTRL" C_RST
           "), compute units (" C_CMD "milk-fpsexec-*" C_RST "),\n");
    printf("                     and external utilities (" C_CMD "milk-commands" C_RST ", " C_CMD
           "milk-cli-help" C_RST ").\n");
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
    printf(C_CMD "  -o, --overwrite    " C_RST "overwrite existing FITS files " C_NOTE
                 "(USE WITH CAUTION)\n" C_RST);
    printf(C_CMD "  -e, --errorexit    " C_RST "exit on error\n");
    printf(C_CMD "  -Z, --idle         " C_RST "only run when X is idle\n");
    printf(C_CMD "  -A, --autocomplete " C_RST "enable inline autocomplete " C_NOTE
                 "(ON by default)\n" C_RST);
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
    printf("  1st arg: Match commands, then images, "
           "then files.\n");
    printf("  Subsequent: Match images, then files.\n");
    printf("  " C_NOTE "History:" C_RST " Commands saved across sessions (Up/Down).\n");
    printf("  " C_NOTE "Autocorrection:" C_RST " Mistyped commands suggest closest match.\n");
    printf("  " C_NOTE "Fuzzy finding:" C_RST " " C_CMD "@fhelp" C_RST " filters interactively.\n");
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
    printf(C_CMD "  @help             " C_RST "Show help topic index\n");
    printf(C_CMD "  @help-<topic>     " C_RST "Show help for a specific topic\n");
    printf(C_CMD "  @cmd? [cmd]       " C_RST "Help for a specific command\n");
    printf(C_CMD "  @m? [module]      " C_RST "List commands in a module\n");
    printf(C_CMD "  @fhelp            " C_RST "Interactive fuzzy command search\n");
    printf(C_CMD "  @fhist            " C_RST "Interactive fuzzy history search\n");
    printf("\n");
    printf(C_HDR "System Info:\n" C_RST);
    printf(C_CMD "  @ci               " C_RST "System info and memory usage\n");
    printf(C_CMD "  mem.listim        " C_RST "List images in memory\n");
    printf("\n");
    printf(C_HDR "File I/O:\n" C_RST);
    printf(C_CMD "  iofits.loadfits   " C_RST "Load FITS file " C_NOTE
                 "(requires CFITSIO)\n" C_RST);
    printf(C_CMD "  iofits.savefits   " C_RST "Save FITS file " C_NOTE
                 "(requires CFITSIO)\n" C_RST);
    printf("\n");
    printf(C_HDR "Session Control:\n" C_RST);
    printf(C_CMD "  quit / exit       " C_RST "Exit the milk shell\n");
    printf(C_CMD "  @sessionlog on|off" C_RST "Enable/disable session log\n");
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
    printf("  " C_CMD "declare -p" C_RST "        List all bash variables\n");
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
    printf("  " C_CMD "@get f.fps.param" C_RST "  Read FPS parameter (or " C_CMD
           "@f.fps.param" C_RST ")\n");
    printf("  " C_CMD "@get q.NAME.prop" C_RST "  Read Sequencer property\n");
    printf("  " C_CMD "@set f.fps.p=val" C_RST "  Write FPS parameter (alias: " C_CMD
           "@fpsset fps p v" C_RST ")\n");
    printf("\n");
    printf(C_HDR "Milk Stream Metadata (@s.<name> shared / @l.<name> local):\n" C_RST);
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
    printf("  " C_CMD "@l.name.<attr>" C_RST "     Session-local image metadata\n");
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
    printf("  Tests: " C_NOTE "-eq -ne -gt -ge -lt -le" C_RST " (numeric), " C_NOTE "= !=" C_RST
           " (string)\n");
    printf("  File tests: " C_NOTE "-f" C_RST " (file), " C_NOTE "-d" C_RST " (dir), " C_NOTE
           "-e" C_RST " (exists)\n");
    printf("  Negate: " C_CMD "[ ! expr ]" C_RST " logical NOT\n");
    printf("  Extended: " C_CMD "[[$s =~ ^[0-9]+$]]" C_RST " regex\n");
    printf("\n");
    printf(C_HDR "Loops:\n" C_RST);
    printf("  " C_CMD "while [$n -lt 10]; do" C_RST " ... " C_CMD "done" C_RST "\n");
    printf("  " C_CMD "for x in a b c; do" C_RST " ... " C_CMD "done" C_RST "\n");
    printf("  " C_CMD "for ((i=0; i<10; i++)); do" C_RST " ... " C_CMD "done" C_RST
           "  ← C-style\n");
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
    printf(C_HDR "Stream & Process Events:\n" C_RST);
    printf("  " C_CMD "@waitfor s.<name> [N] [-t sec]" C_RST
           "  Wait for N new frames via SHM semaphore\n");
    printf("  " C_CMD "@waitfor p.<name> [STATE] [-t]" C_RST
           "  Wait for process state (IDLE/PAUSED/RUNNING)\n");
    printf("  " C_CMD "@on [opts] <t...> -- <cmd>" C_RST
           "      Reactive watcher (--once|--max-hz|--cooldown)\n");
    printf("\n");
    printf(C_HDR "Unified Event Wait:\n" C_RST);
    printf("  " C_CMD "@waitany [-t T] s.<s1> s.<s2>" C_RST
           "   Block until any listed stream updates\n");
    printf("\n");
    printf(C_HDR "Cleanup:\n" C_RST);
    printf("  " C_CMD "@defer <cmd...>" C_RST "     run at exit, last registered first\n");
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
    printf(C_CMD "  @savehistory <file> " C_RST "Save command history\n");
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
    printf(C_HDR "Stream & Image Metadata (@s.<name> shared / @l.<name> local):\n" C_RST);
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
    printf("  " C_CMD "@l.name.<attr>" C_RST "     Session-local image metadata\n");
    printf("\n");
    printf(C_HDR "Waiting for Resources:\n" C_RST);
    printf("  " C_CMD "@waitfor s.<s> [N] [-t T]" C_RST "  Wait for N new frames or stream init\n");
    printf("  " C_CMD "@waitfor f.<f> [-t T]    " C_RST "  Wait for FPS SHM initialization\n");
    printf("  " C_CMD "@waitfor p.<p> [ST] [-t] " C_RST "  Wait for process state (IDLE/PAUSED)\n");
    printf("  " C_CMD "@waitany [-t T] s.<s1>..." C_RST "  Wait until any listed stream updates\n");
    printf("  " C_CMD "@on [opts] <t...> -- cmd " C_RST "  Reactive watcher on stream/FPS/proc\n");
    printf("\n");
    printf(C_HDR "Unified Entity Discovery & Introspection:\n" C_RST);
    printf("  " C_CMD "@ls [--json|-1] [pat...] " C_RST
           "  List streams (s.), local (l.), FPS (f.),\n");
    printf("                           "
           "  processes (p.), and sequencers (q.)\n");
    printf("  " C_CMD "@fpslist [--json] [pat]   " C_RST "  List FPS instances\n");
    printf("  " C_CMD "@streamlist [--json] [pat]" C_RST "  List shared-memory streams\n");
    printf("  " C_CMD "@proclist [--json]        " C_RST "  List active processes\n");
    printf("\n");
    printf(C_HDR "FPS Parameters:\n" C_RST);
    printf("  " C_CMD "@get f.fps.param " C_RST "Read FPS parameter value (or " C_CMD
           "@f.fps.param" C_RST ")\n");
    printf("  " C_CMD "@set f.fps.p=val " C_RST "Write FPS parameter (alias: " C_CMD
           "@fpsset fps p v" C_RST ")\n");
    printf("  " C_CMD "@fparam <fpsname>" C_RST "Interactive FPS parameter editor\n");
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
/* Topic: bash — bash-enhanced mode (source milk.bash) & @ syntax     */
/* ------------------------------------------------------------------ */

/**
 * @brief Print detailed help for bash-enhanced syntax (@hb / help bash).
 */
void help_topic_bash(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "        BASH-ENHANCED MODE & @ SYNTAX  (@hb / bash)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "1. Activation & Session Control (Standard Bash Shell):\n" C_RST);
    printf("  " C_CMD "source milk.bash            " C_RST
           "Enable milk in current bash shell ([milk] prompt)\n");
    printf("  " C_CMD "MILK_QUIET=1 source milk.bash" C_RST
           " Load silently (e.g. in ~/.bashrc or scripts)\n");
    printf("  " C_CMD "MILKCLI_SIGIL=, source ...  " C_RST
           "Use custom sigil instead of '@' (default: @)\n");
    printf("  " C_CMD "@unload                     " C_RST
           "Unload milk.bash, remove @-commands, restore PS1\n");
    printf("\n");
    printf(C_HDR "2. Calling Milk Commands, Batching & Stream Modifiers:\n" C_RST);
    printf("  " C_CMD "@<cmd> [args...]            " C_RST "Run any milk command (e.g. " C_CMD
           "@mk2Dim s.im1 128 128" C_RST ")\n");
    printf("  " C_CMD "@<mod>.<cmd> [args...]      " C_RST "Module-qualified call (e.g. " C_CMD
           "@mem.listim" C_RST ")\n");
    printf("  " C_CMD "@<cmd> --async [args...]    " C_RST
           "Dispatch command asynchronously in background\n");
    printf("  " C_CMD "@batch \"c1; c2\"             " C_RST
           "Run multi-command block or heredoc in 1 round trip\n");
    printf("  " C_CMD "@pipe \"c1 |> c2\"            " C_RST
           "Run |> pipeline in 1 round trip (chains '_' image)\n");
    printf("  " C_CMD "@hc [--sig|-s] <cmd> [pos]  " C_RST
           "Show command help or compact CLICMDARGDEF signature\n");
    printf("  " C_CMD "s.<name>  /  l.<name>       " C_RST
           "Force shared (/dev/shm) or local memory (also " C_CMD "@S: / @L:" C_RST ")\n");
    printf("  " C_CMD "@streamcat [-a arr] <s>     " C_RST
           "Bind pixels to array or stdout (or " C_CMD "| @streamwrite" C_RST ")\n");
    printf("\n");
    printf(C_HDR "3. Core @ Primitives (milk.bash & milk-cli):\n" C_RST);
    printf("  " C_CMD "@ [-v var] '<expr>'         " C_RST
           "Vectorized image/scalar calculator (alias: " C_CMD "@calc" C_RST ")\n");
    printf("    " C_NOTE "Examples:" C_RST " " C_CMD "@ 'im1 = 10; im2 = sqrt(im1) + 2'" C_RST
           "\n");
    printf("              " C_CMD "@ 'im1[0:63, 0:63] = im2 * 0.5'" C_RST "\n");
    printf("              " C_CMD "@ -v rms 'irms(im1)'" C_RST "  |  " C_CMD
           "@calc -v ptv 'imax(im1)-imin(im1)'" C_RST "\n");
    printf("  " C_CMD "@ls [-a arr|--json|-1] [pat]" C_RST
           "Unified table/discovery across s., l., f., p., q.\n");
    printf("  " C_CMD "@get [-v|-a|-A|--json] <sp> " C_RST
           "Read/discover stream (s.), local (l.), FPS (f.), proc (p.), seq (q.)\n");
    printf("    " C_NOTE "Examples:" C_RST " " C_CMD "@get -v nx s.im1.xsize" C_RST "  |  " C_CMD
           "@get -a all 's.*'" C_RST "\n");
    printf("              " C_CMD "@get -A map 's.dm*.xsize'" C_RST "  |  " C_CMD
           "@get --json s.im1" C_RST "\n");
    printf("  " C_CMD "@set <s1>=<v1> [<s2>=<v2>]" C_RST
           "  Write 1 or more FPS params/stream keywords/proc ctrl (globs)\n");
    printf("    " C_NOTE "Examples:" C_RST " " C_CMD "@set f.ao.gain=0.25 f.ao.mult=0.99" C_RST
           "  |  " C_CMD "@set 'f.dm*.xsize' = 128" C_RST "\n");
    printf("  " C_CMD "@snapshot [-o f] [pat...]   " C_RST
           "Save FPS parameter & stream keyword state snapshot\n");
    printf("  " C_CMD "@diff <snap1> [snap2]       " C_RST
           "Diff saved snapshot against live /dev/shm or 2nd snapshot\n");
    printf("  " C_CMD "@restore [-n] <snap> [pat]  " C_RST
           "Restore FPS/stream state from snapshot (" C_CMD "-n" C_RST " = dry-run)\n");
    printf("  " C_CMD "@assert <expr|flags> [msg]  " C_RST
           "Assert math relation or stream/loop health (" C_CMD "--running|--fps-min" C_RST ")\n");
    printf("  " C_CMD "@prompt [watch|preset|...]  " C_RST
           "Configure zero-subshell live telemetry prompt (PS1)\n");
    printf("  " C_CMD "@exists <entity>            " C_RST
           "Exit 0 if stream/local/FPS/proc exists (" C_CMD "s.im1, l.im2, f.ao" C_RST ")\n");
    printf("  " C_CMD "@running <entity>           " C_RST
           "Exit 0 if FPS/proc is actively running\n");
    printf("  " C_CMD "@test '<expr>'              " C_RST "Boolean math/entity test (e.g. " C_CMD
           "if @test 'irms(im1) < 0.5'" C_RST ")\n");
    printf("  " C_CMD "@waitfor <tgt> [N] [-t sec] " C_RST
           "Wait for N new frames or proc state (" C_CMD "@waitfor s.im1 5 -t 2" C_RST ")\n");
    printf("  " C_CMD "@waitany <s1> <s2> [-t sec] " C_RST
           "Wait until any listed stream updates\n");
    printf("  " C_CMD "@grab [--fits f] <s> [N] [o]" C_RST
           "Capture 1/N frames to image/cube/FITS (" C_CMD "--timestamps" C_RST ")\n");
    printf("  " C_CMD "@replay [--fps hz] <in> <s> " C_RST
           "Replay 2D/3D cube or FITS file into SHM stream " C_CMD "<s>" C_RST "\n");
    printf("  " C_CMD "@on [opts] <t...> -- <cmd>  " C_RST "Reactive watcher (" C_CMD
           "--once|--max-hz|--cooldown|--stop" C_RST ")\n");
    printf("  " C_CMD "@defer <cmd...>             " C_RST
           "Register LIFO cleanup command on shell/script EXIT\n");
    printf("  " C_CMD "@source <file.milk> [args]  " C_RST
           "Translate .milk shorthand script and source in bash\n");
    printf("  " C_CMD "@server [subcommand ...]    " C_RST
           "Inspect, start, switch, or stop milk servers (" C_CMD "status|list|use|stop" C_RST
           ")\n");
    printf("  " C_CMD "@log [pattern|options]      " C_RST
           "Inspect/search timestamped server command & output log (" C_CMD "-e|--slow|-f" C_RST
           ")\n");
    printf("\n");
    printf(C_HDR "4. milk-cli Shorthands (Interactive REPL & @source <file.milk>):\n" C_RST);
    printf("  " C_CMD "@s.im1.xsize / @f.ao.gain   " C_RST "Inline property expansion (e.g. " C_CMD
           "echo \"nx=@s.im1.xsize\"" C_RST ")\n");
    printf("  " C_CMD "@f.ao.gain=0.25             " C_RST "Direct parameter/keyword assignment\n");
    printf("  " C_CMD "@f.ao.runstart / @p.ao.pause" C_RST
           "Direct FPS / processinfo action invocation\n");
    printf("  " C_CMD "@{imean(im1)}               " C_RST
           "Inline calculator (bash 5.3: no fork; older bash: $(...))\n");
    printf("  " C_CMD "@ im1[0:9, 0:9] = 10        " C_RST
           "Unquoted calculator expression statement\n");
    printf("\n");
    printf("Full reference: docs/guide/scripting.md (installed docs or the milk repository)\n\n");
}


/* ------------------------------------------------------------------ */
/* Pillar Topics: streams, fps, procinfo, seq                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for ImageStreamIO streams (@h streams / @h stream).
 */
static void help_topic_streams(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "       IMAGESTREAMIO SHARED MEMORY STREAMS (streams)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "1. Stream Namespaces, Modifiers & Slicing:\n" C_RST);
    printf("  " C_CMD "s.<name>  /  @S:<name> " C_RST
           "Force shared memory (/dev/shm/<name>.im.shm)\n");
    printf("  " C_CMD "l.<name>  /  @L:<name> " C_RST "Force session-local memory\n");
    printf("  " C_CMD "@E:<name> /  @N:<name> " C_RST
           "Require existing stream / require new stream\n");
    printf("  " C_CMD "@F:<name>              " C_RST "Auto-load <name>.fits on demand\n");
    printf("  " C_CMD "@TF32:<name> / @TU16:.." C_RST
           "Explicit datatype (F32, F64, U8..U64, I8..I64)\n");
    printf("  " C_CMD "<name>[x0:x1, y0:y1]   " C_RST
           "Sub-region / slice view in expressions & commands\n");
    printf("\n");
    printf(C_HDR "2. Session @ Primitives for Streams & Local Images:\n" C_RST);
    printf("  " C_CMD "@ls s.*  /  @ls l.*    " C_RST
           "List shared-memory streams or session-local images\n");
    printf("  " C_CMD "@get s.<name>.<attr>   " C_RST
           "Read metadata: xsize, ysize, zsize, naxis,\n");
    printf("                         cnt0, cnt1, dtype, location, shared, inode,\n");
    printf("                         fps, age, owner_pid, summary, json, kw.<KEY>\n");
    printf("  " C_CMD "@get l.<name>.<attr>   " C_RST
           "Read session-local image metadata or keywords\n");
    printf("  " C_CMD "@set s.<glob>.kw.<K>=V " C_RST
           "Create/update FITS keyword (supports s. and l. globs)\n");
    printf("  " C_CMD "@assert [flags] s.<n>  " C_RST
           "Assert stream health (--running, --fps-min, --max-age, --dim)\n");
    printf("  " C_CMD "@exists [-r] s.<name>  " C_RST
           "Check if stream/image exists (or writer is alive)\n");
    printf("  " C_CMD "@waitfor s.<name> [N]  " C_RST
           "Wait for N new frames (-t <sec>, --cnt <target>)\n");
    printf("  " C_CMD "@grab [--fits f] <s> N " C_RST
           "Capture 1/N frames to cube or FITS (--timestamps)\n");
    printf("  " C_CMD "@replay [--fps hz] c s " C_RST
           "Replay 3D cube or FITS file into SHM stream <s>\n");
    printf("  " C_CMD "@streamcat [-a arr] <s>" C_RST
           "Bind stream pixels to Bash array or pipe to stdout\n");
    printf("  " C_CMD "@streamwrite <s> [W [H]] -" C_RST
           "Publish pixels from stdin ('-') or arguments to SHM stream\n");
    printf("\n");
    printf(C_HDR "3. Standalone OS Tools:\n" C_RST);
    printf("  " C_CMD "milk-streamCTRL        " C_RST "Interactive TUI stream monitor\n");
    printf("  " C_CMD "milk-termview <stream> " C_RST "Live terminal pixel viewer\n");
    printf("  " C_CMD "milk-stream-info <s>   " C_RST
           "Detailed stream lineage & reader/writer PIDs\n");
    printf("  " C_CMD "milk-stream-help       " C_RST "Full C API & architecture guide\n");
    printf("\n");
}

/**
 * @brief Print help for Function Parameter Structure (@h fps).
 */
static void help_topic_fps(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "       FUNCTION PARAMETER STRUCTURE (fps)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "1. Session @ Primitives for FPS:\n" C_RST);
    printf("  " C_CMD "@ls f.*                " C_RST
           "List active FPS instances with status, PIDs, and param counts\n");
    printf("  " C_CMD "@get f.<name>.<param>  " C_RST
           "Read any parameter value from /dev/shm/<name>.fps.shm\n");
    printf("  " C_CMD "@get f.<name>.<attr>   " C_RST
           "Read metadata: status (IDLE/CONF/RUN), conf, run, pid, nbparam, list\n");
    printf("  " C_CMD "@set f.<glob>.<p> = V  " C_RST
           "Set parameter(s) (supports globs & multi-assignment)\n");
    printf("  " C_CMD "@set f.<name>.conf = X " C_RST
           "Control CONF process: start | stop | step (or @f.<name>.confstart)\n");
    printf("  " C_CMD "@set f.<name>.run  = X " C_RST
           "Control RUN process:  start | stop (or @f.<name>.runstart)\n");
    printf("  " C_CMD "@snapshot / @diff / @restore" C_RST
           " Save, compare, or restore FPS/stream state snapshots\n");
    printf("  " C_CMD "@exists [-r] f.<name>  " C_RST
           "Check if FPS SHM exists (or run process is alive)\n");
    printf("\n");
    printf(C_HDR "2. Standalone OS Tools:\n" C_RST);
    printf("  " C_CMD "milk-fpsCTRL           " C_RST
           "Interactive TUI parameter editor & launcher\n");
    printf("  " C_CMD "milk-fpsexec-list      " C_RST
           "List all standalone V2 compute unit binaries\n");
    printf("  " C_CMD "milk-fps-list          " C_RST
           "List active FPS instances from outside a session\n");
    printf("  " C_CMD "milk-fps-help          " C_RST "Full FPS architecture & C API guide\n");
    printf("\n");
}

/**
 * @brief Print help for Processinfo (@h procinfo / @h proc).
 */
static void help_topic_procinfo(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "            PROCESSINFO TELEMETRY (procinfo)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "1. Session @ Primitives for Processinfo:\n" C_RST);
    printf("  " C_CMD "@ls p.*                " C_RST
           "List active processes with PID, status, loopcnt, and rate\n");
    printf("  " C_CMD "@get p.<name>.<attr>   " C_RST
           "Read telemetry: PID, loopcnt, loopcntMax, cnt0,\n");
    printf("                         status (RUNNING/PAUSED/STEPPING/STOPPED/CRASHED),\n");
    printf("                         ctrl, ctrlval, triggermode, triggerstream,\n");
    printf("                         dt_median_ns, fps_hz, msg\n");
    printf("  " C_CMD "@set p.<name>.ctrl = X " C_RST
           "Control loop: run | pause | step | stop (or 0..3)\n");
    printf("  " C_CMD "@p.<name>.pause|.resume|.step|.stop" C_RST
           " Action-property shorthand for loop control\n");
    printf("  " C_CMD "@set p.<name>.triggermode=M" C_RST
           " Set trigger mode (IMMEDIATE, SEMAPHORE, CNT0, DELAY...)\n");
    printf("  " C_CMD "@waitfor p.<name> [ST] " C_RST
           "Wait until process reaches state (IDLE, PAUSED, RUNNING)\n");
    printf("  " C_CMD "@running p.<name>      " C_RST
           "Exit 0 if processinfo exists and PID is alive\n");
    printf("\n");
    printf(C_HDR "2. Standalone OS Tools:\n" C_RST);
    printf("  " C_CMD "milk-CTRL              " C_RST
           "Unified TUI dashboard (streams, procs, FPS, graph)\n");
    printf("  " C_CMD "milk-procCTRL          " C_RST "Dedicated TUI processinfo controller\n");
    printf("  " C_CMD "milk-procinfo-info <p> " C_RST
           "Detailed processinfo timing & CPU affinity report\n");
    printf("  " C_CMD "milk-procinfo-help     " C_RST "Full processinfo C API & macro guide\n");
    printf("\n");
}

/**
 * @brief Print help for Sequencer (@h seq).
 */
static void help_topic_seq(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "              FPS SEQUENCER ENGINE (seq)\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "1. Session @ Primitives for Sequencer:\n" C_RST);
    printf("  " C_CMD "@get q.<id>.<attr>     " C_RST
           "Read state, active_step, progress, step_count, pid, name\n");
    printf("  " C_CMD "@set q.<id>.action = X " C_RST
           "Control running sequence: pause | resume | abort\n");
    printf("  " C_CMD "@running q.<id>        " C_RST
           "Exit 0 if sequencer <id> is actively running\n");
    printf("\n");
    printf(C_HDR "2. Standalone OS Tools:\n" C_RST);
    printf("  " C_CMD "milk-seq-run <file>    " C_RST "Execute a sequence schedule file\n");
    printf("  " C_CMD "milk-seq-TUI           " C_RST "Interactive TUI sequencer monitor\n");
    printf("  " C_CMD "milk-seq-status        " C_RST "Inspect active sequencers from OS shell\n");
    printf("  " C_CMD "milk-seq-help          " C_RST "Full sequencer DSL & schedule reference\n");
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

    if (strcmp(topic, "bash") == 0 || strcmp(topic, "hb") == 0)
    {
        help_topic_bash();
    }
    else if (strcmp(topic, "stream") == 0 || strcmp(topic, "streams") == 0)
    {
        help_topic_streams();
    }
    else if (strcmp(topic, "fps") == 0)
    {
        help_topic_fps();
    }
    else if (strcmp(topic, "procinfo") == 0 || strcmp(topic, "proc") == 0)
    {
        help_topic_procinfo();
    }
    else if (strcmp(topic, "seq") == 0 || strcmp(topic, "sequencer") == 0)
    {
        help_topic_seq();
    }
    else if (strcmp(topic, "cmdopts") == 0)
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
 * @brief Print only the available-topics list.
 *
 * Shown both in the full help index and as a concise
 * hint when an unknown topic is supplied.
 */
void print_help_topic_list(void)
{
    printf(C_HDR "Available help topics (@h <topic>):\n" C_RST);
    printf("  " C_CMD "bash        " C_RST "Bash-enhanced mode (source milk.bash) & @ syntax "
           "(" C_CMD "@hb" C_RST ")\n");
    printf("  " C_CMD "streams     " C_RST
           "ImageStreamIO shared-memory streams, @S:/@L:, slicing\n");
    printf("  " C_CMD "fps         " C_RST "Function Parameter Structure (@get/@set f.<name>.*)\n");
    printf("  " C_CMD "procinfo    " C_RST
           "Processinfo telemetry & loop control (@get/@set p.*)\n");
    printf("  " C_CMD "seq         " C_RST "FPS Sequencer scheduling & control (@get/@set q.*)\n");
    printf("  " C_CMD "milk        " C_RST
           "Overview of streams, FPS, procinfo, and TUI monitors\n");
    printf("  " C_CMD "variables   " C_RST "Variables, arrays, arithmetic, @s./@f. properties\n");
    printf("  " C_CMD "flowcontrol " C_RST "if/while/for/case/functions and event triggers\n");
    printf("  " C_CMD "scripting   " C_RST "Script files, I/O, pipes, builtins\n");
    printf("  " C_CMD "commands    " C_RST "Built-in CLI commands (ci, listim, loadfits...)\n");
    printf("  " C_CMD "syntax      " C_RST "Interactive REPL features, tab completion, piping\n");
    printf("  " C_CMD "cmdopts     " C_RST "milk-cli command-line flags (-h, -s, -n...)\n");
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
    printf("    {\"name\": \"bash\", \"description\": \"Bash-enhanced mode (source milk.bash) & @ "
           "syntax (@hb)\"},\n");
    printf(
        "    {\"name\": \"cmdopts\", \"description\": \"Command-line flags (-h, -s, -n...)\"},\n");
    printf("    {\"name\": \"syntax\", \"description\": \"Syntax, tab completion, piping\"},\n");
    printf(
        "    {\"name\": \"commands\", \"description\": \"Built-in CLI commands (?, cmd?...)\"},\n");
    printf("    {\"name\": \"variables\", \"description\": \"Variables, arrays, arithmetic\"},\n");
    printf("    {\"name\": \"flowcontrol\", \"description\": \"if/while/for/case/function\"},\n");
    printf("    {\"name\": \"scripting\", \"description\": \"Script files, I/O, builtins\"},\n");
    printf("    {\"name\": \"milk\", \"description\": \"Streams, FPS, milk-specific\"}\n");
    printf("  ],\n");
    printf("  \"quick_reference\": [\n");
    printf("    {\"command\": \"@h / h\", \"args\": \"[topic|cmd|mod]\", \"description\": \"Brief "
           "top-level help or topic/command help\"},\n");
    printf("    {\"command\": \"@hb / hb\", \"args\": \"\", \"description\": \"Detailed help for "
           "bash-enhanced @ syntax\"},\n");
    printf("    {\"command\": \"@hc / cmd?\", \"args\": \"[name]\", \"description\": \"Help for a "
           "specific command\"},\n");
    printf("    {\"command\": \"@hm / m?\", \"args\": \"[module]\", \"description\": \"List "
           "commands in a module\"},\n");
    printf("    {\"command\": \"@hf / fhelp\", \"args\": \"[pattern]\", \"description\": "
           "\"Interactive fuzzy command search\"},\n");
    printf("    {\"command\": \"@unload / exit\", \"args\": \"\", \"description\": \"Unload "
           "milk.bash or exit milk-cli\"}\n");
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
    printf("TOPIC\tbash\t\tBash-enhanced mode (source milk.bash) & @ syntax (@hb)\n");
    printf("TOPIC\tcmdopts\t\tCommand-line flags (-h, -s, -n...)\n");
    printf("TOPIC\tsyntax\t\tSyntax, tab completion, piping\n");
    printf("TOPIC\tcommands\t\tBuilt-in CLI commands (?, cmd?...)\n");
    printf("TOPIC\tvariables\t\tVariables, arrays, arithmetic\n");
    printf("TOPIC\tflowcontrol\t\tif/while/for/case/function\n");
    printf("TOPIC\tscripting\t\tScript files, I/O, builtins\n");
    printf("TOPIC\tmilk\t\tStreams, FPS, milk-specific\n");
    printf("QUICKREF\t@h\t[topic|cmd|mod]\tBrief top-level help\n");
    printf("QUICKREF\t@hb\t\tDetailed help for bash-enhanced @ syntax\n");
    printf("QUICKREF\t@hc / cmd?\t[name]\tHelp for a specific command\n");
    printf("QUICKREF\t@hm / m?\t[module]\tList commands in a module\n");
    printf("QUICKREF\t@hf / fhelp\t[pattern]\tInteractive fuzzy command search\n");
    printf("QUICKREF\t@unload / exit\t\tUnload milk.bash or exit the milk shell\n");
}

/**
 * @brief Print brief top-level help (@h / h / help).
 *
 * Concise overview of milk interactive modes (milk-cli and bash-enhanced
 * mode via `source milk.bash`), essential @-commands, and help topics.
 */
void print_milk_top_help(void)
{
    if (help_format_mode == 1)
    {
        print_milk_cli_help_json();
        return;
    }
    else if (help_format_mode == 2)
    {
        print_milk_cli_help_porcelain();
        return;
    }

    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                   milk \xe2\x80\x94 QUICK HELP\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Bash-Enhanced Mode & Interactive CLI:\n" C_RST);
    printf("  " C_CMD "source milk.bash" C_RST "   Enable milk inside any bash terminal (" C_TITLE
           "[milk]" C_RST " prompt).\n");
    printf("                     Exposes all milk commands & primitives with the " C_CMD "@" C_RST
           " prefix.\n");
    printf("  " C_CMD "milk-cli" C_RST
           "           Launch standalone interactive milk REPL & script engine.\n");
    printf("\n");
    printf(C_HDR "Essential Help & Discovery Commands:\n" C_RST);
    printf("  " C_CMD "@h [topic|cmd|mod]" C_RST
           " Show this help, or help for a topic, command, or module\n");
    printf("  " C_CMD "@hb               " C_RST " Detailed help for bash-enhanced " C_CMD "@" C_RST
           " syntax & primitives\n");
    printf("  " C_CMD "@hm [module]      " C_RST
           " List loaded modules, or list commands in <module>\n");
    printf("  " C_CMD "@hc <command>     " C_RST
           " Show syntax and argument details for <command>\n");
    printf("  " C_CMD "@hf [pattern]     " C_RST
           " Interactive fuzzy search (or regex search) over commands\n");
    printf("  " C_CMD "@unload           " C_RST
           " Unload milk.bash integration from current bash terminal\n");
    printf("\n");
    print_help_topic_list();
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
    if (help_format_mode == 1)
    {
        print_milk_cli_help_json();
        return;
    }
    else if (help_format_mode == 2)
    {
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
    printf("  " C_CMD "source milk.bash" C_RST
           "              Enable milk @-commands in current bash shell\n");
    printf("  " C_CMD "milk-cli" C_RST "                      Start interactive shell\n");
    printf("  " C_CMD "milk-cli -c <cmd>" C_RST "          Execute one command and exit\n");
    printf("  " C_CMD "milk-cli -s <file>" C_RST "         Run a script file, then exit\n");
    printf("  " C_CMD "echo cmds | milk-cli" C_RST "       Read commands from stdin\n");
    printf("  " C_CMD "milk-script <file>" C_RST "         Run a script non-interactively\n");
    printf("\n");
    printf(C_BOLD "DESCRIPTION\n" C_RST);
    printf("  milk-cli is the interactive command-line interface to\n");
    printf("  the milk framework. It provides:\n");
    printf("    \xe2\x80\xa2 A REPL with tab-completion, history and fuzzy search\n");
    printf("    \xe2\x80\xa2 Bash-enhanced mode (source milk.bash) with @-commands\n");
    printf("    \xe2\x80\xa2 Access to all loaded milk/cacao module commands\n");
    printf("    \xe2\x80\xa2 Real-time stream, FPS, and process management\n");
    printf("\n");
    printf(C_BOLD "OPTIONS\n" C_RST);
    printf(C_CMD "  -h, --help           " C_RST "Print this help and exit\n");
    printf(C_CMD "  -v, --version        " C_RST "Print version and exit\n");
    printf(C_CMD "  -i, --info           " C_RST "Print version, paths, and build info\n");
    printf(C_CMD "  -c <cmd>             " C_RST "Execute single command and exit\n");
    printf(C_CMD "  -s <file>            " C_RST
                 "Run a script file, then exit (fed via fifo with -f/-F)\n");
    printf(C_CMD "  -n <name>            " C_RST "Set process name\n");
    printf(C_CMD "  -p <priority>        " C_RST "Set real-time priority (0"
                 "\xe2\x80\x93"
                 "99)\n");
    printf(C_CMD "  -e, --errorexit      " C_RST "Exit on first error\n");
    printf(C_CMD "  -E, --echo-input     " C_RST "Echo each input line (colored)\n");
    printf(C_CMD "  -d <level>           " C_RST "Set debug level at startup\n");
    printf(C_CMD "  -o, --overwrite      " C_RST "Overwrite existing FITS files " C_NOTE
                 "(caution)\n" C_RST);
    printf(C_CMD "  -f, --fifoflag       " C_RST "Enable default FIFO input\n");
    printf(C_CMD "  -F <fifoname>        " C_RST "Specify custom FIFO name\n");
    printf(C_CMD "  -m <monitor>         " C_RST "Set memory monitor stream\n");
    printf(C_CMD "  -Z, --idle           " C_RST "Only run when display is idle\n");
    printf(C_CMD "  -A, --autocomplete   " C_RST "Enable inline autocomplete " C_NOTE
                 "(on by default)\n" C_RST);
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
    printf("available " C_BOLD "within" C_RST " the milk-cli shell or bash-enhanced mode:\n");
    printf("\n");
    print_help_topic_list();
    printf(C_NOTE "From the OS shell:\n" C_RST);
    printf("  $ " C_CMD "milk-cli-help <topic>\n" C_RST);
    printf(C_NOTE "From within milk-cli or bash-enhanced mode:\n" C_RST);
    printf("  > " C_CMD "@h <topic>" C_RST "   (or " C_CMD "@hb" C_RST " for bash syntax)\n");
    printf("\n");
    printf(C_HDR "Quick reference (inside milk-cli / bash):\n" C_RST);
    printf("  " C_CMD "@hb / hb      " C_RST "Detailed help for bash-enhanced @ syntax\n");
    printf("  " C_CMD "@hc / cmd?    " C_RST "Help for a specific command\n");
    printf("  " C_CMD "@hm / m?      " C_RST "List commands in a module\n");
    printf("  " C_CMD "@hf / fhelp   " C_RST "Interactive fuzzy command search\n");
    printf("  " C_CMD "@unload / exit" C_RST "Unload milk.bash / exit the milk shell\n");
    printf("\n");
}
