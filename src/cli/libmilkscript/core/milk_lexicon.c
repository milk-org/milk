// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_lexicon.c
 * @brief Language keywords and bash words (see milk_lexicon.h).
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "milk_lexicon.h"

#define CLI_ONLY MILK_KW_CLI
#define NO_SCRIPT (MILK_KW_CLI | MILK_KW_BASH)

static const milk_keyword_t keywords[] = {
    /* values and expressions */
    { "calc", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@calc [-v <var>[,<var>...] | -a <arr>] \"<expr>[; <expr>...]\"",
      "Evaluate image or scalar expressions",
      "  -v a,b    bind the values of the last statements (an assignment's value is the assigned "
      "one)\n"
      "  -a <arr>  bind the pixels of an image result into an indexed array\n"
      "  Images, calculator variables, the bash variables the text names and entity paths "
      "(s.cam.xsize)\n"
      "  are visible. Slicing (im[x0:x1,y0:y1]), reductions and per-pixel math.",
      "  @calc \"out = im1 + 2.0 * im2\"\n"
      "  @calc -v rms,ptv \"istd(im1); imax(im1) - imin(im1)\"\n"
      "  @calc -a row \"im1[*, 10]\"" },
    { "test", MILK_KW_BUILTIN, MILK_KW_ALL, 1, "@test \"<boolean-expr>\"",
      "Evaluate a boolean expression: exit status 0 if true, 1 if false",
      "  Made for bash conditionals (if, while, &&); no subshell.",
      "  if @test \"imax(im1) > 100.0\"; then echo saturated; fi\n"
      "  while @test \"s.cam0.cnt0 < 50\"; do @sleep 0.01; done" },
    { "assert", MILK_KW_SERVER, MILK_KW_ALL, 1,
      "@assert [<expr> | [-e] [-r] [--fps-min <hz>] [--max-age <s>] [--cnt-min <N>] [--dim "
      "<WxH[xD]>] "
      "<entity>] [\"message\"]",
      "Check an expression or a stream/FPS/process health invariant; fail loudly",
      "  Expression mode: a failed relation prints both sides and returns 1.\n"
      "  Entity mode: existence, running owner, frame rate, age, cnt0 or geometry.",
      "  @assert \"istd(wfs) < 0.5\" \"WFS residual diverged\"\n"
      "  @assert --running --fps-min 500 s.wfs_cam \"camera degraded\"" },
    { "get", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@get [-v <var>|-a <arr>|-A <assoc>|--json] <spec> | @get [<var>=]<spec>...",
      "Read stream, local image, FPS, process or sequencer properties",
      "  -v <var>    bind a scalar into <var>; -a/-A bind lines into an indexed/associative array\n"
      "  <var>=<spec>  bind several in one call; several plain specs print one value per line\n"
      "  --json      all properties as JSON\n"
      "  Globs: s.*, s.<glob>.<attr>.  Streams: xsize ysize zsize naxis cnt0 cnt1 type shared age "
      "kw.<KEY>,\n  min max mean std var tot count, stats (all of them, for -A).\n"
      "  FPS: any parameter, status conf run.  Processes: status loopcnt ctrl pid.  Sequencers: "
      "status.",
      "  @get -v nx s.cam0.xsize\n"
      "  @get nx=s.cam0.xsize ny=s.cam0.ysize\n"
      "  @get -A st s.cam0.stats\n"
      "  @get -A widths 's.dm*.xsize'\n"
      "  @get --json s.cam0" },
    { "set", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@set [--restore] <spec>[=]<value> [<spec>=<value> ...] [comment] | @set --tui [<fps>]",
      "Write stream keywords, FPS parameters, process controls; run FPS/process actions",
      "  Several assignments per call; entity and parameter globs.\n"
      "  --restore  set the old values back when the calling function or @source'd file returns\n"
      "  s.<glob>.kw.<KEY>=v   FITS keyword    f.<glob>.<param>=v   FPS parameter\n"
      "  f.<name>.conf=start|stop|step, f.<name>.run=start|stop (:tmux)   p.<name>.ctrl=pause ...\n"
      "  --tui  browse and edit FPS parameters in a fuzzy finder (milk-cli REPL)",
      "  @set f.loop.gain=0.25 f.loop.mult=0.99\n"
      "  @set --restore f.loop.gain=0\n"
      "  @set 's.cam*.kw.OBSERVER'=Subaru\n"
      "  @set f.loop.run=start:tmux" },
    { "exists", MILK_KW_BUILTIN, MILK_KW_ALL, 1, "@exists [-r] <entity>",
      "Exit status 0 if the stream, local image, FPS, process or sequencer exists",
      "  -r  also require its owner process to be alive (same as @running)",
      "  @exists s.cam0 && echo yes" },
    { "running", MILK_KW_SHELL, MILK_KW_ALL, 1, "@running <entity>",
      "Exit status 0 if the entity exists and its owner process is alive", "",
      "  @running f.loop || @set f.loop.run=start" },
    { "ls", MILK_KW_BUILTIN, MILK_KW_ALL, 1, "@ls [-l|-1|-a <arr>|--json|--local] [<pattern>...]",
      "List streams, local images, FPS, processes and sequencers",
      "  Patterns select a namespace: 's.*' streams, 'f.*' FPS, 'p.*' processes, 'q.*' sequencers.",
      "  @ls\n"
      "  @ls 's.cam*'\n"
      "  @ls -a names 'f.*'" },
    /* events and time */
    { "waitfor", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@waitfor <entity|expr> [N] [-t <sec>] [--cnt <target>] | @waitfor [--next] \"<expr>\" [-t "
      "<sec>]",
      "Sleep until a stream posts N frames, a property has a value, or an expression is true",
      "  Sleeps on the stream semaphores (no CPU). Exit status 124 on timeout.\n"
      "  An expression is evaluated again on each frame of the streams it names (in the shell "
      "when\n"
      "  it is scalar); FPS, process and sequencer values in it are polled. --next: from the next\n"
      "  frame on, not now.",
      "  @waitfor s.cam0 10 -t 2\n"
      "  @waitfor f.loop.status=RUN -t 5\n"
      "  @waitfor \"imean(cam0) > 100\" -t 5" },
    { "waitany", MILK_KW_BUILTIN, MILK_KW_ALL, 1, "@waitany <s1> <s2> [...] [-t <sec>]",
      "Wait until any of the streams updates; its name goes to $MILK_TRIGGER",
      "  Exit status 124 on timeout.", "  @waitany s.cam0 s.cam1 -t 1 && echo \"$MILK_TRIGGER\"" },
    { "on", MILK_KW_SHELL, MILK_KW_ALL, 0,
      "@on [--once] [--scoped] [--max-hz <hz>] [--cooldown <s>] <t1> [t2...] [--] <cmd...> | @on "
      "[...] --when \"<expr>\" [--hold <s>] [--level] <cmd> | @on --every <s> <cmd> | @on "
      "--list|--stop <id>|--clear",
      "Run a command in the background each time a stream or entity updates",
      "  --once fires once; --max-hz/--cooldown throttle; $MILK_TRIGGER names the trigger.\n"
      "  --scoped stops the watcher when the calling function or @source'd file returns.\n"
      "  --when fires when the expression turns true (also if it is true at the start), then "
      "again\n"
      "  once it has been false; --hold <s>: only after it stayed true that long; --level: on\n"
      "  every frame while it is true. --every <s> runs the command periodically, without drift.",
      "  @on --max-hz 10 s.cam0 'info.imstats s.cam0'\n"
      "  @on --when 'imax(cam0) > 60000' --hold 0.5 -- echo saturated\n"
      "  @on --every 1 -- '@get s.cam0.cnt0 >> counts.txt'\n"
      "  @on --stop 1" },
    { "sleep", MILK_KW_BUILTIN, MILK_KW_ALL, 0,
      "@sleep <seconds> | @sleep --every <seconds>|--hz <rate> [--late <var>] <var>",
      "Sleep without starting a process (Ctrl-C interrupts it); pace a loop",
      "  --every/--hz  sleep to the deadline held in <var> (CLOCK_MONOTONIC ns; unset: one period\n"
      "                from now), then move it one period on: a loop runs at that rate, without "
      "drift\n"
      "  --late <var>  the ticks skipped by a late call (it returns at once instead of catching "
      "up)",
      "  @sleep 0.01\n"
      "  unset t; while :; do @sleep --hz 100 t; read_sensor; done" },
    /* data */
    { "grab", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@grab [-t <sec>] [--fits <out.fits>] [--timestamps <out.txt>] <stream> [N] [dst_img] | "
      "@grab --stdout [-a <arr>|--raw] <stream>",
      "Capture 1 or N consecutive frames into a local image or cube, or a FITS file",
      "  --stdout  write the pixels to stdout as text (--raw: binary), or bind them into an array",
      "  @grab -t 1 s.cam0 l.dark\n"
      "  @grab --fits telem.fits --timestamps telem.txt s.cam0 100\n"
      "  @grab --stdout --raw s.cam0 | python3 analyze.py" },
    { "replay", MILK_KW_SERVER, MILK_KW_ALL, 1,
      "@replay [--fps <hz>|--dt <sec>] [--count <N>] [--loop] <cube_or_fits> <dst_stream> | "
      "@replay --stdin [--raw] [--dtype <type>] <stream> [xsize [ysize [zsize]]]",
      "Play an image cube or FITS file into a stream at a given rate",
      "  Runs in a copy of the server: images it creates are not kept.\n"
      "  --stdin  write the values read from stdin (--raw: binary) into the stream, creating it if "
      "needed",
      "  @replay --fps 200 telem.fits cam0_sim\n"
      "  python3 gen.py | @replay --stdin --dtype F32 cam0 128 128" },
    { "snapshot", MILK_KW_SERVER, MILK_KW_ALL, 1, "@snapshot [-o <file>] [<pattern>...]",
      "Save FPS parameters and stream keywords", "",
      "  @snapshot -o base.snap 'f.loop*' 's.cam*'" },
    { "diff", MILK_KW_SERVER, MILK_KW_ALL, 1, "@diff <snap_a> [snap_b]",
      "Compare a snapshot with the live state or another snapshot (status 1 if they differ)", "",
      "  @diff base.snap" },
    { "restore", MILK_KW_SERVER, MILK_KW_ALL, 1,
      "@restore [-n] <snap_file> [<pattern>...] | @restore --undo [-n] [N]",
      "Apply a snapshot back to the live state (-n: dry run), or undo this shell's last @set calls",
      "  --undo [N]  set back the values the last N @set calls of this shell overwrote (FPS\n"
      "              parameters, stream keywords), where they still hold what was written; the\n"
      "              others are left as they are, with a warning (status 1). -n lists them.",
      "  @restore -n base.snap\n"
      "  @restore --undo 3" },
    { "pipe", MILK_KW_SERVER, MILK_KW_ALL, 1,
      "@pipe \"<cmd1> |> <cmd2> |> ...\" | @pipe \"<cmd1>; <cmd2>\"",
      "Run several milk commands and expressions in one request; '_' is the image the previous "
      "stage made",
      "",
      "  @pipe \"mem.mk2Dim p 32 32 |> _ = 4 |> q = _ * 3\"\n  @pipe \"mem.mk2Dim b1 32 32; b1 = "
      "5\"" },
    /* program and session */
    { "source", MILK_KW_SHELL, MILK_KW_ALL, 0, "@source [--once] <file.milk> [args...]",
      "Translate and run a milk script in the current shell",
      "  --once  skip a file this shell already sourced (by real path)",
      "  @source setup.milk cam0\n  @source --once lib.milk" },
    { "defer", MILK_KW_BUILTIN, MILK_KW_ALL, 0,
      "@defer [--exit] <command...> | @defer --restore <spec>...",
      "Run a command when the calling function or @source'd file returns (last registered first)",
      "  At the top level, or with --exit, the command runs when the shell or script exits.\n"
      "  One argument is a command string (as for trap); several are kept as quoted words.\n"
      "  --restore  read the values now, set them back when the scope ends: an entity path,\n"
      "             <entity>.* (all its parameters or keywords) or <ns>.<glob>.<attr>\n"
      "  Commands that Ctrl-C or a set -e failure skipped run before the next prompt, or at exit.",
      "  @defer mem.rm tmp_im\n"
      "  @defer --restore 'f.loop.*'\n"
      "  @defer --exit 'echo done'" },
    { "args", MILK_KW_BUILTIN, MILK_KW_ALL, 0,
      "@args \"<name>:<type>[=<default>][:<help>]\"... -- \"$@\"",
      "Declare the arguments of a function or script; they are checked like a module command's",
      "  Types: int uint float bool (1 or 0) str. Without a default a value is required. Values "
      "in\n"
      "  order or as .name=value, '.' keeps a default, -h prints the usage. The names become "
      "locals\n"
      "  of the function; after -h or a bad value the function returns (status 0 or 2). Commands\n"
      "  in share/milk/commands.d and ~/.config/milk/commands.d (*.bash) are module.cmd functions.",
      "  measure() { @args 'cam:str:camera stream' 'n:int=100:frames' -- \"$@\"; echo \"$cam $n\"; "
      "}\n"
      "  measure s.cam0 .n=50" },
    { "server", MILK_KW_SHELL, MILK_KW_ALL, 1,
      "@server [status|list|start [name]|use <name>|local|stop [name|--all]|log|attach]",
      "Show, start, switch to or stop milk-cli servers",
      "  local returns to the shell's own server (inside milk-cli: the session server).",
      "  @server use bench1\n"
      "  @server stop bench1" },
    { "module", MILK_KW_SERVER, MILK_KW_ALL, 0,
      "@module [list [--all]] | @module load <name|file.so> [--as <short>]",
      "List the loaded modules, or load one (commands of installed modules load on first use)",
      "  --all   also the installed modules not loaded yet\n"
      "  --as    register the module's commands under another short name",
      "  @module list\n  @module load milkfft" },
    { "config", MILK_KW_SHELL, MILK_KW_ALL, 0, "@config [<key> [<value>...]]",
      "Show or change a session setting",
      "  digits <N>               significant digits of printed results (all calc paths)\n"
      "  precision single|double  default floating-point precision of new images\n"
      "  procinfo on|off          processinfo telemetry for commands that support it\n"
      "  prompt ...               the prompt: milk.bash PS1 telemetry (watch, preset, ...) or the "
      "milk-cli\n"
      "                           REPL format (%u %h %d %t %n)\n"
      "  inline on|off            expand @{...} in lines typed at an interactive milk.bash prompt\n"
      "  synhl, syndiag, indent, showmatch on|off; sessionlog on|off|<file>   (milk-cli REPL)",
      "  @config digits 4\n  @config prompt watch f.loop.run\n  @config" },
    { "unload", MILK_KW_SHELL, MILK_KW_BASH, 0, "@unload",
      "Remove milk from this bash shell: stop its server and watchers, unset the functions", "",
      "  @unload" },
    /* introspection */
    { "log", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@log [-n <N>|-a] [-g <pat>] [-e] [--slow <ms>] [--since <dur>] [-f] [-s <name>] [--json] "
      "[--clear]",
      "Show the server's recent commands with timing and status", "", "  @log -n 20\n  @log -e" },
    { "help", MILK_KW_SERVER, MILK_KW_ALL, 0,
      "@help [<keyword|namespace|module|module.cmd|topic>] [--json|--porcelain] | @help -k <regex> "
      "| @help --tui",
      "Show help on keywords, entity namespaces, modules, commands and topics",
      "  Every command and keyword also answers -h/--help (@calc, @test, @assert: --help).\n"
      "  -k <regex>  search keywords and command descriptions\n"
      "  --tui       browse the help in a fuzzy finder (milk-cli REPL)\n"
      "  Topics: bash syntax variables flowcontrol scripting milk cmdopts readline streams fps "
      "procinfo seq",
      "  @help\n  @help iofits.loadfits\n  @help s\n  @help -k fits" },
    { "history", MILK_KW_SERVER, CLI_ONLY, 0,
      "@history [N] | search <text> | --global [N] | --local [N] | save <file> | --stats | --tui",
      "Show, search or save the milk-cli command history",
      "  --global  all sessions    --local  this session    --stats  command use counts\n"
      "  --tui     browse the history in a fuzzy finder",
      "  @history 20\n  @history search loadfits\n  @history save session.milk" },
    { NULL, MILK_KW_BUILTIN, 0, 0, NULL, NULL, NULL, NULL }
};

static const char *const bash_words[] = {
    "if",       "elif",    "else",    "fi",     "for",      "while",   "until",   "do",
    "done",     "case",    "esac",    "select", "function", ".",       "source",  "break",
    "continue", "return",  "true",    "false",  "exit",     "shift",   "set",     "export",
    "readonly", "local",   "declare", "let",    "eval",     "type",    "command", "trap",
    "time",     "timeout", "wait",    "printf", "echo",     "getopts", "mapfile", "alias",
    "unalias",  "pushd",   "popd",    "dirs",   "[[",       NULL
};

const milk_keyword_t *milk_lexicon_keywords(void)
{
    return keywords;
}

const milk_keyword_t *milk_lexicon_keyword(const char *name)
{
    if (name == NULL)
    {
        return NULL;
    }
    if (name[0] == '@')
    {
        name++;
    }
    for (const milk_keyword_t *k = keywords; k->name != NULL; k++)
    {
        if (strcmp(k->name, name) == 0)
        {
            return k;
        }
    }
    return NULL;
}

unsigned milk_lexicon_frontend(const char *engine)
{
    if (engine != NULL && strcmp(engine, "cli") == 0)
    {
        return MILK_KW_CLI;
    }
    if (engine != NULL && strcmp(engine, "script") == 0)
    {
        return MILK_KW_SCRIPT;
    }
    return MILK_KW_BASH;
}

const char *const *milk_lexicon_bash_words(void)
{
    return bash_words;
}

int milk_lexicon_help_arg(const milk_keyword_t *k, const char *arg)
{
    if (arg == NULL || arg[0] != '-')
    {
        return 0;
    }
    return strcmp(arg, "--help") == 0 ||
           (strcmp(arg, "-h") == 0 && strcmp(k->name, "calc") != 0 &&
            strcmp(k->name, "test") != 0 && strcmp(k->name, "assert") != 0);
}

void milk_lexicon_print_help(const milk_keyword_t *k, FILE *fp)
{
    fprintf(fp, "\n\033[32m@%s\033[0m [language keyword]\n\t\033[32m%s\n\033[0m", k->name,
            k->summary);
    fprintf(fp, "\n  \033[1mSynopsis:\033[0m\n    %s\n", k->synopsis);
    if (k->details[0] != '\0')
    {
        fprintf(fp, "\n  \033[1mDetails:\033[0m\n%s\n", k->details);
    }
    if (k->examples[0] != '\0')
    {
        fprintf(fp, "\n  \033[33mExamples:\033[0m\n\033[1m%s\033[0m\n", k->examples);
    }
    fprintf(fp, "\n");
}
