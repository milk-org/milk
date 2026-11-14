// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_keywords.c
 * @brief Language keywords and bash words (see milk_keywords.h).
 *
 * The keyword table is one static array of milk_keyword_t (synopsis, options, help text,
 * examples and completion words for each keyword); a lookup scans it. The options of a keyword
 * are written once, in its opts_<name> list: milk_kw_parse() reads them from the arguments, the
 * help prints them and completion offers them.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_keywords.h"

#define CLI_ONLY MILK_KW_CLI
#define NO_SCRIPT (MILK_KW_CLI | MILK_KW_BASH)

static const milk_kw_opt_t opts_calc[] = {
    { "-v", "<var>[,<var>...]",
      "bind the values of the last statements (an assignment's value is the\nassigned one)" },
    { "-a", "<arr>", "bind the pixels of an image result into an indexed array" },
    { "--lib", "<file>", "load user C extension (.c or .so) into the session" },
    { "--list", NULL,
      "list built-in operators and functions & loaded C extensions\n(a name pattern may follow)" },
    { "--json", NULL, "with --list: JSON output" },
    { "--explain", NULL,
      "compile the statements as a unit would and print how each one runs\n(fused pass with shape, "
      "precision and threads; merged; scalar\nstatement; library step); nothing runs" },
    { "--repl", NULL, NULL },
    { "--tokens", NULL, "!use: --list" },
    { "-l", "<file>", "!use: --lib" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_assert[] = {
    { "--exists", NULL, "the entity exists (the check of an entity given alone)" },
    { "-r --running", NULL, "its owner process is alive" },
    { "--fps-min", "<hz>", "its frame rate is at least <hz>" },
    { "--max-age", "<s>", "its last frame is at most <s> seconds old" },
    { "--cnt-min", "<N>", "its cnt0 is at least N" },
    { "--dim", "<WxH[xD]>", "its size" },
    { "-m --msg", "<text>", "the message of a failure (also: the last argument)" },
    { "-e", NULL, "!use: --exists" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_get[] = { { "-v", "<var>", "bind a scalar into <var>" },
                                          { "-a", "<arr>", "bind the lines into an indexed array" },
                                          { "-A", "<assoc>",
                                            "bind the lines into an associative array" },
                                          { "--json -j", NULL, "all properties as JSON" },
                                          { "--watch", "[T]",
                                            "the values every T s (0.5) until Ctrl-C: in place at "
                                            "a terminal, else\none line per refresh" },
                                          { NULL, NULL, NULL } };

static const milk_kw_opt_t opts_set[] = {
    { "--restore", NULL,
      "set the old values back when the calling function or @source'd file\nreturns" },
    { "--tui", NULL, "browse and edit FPS parameters in a fuzzy finder (milk-cli REPL)" },
    { "--flush-undo", NULL, NULL },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_exists[] = { { "-r --running", NULL,
                                               "also require its owner process to be alive" },
                                             { NULL, NULL, NULL } };

static const milk_kw_opt_t opts_ls[] = { { "-l --long", NULL, "the details of each entity" },
                                         { "-1", NULL, "names only (the default)" },
                                         { "-q --quiet", NULL,
                                           "names without their namespace prefix" },
                                         { "--json -j", NULL, "JSON" },
                                         { "--local", NULL, "the session's local images" },
                                         { "-a", "<arr>", "bind the names into an indexed array" },
                                         { NULL, NULL, NULL } };

static const milk_kw_opt_t opts_waitfor[] = {
    { "-t --timeout", "<sec>", "give up after <sec> s (exit status 124)" },
    { "--cnt", "<target>", "a stream: until its cnt0 reaches <target>" },
    { "--sem", "<n>", "a stream: wait on its semaphore <n>" },
    { "--next", NULL, "an expression: from the next frame on, not now" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_waitany[] = { { "-t --timeout", "<sec>",
                                                "give up after <sec> s (exit status 124)" },
                                              { NULL, NULL, NULL } };

static const milk_kw_opt_t opts_on[] = {
    { "--once", NULL, "fire once" },
    { "--scoped", NULL, "stop the watcher when the calling function or @source'd file returns" },
    { "--max-hz", "<hz>", "fire at most <hz> times per second" },
    { "--cooldown", "<s>", "wait <s> seconds after each run" },
    { "-t --timeout", "<sec>", "stop the watcher after <sec> s without an update" },
    { "--when", "<expr>",
      "fire when the expression turns true (also if it is true at the start),\nthen again once it "
      "has been false" },
    { "--hold", "<s>", "with --when: only after it stayed true that long" },
    { "--level", NULL, "with --when: on every frame while it is true" },
    { "--every", "<s>", "run the command periodically, without drift" },
    { "--list", NULL, "list the watchers" },
    { "--stop", "<id>", "stop one" },
    { "--clear", NULL, "stop them all" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_sleep[] = {
    { "--every", "<seconds>",
      "sleep to the deadline held in <var> (CLOCK_MONOTONIC ns; unset: one\nperiod from now), then "
      "move it one period on: a loop runs at that\nrate, without drift" },
    { "--hz", "<rate>", "the same, for <rate> periods per second" },
    { "--late", "<var>",
      "the ticks skipped by a late call (it returns at once instead of\ncatching up)" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_grab[] = {
    { "-t --timeout", "<sec>", "give up after <sec> s" },
    { "--fits", "<out.fits>", "write the frames to a FITS file" },
    { "--timestamps --ts", "<out.txt>", "write the time of each frame to a text file" },
    { "--stdout", NULL, "write the pixels to stdout as text, or bind them into an array" },
    { "--raw", NULL, "with --stdout: binary" },
    { "-a", "<arr>", "with --stdout: bind the pixels into an indexed array" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_replay[] = {
    { "--fps", "<hz>", "frames per second" },
    { "-n --count", "<N>", "stop after N frames" },
    { "--loop", NULL, "start again at the end" },
    { "--stdin", NULL, "write the values read from stdin into the stream, creating it if\nneeded" },
    { "--raw", NULL, "with --stdin: binary" },
    { "--dtype", "<type>", "with --stdin: the type of a stream it creates" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_snapshot[] = { { "-o --output", "<file>",
                                                 "write to <file> (default: stdout)" },
                                               { NULL, NULL, NULL } };

static const milk_kw_opt_t opts_restore[] = {
    { "--dry-run", NULL, "only list the changes" },
    { "--undo", "[N]",
      "set back the values the last N @set calls of this shell overwrote\n(FPS parameters, stream "
      "keywords), where they still hold what was\nwritten; the others are left as they are, with a "
      "warning (status 1)" },
    { "-n", NULL, "!use: --dry-run" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_source[] = {
    { "--once", NULL, "skip a file this shell already sourced (by real path)" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_defer[] = {
    { "--exit", NULL, "run the command when the shell or script exits" },
    { "--restore", NULL,
      "read the values now, set them back when the scope ends: an entity\npath, <entity>.* (all "
      "its parameters or keywords) or\n<ns>.<glob>.<attr>" },
    { "--return", NULL, NULL },
    { "--orphans", NULL, NULL },
    { "--exit-run", NULL, NULL },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_session[] = {
    { "-d", NULL, "new: do not use the new session" },
    { "--idle", "<T>",
      "new: it ends after T (90, 30s, 10m, 2h, 1d) without a command while no\nshell uses it" },
    { "--all", NULL, "end: every session that ls lists" },
    { "--units", NULL, "end: and the units started from it" },
    { "--json", NULL, "ls: JSON" },
    { "--tmux -tmux", NULL,
      "!sessions start detached, their output in the file @session status names" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_module[] = {
    { "--all", NULL, "list: also the installed modules not loaded yet" },
    { "--as", "<short>", "load: register the module's commands under another short name" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_log[] = {
    { "-n --last", "<N>", "the last N matching entries (20)" },
    { "--all", NULL, "every entry in the ring buffer" },
    { "-g --grep", "<regex>", "case-insensitive extended regex on the command and its output" },
    { "-e --errors", NULL, "failed commands only (status not 0)" },
    { "--slow", "<ms>", "commands slower than <ms> ms only" },
    { "--since", "<spec>", "since a relative time (30s, 5m, 1h) or HH:MM:SS" },
    { "-f --follow", NULL, "live-tail new commands and outputs" },
    { "-s --session", "<name>", "the log of session <name>" },
    { "--json", NULL, "JSON Lines" },
    { "--clear", NULL, "clear the session's ring buffer" },
    { "-a", NULL, "!use: --all" },
    { "--server", NULL, "!use: --session <name> (or -s <name>)" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_help[] = {
    { "--json", NULL, "the help as JSON" },
    { "--porcelain", NULL, "the help as plain lines, for programs" },
    { "-k", "<regex>", "search keywords and command descriptions" },
    { "--tui", NULL, "browse the help in a fuzzy finder (milk-cli REPL)" },
    { NULL, NULL, NULL }
};

static const milk_kw_opt_t opts_history[] = {
    { "--local", NULL, "this session (the default)" },
    { "--global", NULL, "all sessions" },
    { "--stats", NULL, "command statistics (milk-cli prompt)" },
    { "--tui", NULL, "browse the history in a fuzzy finder (milk-cli prompt)" },
    { NULL, NULL, NULL }
};

static const milk_keyword_t keywords[] = {
    /* values and expressions */
    { "calc", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@calc [-v <var>[,<var>...] | -a <arr>] [--lib <file.c|so>] [--list [<pat>] [--json]] "
      "[--explain] [\"<expr>[; <expr>...]\"]",
      "Evaluate image or scalar expressions (no args in milk-cli: toggle calc mode)",
      "  (no args)   toggle interactive calc mode in milk-cli (exit/quit/@calc to leave)\n  "
      "Images, calculator variables, the bash variables the text names and entity paths "
      "(s.cam.xsize)\n  are visible. Slicing (im[x0:x1,y0:y1]), reductions and per-pixel math.",
      "  @calc \"out = im1 + 2.0 * im2\"\n  @calc -v rms,ptv \"istd(im1); imax(im1) - "
      "imin(im1)\"\n  @calc -a row \"im1[*, 10]\"\n  @calc --list stencil\n  @calc --explain \"out "
      "= cam * gain; z = cplx(out, out)\"",
      opts_calc, NULL },
    { "test", MILK_KW_BUILTIN, MILK_KW_ALL, 1, "@test \"<boolean-expr>\"",
      "Evaluate a boolean expression: exit status 0 if true, 1 if false",
      "  Made for bash conditionals (if, while, &&); no subshell.",
      "  if @test \"imax(im1) > 100.0\"; then echo saturated; fi\n  while @test \"s.cam0.cnt0 < "
      "50\"; do @sleep 0.01; done",
      NULL, NULL },
    { "assert", MILK_KW_SERVER, MILK_KW_ALL, 1,
      "@assert [<expr> | [--exists] [-r] [--fps-min <hz>] [--max-age <s>] [--cnt-min <N>] [--dim "
      "<WxH[xD]>] <entity>] [\"message\"]",
      "Check an expression or a stream/FPS/process health invariant; fail loudly",
      "  Expression mode: a failed relation prints both sides and returns 1.\n  Entity mode: "
      "existence, running owner, frame rate, age, cnt0 or geometry.",
      "  @assert \"istd(wfs) < 0.5\" \"WFS residual diverged\"\n  @assert --running --fps-min 500 "
      "s.wfs_cam \"camera degraded\"",
      opts_assert, NULL },
    { "get", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@get [-v <var>|-a <arr>|-A <assoc>|--json] <spec> | @get [<var>=]<spec>... | @get --watch "
      "[T] <spec>...",
      "Read stream, local image, FPS or process properties",
      "  <var>=<spec>  bind several in one call; several plain specs print one value per line\n  "
      "Globs: s.*, s.<glob>.<prop>.  Streams: the properties of @help streams (stats: all the\n  "
      "statistics, for -A).\n  FPS: any parameter; status (or run) and conf: stopped starting "
      "running stopping exited\n  failed; session; run.out, conf.out (last output lines).  "
      "Processes: status loopcnt ctrl pid.",
      "  @get -v nx s.cam0.xsize\n  @get nx=s.cam0.xsize ny=s.cam0.ysize\n  @get -A st "
      "s.cam0.stats\n  @get -A widths 's.dm*.xsize'\n  @get --json s.cam0",
      opts_get, NULL },
    { "set", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@set [--restore] <spec>[=]<value> [<spec>=<value> ...] [comment] | @set --tui [<fps>]",
      "Write stream keywords, FPS parameters, process controls; run FPS/process actions",
      "  Several assignments per call; entity and parameter globs.\n  s.<glob>.kw.<KEY>=v   FITS "
      "keyword    f.<glob>.<param>=v   FPS parameter\n  f.<name>.conf=start|stop|step, "
      "f.<name>.run=start|stop (returns once done)   p.<name>.ctrl=pause",
      "  @set f.loop.gain=0.25 f.loop.mult=0.99\n  @set --restore f.loop.gain=0\n  @set "
      "'s.cam*.kw.OBSERVER'=Subaru\n  @set f.loop.run=start",
      opts_set, NULL },
    { "exists", MILK_KW_BUILTIN, MILK_KW_ALL, 1, "@exists [-r] <entity>",
      "Exit status 0 if the stream, local image, FPS or process exists", "",
      "  @exists s.cam0 && echo yes\n  @exists -r f.loop || @set f.loop.run=start", opts_exists,
      NULL },
    { "ls", MILK_KW_BUILTIN, MILK_KW_ALL, 1, "@ls [-l|-1|-a <arr>|--json|--local] [<pattern>...]",
      "List streams, local images, FPS and processes",
      "  Patterns select a namespace: 's.*' streams, 'f.*' FPS, 'p.*' processes.",
      "  @ls\n  @ls 's.cam*'\n  @ls -a names 'f.*'", opts_ls, NULL },
    /* events and time */
    { "waitfor", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@waitfor <entity|expr> [N] [-t <sec>] [--cnt <target>] | @waitfor [--next] \"<expr>\" [-t "
      "<sec>]",
      "Sleep until a stream posts N frames, a property has a value, or an expression is true",
      "  Sleeps on the stream semaphores (no CPU).\n  An expression is evaluated again on each "
      "frame of the streams it names (in the shell when\n  it is scalar); FPS and process values "
      "in it are polled.\n  f.<name> running|stopped: the unit's run process (no state: the FPS "
      "exists).",
      "  @waitfor s.cam0 10 -t 2\n  @waitfor f.loop running -t 5\n  @waitfor \"imean(cam0) > 100\" "
      "-t 5",
      opts_waitfor, NULL },
    { "waitany", MILK_KW_BUILTIN, MILK_KW_ALL, 1, "@waitany <s1> <s2> [...] [-t <sec>]",
      "Wait until any of the streams updates; its name goes to $MILK_TRIGGER", "",
      "  @waitany s.cam0 s.cam1 -t 1 && echo \"$MILK_TRIGGER\"", opts_waitany, NULL },
    { "on", MILK_KW_SHELL, MILK_KW_ALL, 0,
      "@on [--once] [--scoped] [--max-hz <hz>] [--cooldown <s>] <t1> [t2...] [--] <cmd...> | @on "
      "[...] --when \"<expr>\" [--hold <s>] [--level] <cmd> | @on --every <s> <cmd> | @on "
      "--list|--stop <id>|--clear",
      "Run a command in the background each time a stream or entity updates",
      "  $MILK_TRIGGER names the trigger.",
      "  @on --max-hz 10 s.cam0 'info.imstats s.cam0'\n  @on --when 'imax(cam0) > 60000' --hold "
      "0.5 -- echo saturated\n  @on --every 1 -- '@get s.cam0.cnt0 >> counts.txt'\n  @on --stop 1",
      opts_on, NULL },
    { "sleep", MILK_KW_BUILTIN, MILK_KW_ALL, 0,
      "@sleep <seconds> | @sleep --every <seconds>|--hz <rate> [--late <var>] <var>",
      "Sleep without starting a process (Ctrl-C interrupts it); pace a loop", "",
      "  @sleep 0.01\n  unset t; while :; do @sleep --hz 100 t; read_sensor; done", opts_sleep,
      NULL },
    /* data */
    { "grab", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@grab [-t <sec>] [--fits <out.fits>] [--timestamps <out.txt>] <stream> [N] [dst_img] | "
      "@grab --stdout [-a <arr>|--raw] <stream>",
      "Capture 1 or N consecutive frames into a local image or cube, or a FITS file", "",
      "  @grab -t 1 s.cam0 l.dark\n  @grab --fits telem.fits --timestamps telem.txt s.cam0 100\n  "
      "@grab --stdout --raw s.cam0 | python3 analyze.py",
      opts_grab, NULL },
    { "replay", MILK_KW_SERVER, MILK_KW_ALL, 1,
      "@replay [--fps <hz>] [--count <N>] [--loop] <cube_or_fits> <dst_stream> | @replay --stdin "
      "[--raw] [--dtype <type>] <stream> [xsize [ysize [zsize]]]",
      "Play an image cube or FITS file into a stream at a given rate",
      "  Runs in a copy of the server: images it creates are not kept.",
      "  @replay --fps 200 telem.fits cam0_sim\n  python3 gen.py | @replay --stdin --dtype F32 "
      "cam0 128 128",
      opts_replay, NULL },
    { "snapshot", MILK_KW_SERVER, MILK_KW_ALL, 1, "@snapshot [-o <file>] [<pattern>...]",
      "Save FPS parameters and stream keywords", "", "  @snapshot -o base.snap 'f.loop*' 's.cam*'",
      opts_snapshot, NULL },
    { "diff", MILK_KW_SERVER, MILK_KW_ALL, 1, "@diff <snap_file> [<pattern>...]",
      "Compare a snapshot with the live state (status 1 if they differ)", "", "  @diff base.snap",
      NULL, NULL },
    { "restore", MILK_KW_SERVER, MILK_KW_ALL, 1,
      "@restore [--dry-run] <snap_file> [<pattern>...] | @restore --undo [--dry-run] [N]",
      "Apply a snapshot back to the live state (--dry-run: only list the changes), or undo this "
      "shell's last @set calls",
      "", "  @restore --dry-run base.snap\n  @restore --undo 3", opts_restore, NULL },
    { "pipe", MILK_KW_SERVER, MILK_KW_ALL, 1,
      "@pipe \"<cmd1> |> <cmd2> |> ...\" | @pipe \"<cmd1>; <cmd2>\"",
      "Run several milk commands and expressions in one request; '_' is the image the previous "
      "stage made",
      "",
      "  @pipe \"mem.mk2Dim p 32 32 |> _ = 4 |> q = _ * 3\"\n  @pipe \"mem.mk2Dim b1 32 32; b1 = "
      "5\"",
      NULL, NULL },
    /* program and session */
    { "source", MILK_KW_SHELL, MILK_KW_ALL, 0, "@source [--once] <file.milk> [args...]",
      "Translate and run a milk script in the current shell", "",
      "  @source setup.milk cam0\n  @source --once lib.milk", opts_source, NULL },
    { "defer", MILK_KW_BUILTIN, MILK_KW_ALL, 0,
      "@defer [--exit] <command...> | @defer --restore <spec>...",
      "Run a command when the calling function or @source'd file returns (last registered first)",
      "  At the top level the command runs when the shell or script exits.\n  One argument is a "
      "command string (as for trap); several are kept as quoted words.\n  Commands that Ctrl-C or "
      "a set -e failure skipped run before the next prompt, or at exit.",
      "  @defer mem.rm tmp_im\n  @defer --restore 'f.loop.*'\n  @defer --exit 'echo done'",
      opts_defer, NULL },
    { "args", MILK_KW_BUILTIN, MILK_KW_ALL, 0,
      "@args \"<name>:<type>[=<default>][:<help>]\"... -- \"$@\"",
      "Declare the arguments of a function or script; they are checked like a module command's",
      "  Types: int uint float bool (1 or 0) str stream fits. Without a default a value is "
      "required.\n  Values in order or as .name=value, '.' keeps a default, -h prints the usage. "
      "The names become\n  locals of the function; after -h or a bad value the function returns "
      "(status 0 or 2).\n  Commands in share/milk/commands.d and ~/.config/milk/commands.d "
      "(*.bash) are module.cmd\n  functions.",
      "  measure() { @args 'cam:str:camera stream' 'n:int=100:frames' -- \"$@\"; echo \"$cam $n\"; "
      "}\n  measure s.cam0 .n=50",
      NULL, NULL },
    { "session", MILK_KW_SHELL, MILK_KW_ALL, 1,
      "@session [status [<name>] | ls [--json] | new <name> [-d] [--idle <T>] | use <name> | leave "
      "| who [<name>] | end [<name>|--all] [--units] | prune | save <file> | restore <file>]",
      "Show, start, join, leave or end sessions: the milk-cli servers that hold images and "
      "variables",
      "  status [<name>]        this shell's session, or session <name>\n  ls                     "
      "this user's sessions in this MILK_SHM_DIR (ORPHANED: the shell that\n                       "
      "  created it has exited)\n  new <name>             start session <name> and use it; fails "
      "if it runs\n  use <name>             use session <name>; fails if it does not run\n  leave  "
      "                use this shell's own session again\n  who [<name>]           the shells "
      "that use a session\n  end [<name>]           end a session, the one in use by default\n  "
      "prune                  remove what ended sessions and exited shells left\n  save <file>     "
      "       write the session's state as a milk script, images in <file>.d/\n  restore <file>    "
      "     run that script in this shell's session (units: listed, not started)\n  A shell's own "
      "session starts on its first command and ends with the shell (inside milk-cli:\n  "
      "milk-cli's); scripts started from the shell use it too. A named one is shared by the "
      "shells\n  that use it and runs until it is ended. A session belongs to the MILK_SHM_DIR it "
      "started with.\n  milk-cli --session NAME starts a REPL whose bash uses session NAME.",
      "  @session new bench1 --idle 2h\n  @session use bench1\n  @session end bench1 --units",
      opts_session, "status ls new use leave who end prune save restore" },
    { "module", MILK_KW_SERVER, MILK_KW_ALL, 0,
      "@module [list [--all]] | @module load <name|file.so> [--as <short>]",
      "List the loaded modules, or load one (commands of installed modules load on first use)", "",
      "  @module list\n  @module load milkfft", opts_module, "list load" },
    { "config", MILK_KW_SHELL, MILK_KW_ALL, 0, "@config [<key> [<value>...]]",
      "Show or change a session setting",
      "  digits <N>               significant digits of printed results, for this shell\n  "
      "precision single|double  precision of the images commands create (calculator results "
      "follow\n                           their operands)\n  procinfo on|off          processinfo "
      "telemetry for commands that support it\n  prompt ...               the prompt: milk.bash "
      "PS1 telemetry (watch, preset, ...) or the milk-cli\n                           REPL format "
      "(%u %h %d %t %n)\n  inline on|off            expand @{...} in lines typed at an interactive "
      "milk.bash prompt\n  synhl, syndiag, indent, showmatch, calc on|off; sessionlog "
      "on|off|<file>   (milk-cli REPL)",
      "  @config digits 4\n  @config prompt watch f.loop.run\n  @config", NULL,
      "digits precision procinfo prompt inline synhl syndiag indent showmatch calc "
      "sessionlog\nprompt: status watch unwatch clear preset on off render\npreset: minimal std "
      "ao\n*: single double on off" },
    { "unload", MILK_KW_SHELL, MILK_KW_BASH, 0, "@unload",
      "Remove milk from this bash shell: stop its session and watchers, unset the functions", "",
      "  @unload", NULL, NULL },
    /* introspection */
    { "log", MILK_KW_BUILTIN, MILK_KW_ALL, 1,
      "@log [-n <N>|--all] [-g <pat>] [-e] [--slow <ms>] [--since <dur>] [-f] [-s <name>] [--json] "
      "[--clear] [<pattern>]",
      "Show the session's recent commands with timing and status", "", "  @log -n 20\n  @log -e",
      opts_log, NULL },
    { "help", MILK_KW_SERVER, MILK_KW_ALL, 0,
      "@help [<keyword|namespace|module|module.cmd|topic>] [--json|--porcelain] | @help -k <regex> "
      "| @help --tui",
      "Show help on keywords, entity namespaces, modules, commands and topics",
      "  Every command and keyword also answers -h/--help (@calc, @test, @assert: --help).\n  "
      "Topics: bash syntax variables flowcontrol scripting milk cmdopts readline streams fps "
      "procinfo seq",
      "  @help\n  @help iofits.loadfits\n  @help s\n  @help -k fits", opts_help, NULL },
    { "history", MILK_KW_SHELL, MILK_KW_ALL, 0,
      "@history [--local|--global] [N] | search <text> | save <file> | --stats | --tui",
      "Show the lines typed at milk-cli prompts in this session (--global: in all sessions)",
      "  N         the last N lines (20)\n  search and save are the milk-cli prompt's, on its "
      "readline history",
      "  @history 20\n  @history --global\n  @history search loadfits", opts_history, NULL },
    { NULL, MILK_KW_BUILTIN, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL }
};

static const char *const bash_words[] = {
    "if",       "elif",    "else",    "fi",     "for",      "while",   "until",   "do",
    "done",     "case",    "esac",    "select", "function", ".",       "source",  "break",
    "continue", "return",  "true",    "false",  "exit",     "shift",   "set",     "export",
    "readonly", "local",   "declare", "let",    "eval",     "type",    "command", "trap",
    "time",     "timeout", "wait",    "printf", "echo",     "getopts", "mapfile", "alias",
    "unalias",  "pushd",   "popd",    "dirs",   "[[",       NULL
};

/**
 * milk_lexicon_keywords - Return the NULL-terminated table of milk language keywords
 *
 * Provides read-only access to the master `keywords[]` descriptor array used for
 * dispatch, help generation, syntax highlighting, and tab completion.
 *
 * Return: Pointer to the first element of the `keywords` array
 */
const milk_keyword_t *milk_lexicon_keywords(void)
{
    return keywords;
}

/**
 * milk_lexicon_keyword - Look up a milk language keyword descriptor by name
 * @name: Keyword name with or without a leading `@` (e.g. `"@get"` or `"get"`)
 *
 * Searches the `keywords[]` table for an entry matching @name.
 *
 * Return: Pointer to the matching `milk_keyword_t` descriptor, or NULL if not a keyword
 */
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

/**
 * milk_lexicon_frontend - Map a front-end name to its `MILK_KW_*` bitmask flag
 * @frontend: Front-end name (`"cli"`, `"script"`, or `"bash"`/NULL)
 *
 * Translates the current front-end identifier into a bitmask used to check whether
 * a keyword (such as `@history` or `@unload`) is supported in that environment.
 *
 * Return: `MILK_KW_CLI`, `MILK_KW_SCRIPT`, or `MILK_KW_BASH`
 */
unsigned milk_lexicon_frontend(const char *frontend)
{
    if (frontend != NULL && strcmp(frontend, "cli") == 0)
    {
        return MILK_KW_CLI;
    }
    if (frontend != NULL && strcmp(frontend, "script") == 0)
    {
        return MILK_KW_SCRIPT;
    }
    return MILK_KW_BASH;
}

/**
 * milk_lexicon_bash_words - Return the NULL-terminated list of reserved bash words
 *
 * Used by the milk REPL syntax highlighter and completer to recognize shell control
 * structures and builtins.
 *
 * Return: Pointer to the NULL-terminated `bash_words` string array
 */
const char *const *milk_lexicon_bash_words(void)
{
    return bash_words;
}

/**
 * milk_lexicon_help_arg - Check whether a first argument requests help for keyword @k
 * @k:   Keyword descriptor being invoked
 * @arg: First argument token passed to the keyword
 *
 * Recognizes `--help` for all keywords and `-h` for keywords where `-h` is not an
 * expression or option flag (all except `@calc`, `@test`, and `@assert`).
 *
 * Return: 1 if @arg is a help flag for @k, 0 otherwise
 */
int milk_lexicon_help_arg(const milk_keyword_t *k, const char *arg)
{
    if (arg == NULL || arg[0] != '-')
    {
        return 0;
    }
    return strcmp(arg, "--help") == 0 || (strcmp(arg, "-h") == 0 && !milk_lexicon_takes_expr(k));
}

/**
 * milk_lexicon_takes_expr - Check whether the arguments of keyword @k are calculator text
 * @k: Keyword descriptor
 *
 * Return: 1 for `@calc`, `@test` and `@assert`, 0 otherwise
 */
int milk_lexicon_takes_expr(const milk_keyword_t *k)
{
    return strcmp(k->name, "calc") == 0 || strcmp(k->name, "test") == 0 ||
           strcmp(k->name, "assert") == 0;
}

/**
 * milk_lexicon_print_list - Print the keywords, one per line
 * @fp:            Output stream
 * @callable_only: Only the keywords milk-call implements
 * @synopsis:      Each synopsis, its summary on the next line, instead of the name and summary
 *
 * Return: void
 */
void milk_lexicon_print_list(FILE *fp, int callable_only, int synopsis)
{
    for (const milk_keyword_t *k = keywords; k->name != NULL; k++)
    {
        if (callable_only && !k->callable)
        {
            continue;
        }
        if (synopsis)
        {
            fprintf(fp, "  %s\n      %s\n", k->synopsis, k->summary);
        }
        else
        {
            fprintf(fp, "  @%-10s %s\n", k->name, k->summary);
        }
    }
}

/**
 * milk_lexicon_usage - Print the usage line of keyword @name (for a usage error)
 * @name: Keyword name, with or without the sigil
 * @fp:   Output stream
 *
 * Return: 2, the status of a usage error
 */
int milk_lexicon_usage(const char *name, FILE *fp)
{
    const milk_keyword_t *k = milk_lexicon_keyword(name);
    if (k != NULL)
    {
        fprintf(fp, "usage: %s\n", k->synopsis);
    }
    return 2;
}

/** opt_name() - 1 if @w (@wl chars) is one of the spellings of option @o */
static int opt_name(const milk_kw_opt_t *o, const char *w, size_t wl)
{
    for (const char *p = o->names; *p != '\0'; p += strcspn(p, " "), p += (*p == ' '))
    {
        if (strcspn(p, " ") == wl && strncmp(p, w, wl) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/** opt_find() - index in k->opts of the option word @w is (`-t`, `--timeout`, `--timeout=3`:
 * *@eq then points past the '='), or -1 */
static int opt_find(const milk_keyword_t *k, const char *w, const char **eq)
{
    size_t wl = strlen(w);
    size_t el = strcspn(w, "=");
    *eq       = NULL;
    for (int r = 0; k->opts != NULL && k->opts[r].names != NULL; r++)
    {
        if (opt_name(&k->opts[r], w, wl))
        {
            return r;
        }
        if (el < wl && w[1] == '-' && k->opts[r].arg != NULL && opt_name(&k->opts[r], w, el))
        {
            *eq = w + el + 1;
            return r;
        }
    }
    return -1;
}

/** kw_bad() - report a usage error of keyword @a->k: "@<kw>: <msg><what>", its usage line; 1 */
static int kw_bad(milk_kw_args_t *a, const char *msg, const char *what)
{
    fprintf(stderr, "@%s: %s%s\n", a->k->name, msg, what);
    a->rc = milk_lexicon_usage(a->k->name, stderr);
    return 1;
}

int milk_kw_parse(const char *name, unsigned flags, int *argc, char **argv, milk_kw_args_t *a)
{
    int n    = *argc;
    int np   = 0;
    int text = 0;

    memset(a, 0, sizeof(*a));
    a->k = milk_lexicon_keyword(name);
    text = milk_lexicon_takes_expr(a->k);
    for (int i = 0; i < n; i++)
    {
        char                *w    = argv[i];
        const char          *eq   = NULL;
        const char          *v    = "";
        char                *end  = NULL;
        int                  dash = (w[0] == '-' && w[1] != '\0');
        int                  r    = dash ? opt_find(a->k, w, &eq) : -1;
        const milk_kw_opt_t *o    = (r >= 0) ? &a->k->opts[r] : NULL;
        int                  need = (o != NULL && o->arg != NULL && o->arg[0] != '[' && eq == NULL);

        if (strcmp(w, "--") == 0 || (text && need && i + 1 >= n))
        {
            o = NULL; /* "--": the rest are arguments; `@calc -a`: minus a */
        }
        else if (o == NULL && dash && !text && strpbrk(w, " \t\n") == NULL &&
                 (strtod(w, &end), *end != '\0'))
        {
            if (strcmp(w, "--help") == 0 || strcmp(w, "-h") == 0)
            {
                milk_lexicon_print_help(a->k, stdout);
                return 1; /* a->rc is 0 */
            }
            return kw_bad(a, "unknown option ", w);
        }
        if (o == NULL)
        {
            int rest = (strcmp(w, "--") == 0 || (flags & MILK_KW_OPTS_FIRST));
            for (i += (strcmp(w, "--") == 0); i < n; i++)
            {
                argv[np++] = argv[i];
                if (!rest)
                {
                    break;
                }
            }
            continue;
        }
        if (o->help != NULL && o->help[0] == '!')
        {
            fprintf(stderr, "@%s: %s was removed; %s\n", a->k->name, w, o->help + 1);
            a->rc = 2;
            return 1;
        }
        if (eq != NULL)
        {
            v = eq;
        }
        else if (need && i + 1 >= n)
        {
            return kw_bad(a, w, " needs a value");
        }
        else if (need || (o->arg != NULL && i + 1 < n &&
                          strchr("0123456789.", argv[i + 1][0]) != NULL && argv[i + 1][0] != '\0'))
        {
            v = argv[++i];
        }
        if (r >= MILK_KW_NOPT || a->nocc >= MILK_KW_NOCC)
        {
            return kw_bad(a, "too many options at ", w);
        }
        a->val[r]             = v;
        a->occ[a->nocc].opt   = r;
        a->occ[a->nocc++].val = v;
    }
    *argc = np;
    return 0;
}

const char *milk_kw_opt(const milk_kw_args_t *a, const char *opt)
{
    const char *eq = NULL;
    int         r  = opt_find(a->k, opt, &eq);
    return (r >= 0 && r < MILK_KW_NOPT) ? a->val[r] : NULL;
}

int milk_kw_only(const milk_kw_args_t *a, const char *what, const char *allowed)
{
    milk_kw_opt_t ok = { allowed, NULL, NULL };
    for (int i = 0; i < a->nocc; i++)
    {
        const char *nm = a->k->opts[a->occ[i].opt].names;
        if (!opt_name(&ok, nm, strcspn(nm, " ")))
        {
            fprintf(stderr, "@%s: %s does not take %.*s\n", a->k->name, what,
                    (int) strcspn(nm, " "), nm);
            return milk_lexicon_usage(a->k->name, stderr);
        }
    }
    return 0;
}

size_t milk_lexicon_options(const milk_keyword_t *k, char *buf, size_t sz)
{
    size_t pos = 0;
    buf[0]     = '\0';
    for (const milk_kw_opt_t *o = k->opts; o != NULL && o->names != NULL; o++)
    {
        if (o->help != NULL && o->help[0] != '!' && pos + strlen(o->names) + 2 < sz)
        {
            pos += (size_t) snprintf(buf + pos, sz - pos, "%s ", o->names);
        }
    }
    if (pos + 8 < sz)
    {
        pos += (size_t) snprintf(buf + pos, sz - pos, "--help");
    }
    return pos;
}

/** print_options() - the options of @k that the help shows: "-t, --timeout <sec>  text" */
static void print_options(const milk_keyword_t *k, FILE *fp)
{
    int any = 0;
    for (const milk_kw_opt_t *o = k->opts; o != NULL && o->names != NULL; o++)
    {
        char left[96];
        int  n = 0;
        if (o->help == NULL || o->help[0] == '!')
        {
            continue;
        }
        if (!any++)
        {
            fprintf(fp, "\n  \033[1mOptions:\033[0m\n");
        }
        for (const char *p = o->names; *p != '\0' && n < 80; p++)
        {
            n += snprintf(left + n, sizeof(left) - (size_t) n, (*p == ' ') ? ", " : "%c", *p);
        }
        snprintf(left + n, sizeof(left) - (size_t) n, "%s%s", (o->arg != NULL) ? " " : "",
                 (o->arg != NULL) ? o->arg : "");
        int own = (strlen(left) > 22); /* too long for the column: on its own line */
        if (own)
        {
            fprintf(fp, "    %s\n", left);
        }
        for (const char *h = o->help; h != NULL;
             h             = (h[strcspn(h, "\n")] != '\0') ? h + strcspn(h, "\n") + 1 : NULL)
        {
            fprintf(fp, "    %-22s  %.*s\n", (h == o->help && !own) ? left : "",
                    (int) strcspn(h, "\n"), h);
        }
    }
}

/**
 * milk_lexicon_print_help - Format and print full help for a language keyword
 * @k:  Keyword descriptor to display
 * @fp: Output file stream
 *
 * Prints the keyword's summary, synopsis, details, and examples with ANSI color
 * formatting to @fp.
 *
 * Return: void
 */
void milk_lexicon_print_help(const milk_keyword_t *k, FILE *fp)
{
    fprintf(fp, "\n\033[32m@%s\033[0m [language keyword]\n\t\033[32m%s\n\033[0m", k->name,
            k->summary);
    fprintf(fp, "\n  \033[1mSynopsis:\033[0m\n    %s\n", k->synopsis);
    print_options(k, fp);
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
