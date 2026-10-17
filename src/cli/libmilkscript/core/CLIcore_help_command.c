// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    CLIcore_help_command.c
 * @brief   Per-command help and search functions
 *
 * Contains help_command(), command_info_search(),
 * help(), and help_module().
 *
 * @see CLIcore_help.c for system info and listings.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <libgen.h>
#include <malloc.h>

#include <fitsio.h>
#include <regex.h>

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include "CLIcore.h"

#include "fps.h"
#include "fps_connect.h"

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
#ifndef COLORARGCLI
#    define COLORARGCLI "\033[36m"
#endif
#ifndef COLORARGnotCLI
#    define COLORARGnotCLI "\033[35m"
#endif

extern int help_format_mode;

/**
 * @brief Check if a 64-bit flag contains a specific bit.
 */
static int checkFlag64(uint64_t flags, uint64_t testflag, char *flagdescription)
{
    int rval = 0;

    // printf("--------- flags: %ld\n", flags);

    if (flags & testflag)
    {
        rval = 1;
        printf("    [%c[%d;%dm ON%c[%dm]  %s\n", (char) 27, 1, 32, (char) 27, 0, flagdescription);
    }
    else
    {
        rval = 0;
        printf("    [%c[%d;%dmOFF%c[%dm]  %s\n", (char) 27, 1, 31, (char) 27, 0, flagdescription);
    }
    return rval;
}

typedef struct
{
    const char *name;
    const char *alias;
    const char *synopsis;
    const char *summary;
    const char *details;
    const char *examples;
} milk_bash_primitive_help_t;

static const milk_bash_primitive_help_t g_milk_bash_help[] = {
    { "calc", "=", "@ [-v <var>] \"<expr>\"  |  @calc [-v <var>] \"<expr>\"",
      "Evaluate an image/tensor or scalar expression in the persistent Milk server",
      "  - Options:\n"
      "      -v <var>  Bind scalar result directly into Bash variable <var> (zero subshell)\n"
      "  - Automatically syncs referenced and assigned Bash scalar variables.\n"
      "  - Supports 2D/3D slicing (im[x0:x1,y0:y1]), reductions (sum, mean, rms,\n"
      "    min, max, median), element-wise math, and SHM entity attributes (@s., @f.).",
      "  @ \"out = im1 + 2.0 * im2\"\n"
      "  @ -v rms_val \"istd(im1)\"\n"
      "  @calc -v ptv \"imax(im1) - imin(im1)\"" },
    { "test", NULL, "@test \"<boolean-expr>\"",
      "Evaluate a Milk boolean expression and return exit status 0 (true) or 1 (false)",
      "  - Designed for native Bash conditionals (if / while / &&).\n"
      "  - Supports image reductions and SHM entity comparisons without subshells.",
      "  if @test \"max(im1) > 100.0\"; then echo \"saturated\"; fi\n"
      "  while @test \"@s.cam0.cnt0 < 50\"; do sleep 0.01; done" },
    { "get", NULL, "@get [-v <var>|-a <arr>|-A <assoc>|-j|--json] <spec>",
      "Read or discover shared-memory stream, FPS, processinfo, or sequencer attributes",
      "  - Options:\n"
      "      -v <var>    Bind scalar output directly into Bash variable <var>\n"
      "      -a <arr>    Bind multi-line output into indexed Bash array <arr>\n"
      "      -A <assoc>  Bind 'key val' output into associative Bash array <assoc>\n"
      "      -j, --json  Format output as JSON (also via '<entity>.json' suffix)\n"
      "  - Discovery & Globbing:\n"
      "      s.*, f.*, p.*, q.*   List active SHM entities matching glob pattern\n"
      "      s.<glob>.<attr>      Query <attr> across all matching entities\n"
      "  - Stream (s.<name>.): xsize, ysize, zsize, naxis, cnt0, cnt1, dtype,\n"
      "                        location, shared, inode, fps, age, owner_pid,\n"
      "                        summary, json, kw.<KEY>\n"
      "  - FPS    (f.<name>.): <param>, status, conf, run, pid, nbparam, list, json\n"
      "  - Proc   (p.<name>.): PID, loopcnt, loopcntMax, cnt0, status, ctrlval,\n"
      "                        triggermode, triggerstream, dt_median_ns, msg\n"
      "  - Seq    (q.<id>.):   state, active_step, progress, step_count, pid, name",
      "  @get -v nx s.cam0.xsize\n"
      "  @get -a streams 's.*'\n"
      "  @get -A widths 's.dm*.xsize'\n"
      "  @get --json s.cam0" },
    { "set", NULL, "@set <spec1> [=] <val1> [<spec2> [=] <val2> ...] [comment]",
      "Modify shared-memory stream keywords, FPS parameters/lifecycle, processinfo, or sequencers",
      "  - Supports multi-assignment in one call and globbed entity/parameter matching.\n"
      "  - Stream (s.<glob>.kw.<KEY>): create or update FITS keyword (auto-typed)\n"
      "  - FPS    (f.<glob>.<pglob>):  set any FPS parameter (all types incl. TIMESPEC)\n"
      "  - FPS    (f.<name>.conf):     start | stop | step (or :tmux to dispatch via tmux)\n"
      "  - FPS    (f.<name>.run):      start | stop (or :tmux to dispatch via tmux)\n"
      "  - Proc   (p.<name>.ctrlval):  0 (run), 1 (pause), 2 (single step), 3 (exit)\n"
      "  - Seq    (q.<id>.action):     pause | resume | abort",
      "  @set f.loopctrl.gain = 0.25 f.loopctrl.mult = 0.99\n"
      "  @set 'f.loop*.xsize' = 128 's.cam*.kw.OBSERVER' = \"Subaru\"\n"
      "  @set f.loopctrl.conf = start\n"
      "  @set f.loopctrl.run = start:tmux" },
    { "exists", NULL, "@exists [-r] [s.|f.|p.|q.]<name>",
      "Check whether a shared-memory stream, FPS, processinfo, or sequencer exists",
      "  - Returns exit code 0 if present in MILK_SHM_DIR (/dev/shm), 1 otherwise.\n"
      "  - With -r, also verifies that the owning process PID is currently alive.",
      "  if @exists s.cam0; then @info.imstats @S:cam0; fi\n"
      "  if @exists -r f.loopctrl; then echo \"active\"; fi" },
    { "running", NULL, "@running [f.|p.|q.|s.]<name>",
      "Check whether a shared-memory entity exists and its owner PID is alive",
      "  - Equivalent to '@exists -r <name>'. Returns exit code 0 if running, 1 otherwise.",
      "  if @running f.loopctrl; then @set f.loopctrl.gain = 0.1; fi" },
    { "waitfor", NULL, "@waitfor <stream> [N] [-s <sem>] [-t|--timeout <sec>] [--cnt <target>]",
      "Block until a shared-memory stream posts N new frames or reaches target cnt0",
      "  - Uses ImageStreamIO semaphores for sub-microsecond wakeup latency.\n"
      "  - Returns exit code 0 on wakeup, 124 on timeout.",
      "  @waitfor cam0 10 --timeout 2.0\n"
      "  @waitfor cam0 --cnt 1000 -t 5.0" },
    { "waitany", NULL, "@waitany <s1> <s2> [...] [-t|--timeout <sec>]",
      "Wait until any of the specified streams updates and report the first triggered stream",
      "  - Prints the triggered stream name (or binds REPLY if declared). Returns 124 on timeout.",
      "  ready=$(@waitany cam0 cam1 --timeout 1.0)" },
    { "grab", NULL,
      "@grab [-t <sec>] [--fits <out.fits>] [--timestamps <out.txt>] <stream> [N] [dst_img]",
      "Capture 1 or N consecutive frames from a stream into a local image/cube or FITS file",
      "  - Atomically snapshots <stream> (single frame or N-frame 3D cube) into <dst_img>.\n"
      "  - With --fits <out.fits>, saves the captured 2D frame or 3D cube directly to FITS.\n"
      "  - With --timestamps <out.txt>, writes per-frame cnt0, cnt1, atime_ns, and mono_ns.",
      "  @grab --timeout 1.0 cam0 dark_frame\n"
      "  @grab --fits telem.fits --timestamps telem.txt cam0 100" },
    { "replay", NULL,
      "@replay [--fps <hz>|--dt <sec>] [--count <N>] [--loop] <src_cube_or_fits> <dst_stream>",
      "Replay a 2D/3D image, SHM stream, or FITS cube into a shared-memory stream at target rate",
      "  - Publishes each slice to <dst_stream> with write flag, cnt0, atime, and semaphores.\n"
      "  - Options:\n"
      "      --fps <hz>     Target playback frame rate in Hz (default: 100)\n"
      "      --dt <sec>     Target inter-frame interval in seconds\n"
      "      --count <N>    Stop after publishing N total frames\n"
      "      --loop         Repeat cube playback until --count is reached",
      "  @replay --fps 200 telem.fits cam0_sim\n"
      "  @replay --fps 500 --loop --count 1000 calib_cube wfs_sim" },
    { "snapshot", NULL, "@snapshot [-o <file>] [<pattern...>]",
      "Save a human-readable snapshot of FPS parameters and stream FITS keywords",
      "  - Captures writable FPS parameters (f.<glob>) and stream keywords (s.<glob>).\n"
      "  - Writes to <file> when -o/--output is specified, or stdout otherwise.",
      "  @snapshot -o baseline.snap\n"
      "  @snapshot -o ao_loop.snap 'f.loop*' 's.cam*'" },
    { "diff", NULL, "@diff <snap_a> [snap_b]",
      "Compare a saved snapshot against live /dev/shm state or a second snapshot file",
      "  - Prints modified (~ key: old -> new), added (+ key: val), and removed (- key: val) "
      "items.\n"
      "  - Returns exit code 0 when identical, 1 when differences exist.",
      "  @diff baseline.snap\n"
      "  @diff before.snap after.snap" },
    { "restore", NULL, "@restore [--dry-run|-n] <snap_file> [<pattern...>]",
      "Restore FPS parameters and stream FITS keywords from a saved snapshot file",
      "  - Applies saved values back to live /dev/shm entities (optionally filtered by "
      "<pattern>).\n"
      "  - With -n / --dry-run, prints the @set operations that would be applied without writing.",
      "  @restore --dry-run baseline.snap\n"
      "  @restore baseline.snap 'f.loop*'" },
    { "assert", NULL,
      "@assert [<expr> | [--exists|-e] [--running|-r] [--fps-min <hz>] [--max-age <s>] "
      "[--cnt-min <N>] [--dim <WxH[xD]>] <entity>] [\"message\"]",
      "Assert a Milk boolean expression or real-time stream/FPS/process health invariant",
      "  - Expression mode: evaluates <expr>; on top-level relational failure (<, <=, >, >=, ==, "
      "!=)\n"
      "    automatically prints actual LHS vs RHS values to stderr and returns exit code 1.\n"
      "  - Entity mode: verifies existence, running owner PID, frame rate, age, cnt0, or geometry.",
      "  @assert \"istd(wfs_im) < 0.5\" \"WFS residual diverged\"\n"
      "  @assert --running --fps-min 500 --dim 128x128 s.wfs_cam \"Camera degraded\"" },
    { "prompt", NULL,
      "@prompt [status | watch [lbl=]<spec>... | unwatch <tgt>... | clear | "
      "preset <minimal|std|ao> | on | off | render]",
      "Configure the zero-subshell dynamic telemetry Bash prompt (PS1)",
      "  - Reads watched /dev/shm properties via in-process C builtins (zero subshells per "
      "prompt).\n"
      "  - Color-codes FPS status/run badges (green=ON/RUN, yellow=CONF, amber=OFF/IDLE).",
      "  @prompt watch f.loopctrl.run gain=f.loopctrl.gain s.wfs_cam.kw.MODE\n"
      "  @prompt preset ao\n"
      "  @prompt status" },
    { "defer", NULL, "@defer <command...>",
      "Register a cleanup command on the LIFO exit stack for the current Bash shell/script",
      "  - Executed automatically in reverse (LIFO) order on shell/script EXIT.",
      "  @defer @mem.rm tmp_im" },
    { "on", NULL,
      "@on [--once|-1] [--max-hz <hz>] [--cooldown <s>] <t1> [t2...] [--] <cmd...>  |  "
      "@on --list|--stop <id>|--clear",
      "Run a background reactive command when one or more streams/entities update",
      "  - Options:\n"
      "      -1, --once        Fire once on the next event and exit automatically\n"
      "      --max-hz <rate>   Throttle execution to at most <rate> Hz\n"
      "      --cooldown <sec>  Enforce minimum <sec> seconds between triggers\n"
      "      --list            List active watchers with ID, PID, options, and targets\n"
      "      -s, --stop <id>   Stop a specific watcher by ID, PID, or target name\n"
      "      --clear           Stop all active watchers in this session\n"
      "  - Exports MILK_TRIGGER with the name of the stream/entity that fired.",
      "  @on --max-hz 10 cam0 '@info.imstats @S:cam0'\n"
      "  @on --once cam0 cam1 -- 'echo \"Triggered by $MILK_TRIGGER\"'\n"
      "  @on --stop 1" },
    { "batch", NULL, "@batch \"<cmd1>; <cmd2>; ...\"  |  @batch << 'EOF' ... EOF",
      "Execute multiple Milk commands in a single client-server IPC round trip",
      "  - Eliminates per-command socket latency when running multi-step command blocks.\n"
      "  - Accepts semicolon/newline-separated arguments or a standard input heredoc.",
      "  @batch \"mem.mk2Dim b1 32 32; b1 = 5; b2 = b1 * 2\"\n"
      "  @batch << 'EOF'\n"
      "  mem.mk2Dim a 64 64\n"
      "  a = 10\n"
      "  EOF" },
    { "pipe", NULL, "@pipe \"<cmd1> |> <cmd2> |> ...\"",
      "Execute a |> pipeline in a single IPC round trip with automatic '_' image propagation",
      "  - Chains stages separated by '|>' inside the persistent server.\n"
      "  - Automatically binds '_' to the image produced or modified by the preceding stage.",
      "  @pipe \"mem.mk2Dim psrc 32 32 |> psrc = 4 |> pdst = _ * 3\"" },
    { "hc", NULL, "@hc [--sig|-s] <command> [arg_pos]",
      "Display detailed command help or compact CLICMDARGDEF argument signature hint",
      "  - Without --sig, prints full synopsis, argument types, and examples for <command>.\n"
      "  - With --sig (or -s), prints a one-line argument signature highlighting [arg_pos].",
      "  @hc mem.mk2Dim\n"
      "  @hc --sig mem.mk2Dim 2" },
    { "source", NULL, "@source <script.milk> [args...]",
      "Translate and execute a legacy .milk script inside the current Bash session",
      "  - Converts legacy .milk syntax on the fly via 'milk-call translate --legacy'.",
      "  @source my_pipeline.milk" },
    { "server", NULL,
      "@server [status|list|start [name]|use <name>|local|detach|stop [name|--all]|log]",
      "Manage persistent milk-cli --server daemons and switch active session attachment",
      "  - status         Show current session server PID, socket, and SHM log path\n"
      "  - list (or ls)   List all live milk-cli servers for the current user\n"
      "  - start [name]   Start a persistent named server (default: 'default')\n"
      "  - use <name>     Attach current Bash shell to named server (starts if needed)\n"
      "  - local / detach Detach from shared server and use a shell-private server\n"
      "  - stop [name]    Stop a named server (or '--all' to stop all servers)",
      "  @server list\n"
      "  @server use bench1\n"
      "  @server stop bench1" },
    { "log", NULL,
      "@log [-n <N>|-a] [-g <pat>] [-e] [--slow <ms>] [--since <dur>] [-f] [-s <name>] [--json] "
      "[--clear]",
      "Inspect or follow the lock-free shared-memory command execution ring buffer",
      "  - -n, --last <N>    Show last N entries (default: 25)\n"
      "  - -a, --all         Show all buffered entries (up to 4096)\n"
      "  - -g, --grep <pat>  Filter commands by substring\n"
      "  - -e, --errors      Show only failed commands (rc != 0)\n"
      "  - --slow <ms>       Show only commands slower than <ms> milliseconds\n"
      "  - --since <dur>     Show entries within duration (e.g. 30s, 5m, 1h)\n"
      "  - -f, --follow      Stream new log entries in real time\n"
      "  - -s, --server <id> Inspect a specific named server's log\n"
      "  - --json / --clear  Emit JSON Lines or clear the ring buffer",
      "  @log -n 20\n"
      "  @log -e --slow 5.0" },
    { "streamcat", NULL, "@streamcat [-a <arr>|--raw] <stream_or_img>",
      "Read pixel data of a shared-memory stream or local image into an array or stdout",
      "  - With -a <arr>, binds pixel values directly into Bash array <arr> (zero subshell).\n"
      "  - Without --raw, prints formatted text values.\n"
      "  - With --raw, writes raw binary pixel bytes for piping to external processes.",
      "  @streamcat -a pix cam0\n"
      "  @streamcat cam0 | head -n 20\n"
      "  @streamcat --raw cam0 | python3 analyze.py" },
    { "streamwrite", NULL,
      "@streamwrite [--raw] [--dtype <type>] <stream> [xsize [ysize [zsize]]] [-|values...]",
      "Write values (or stdin with '-') into a shared-memory stream, creating it if needed",
      "  - Supported --dtype values: F32 (default), F64, U8, U16, U32, U64, I8, I16, I32, I64.\n"
      "  - Properly sets write flag, increments cnt0, and posts ImageStreamIO semaphores.",
      "  python3 gen_frame.py | @streamwrite --dtype F32 cam0 128 128 -\n"
      "  @streamwrite cam0 1 2 3 4" },
    { "unload", NULL, "@unload",
      "Shut down the local session server and remove all @ functions and PATH symlinks",
      "  - Stops active '@on' watchers, restores PS1 and PATH, and unsets @ shims.", "  @unload" },
    { NULL, NULL, NULL, NULL, NULL, NULL }
};

/**
 * @brief Print detailed help if @cmdkey is a `milk.bash` `@` primitive.
 *
 * @param cmdkey Command name (with or without leading `@`)
 * @return 0 if matched and printed, -1 otherwise
 */
static int help_milk_bash_primitive(const char *cmdkey)
{
    if (cmdkey == NULL)
    {
        return -1;
    }
    if (strcmp(cmdkey, "@") == 0)
    {
        cmdkey = "calc";
    }
    else if (cmdkey[0] == '@' && cmdkey[1] != '\0')
    {
        cmdkey++;
    }

    for (int i = 0; g_milk_bash_help[i].name != NULL; i++)
    {
        if (strcmp(cmdkey, g_milk_bash_help[i].name) == 0 ||
            (g_milk_bash_help[i].alias != NULL && strcmp(cmdkey, g_milk_bash_help[i].alias) == 0))
        {
            printf("\n");
            printf(COLORCMD "@%s" COLORRESET " [milk.bash Primitive]\n"
                            "\t" COLORINFO "%s\n" COLORRESET,
                   g_milk_bash_help[i].name, g_milk_bash_help[i].summary);
            printf("\n  " C_BOLD "Synopsis:" C_RST "\n    %s\n", g_milk_bash_help[i].synopsis);
            printf("\n  " C_BOLD "Details:" C_RST "\n%s\n", g_milk_bash_help[i].details);
            printf("\n  " C_NOTE "Examples:" C_RST "\n" C_BOLD "%s" C_RST "\n\n",
                   g_milk_bash_help[i].examples);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief command help\n
 *
 * @param[in] cmdkey Commmand name
 */

errno_t help_command(const char *__restrict cmdkey)
{
    int cOK = 0;

    if (cmdkey != NULL && cmdkey[0] == '@' && cmdkey[1] != '\0')
    {
        cmdkey++;
    }

    if (help_milk_bash_primitive(cmdkey) == 0)
    {
        return RETURN_SUCCESS;
    }

    for (unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {
        const char *dot = strchr(data.cmd[cmdi].key, '.');
        int is_match    = (!strcmp(cmdkey, data.cmd[cmdi].key)) ||
                          (strchr(cmdkey, '.') == NULL && dot != NULL && !strcmp(cmdkey, dot + 1));
        if (is_match)
        {
            printf("\n");
            printf(COLORCMD "%s" COLORRESET " in %s [%s]\n"
                            "\t" COLORINFO "%s\n" COLORRESET,
                   data.cmd[cmdi].key, data.cmd[cmdi].module,
                   data.module[data.cmd[cmdi].moduleindex].shortname, data.cmd[cmdi].info);

            printf("\t\033[33mexample>\033[0m"
                   " \033[1m%s\033[0m\n",
                   data.cmd[cmdi].example);
            printf("\t\033[2msrc: %s\033[0m"
                   "\n",
                   data.cmd[cmdi].srcfile);

            int FPSsupport =
                checkFlag64(data.cmd[cmdi].cmdsettings.flags, CLICMDFLAG_FPS, "FPS support");
            if (FPSsupport == 1)
            {
                if (checkFlag64(data.cmd[cmdi].cmdsettings.flags, CLICMDFLAG_PROCINFO,
                                "processinfo support (..procinfo 0/1)") == 1)
                {
                    printf("        loopcntMax         : %ld\n",
                           data.cmd[cmdi].cmdsettings.procinfo_loopcntMax);
                    printf("      Triggering:\n");

                    printf("        triggermode        : %d ",
                           data.cmd[cmdi].cmdsettings.triggermode);
                    switch (data.cmd[cmdi].cmdsettings.triggermode)
                    {
                    case PROCESSINFO_TRIGGERMODE_IMMEDIATE:
                        printf("IMMEDIATE");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT0:
                        printf("CNT0");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT1:
                        printf("CNT1");
                        break;
                    case PROCESSINFO_TRIGGERMODE_SEMAPHORE:
                        printf("SEMAPHORE");
                        break;
                    case PROCESSINFO_TRIGGERMODE_DELAY:
                        printf("DELAY");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT2:
                        printf("CNT2");
                        break;
                    default:
                        printf("unknown");
                        break;
                    }
                    printf("\n");

                    printf("        triggerstreamname  : %s\n",
                           data.cmd[cmdi].cmdsettings.triggerstreamname);

                    printf("        semindexrequested  : %d\n",
                           data.cmd[cmdi].cmdsettings.semindexrequested);

                    printf("        triggerdelay       : "
                           "%lld.%09ld\n",
                           (long long) data.cmd[cmdi].cmdsettings.triggerdelay.tv_sec,
                           data.cmd[cmdi].cmdsettings.triggerdelay.tv_nsec);

                    printf("        triggertimeout     : "
                           "%lld.%09ld\n",
                           (long long) data.cmd[cmdi].cmdsettings.triggertimeout.tv_sec,
                           data.cmd[cmdi].cmdsettings.triggertimeout.tv_nsec);

                    printf("      Resources:\n");
                    printf("        RT_priority        : %d\n",

                           data.cmd[cmdi].cmdsettings.RT_priority);

                    printf("        CPUmask            : ");

                    int nproc = sysconf(_SC_NPROCESSORS_ONLN);
                    for (int cpu = 0; cpu < nproc; cpu++)
                    {
                        printf(" %d", CPU_ISSET(cpu, &data.cmd[cmdi].cmdsettings.CPUmask));
                    }
                    printf("\n");
                    printf("        MeasureTiming      : %d\n",
                           data.cmd[cmdi].cmdsettings.procinfo_MeasureTiming);
                }
            }

            printf("\n");
            printf("  CLI call arguments:\n");
            //printf("  CLI#       tagname             Value         description\n");

            int CLIargcnt = 0;
            for (int argi = 0; argi < data.cmd[cmdi].nbparam; argi++)
            {
                char valuestring[STRINGMAXLEN_CLICMDARG] = "???";

                switch (data.cmd[cmdi].argdata[argi].type)
                {
                case CLIARG_FLOAT32:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[float32]  %f",
                                   data.cmd[cmdi].argdata[argi].val.f32);
                    break;

                case CLIARG_FLOAT64:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[float64]  %lf",
                                   data.cmd[cmdi].argdata[argi].val.f64);
                    break;

                case CLIARG_ONOFF:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[ONOFF]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.ui64);
                    break;

                case CLIARG_INT32:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[int32]  %d",
                                   data.cmd[cmdi].argdata[argi].val.i32);
                    break;

                case CLIARG_UINT32:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[uint32]  %u",
                                   data.cmd[cmdi].argdata[argi].val.ui32);
                    break;

                case CLIARG_INT64:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[int64]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.i64);
                    break;

                case CLIARG_UINT64:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[uint64]  %lu",
                                   data.cmd[cmdi].argdata[argi].val.ui64);
                    break;

                case CLIARG_STR_NOT_IMG:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[STRnI]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case CLIARG_IMG:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[STREAM]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_FITSFILENAME:
                case FPTYPE_FILENAME:
                case FPTYPE_DIRNAME:
                case FPTYPE_EXECFILENAME:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[FILE ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_STRING:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[STRING]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_PID:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[ PID ]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.i64);
                    break;

                case FPTYPE_TIMESPEC:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[TIME ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_PROCESS:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[PROCESS]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_FPSNAME:
                    SNPRINTF_CHECK(valuestring, STRINGMAXLEN_CLICMDARG, "[FPSNAME]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;
                }

                if (data.cmd[cmdi].argdata[argi].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
                {
                    printf("%6d   " COLORARGCLI " %-16s" COLORRESET " %-24s %s\n", CLIargcnt,
                           data.cmd[cmdi].argdata[argi].fpstag, valuestring,
                           data.cmd[cmdi].argdata[argi].descr);
                    CLIargcnt++;
                }
                else
                {
                    printf("[hidden] " COLORARGnotCLI " %-16s" COLORRESET " %-24s %s\n",
                           data.cmd[cmdi].argdata[argi].fpstag, valuestring,
                           data.cmd[cmdi].argdata[argi].descr);
                }
            }

            printf("\n");

            cOK = 1;
        }
    }

    int foundsubstring  = 0;
    int foundregexmatch = 0;
    if (cOK == 0)
    {
        printf("Command \"%s\" does not exist. Partial matches:\n", cmdkey);

        regex_t regex;
        int     reti;
        /* Compile regular expression */
        reti = regcomp(&regex, cmdkey, REG_EXTENDED);
        if (reti)
        {
            PRINT_ERROR("Could not compile regex : \"%s\"", cmdkey);
            exit(1);
        }
        int        maxGroups = 8;
        regmatch_t groupArray[maxGroups];

        for (unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
        {
            int matchsubstring = 0;
            // look for substring match

            if (strstr(data.cmd[cmdi].key, cmdkey) != NULL)
            {
                foundsubstring = 1;
                matchsubstring = 1;
                printf(COLORCMD "%s" COLORRESET " in %s [%s]\n\t" COLORINFO "%s\n" COLORRESET,
                       data.cmd[cmdi].key, data.cmd[cmdi].module,
                       data.module[data.cmd[cmdi].moduleindex].shortname, data.cmd[cmdi].info);
            }

            // Regular expression search
            if (matchsubstring == 0)
            {
                // Regular expression search
                reti = regexec(&regex, data.cmd[cmdi].key, maxGroups, groupArray, 0);
                if (!reti)
                {
                    foundregexmatch = 1;
                    printf(COLORCMD "%s" COLORRESET " in %s [%s]\n\t" COLORINFO "%s\n" COLORRESET,
                           data.cmd[cmdi].key, data.cmd[cmdi].module,
                           data.module[data.cmd[cmdi].moduleindex].shortname, data.cmd[cmdi].info);

                    char        *cursor = data.cmd[cmdi].key;
                    unsigned int offset = 0;
                    for (int g = 0; g < maxGroups; g++)
                    {
                        if (groupArray[g].rm_so == (regoff_t) ((size_t) -1))
                        {
                            break; // No more groups
                        }

                        if (g == 0)
                        {
                            offset = groupArray[g].rm_eo;
                        }

                        char cursorCopy[strlen(cursor) + 1];
                        memcpy(cursorCopy, cursor, strlen(cursor) + 1);
                        cursorCopy[groupArray[g].rm_eo] = 0;
                        /*printf("\t    Match Group %u: [%2u-%2u]: %s\n",
                               g, groupArray[g].rm_so, groupArray[g].rm_eo,
                               cursorCopy + groupArray[g].rm_so);*/
                    }
                    cursor += offset;
                }
                else if (reti == REG_NOMATCH)
                {
                    //puts("No match");
                }
                else
                {
                    char msgbuf[100];
                    regerror(reti, &regex, msgbuf, sizeof(msgbuf));
                    PRINT_ERROR("Regex match failed: %s", msgbuf);
                    exit(1);
                }
            }
        }

        regfree(&regex);

        if (foundsubstring == 0)
        {
            if (foundregexmatch == 0)
            {
                printf("\tNo substring or regex "
                       "match to \"%s\"\n",
                       cmdkey);
            }
        }
        return RETURN_FAILURE;
    }

    return RETURN_SUCCESS;
}


/**
 * @brief search for string in command info
 *
 */

errno_t command_info_search(const char *restrict searchstring)
{
    int foundsubstring  = 0;
    int foundregexmatch = 0;
    int colorcodecmd    = 31; // red
    int colorcodeinfo   = 32; // green

    regex_t regex;
    /* Compile regular expression */
    int reti = regcomp(&regex, searchstring, REG_EXTENDED);
    if (reti)
    {
        PRINT_ERROR("Could not compile regex : \"%s\"", searchstring);
        exit(1);
    }
    int        maxGroups = 8;
    regmatch_t groupArray[maxGroups];

    for (unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {
        int matchsubstring = 0;
        // look for substring match

        if (strstr(data.cmd[cmdi].info, searchstring) != NULL)
        {
            foundsubstring = 1;
            matchsubstring = 1;
            printf("%c[%d;%dm%s%c[%dm in %s [%s]\n\t%c[%d;%dm%s%c[%dm\n", (char) 27, 1,
                   colorcodecmd, data.cmd[cmdi].key, (char) 27, 0, data.cmd[cmdi].module,
                   data.module[data.cmd[cmdi].moduleindex].shortname, (char) 27, 1, colorcodeinfo,
                   data.cmd[cmdi].info, (char) 27, 0);
        }

        // Regular expression search
        if (matchsubstring == 0)
        {
            // Regular expression search
            reti = regexec(&regex, data.cmd[cmdi].info, maxGroups, groupArray, 0);
            if (!reti)
            {
                foundregexmatch = 1;

                printf("%c[%d;%dm%s%c[%dm in %s "
                       "[%s]\n\t%c[%d;%dm%s%c[%dm\n",
                       (char) 27, 1, colorcodecmd, data.cmd[cmdi].key, (char) 27, 0,
                       data.cmd[cmdi].module, data.module[data.cmd[cmdi].moduleindex].shortname,
                       (char) 27, 1, colorcodeinfo, data.cmd[cmdi].info, (char) 27, 0);

                char        *cursor = data.cmd[cmdi].info;
                unsigned int offset = 0;
                for (int g = 0; g < maxGroups; g++)
                {
                    if (groupArray[g].rm_so == (regoff_t) ((size_t) -1))
                    {
                        break; // No more groups
                    }

                    if (g == 0)
                    {
                        offset = groupArray[g].rm_eo;
                    }

                    char cursorCopy[strlen(cursor) + 1];
                    memcpy(cursorCopy, cursor, strlen(cursor) + 1);
                    cursorCopy[groupArray[g].rm_eo] = 0;
                    /*printf("\t    Match Group %u: [%2u-%2u]: %s\n",
                           g, groupArray[g].rm_so, groupArray[g].rm_eo,
                           cursorCopy + groupArray[g].rm_so);*/
                }
                cursor += offset;
            }
            else if (reti == REG_NOMATCH)
            {
                //puts("No match");
            }
            else
            {
                char msgbuf[100];
                regerror(reti, &regex, msgbuf, sizeof(msgbuf));
                PRINT_ERROR("Regex match failed: %s", msgbuf);
                exit(1);
            }
        }
    }

    regfree(&regex);

    if (foundsubstring == 0)
    {
        if (foundregexmatch == 0)
        {
            printf("\tNo substring or regex match to \"%s\"\n", searchstring);
        }
    }

    return RETURN_SUCCESS;
}


/**
 * @brief Top-level 'help' command handler.
 *
 * Dispatches to topic-specific help pages or the
 * general CLI help summary. Supports --json and
 * --porcelain output modes for machine-readable
 * output.
 */
errno_t help()
{
    int         json_mode      = 0;
    int         porcelain_mode = 0;
    const char *topic          = NULL;

    for (int arg = 1; arg < data.cmdNBarg; arg++)
    {
        if (data.cmdargtoken[arg].type == CMDARGTOKEN_TYPE_STRING ||
            data.cmdargtoken[arg].type == CMDARGTOKEN_TYPE_RAWSTRING)
        {
            if (strcmp(data.cmdargtoken[arg].val.string, "--json") == 0)
            {
                json_mode = 1;
            }
            else if (strcmp(data.cmdargtoken[arg].val.string, "--porcelain") == 0)
            {
                porcelain_mode = 1;
            }
            else
            {
                topic = data.cmdargtoken[arg].val.string;
            }
        }
    }

    help_format_mode = json_mode ? 1 : (porcelain_mode ? 2 : 0);

    if (topic != NULL)
    {
        if (topic[0] == '@' && topic[1] != '\0')
        {
            topic++;
        }
        if (help_topic_dispatch(topic) != 0)
        {
            int matched = 0;
            if (help_milk_bash_primitive(topic) == 0)
            {
                matched = 1;
            }
            for (unsigned int cmdi = 0; !matched && cmdi < data.NBcmd; cmdi++)
            {
                const char *dot = strchr(data.cmd[cmdi].key, '.');
                if (strcmp(topic, data.cmd[cmdi].key) == 0 ||
                    (strchr(topic, '.') == NULL && dot != NULL && strcmp(topic, dot + 1) == 0))
                {
                    help_command(topic);
                    matched = 1;
                    break;
                }
            }
            if (!matched)
            {
                for (int m = 0; m < data.NBmodule; m++)
                {
                    if (strcmp(topic, data.module[m].name) == 0 ||
                        strcmp(topic, data.module[m].shortname) == 0)
                    {
                        list_commands_module(topic);
                        matched = 1;
                        break;
                    }
                }
            }
            if (!matched && help_format_mode == 0)
            {
                printf(C_ERR "Unknown help topic: \"%s\"" C_RST "\n\n", topic);
                print_help_topic_list();
            }
        }
    }
    else
    {
        print_milk_top_help();
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Display readline keybinding help.
 *
 * Pages through the doc/helpreadline.md file
 * using the system pager.
 */
errno_t helpreadline()
{
    EXECUTE_SYSTEM_COMMAND_NOCHECK("more %s/src/CommandLineInterface/doc/helpreadline.md",
                                   dcsourcedir);

    return RETURN_SUCCESS;
}

/**
 * @brief Show detailed help for a specific command.
 *
 * If a command name is provided as argument 1,
 * prints its full help (syntax, arguments, flags).
 * If no argument, falls back to list_commands().
 */
errno_t help_cmd()
{
    if ((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_EXISTINGIMAGE) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_COMMAND) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        const char *cname = data.cmdargtoken[1].val.string;
        if (cname[0] == '@' && cname[1] != '\0')
        {
            cname++;
        }
        help_command(cname);
    }
    else
    {
        list_commands();
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Search command descriptions for a keyword.
 *
 * If a search string is provided as argument 1,
 * calls command_info_search() to find matching
 * commands. If no argument, lists all commands.
 */
errno_t cmdinfosearch()
{
    if ((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_EXISTINGIMAGE) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_COMMAND) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        command_info_search(data.cmdargtoken[1].val.string);
    }
    else
    {
        list_commands();
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Show module info or list all loaded modules.
 *
 * If a module name is provided as argument 1,
 * shows that module's commands. Otherwise lists
 * all loaded modules with version and description.
 */
errno_t help_module()
{
    if ((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_EXISTINGIMAGE) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_COMMAND) ||
        (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        list_commands_module(data.cmdargtoken[1].val.string);
    }
    else
    {
        printf("\n");
        printf("%ld modules loaded\n", data.NBmodule);
        printf("%2s  %10s %32s %10s %7s    %20s %s\n", "#", "shortname", "Name", "Package",
               "Version", "last compiled", "description");
        printf("--------------------------------------------------------------"
               "-----------------------------------------"
               "-------\n");
        for (long i = 0; i < data.NBmodule; i++)
        {
            printf("%2ld %10s \033[1m%32s\033[0m %10s %2d.%02d.%02d    "
                   "%11s %8s  %s\n",
                   i, data.module[i].shortname, data.module[i].name, data.module[i].package,
                   data.module[i].versionmajor, data.module[i].versionminor,
                   data.module[i].versionpatch, data.module[i].datestring,
                   data.module[i].timestring, data.module[i].info);
        }
        printf("--------------------------------------------------------------"
               "-----------------------------------------"
               "\n");
        printf("\n");
    }

    return RETURN_SUCCESS;
}
