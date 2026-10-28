---
tags:
  - cli
  - scripting
---

# Scripting

A milk script is a bash script with three additions:

- module commands, written `module.command` (`mem.mk2Dim`, `iofits.loadfits`);
- language keywords, written `@keyword` (`@calc`, `@get`, `@waitfor`);
- the `@{…}` expansion, which reads an entity property or computes an expression inside a word.

Scripts run in `milk-cli` (interactively, with `-s`, `-c` or piped input), in `milk-script`, or in
any bash shell that has loaded `milk.bash`. Everything else is bash, unchanged: variables, arrays,
`$(( ))`, `[[ ]]`, functions, `if`/`for`/`while`/`case`, pipes, redirections, heredocs, `trap`,
background jobs. A binary run by absolute path finds its own install:
when `MILK_INSTALLDIR` is unset it is set from the executable's location, so modules, `milk.bash` and
the `milk-fpsexec-*` units all come from that install. To work with several builds, source the
`bin/milk.bash` of the one you want (it sets `MILK_INSTALLDIR`, `PATH` and `LD_LIBRARY_PATH`) rather
than editing those variables by hand.

See also: [Calculator & Compute Units](calculator.md) · [Sessions](sessions.md) ·
[CLI Syntax](CLIcore.md) · [Stream Slicing](slicing.md) ·
[C Extensions (`milk_calc_ext.h`)](calc_c_extensions.md)

## The Language at a Glance

| You write                     | It is                                                                     |
| ----------------------------- | ------------------------------------------------------------------------- |
| `word args`                   | bash                                                                      |
| `module.command args`         | a module command: a bash function defined from the install catalog       |
| `@keyword args`               | one of the 28 [keywords](#keywords)                                      |
| `@{expr}` inside a word       | an entity path reads a property, anything else goes to the calculator    |
| `\@{`                         | a literal `@{`                                                            |
| `<anything> -h`, `--help`     | help                                                                      |

`@` has no other meaning. `user@host`, `curl -d @file`, `git checkout @{-1}` and `"HEAD@{1}"` are
passed to bash as written: an `@{` right after a letter, digit or `_` is not expanded, nor is one
that holds only an integer (git's `@{1}`, `@{-1}`). To expand right after a letter, quote the
expansion: `img"@{n}"`. Inside calculator text, entities are written without `@`:
`@calc "x = s.cam.xsize * 2"`.

| Guide Page | What It Covers |
| --- | --- |
| **[Scripting (this page)](scripting.md)** | Language rules, module commands, `@args`, entities (`s.`, `l.`, `f.`, `p.`), waiting/events (`@waitfor`, `@grab`, `@on`, `@defer`), exit codes, cost summary, migration, and complete examples |
| **[Calculator & Compute Units](calculator.md)** | `@calc` / `@test` / `@pipe` expressions, full operator & function tables (math, reductions, stencils, BLAS/LAPACK), `milk-fpsexec-milk-calc` / `milk-fpsexec-milk-script` units, and SIMD/OpenMP fusion rules |
| **[Sessions](sessions.md)** | `@session` lifecycle (`new`, `use`, `leave`, `ls`, `end`, `save`/`restore`), `source milk.bash`, multi-shell concurrency, and `@snapshot`/`@diff`/`@restore`/`@log` |
| **[Stream Slicing](slicing.md)** | Zero-copy image and stream views (`im[x0:x1, :]`, strided/binned slices) in calculator expressions and stream modifiers |
| **[C Extensions (`milk_calc_ext.h` & `milk-cc`)](calc_c_extensions.md)** | Extending `@calc` and `milk.calc` with auto-compiled `.c` kernels (`-l file.c`) and compiling standalone `MILK_RUNNABLE` units (`milk-cc`) |

## Running Scripts

| How                                 | What happens                                                    |
| ----------------------------------- | --------------------------------------------------------------- |
| `milk-cli -s script.milk [args]`    | runs the file and exits with its status                         |
| `milk-cli -c 'commands'`            | runs the string and exits with its status                       |
| `cmds \| milk-cli`                  | runs each input line; `$?` carries over from line to line       |
| `#!/usr/bin/env milk-script`        | standalone runner, no readline (`milk-script -e` stops on errors) |
| `@source [--once] file.milk [args]` | runs a file in the current shell, `$1`... set, `@{…}` expanded  |
| `source milk.bash`                  | adds milk to an ordinary bash shell ([Sessions](sessions.md#using-milk-from-plain-bash)) |

A plain `source file.milk` runs the file as bash: its `@{…}` stay as written. `~/.milkrc` runs only
when milk-cli starts an interactive prompt: not with `-c`, `-s` or piped input, and not in
milk-script, a session's server or plain bash. `@history` shows the lines typed at milk-cli prompts,
or piped into milk-cli, in the session the shell uses (`--global`: in all sessions), from any
front-end; `@history save file` writes the prompt's history to a file that can be run again. The
log they come from, `~/.milk_history_log`, is never trimmed: delete it to start afresh.

## Keywords

| Group         | Keywords                                                                        |
| ------------- | ------------------------------------------------------------------------------- |
| values        | `@calc` `@test` `@assert` `@get` `@set` `@exists` `@running` `@ls`              |
| events        | `@waitfor` `@waitany` `@on` `@sleep`                                            |
| data          | `@grab` `@replay` `@snapshot` `@diff` `@restore` `@pipe`                        |
| program       | `@source` `@defer` `@args` `@module` `@config` `@session` `@unload`              |
| introspection | `@help` `@log` `@history`                                                       |

`@help` with no argument lists them; `@help <keyword>` and `<keyword> --help` describe one. The
keywords that read or write shared memory and evaluate scalars are bash builtins (`type -t @get`),
so a call costs no shell function; the others are functions.
`@help -k <regex>` searches commands, keywords and topics. The expression keywords (`@calc`,
`@test`, `@assert`) take `--help` only, as `-h` is a valid expression. `@history` and the REPL
settings of `@config` exist only at the milk-cli prompt.

## Module Commands

Each command has one name, `<module>.<command>`. It is a bash function in every front-end, defined
from the catalog written at install (`share/milk/catalog.bash`); `libexec/milk/bin` holds links for
external tools (`xargs`, `timeout`). A module is loaded the first time one of its commands runs.
`@module list [--all]` shows the modules, and `@module load <name|file.so> [--as <short>]` loads a
module ahead of time or one that is not installed.

```bash
mem.mk2Dim im 512 512              # make a 2-D image
mem.mk2Dim -h                      # its help: arguments, defaults, presets
@help mem                          # every command of a module
```

### Arguments

```bash
mem.mk2Dim im 64 64                # values in order
mem.mk2Dim .ysize=32 im 64         # .tag=value anywhere; the other values fill the rest in order
mem.mk2Dim im . 32                 # '.' is the default value
mem.mk2Dim --default .xsize=128    # session default (does not run); '.tag=.' clears it
mem.mk2Dim --save=big im 2048 2048 # run, and keep the values as preset "big"
mem.mk2Dim --load=big im2          # start from the preset
mem.mk2Dim --no-run im 64 64       # print the resolved call, run nothing
```

- Every call starts from the declared defaults overlaid with the session defaults: nothing carries
  over from one call to the next, and two shells never share argument state.
- A value is taken from, first to last: `.tag=value` or its position, the `--load` preset, the
  session default, the declared default.
- A missing, invalid or extra value is a usage error: the command does not run and exits with
  status 2. `-0.5` is a value: `-h` is the only short option, and after `--` every word is a value.
- A number argument is a literal or a calculator expression (`2*n0`, with calculator variables).
  It never reads bash variables: write `$n`.
- Presets are files under `${XDG_STATE_HOME:-~/.local/state}/milk/args/<command>/<NAME>`, one
  `.tag=value` per line.
- In an image argument, `"@L:name"` makes a local image and `"@S:name"` a shared stream.

### Your Own Commands

`@args` declares the arguments of a function or script; they are checked as a module command's are:

```bash
# Average frames of a camera
cam.average() {
    @args 'cam:str:camera stream' 'n:int=100:frames to average' -- "$@"
    @grab "s.${cam}" "${n}" "@L:avg_cube"
    @calc "avg = imean(avg_cube, 2)"
}
cam.average cam0 .n=50       # values in order, or .name=value; '.' keeps a default
cam.average -h               # the usage, from the declaration
```

- A declaration is `name:type[=default][:help]`, with type `int` `uint` `float` `bool` (bound as
  1 or 0), `str`, `stream` or `fits`. Without a default a value is required.
- The names become local variables of the function (globals at a script's top level). After `-h`,
  or a missing, invalid or extra value, the function returns (status 0 or 2); a script ends.
- Without the libmilkbash builtins (`MILK_NO_BUILTINS=1`), `@args` assigns plain variables and
  only returns its status: write `@args ... -- "$@" || return`, and declare the names `local`.
- `*.bash` files in `$MILK_INSTALLDIR/share/milk/commands.d` and
  `${XDG_CONFIG_HOME:-~/.config}/milk/commands.d` are loaded with milk.bash (in plain bash, milk-cli
  and milk-script). The `module.cmd` functions they define are commands like the others: completion,
  `@unload`, and `@help` (the comment line right above a definition is its description).

## Entities

An entity path is `<namespace>.<name>.<property>`:

| Namespace        | Refers to                                   |
| ---------------- | ------------------------------------------- |
| `s.` `stream.`   | shared-memory stream (`$MILK_SHM_DIR`)      |
| `l.` `local.`    | image local to the milk-cli process         |
| `f.` `fps.`      | function parameter structure (FPS)          |
| `p.` `proc.`     | processinfo loop                            |

Images created by `mem.mk2Dim` and other commands are shared streams by default.

### Reading

```bash
@get s.cam.xsize                   # print a property
@get -v nx s.cam.xsize             # bind it to a bash variable (no subshell)
@get nx=s.cam.xsize ny=s.cam.ysize # several values in one call
@get -A st s.cam.stats             # pixel statistics: ${st[mean]} ${st[std]} ...
@get -a names 's.*'                # matching entity names into an indexed array
@get -A sizes 's.dm*.xsize'        # name -> value into an associative array
@get --json s.cam                  # all properties as JSON (also f.<fps>, p.<proc>)
@get --watch 1 s.cam.cnt0 p.loop.hz # every second until Ctrl-C (default 0.5 s)
nx=@{s.cam.xsize}                  # inline, inside a word (not right after a letter)
echo "frame @{s.cam.cnt0} of @{s.cam.xsize} x @{s.cam.ysize}"
n=cam; echo "@{s.${n}.xsize}"      # bash variables inside the path
echo "\@{s.cam.xsize}"             # escaped: printed as is
```

Common properties:

- streams (`s.`, `l.`): `xsize` `ysize` `zsize` `naxis` `nelem` `type` `typename` `cnt0` `cnt1`
  `sem` `shared` `age` `fps` `ownerPID` `kw.<KEY>` (or `k.<KEY>`, FITS keyword), `kw.*`, `list`,
  `summary`, and the pixel statistics `min` `max` `mean` `std` `var` `tot` `count` (non-zero
  pixels), `stats` (all of them, plus `torn`, 1 when three attempts 20 µs apart all overlapped a
  write and the values come from the last one; computed in one SIMD pass with a `(cnt0, write)`
  torn-frame guard and cached per `(inode, cnt0, 16-pixel sample)` so reading several properties
  on the same frame scans the pixels only once: a writer that changes pixels in place without
  advancing `cnt0` or the write time can get a cached value when those 16 pixels did not change);
- FPS (`f.`): any parameter (`f.loop.gain`), plus `status` (or `run`) and `conf`, the state of the
  unit's run and conf processes: `stopped` `starting` `running` `stopping` `exited` `failed`;
  `session`; `run.out` and `conf.out`, the last 20 lines the unit wrote (`milk-fps-log` shows more);
  `confpid` `runpid` `nbparam` `list` `description`; `ok` `errors` `msg`, the last parameter check
  (confstep, confstart or runstart; `@set f.X.conf=step` runs one): `ok` is 1 if it passed (0 before
  any check), `errors` counts the parameters that failed, `msg` gives one `<parameter>: <message>`
  line each. An exact parameter name wins over these. `procinfo.triggermode` reads as a name and
  takes a name or a number: `immediate` `cnt0` `cnt1` `semaphore` `delay` `semaphore-timeout` `cnt2`;
- processes (`p.`): `status` `loopcnt` `loopfreq` (or `hz`) `dt` `dtmax` `dt_median_ns` `dt_max_ns`
  `exectime` `exec_max` `jitter` (all from the process's timing ring: median and maximum iteration
  interval in µs, and in ns for the `_ns` ones, median and maximum busy time in µs, largest deviation
  of the interval from its median in µs) `trigmode` (a name, as above) `ctrl` `pid` `missedframes`
  `timeouts` (trigger waits that timed out) `triggerstream` `unit` (the unit the process runs)
  `statusmsg`. `@waitfor p.<name>` waits until the process stops; a clean exit, which removes its
  processinfo file, counts.

`@get --json <entity>` lists everything an entity provides.

### Writing and Actions

```bash
@set f.loop.gain=0.25              # one value
@set f.loop.gain = 0.25            # same, '=' may be spaced
@set f.loop.gain=0.25 f.loop.mult=0.99   # several in one call
@set 'f.dm*.xsize' = 128           # every match of a glob
@set s.cam.kw.OBS=night "comment"  # FITS keyword, optional comment
@set f.loop.conf=start             # FPS: conf=start|stop|step, run=start|stop, once done
@set p.loop.ctrl=pause             # processinfo control: run pause step stop ...
@set p.loop.pause                  # same, as an action
@set --restore f.loop.gain=0       # the old value comes back when the calling function returns
@set --tui loop                    # browse and edit FPS parameters (milk-cli prompt)
```

`@get --watch [T] <spec>...` redraws the values in place at a terminal; when its output is not a
terminal it prints one tab-separated line per refresh, for a log. Ctrl-C ends it, with status 0.

A parameter sweep is a loop: `@set --restore` puts the parameter back when the function returns, and
`@calc -v` binds what each step measures.

```bash
sweep_gain() {
    @set --restore f.loop.gain=0.1     # the gain comes back when sweep_gain returns
    local g rms
    for g in 0.1 0.2 0.5 1.0; do
        @set f.loop.gain=$g
        @waitfor s.wfs_resid 100 -t 5  # let 100 frames go by
        @calc -v rms "istd(wfs_resid)"
        printf '%s\t%s\n' "$g" "$rms"
    done
}
```

### Existence and Listing

```bash
@exists s.cam && echo yes          # exit status 0 if the entity exists
@running f.loop                    # 0 if the FPS or process runs
@ls                                # table of streams, FPS and processes
@ls 's.cam*'                       # one namespace, with a glob
@ls -a list 'f.*'                  # names into an array (also -1, -l)
@ls --json                         # JSON snapshot
```

## Calculator

```bash
@calc "cam = cam * 0 + 5"          # image arithmetic, in place
@calc -v m "imean(cam)"            # bind the result to a bash variable
@calc -v rms,ptv 'istd(cam); imax(cam) - imin(cam)'   # the values of the last statements
@calc -a row "cam[*, 10]"          # the pixels of an image into a bash array
@calc "m, s, lo, hi = istats(cam)" # single-pass mean, std, min, max unpacking
@calc "z = (cam - imean(cam)) / istd(cam)"            # multi-reduction in one fused expression
@calc "clean = medfilt3(cam - dark)"                  # exact 3x3 SIMD median filter
@calc "dm = 0.99 * dm - 0.3 * matmul(Cmat, wfs)"      # lowered to single in-place BLAS gemv
@calc "cam[0:31, 0:31] = 0"        # slices: see slicing.md
echo "mean=@{imean(cam)}"          # inline
if @test "imean(cam) > 1"; then echo bright; fi        # status 0 if true, 1 if false
@pipe "mem.mk2Dim p 32 32 |> _ = 4 |> _ *= 2.5"       # chain stages in one request
```

- Names in an expression are images, calculator variables, and bash variables. A bash variable wins
  over a calculator variable of the same name; its value is data (`s="sqrt(16)"` is the string, not
  `4`). Calculator variables come back to bash only through `-v` (or `@{…}`): after `@calc "x = 5"`,
  bash has no `$x` — write `@calc -v x "5"` or `x=@{5}`.
- Scalar-only expressions (numbers, math functions, `s.`/`f.`/`p.` properties) evaluate in-process
  in the shell (~2 µs); image expressions execute in a single fused SIMD/OpenMP pass or in-place
  BLAS/LAPACK call in the session server.
- New images produced by an expression are `FLOAT` (or `DOUBLE` when an operand is `f64(...)`);
  in-place updates (`+=`, scalar fill, or writing an existing shared stream) preserve the
  destination's datatype.

For the complete operator table, function reference (math, reductions, `istats`/`icentroid`, axis
reductions, `shift`/`lut`, `medfilt3/5/7`, `gauss`, `interp2`, `pixremap`/`pixunmap`, `rprof`,
BLAS/LAPACK solvers, complex images, and `def` inline functions), and more examples, see
**[Calculator & Scripted Compute Units](calculator.md#calculator-calc-test-pipe)**.

## Waiting, Events and Cleanup

```bash
@waitfor s.cam                     # next frame
@waitfor s.cam 10 -t 2             # 10 frames, timeout 2 s (status 124 on timeout)
@waitfor s.cam --cnt 5000          # until cnt0 reaches 5000
@waitfor f.loop.gain=0.5 -t 5     # until a property has a value
@waitfor f.loop running -t 5       # until the unit's loop runs (stopped: until no process runs)
@waitfor "imean(cam) > 100" -t 5   # until an expression is true, checked on each frame of cam
@waitany s.cam s.wfs -t 1          # whichever updates first; its name is in $MILK_TRIGGER
@grab s.cam 100 "@L:cube"          # 100 consecutive frames into a cube
@grab --fits cube.fits --timestamps t.txt s.cam 100
@grab --stdout -a pix s.cam        # the next frame's pixels into a bash array
@replay --fps 100 --loop cube.fits sim   # play a cube or FITS file into stream sim
@on s.cam 'echo "new frame on $MILK_TRIGGER"'      # run on every update (background)
@on --once --max-hz 10 s.cam s.wfs -- process_frame
@on --when 'imax(cam) > 60000' --hold 0.5 -- echo saturated   # when it turns true (and stays)
@on --when 'imean(cam) > 100' --level -- log_frame                # every frame while it is true
@on --every 10 -- '@get s.cam.cnt0 >> counts.txt'                 # a periodic job
@on --list ; @on --stop s.cam ; @on --clear
@sleep 0.05                        # sleep without starting a process
unset t; while :; do @sleep --hz 100 t; process_frame; done   # a loop paced at 100 Hz
@sleep --every 0.5 --late missed t # $missed: ticks a late call skipped
```

`@sleep --every T VAR` (or `--hz F VAR`) sleeps to the deadline held in `VAR`, an integer count of
nanoseconds on the monotonic clock, then moves it one period on; an unset `VAR` starts one period
from now. The loop keeps its rate whatever its body costs, without drift. A call that comes after its
deadline returns at once and skips the ticks it missed instead of running them back to back.

`@waitfor` and `@waitany` sleep on the stream semaphores (keeping the stream mapping and semaphore
slot pinned for the duration of the wait), so they use no CPU while waiting. An expression is
evaluated again on each new frame of the streams it names, in the shell when it is scalar; FPS
and process values in it are polled. `@waitfor --next "<expr>"` starts from the next
frame. `@grab` copies frames outside the server command lock and records the total number of
skipped frames (`cnt0` gaps $> 1$ between captured frames) in the output cube's `DROPPED` FITS
keyword (`s.<cube>.kw.DROPPED` or `s.<cube>.k.DROPPED`). With `--`, `@on` runs the words after it
as a command; use the quoted form when the command refers to `$MILK_TRIGGER`. `@on --when` fires
when the expression turns true, also when it is already true at the start, then again once it has
been false. A watcher ends with the shell, or subshell, that started it, also when that shell is
killed; a watcher that must outlive its shell is a unit (`milk.script`). This takes milk.bash's
builtins (libmilkbash, loaded unless `MILK_NO_BUILTINS=1`): without them, a watcher outlives its
shell.

`@defer` runs a command when the function or `@source`'d file that registered it returns, last
registered first:

```bash
measure() {
    local tmp=/tmp/measure.$$
    @defer 'rm -f "$tmp"'              # runs when measure returns, with its locals
    @set --restore f.loop.gain=0       # set now; the old gain comes back when measure returns
    @defer --restore 's.cam.kw.*'      # every keyword of s.cam comes back
    @on --scoped s.cam -- process_frame    # the watcher stops when measure returns
    @defer --exit 'echo session done'  # at exit instead
    @waitfor s.cam 100 -t 5
}
@defer rm -f /tmp/session.log          # at the top level: when the shell or script exits
```

- A plain `source` is not a scope: entries registered in a file sourced from a function run when the
  function returns. At the top level, and with `--exit`, entries run at exit, after the scoped ones.
- Each entry runs once. A function's return status is kept.
- After Ctrl-C or a `set -e` failure, the skipped entries run before the next prompt or milk-cli
  command, or at exit. A subshell runs only the entries it registered itself.
- One argument is a command string, as for `trap`; several words are kept quoted.
- `@defer` uses the RETURN trap, and the EXIT trap. A user EXIT trap set before the first
  `@defer` runs after the entries; a RETURN trap set by the script replaces `@defer`'s.
- `@unload` runs the entries left and gives the EXIT trap back.

## Compute Units Without C

Two compute units run on each frame of a trigger stream (or on a `-loopd SEC` timer) without C
code: `milk-fpsexec-milk-calc` runs compiled calculator programs, and `milk-fpsexec-milk-script`
runs a bash snippet.

```bash
# out = (cam - dark) * gain on each frame of cam; gain is a parameter of FPS dsub, set to 2.5
milk-fpsexec-milk-calc -n dsub -t cam -p "gain:float=1" -e "out = (cam - dark) * gain" fpsinit 2.5
milk-fpsexec-milk-calc -n dsub runstart         # in the background, returns once the loop runs
milk-fpsexec-milk-calc -n dsub set 3.0          # read by the next frame (same as @set f.dsub.gain=3)
@set f.dsub.unit.expr="out = cam * gain"        # compiled at the next frame (or: -e "..." set)
@get f.dsub.unit.error                          # why the loop stopped (empty while it runs)
milk-fpsexec-milk-calc -n dsub runstop

# one-time initialization (-i) before frame 0, then per-frame leaky integrator, in this terminal
milk-fpsexec-milk-calc -n ema -t cam -p "a:float=0.1" -i "acc = cam * 0" \
    -e "acc = (1 - a) * acc + a * cam" exec .a=0.05

# a bash snippet on each frame of cam, in a shell that loaded milk.bash once
milk-fpsexec-milk-script -n mon -t cam -c '@get s.cam.mean >> cam.log' fpsinit
milk-fpsexec-milk-script -n mon runstart
```

For the full `.unit.*` option table, per-frame variables (`_n`, `_t`, `_dt`, `_cnt0`, `_ts`,
`_missed`, `_torn`), torn-frame protection, `.calc` program constructs (`if`, `for`, `local`,
`def`, `hist`/`prev`, `skip`, `post`, `zeros`), and OpenMP/deadline tuning, see
**[Calculator & Scripted Compute Units — Compute Units Without C](calculator.md#compute-units-without-c)**.

## State, Assertions and Logs

```bash
@snapshot -o base.json 'f.loop*' 's.cam*'     # FPS parameters and stream keywords to JSON
@diff base.json                               # compare with the live state
@restore -n base.json                         # dry run, then without -n to apply
@restore --undo                               # set back what this shell's last @set changed
@assert "istd(wfs) < 0.5" "residual diverged" # fails with the values of both sides
@assert --running --fps-min 500 s.cam "camera stalled"
@assert --max-age 0.1 --cnt-min 100 --dim 128x128 s.cam   # last frame at most 0.1 s old, ...
@assert s.dark -m "no dark"                   # an entity alone: it exists (-e, --exists)
@log -n 20                                    # recent milk commands, timing and output
@session                                      # the session in use (see Sessions)
@config prompt preset std                     # live telemetry in the prompt
@config                                       # digits, precision, procinfo
```

See [Sessions — Snapshots, Undo, Assertions, and Logs](sessions.md#snapshots-undo-assertions-and-logs)
for `@restore --undo`, `@session save`, and `@log` details.

## Errors and Exit Status

| Status | Meaning                                                         |
| ------ | --------------------------------------------------------------- |
| 0      | success                                                         |
| 1      | the command failed                                              |
| 2      | usage error: bad or missing argument, invalid value, syntax     |
| 124    | timeout (`@waitfor`, `@waitany`, `@grab -t`)                    |
| 127    | unknown command (the message names the replacement of an old spelling) |
| 130    | interrupted (Ctrl-C)                                            |
| 255    | the session cannot be reached (it was ended), or speaks another protocol version; `@get`, `@exists` and `@ls` read shared memory directly and do not need it |

- `milk-cli -s`, `milk-cli -c` and piped input exit with the status of the last command, and `$?`
  carries over from one input line to the next.
- A failed `@{…}` expands to an empty string after its message: `milk: @{s.cam.xsize}: inline
  expansion failed` for an entity, the calculator's error for an expression. An assignment such as
  `x=@{s.cam.xsize}` also returns its status. With `MILK_STRICT=1`
  the failure ends the script with that status, as with `set -u`; at a prompt it is only reported.
- Calculator errors are printed even in quiet mode (`milk: calc error: ... in '<expr>'`).
- Ctrl-C interrupts the running command line, including a milk command executing inside milk-cli.
  Shell variables, functions and background jobs survive. A script that sets its own `trap ... INT`
  handles Ctrl-C itself.
- If the bash process dies (killed, or recursion deeper than `FUNCNEST`, 5000 by default), milk-cli
  says so. A script stops there; at the prompt, the next command starts a new bash, without the
  previous shell state.

## Cost Model

Measured on a recent desktop machine (see `test_perf_latency_ceilings` in
`python/tests/cli/perf_ceilings_test.py` and `python/tests/perf/calc_bash_bench.py`):

| Form                                  | Time per call | Note                                     |
| ------------------------------------- | ------------- | ---------------------------------------- |
| `@calc -v y "1+2"` (scalar only)      | ~2 µs         | evaluated in the shell                   |
| `@get -v x s.cam.xsize`               | ~3 µs         | read from shared memory, no request      |
| `@set s.cam.kw.FOO=1`                 | ~5 µs         |                                          |
| `cmds \| milk-cli -q` (per line)      | ~2–5 µs       | non-TTY stdin lines batched without readline |
| `f() { @defer …; }; f`                | ~15 µs        | registered, then run when `f` returns    |
| `x=@{s.cam.xsize}`, `y=@{1+2}`        | ~2–3 µs       | a statement of its own: bound with `-v`  |
| `"frame @{s.cam.cnt0}"` (in a word)   | ~15 µs        | bash's `${\| …; }` substitution, no fork |
| `@calc "cam = cam * 2"` (images)      | ~30 µs        | one request to the server (32-entry LRU statement cache) |
| `mem.listim`, other milk commands     | ~75 µs        | one request to the server                |
| `y=$(@calc 1+2)`                      | ~0.8 ms       | `$( )` forks a subshell: avoid in loops  |

Prefer `-v`, `-a`, `-A` and `NAME=@{…}` over `$(…)` and over `@{…}` inside longer words in loops,
group work with `@pipe`, and use `@sleep` rather than `sleep` for short waits. On bash older than
5.3, `@{…}` inside a word falls back to `$(…)` and costs a fork; `NAME=@{…}` does not. For image
expression fusion, slice/broadcast views, and temporary pool reuse, see
**[Calculator & Scripted Compute Units — Fusion & Cost Model](calculator.md#fusion-cost-model)**.

## Sessions

Each shell (`milk-cli`, `milk-script`, or plain bash after `source milk.bash`) talks to a
`milk-cli --server` session that holds local images (`l.`), calculator variables, and settings, or
shares a named session across multiple shells (`@session new NAME`, `@session use NAME`). For
named-session lifecycle, `source milk.bash`, multi-shell concurrency, and `@session save`/`restore`,
see **[Sessions](sessions.md)**.

## Migrating Old Scripts

Old spellings fail with a message that names their replacement: status 127 for a removed command
word, 2 for a removed option or verb. Nothing rewrites a file: change it from the message, or from
this list:

- `@server` → `@session`, with `start` → `new` (or `use` for a session that runs), `list` → `ls`,
  `local` → `leave`, `stop` → `end`, `attach` → `log -f`; `@log --server` → `@log --session`;
  `source milk.bash --server NAME` → `source milk.bash --session NAME` (`-s` is not milk's any more);
- `@listim`, `@mem.listim`, `listim` → `mem.listim`; `@ci` → `milk.ci`;
- `@ x = e` → `@calc "x = e"`, or `@calc -v x "e"` to read the value into `$x`;
- `@s.cam.xsize` → `@get s.cam.xsize`, `@f.loop.gain=0.2` → `@set f.loop.gain=0.2`, inline
  `@s.cam.cnt0` and `${@s.cam.xsize}` → `@{s.cam.cnt0}`, `@s.x` inside calculator text → `s.x`;
- `@batch` → `@pipe`, `@streamcat` → `@grab --stdout`, `@streamwrite` → `@replay --stdin`,
  `@usleep` → `@sleep`, `@procwait` → `@waitfor p.<name>.status=...`;
- `@mload`, `@mloadas` → `@module load [--as]`; `@dpdigits` → `@config digits`; the history and
  help words → `@history`, `@help`, `-h`;
- `include_once` → `@source --once`, `source x.milk` → `@source x.milk`, `exitCLI` → `exit`;
- `cmd .tag value` → `cmd --default .tag=value`.

### From milk-seq

The sequencer (`milk-seq`, its `.seq` files, the `seq.*` commands, the `q.` properties and the
fpsCTRL scheduler screen) was removed: a queued workflow is a milk script. The old commands and
`@get q.<name>.<prop>` fail with a message that points here.

| milk-seq                                     | milk script                                                      |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `repeat N` … `endrepeat`                     | `for i in $(seq N); do …; done`                                  |
| `if_fps_status X running` … `endif`          | `if @running f.X; then …; fi`, tested when reached               |
| `wait_fps X running`, `wait_fps X norun`     | `@waitfor f.X running -t <sec>`, `@waitfor f.X stopped -t <sec>` |
| `runwait X`                                  | `@waitfor f.X stopped -t <sec>`                                  |
| `confstart X`, `runstart X`, `runstop X`     | `@set f.X.conf=start`, `@set f.X.run=start`, `@set f.X.run=stop` |
| `setval X.p v`, `getval X.p`, `fwrval X.p F` | `@set f.X.p=v`, `@get f.X.p`, `@get f.X.p >> F`                  |
| `fpsrm X`                                    | `milk-fps-rm X`                                                  |
| `on_error abort`, `skip`, `retry`            | `set -e`; `cmd \|\| true`; a loop around the command             |
| `include F`, `set V x`                       | `@source F`, `V=x`                                               |
| `wait_seq S idle`                            | `wait` for a background script                                   |

A workflow that `seq.start` ran in the background is a background script (`./flow.milk &`). Instead
of `seq.submit` or `seq_send`, other shells send commands to a named session (`@session use
<name>`), and scripts in different shells synchronize with `@waitfor` on an FPS parameter that one
of them sets.

## Examples

Monitor a stream and log its mean once per second:

```bash
#!/usr/bin/env milk-script
mem.mk2Dim "@S:demo_cam" 64 64
for i in 1 2 3; do
    @calc "demo_cam = $i * 10"
    @calc -v m "imean(demo_cam)"
    printf 'frame %s mean=%s\n' "@{s.demo_cam.cnt0}" "$m"
    @sleep 1
done
```

Capture a cube from a live stream and save statistics:

```bash
#!/usr/bin/env milk-script
@defer echo "done"
@waitfor s.wfs -t 5 || { echo "no frames"; exit 1; }
@grab s.wfs 50 "@L:wfs_cube"
@calc -v rms "istd(wfs_cube)"
@set s.wfs.kw.RMS="$rms" "rms of the last 50 frames"
echo "rms=$rms over @{l.wfs_cube.zsize} frames"
```

Tune an FPS gain until a stream settles:

```bash
#!/usr/bin/env milk-script
@running f.loop || @set f.loop.run=start
for g in 0.1 0.2 0.3 0.4; do
    @set f.loop.gain="$g"
    @waitfor s.dmdisp 20 -t 2
    if @test "istd(dmdisp) < 0.05"; then
        echo "gain $g is stable"
        break
    fi
done
```

Start a chain of units, each once the one before it has written a frame:

```bash
#!/usr/bin/env milk-script
milk-fpsexec-imggen-mkrandom -procinfo -loopd 0.01 fpsinit random_in 512 512 .distrib=0
milk-fpsexec-imggen-mkrandom -loopd 0.01 runstart || exit 1   # returns once the loop runs
@waitfor s.random_in -t 15 || exit 1                          # its first frame
milk-fpsexec-imgfilt-gaussfilt -procinfo -loops fpsinit random_in blurred_out 5.0 15
milk-fpsexec-imgfilt-gaussfilt -loops runstart || exit 1
@waitfor s.blurred_out -t 15 || exit 1
```

`runstart` returns once the unit's loop runs, and `@waitfor s.<out>` waits for the unit's first
frame, so the next unit starts on an input that has data: no `sleep` is needed. `milk-demopipeline`
starts its nine units this way (`milk-demopipeline cleanup` stops them).

## Syntax Highlighting (Neovim)

The `tree-sitter-milkcli` package provides a [tree-sitter](https://tree-sitter.github.io/) grammar
for `.milk` script files, enabling rich syntax highlighting in **Neovim** (≥ 0.9) and other
tree-sitter-capable editors.

### Quick Setup

From the milk source tree:

```bash
cd tree-sitter-milkcli

# 1. Build the parser from the grammar
./scripts/build.sh

# 2. Install into Neovim (queries + config)
./scripts/nvim-install.sh

# 3. Open a .milk file — highlighting is active
nvim examples/demo.milk
```

The install script copies highlight queries to `~/.config/nvim/after/queries/milkcli/` and creates a
Lua config snippet at `~/.config/nvim/plugin/milkcli-treesitter.lua`.

### What Gets Highlighted

| Syntax Element  | Highlight Group        | Example                                  |
| --------------- | ---------------------- | ---------------------------------------- |
| Flow control    | `@keyword`             | `if`, `for`, `while`, `fi`               |
| Shell builtins  | `@function.builtin`    | `echo`, `export`, `source`               |
| milk keywords   | `@function.macro`      | `@calc`, `@waitfor`, `@source`           |
| Module commands | `@function.call`       | `mem.listim` (builtin color, see below)  |
| Entity paths    | `@property`            | `s.cam.xsize`, `"@{f.${loop}.gain}"`     |
| Expansion       | `@punctuation.special` | `@{imean(cam)}`                          |
| Variables       | `@variable.builtin`    | `$VAR`, `${VAR}`                         |
| Strings         | `@string`              | `"hello"`, `'literal'`                   |
| Numbers         | `@number`              | `42`, `3.14`                             |
| Comments        | `@comment`             | `# comment`                              |

Heredocs are not parsed by the grammar: their body is highlighted as ordinary commands.

### Module Commands

Module commands are listed in the install catalog, not in the grammar. To highlight them, run the
generator script after installing new milk modules:

```bash
# Generate and install module highlight queries
./scripts/gen-module-highlights.sh --install

# Or generate to stdout for inspection
./scripts/gen-module-highlights.sh --stdout
```

### Management

```bash
# Check installation status (dry-run)
./scripts/nvim-install.sh --check

# Remove from Neovim
./scripts/nvim-install.sh --uninstall

# Rebuild parser after grammar changes
./scripts/build.sh
./scripts/build.sh --test    # build + parse example
./scripts/build.sh --clean   # remove generated files
```

<!-- prettier-ignore -->
!!! info "Requirements"
    - **Node.js** — for the `tree-sitter-cli` build tool
    - **Neovim ≥ 0.9** — with built-in tree-sitter support
    - **nvim-treesitter** plugin — installed automatically by `nvim-install.sh`

---

## Bringing Your Own Compiled C Code (`milk_calc_ext.h` & `milk-cc`)

Users can extend `@calc` and `milk-fpsexec-milk-calc` with compiled C code (`-l <file.c|file.so>`)
or compile standalone `MILK_RUNNABLE` compute units with a single command (`milk-cc <unit.c>`),
without editing CMake files.

| Tier | Registration Macro | Best For |
| ---- | ------------------ | -------- |
| **1a: 1D SIMD Block Kernel** | `MILK_CALC_REG_BLK`, `MILK_CALC_REG_BLK2` | Per-pixel non-linearities, LUTs, 2-image blends fused into the L1 tile loop (`FUSE_BLK = 1024`) |
| **1b: 2D Neighborhood Stencil** | `MILK_CALC_REG_STENCIL` | $(2R+1)\times(2R+1)$ local neighborhood filters ($R \in \{1,2,3\}$) with pre-clamped row pointers |
| **1c: Whole-Image Kernel** | `MILK_CALC_REG_IMG` | Single-function 1-in / 1-out global image algorithms (column/row de-meaning, custom transforms) |
| **2: Stateful Multi-Port Operator** | `MILK_CALC_OP_DECLARE`, `MILK_CALC_REG_OP` | Multi-input / multi-output algorithms with `init` / `exec` / `free` state and custom output shapes |
| **3: Standalone `MILK_RUNNABLE`** | `milk-cc <unit.c>` | Full standalone `milk-fpsexec-*` binary with custom semaphore triggering |

For auto-compilation/caching (`-l`), `MILK_CALC_PARAMS` exposure rules, and full C examples of each
tier, see **[C Extensions (`milk_calc_ext.h` & `milk-cc`)](calc_c_extensions.md)**.

---

← [CLI Syntax](CLIcore.md) · [Calculator & Compute Units](calculator.md) · [Sessions](sessions.md) · [C Extensions (`milk_calc_ext.h`)](calc_c_extensions.md) · [Documentation Index](../index.md)
