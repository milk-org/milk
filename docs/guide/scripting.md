---
tags:
  - cli
  - scripting
---

# Scripting

A milk script is a bash script with three additions:

- module commands, written `module.command` (`mem.mk2Dim`, `iofits.loadfits`);
- language keywords, written `@keyword` (`@calc`, `@get`, `@waitfor`);
- the `@{…}` expansion, which reads an entity property or computes an expression inside any word.

Scripts run in `milk-cli` (interactively, with `-s`, `-c` or piped input), in `milk-script`, or in
any bash shell that has loaded `milk.bash`. Everything else is bash, unchanged: variables, arrays,
`$(( ))`, `[[ ]]`, functions, `if`/`for`/`while`/`case`, pipes, redirections, heredocs, `trap`,
background jobs.

See also: [CLI Syntax](CLIcore.md) · [Streams](streams.md) · [FPS](fps.md) ·
[Slicing](slicing.md) · [What's New](../whatsnew.md)

## The Language at a Glance

| You write                     | It is                                                                     |
| ----------------------------- | ------------------------------------------------------------------------- |
| `word args`                   | bash                                                                      |
| `module.command args`         | a module command: a bash function defined from the install catalog       |
| `@keyword args`               | one of the 27 [keywords](#keywords)                                      |
| `@{expr}` inside a word       | an entity path reads a property, anything else goes to the calculator    |
| `\@{`                         | a literal `@{`                                                            |
| `<anything> -h`, `--help`     | help                                                                      |

`@` has no other meaning. `user@host`, `curl -d @file`, `git checkout @{-1}` and `"HEAD@{1}"` (an
`@{` after a letter) are passed to bash as written. Inside calculator text, entities are written
without `@`: `@calc "x = s.cam.xsize * 2"`.

## Running Scripts

| How                                 | What happens                                                    |
| ----------------------------------- | --------------------------------------------------------------- |
| `milk-cli -s script.milk [args]`    | runs the file and exits with its status                         |
| `milk-cli -c 'commands'`            | runs the string and exits with its status                       |
| `cmds \| milk-cli`                  | runs each input line; `$?` carries over from line to line       |
| `#!/usr/bin/env milk-script`        | standalone runner, no readline (`milk-script -e` stops on errors) |
| `@source [--once] file.milk [args]` | runs a file in the current shell, `$1`... set, `@{…}` expanded  |
| `source milk.bash`                  | adds milk to an ordinary bash shell ([below](#using-milk-from-plain-bash)) |

A plain `source file.milk` runs the file as bash: its `@{…}` stay as written. The interactive
milk-cli prompt sources `~/.milkrc` at startup; `@history save file` writes the session history to
a file that can be run again.

## Keywords

| Group         | Keywords                                                                        |
| ------------- | ------------------------------------------------------------------------------- |
| values        | `@calc` `@test` `@assert` `@get` `@set` `@exists` `@running` `@ls`              |
| events        | `@waitfor` `@waitany` `@on` `@sleep`                                            |
| data          | `@grab` `@replay` `@snapshot` `@diff` `@restore` `@pipe`                        |
| program       | `@source` `@defer` `@module` `@config` `@server` `@unload`                       |
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

## Entities

An entity path is `<namespace>.<name>.<property>`:

| Namespace        | Refers to                                   |
| ---------------- | ------------------------------------------- |
| `s.` `stream.`   | shared-memory stream (`$MILK_SHM_DIR`)      |
| `l.` `local.`    | image local to the milk-cli process         |
| `f.` `fps.`      | function parameter structure (FPS)          |
| `p.` `proc.`     | processinfo loop                            |
| `q.` `fpsseq.`   | FPS sequencer                               |

Images created by `mem.mk2Dim` and other commands are shared streams by default.

### Reading

```bash
@get s.cam.xsize                   # print a property
@get -v nx s.cam.xsize             # bind it to a bash variable (no subshell)
@get -a names 's.*'                # matching entity names into an indexed array
@get -A sizes 's.dm*.xsize'        # name -> value into an associative array
@get --json s.cam                  # all properties as JSON (also f.<fps>, p.<proc>)
nx=@{s.cam.xsize}                  # inline, in any word
echo "frame @{s.cam.cnt0} of @{s.cam.xsize} x @{s.cam.ysize}"
n=cam; echo "@{s.${n}.xsize}"      # bash variables inside the path
echo "\@{s.cam.xsize}"             # escaped: printed as is
```

Common properties:

- streams (`s.`, `l.`): `xsize` `ysize` `zsize` `naxis` `nelem` `type` `typename` `cnt0` `cnt1`
  `sem` `shared` `age` `fps` `ownerPID` `kw.<KEY>` (FITS keyword), `kw.*`, `list`, `summary`;
- FPS (`f.`): any parameter (`f.loop.gain`), plus `status` `conf` `run` `confpid` `runpid` `nbparam`
  `list` `description`;
- processes (`p.`): `status` `loopcnt` `loopfreq` `ctrl` `pid` `exectime` `missedframes`
  `triggerstream` `tmux` `statusmsg`;
- sequencers (`q.`): `status` `step_count` `active_step` `completed` `errors` `last_error`.

`@get --json <entity>` lists everything an entity provides.

### Writing and Actions

```bash
@set f.loop.gain=0.25              # one value
@set f.loop.gain = 0.25            # same, '=' may be spaced
@set f.loop.gain=0.25 f.loop.mult=0.99   # several in one call
@set 'f.dm*.xsize' = 128           # every match of a glob
@set s.cam.kw.OBS=night "comment"  # FITS keyword, optional comment
@set f.loop.conf=start             # FPS: conf=start|stop|step, run=start|stop
@set p.loop.ctrl=pause             # processinfo control: run pause step stop ...
@set p.loop.pause                  # same, as an action
@set --tui loop                    # browse and edit FPS parameters (milk-cli prompt)
```

### Existence and Listing

```bash
@exists s.cam && echo yes          # exit status 0 if the entity exists
@running f.loop                    # 0 if the FPS or process runs
@ls                                # table of streams, FPS, processes, sequencers
@ls 's.cam*'                       # one namespace, with a glob
@ls -a list 'f.*'                  # names into an array (also -1, -l)
@ls --json                         # JSON snapshot
```

## Calculator

```bash
@calc "cam = cam * 0 + 5"          # image arithmetic, in place
@calc -v m "imean(cam)"            # bind the result to a bash variable
@calc 'rms = istd(cam); ptv = imax(cam) - imin(cam)'   # several statements
@calc "cam[0:31, 0:31] = 0"        # slices: see slicing.md
echo "mean=@{imean(cam)}"          # inline
if @test "imean(cam) > 1"; then echo bright; fi        # status 0 if true, 1 if false
```

- Names in an expression are images, calculator variables and bash variables; an unknown name is an
  error. Quote string literals: `@calc -v n 'strlen("abc")'`.
- A bash variable is read only by calculator text (`@calc`, `@test`, `@assert`, `@pipe`, `@{…}`),
  and only when the expression names it. Its value is data: `s="sqrt(16)"` is the string, not 4. A
  bash variable wins over a calculator variable of the same name.
- Calculator variables come back to bash only through `-v` (and `@{…}`): after
  `@calc "x = 5"`, bash has no `$x`. Write `@calc -v x "5"`, or read it later with
  `@calc -v x "x"`. Commands that set calculator variables for their caller declare them:
  `info.imstats im` sets `imsize0`... `vmin` `vmax` `vmean` `vrms` and the percentiles in bash.
- An expression made only of numbers, operators, math functions and `s.`/`f.`/`p.`/`q.` properties
  is evaluated in the shell itself, in a few microseconds, without a request to the server.
- `@config digits N` sets the number of significant digits of printed results.
- A pixel of an integer image reads and writes as an integer (`im[3,4] % 10`, `im[0,0] & 0xff`).
- `@pipe "c1; c2"` runs several commands and expressions in one request; `@pipe "c1 |> c2"` also
  chains the image each stage makes as `_`:

```bash
@pipe "mem.mk2Dim a 64 64; a = 5; a *= 2"
@pipe "mem.mk2Dim p 32 32 |> _ = 4 |> _ *= 2.5"
```

Operators, loosest first:

| Operators                       | Notes                                                            |
| ------------------------------- | ---------------------------------------------------------------- |
| `=` `+=` `-=` `*=` `/=`, `?:`   | `c ? a : b` with a scalar `c` evaluates only the branch it takes |
| `\|\|`, `&&`                    | skip the right side once the result is known: `0 && (x = 1)`     |
| `\|`, `^^`, `&`                 | bitwise or, xor, and                                             |
| `==` `!=`, `<` `<=` `>` `>=`    | 1 or 0; on images, a 0/1 mask per pixel                          |
| `<<` `>>`, `+` `-`, `*` `/` `%` | `"a" + "b"` and `"frame" + 3` concatenate strings                |
| `^`                             | power, right-associative                                         |
| unary `-` `~` `!`               | bind tighter than `^`: `-2^2` is 4                               |

Integer operands give an integer result, except `/`; an integer result that overflows becomes a
double. Numbers may be written `0x1f`, `0o17`, `0b101`.

Functions (the math ones also apply per pixel to an image):

| Kind             | Functions                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Math             | `sin cos tan asin acos atan exp ln log sqrt cbrt ceil floor round abs posi`              |
| Math, 2-3 args   | `atan2(y,x)`, `fmod(a,b)`, `min(a,b)`, `max(a,b)`, `trunc(x,lo,hi)` (clamp)              |
| Image to number  | `itot imin imax imean istd ivar imedian norm`, `perc(im,f)`, `dot(a,b)`; `irms` = `istd` |
| Image to integer | `icount(im)` (non-zero pixels), `iargmax(im)`, `iargmin(im)` (index `x + y*xsize`)       |
| Along one axis   | `itot imin imax imean istd ivar` plus an axis: `imean(cube, 2)` is the mean frame        |
| Image to image   | `where(c,a,b)`, `matmul(a,b)`                                                            |
| Strings          | `strlen toupper tolower`, `substr(s,start,len)`, `replace(s,old,new)`, `hex oct bin`     |
| Complex images   | `cplx(re,im)`, `real imag abs arg conj`; `+ - * /` and `-z` work on them                 |

Pixels of complex images go through those functions (`abs(z[3,4])` is a number); the other
functions and operators reject complex images. Images combined pixel by pixel must have the same
size on each axis, or size 1 (a 2-D image applies to every frame of a cube).

## Waiting, Events and Cleanup

```bash
@waitfor s.cam                     # next frame
@waitfor s.cam 10 -t 2             # 10 frames, timeout 2 s (status 124 on timeout)
@waitfor s.cam --cnt 5000          # until cnt0 reaches 5000
@waitfor f.loop.status=RUN -t 5    # until a property has a value
@waitfor "imean(cam) > 100" -t 5   # until an expression is true
@waitany s.cam s.wfs -t 1          # whichever updates first; its name is in $MILK_TRIGGER
@grab s.cam 100 "@L:cube"          # 100 consecutive frames into a cube
@grab --fits cube.fits --timestamps t.txt s.cam 100
@grab --stdout -a pix s.cam        # the next frame's pixels into a bash array
@replay --fps 100 --loop cube.fits sim   # play a cube or FITS file into stream sim
@on s.cam 'echo "new frame on $MILK_TRIGGER"'      # run on every update (background)
@on --once --max-hz 10 s.cam s.wfs -- process_frame
@on --list ; @on --stop s.cam ; @on --clear
@defer rm -f /tmp/work.fits        # run at exit, last registered first
@sleep 0.05                        # sleep without starting a process
```

`@waitfor` and `@waitany` sleep on the stream semaphores, so they use no CPU while waiting. With
`--`, `@on` runs the words after it as a command; use the quoted form when the command refers to
`$MILK_TRIGGER`.

## State, Assertions and Logs

```bash
@snapshot -o base.json 'f.loop*' 's.cam*'     # FPS parameters and stream keywords to JSON
@diff base.json                               # compare with the live state
@restore -n base.json                         # dry run, then without -n to apply
@assert "istd(wfs) < 0.5" "residual diverged" # fails with the values of both sides
@assert --running --fps-min 500 s.cam "camera stalled"
@log -n 20                                    # recent milk commands, timing and output
@log --errors
@server status                                # server in use; also list, use, stop
@config prompt preset std                     # live telemetry in the prompt
@config                                       # digits, precision, procinfo
```

## Errors and Exit Status

| Status | Meaning                                                         |
| ------ | --------------------------------------------------------------- |
| 0      | success                                                         |
| 1      | the command failed                                              |
| 2      | usage error: bad or missing argument, invalid value, syntax     |
| 124    | timeout (`@waitfor`, `@waitany`, `@grab -t`)                    |
| 127    | unknown command (the message names the replacement of an old spelling) |
| 130    | interrupted (Ctrl-C)                                            |
| 255    | the server cannot be reached, or speaks another protocol version |

- `milk-cli -s`, `milk-cli -c` and piped input exit with the status of the last command, and `$?`
  carries over from one input line to the next.
- A failed `@{…}` prints `milk: @{s.cam.xsize}: inline expansion failed` and expands to an empty
  string; an assignment such as `x=@{s.cam.xsize}` also returns its status. With `MILK_STRICT=1`
  the failure ends the script with that status, as with `set -u`; at a prompt it is only reported.
- Calculator errors are printed even in quiet mode (`milk: calc error: ... in '<expr>'`).
- Ctrl-C interrupts the running command line, including a milk command executing inside milk-cli.
  Shell variables, functions and background jobs survive. A script that sets its own `trap ... INT`
  handles Ctrl-C itself.
- If the bash process dies (killed, or recursion deeper than `FUNCNEST`, 5000 by default), milk-cli
  says so. A script stops there; at the prompt, the next command starts a new bash, without the
  previous shell state.

## Concurrent Commands

milk commands run one at a time in the server. `@waitfor` and `@grab` wait in the background, so
other commands keep running meanwhile, and they see live state: a local image that another command
changes, or the cube that `@grab` fills. A `SHUTDOWN` request is honoured even while a command is
running.

A command that has to run alongside another one with its own input and output runs in a copy of the
process: a later stage of a pipeline such as `@grab --stdout s.cam | @replay --stdin out`, a
`milkcall --async` command, and `@replay`. Shared-memory streams it writes are visible to everyone,
but local images and variables it creates are not kept.

## Cost Model

Measured on a recent desktop machine (see `test_latency_benchmark` in the regression suite):

| Form                                  | Time per call | Note                                     |
| ------------------------------------- | ------------- | ---------------------------------------- |
| `@calc -v y "1+2"` (scalar only)      | ~2 µs         | evaluated in the shell                   |
| `@get -v x s.cam.xsize`               | ~3 µs         | read from shared memory, no request      |
| `@set s.cam.kw.FOO=1`                 | ~5 µs         |                                          |
| `x=@{s.cam.xsize}`, `y=@{1+2}`        | ~2–3 µs       | a statement of its own: bound with `-v`  |
| `"frame @{s.cam.cnt0}"` (in a word)   | ~15 µs        | bash's `${| …; }` substitution, no fork  |
| `@calc "cam = cam * 2"` (images)      | ~30 µs        | one request to the server                |
| `mem.listim`, other milk commands     | ~75 µs        | one request to the server                |
| `y=$(@calc 1+2)`                      | ~0.8 ms       | `$( )` forks a subshell: avoid in loops  |

An expression made of same-size images, numbers, arithmetic, comparisons and the per-pixel math
functions runs in one pass, however many operators it has (1024×1024 floats:
`x = (b*2 + c*3 - a) / 4` ~0.2 ms, `a += 1` ~0.2 ms), and assigning it to an existing image of
the same size and type writes in place. Slices, broadcasting, reductions and complex images take
one pass per operator.

Prefer `-v`, `-a`, `-A` and `NAME=@{…}` over `$(…)` and over `@{…}` inside longer words in loops,
group work with `@pipe`, and use `@sleep` rather than `sleep` for short waits. On bash older than
5.3, `@{…}` inside a word falls back to `$(…)` and costs a fork; `NAME=@{…}` does not.

## Using milk from Plain Bash

`source milk.bash` adds the module commands, keywords, entities and calculator to any bash shell or
script, without starting a process; they talk to a `milk-cli --server` started on first use.
`MILK_QUIET=1` loads silently, and `@unload` removes everything again.

- `@source file.milk` runs a milk script in that shell, with its `@{…}` expanded.
- `@{…}` typed at an interactive prompt is expanded once `@config inline on` is set: Enter then
  translates the line before running it (`@config inline off` restores Enter).
- The shell keeps its own state: `cd`, variables and functions are those of your shell, while
  images and calculator variables live in the server.
- The milk-cli prompt settings (`@config synhl`, `@history`, ...) exist only in milk-cli.

## Migrating Old Scripts

Old spellings fail with a message that names their replacement. `milk-script --check file` lists
them, and `milk-script --modernize file` rewrites the file in place; running it again changes
nothing. It rewrites:

- `@listim`, `@mem.listim`, `listim` → `mem.listim`; `@ci` → `milk.ci`;
- `@ x = e` → `@calc "x = e"`, or `@calc -v x "e"` when the file reads `$x`;
- `@s.cam.xsize` → `@get s.cam.xsize`, `@f.loop.gain=0.2` → `@set f.loop.gain=0.2`, inline
  `@s.cam.cnt0` and `${@s.cam.xsize}` → `@{s.cam.cnt0}`, `@s.x` inside calculator text → `s.x`;
- `@batch` → `@pipe`, `@streamcat` → `@grab --stdout`, `@streamwrite` → `@replay --stdin`,
  `@usleep` → `@sleep`, `@procwait` → `@waitfor p.<name>.status=...`;
- `@mload`, `@mloadas` → `@module load [--as]`; `@dpdigits` → `@config digits`; the history and
  help words → `@history`, `@help`, `-h`;
- `include_once` → `@source --once`, `source x.milk` → `@source x.milk`, `exitCLI` → `exit`;
- `cmd .tag value` → `cmd --default .tag=value`.

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

← [CLI Syntax](CLIcore.md) · [Documentation Index](../index.md)
