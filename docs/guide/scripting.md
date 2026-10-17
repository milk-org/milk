---
tags:
  - cli
  - scripting
---

# Scripting

A milk script is a bash script with a few additions for milk: commands, entities (streams, local
images, FPS, processes, sequencers) and an image calculator, all written with the `@` sigil. It runs
in `milk-cli` (interactively, with `-s`, `-c` or piped input), in `milk-script`, or in any bash shell
that has loaded `milk.bash`.

`milk-cli` translates the `@` forms and hands everything to a bash process it supervises; bash then
calls back into `milk-cli` for each milk command. So every bash feature works as in bash: variables,
arrays, `$(( ))`, `[[ ]]`, functions, `if`/`for`/`while`/`case`, pipes, redirections, heredocs,
`trap`, background jobs.

This page is the reference for the `@` additions. See also: [CLI Syntax](CLIcore.md) ·
[Streams](streams.md) · [FPS](fps.md) · [Slicing](slicing.md) · [What's New](../whatsnew.md)

## Running Scripts

| How                         | What happens                                                          |
| --------------------------- | --------------------------------------------------------------------- |
| `milk-cli -s script.milk`   | runs the file and exits with its status                               |
| `milk-cli -c 'commands'`    | runs the string and exits with its status                             |
| `cmds | milk-cli`           | runs each input line; `$?` carries over from line to line             |
| `#!/usr/bin/env milk-script` | standalone runner, no readline (`milk-script -e` stops on errors)   |
| `source file.milk [args]`   | inside milk-cli: runs the file in the current shell, with `$1`... set |
| `include_once file.milk`    | sources a file only the first time                                    |
| `source milk.bash`          | adds the `@` commands to an ordinary bash shell (see below)           |

At interactive startup, `~/.milkrc` is sourced if it exists. `@savehistory file` writes the session
history to a file that can be run again.

## Calling milk Commands

The same rule applies at the prompt, with `-c`, on piped input, in scripts and in sourced files:

- `module.command args`: a module-qualified name (`mem.listim`, `iofits.saveFITS`) needs no sigil.
- `@command args`: the sigil form works for every command, including short names (`@listim`),
  commands without a module (`@streamlist`, `@ci`), prompt built-ins (`@synhl off`, `@help`) and help
  shorthands (`@?`, `@m?`, `@cmd? mem.listim`).

Any other word goes to bash unchanged: `listim` without `@` is an ordinary (usually unknown) shell
command.

## Entities

An entity path is `<namespace>.<name>.<property>`:

| Namespace                  | Refers to                                   |
| -------------------------- | ------------------------------------------- |
| `s.` `stream.`             | shared-memory stream (`/dev/shm`)           |
| `l.` `local.`              | image local to the milk-cli process         |
| `f.` `fps.`                | function parameter structure (FPS)          |
| `p.` `proc.` `procinfo.`   | processinfo loop                            |
| `q.` `seq.` `fpsseq.`      | FPS sequencer                               |

Images created by `mk2Dim` and other commands are shared streams by default. Use the `@L:name`
modifier in a command argument for a local image (`@mk2Dim "@L:buf" 64 64`), and `@S:name` to force a
shared one.

### Reading

```bash
@get s.cam.xsize                   # print a property
@get -v nx s.cam.xsize             # bind it to a bash variable (no subshell)
@get -a names 's.*'                # matching entity names into an indexed array
@get -A sizes 's.dm*.xsize'        # name -> value into an associative array
@get --json s.cam                  # all properties as JSON (also f.<fps>, p.<proc>)
nx=@s.cam.xsize                    # inline expansion in an assignment
echo "frame @s.cam.cnt0 of @s.cam.xsize x @s.cam.ysize"   # inline, also inside "..."
echo "\@s.cam.xsize"               # escaped: printed as is
n=cam; echo "@s.${n}.xsize"        # bash variables inside the path
```

Common properties:

- streams (`s.`, `l.`): `xsize` `ysize` `zsize` `naxis` `nelem` `type` `typename` `cnt0` `cnt1` `sem`
  `shared` `age` `fps` `ownerPID` `kw.<KEY>` (FITS keyword), `kw.*`, `list`, `summary`;
- FPS (`f.`): any parameter (`f.loop.gain`), plus `status` `conf` `run` `confpid` `runpid` `nbparam`
  `list` `description`;
- processes (`p.`): `status` `loopcnt` `loopfreq` `ctrl` `pid` `exectime` `missedframes`
  `triggerstream` `tmux` `statusmsg`;
- sequencers (`q.`): `status` `step_count` `active_step` `completed` `errors` `last_error`.

`@get --json <entity>` lists everything an entity provides. A failed inline expansion prints an
error and expands to nothing (see [Errors and Exit Status](#errors-and-exit-status)).

### Writing and Actions

```bash
@set f.loop.gain=0.25              # one value
@set f.loop.gain = 0.25            # same, '=' may be spaced
@set f.loop.gain=0.25 f.loop.mult=0.99   # several in one call
@set 'f.dm*.xsize' = 128           # every match of a glob
@set s.cam.kw.OBS=night "comment"  # FITS keyword, optional comment
@set p.loop.ctrl=pause             # processinfo control: run pause step stop ...
@f.loop.confstart                  # FPS actions: confstart confstop runstart runstop reset
@p.loop.pause                      # process actions: run resume pause step incr stop exit kill
```

`@f.loop.gain=0.25` and `fpsset loop gain 0.25` are aliases of `@set`.

### Existence and Listing

```bash
@exists s.cam && echo yes          # exit status 0 if the entity exists
@running f.loop                    # 0 if the FPS or process runs
@ls                                # table of streams, FPS, processes, sequencers
@ls 's.cam*'                       # one namespace, with a glob
@ls -a list 'f.*'                  # names into an array (also -1, --json)
@milkquery --fps                   # JSON snapshot (also --streams, --procs)
```

## Calculator

```bash
@calc "cam = cam * 0 + 5"          # image arithmetic, in place
@calc -v m "imean(cam)"            # bind the result to a bash variable
@calc 'rms = irms(cam); ptv = imax(cam) - imin(cam)'   # several statements
@calc "cam[0:31, 0:31] = 0"        # slices: see slicing.md
echo "mean=@{imean(cam)}"          # inline calculator
@ x = 2 * 21                       # statement form (alias of @calc)
if @test "imean(cam) > 1"; then echo bright; fi        # status 0 if true, 1 if false
```

- Images, calculator variables and bash variables are all visible to expressions. An unknown name is
  an error; quote string literals: `@calc -v n 'strlen("abc")'`.
- An expression made only of numbers, operators and math functions is evaluated in the shell itself
  (a few microseconds).
- `@dpdigits N` sets the number of significant digits of printed results.
- `@batch "c1; c2"` runs several commands and expressions in one request; `@pipe "c1 |> c2"` also
  chains the image each stage makes as `_`:

```bash
@batch "mem.mk2Dim a 64 64; a = 5; a *= 2"
@pipe "mem.mk2Dim p 32 32 |> _ = 4 |> _ *= 2.5"
```

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
@replay --fps 100 --loop cube.fits sim   # play a cube or FITS file into stream sim
@on s.cam 'echo "new frame on $MILK_TRIGGER"'      # run on every update (background)
@on --once --max-hz 10 s.cam s.wfs -- process_frame
@on --list ; @on --stop s.cam ; @on --clear
@defer rm -f /tmp/work.fits        # run at exit, last registered first
milksleep 0.05                     # sleep without starting a process
```

`@waitfor` and `@waitany` sleep on the stream semaphores, so they use no CPU while waiting. With
`--`, `@on` runs the words after it as a command; use the quoted form when the command refers to
`$MILK_TRIGGER`.

## State, Assertions and Logs

```bash
@snapshot -o base.json 'f.loop*' 's.cam*'     # FPS parameters and stream keywords to JSON
@diff base.json                               # compare with the live state
@restore -n base.json                         # dry run, then without -n to apply
@assert "irms(wfs) < 0.5" "residual diverged" # fails with the values of both sides
@assert --running --fps-min 500 s.cam "camera stalled"
@log -n 20                                    # recent milk commands, timing and output
@log --errors
@server status                                # server in use; also list, use, stop
@prompt preset std                            # live telemetry in the prompt
```

## Errors and Exit Status

- `milk-cli -s`, `milk-cli -c` and piped input exit with the status of the last command, and `$?`
  carries over from one input line to the next.
- A failed inline expansion (`@s.cam.xsize` for a missing stream) prints
  `milk: @s.cam.xsize: inline expansion failed` and expands to an empty string. An assignment such
  as `x=@s.cam.xsize` also returns its non-zero status. With `MILK_STRICT=1` the failure is fatal: the
  script exits with that status, as with `set -u`. At the interactive prompt it is only reported.
- Calculator errors are printed even in quiet mode (`milk: calc error: ... in '<expr>'`). An
  undefined name is an error: quote string literals (`@calc 'strlen("abc")'`).
- When the server cannot be reached or speaks another protocol version, milk commands print a
  message and return 255, so `@test` never reads a lost connection as "false".
- Ctrl-C interrupts the running command line, including a milk command executing inside milk-cli,
  and returns status 130. Shell variables, functions and background jobs survive. A script that sets
  its own `trap ... INT` handles Ctrl-C itself.
- If the bash process dies (killed, or recursion deeper than `FUNCNEST`, 5000 by default),
  milk-cli says so. A script stops there; at the prompt, the next command starts a new bash, without
  the previous shell state.

## Concurrent Commands

milk commands run one at a time in the milk-cli process. `@waitfor` and `@grab` wait in the
background, so other commands keep running meanwhile, and they see live state: a local image that
another command changes, or the cube that `@grab` fills. A `SHUTDOWN` request is honoured even while
a command is running.

An expression made only of numbers, operators and math functions (`@calc -v y "2*sin(0.5)"`,
`@test "3 > 2"`, `@{1+2}`) is evaluated in the shell itself, without a request to milk-cli, so it does
not appear in `@log`. `@dpdigits` applies to these results too. Use `milksleep 0.05` for short sleeps
in loops: it does not start a process, and Ctrl-C still interrupts it.

A command that has to run alongside another one with its own input and output runs in a copy of the
process: a later stage of a pipeline such as `@streamcat s.cam | @streamwrite out`, a
`milk-call call --async` command, and `@replay`. Shared-memory streams it writes are visible to
everyone, but local images and variables it creates are not kept.

## Cost Model

Measured on a recent desktop machine (see `test_latency_benchmark` in the regression suite):

| Form                                  | Time per call | Note                                     |
| ------------------------------------- | ------------- | ---------------------------------------- |
| `@calc -v y "1+2"` (scalar only)      | ~4 µs         | evaluated in the shell                   |
| `@get -v x s.cam.xsize`               | ~5–8 µs       | read from shared memory, no request      |
| `@set s.cam.kw.FOO=1`                 | ~7 µs         |                                          |
| `y=@{1+2}`, `x=@s.cam.xsize` (inline) | ~14–16 µs     | bash's `${| …; }` substitution, no fork  |
| `@calc "cam = cam * 2"` (images)      | ~30 µs        | one request to milk-cli                  |
| `@listim`, other milk commands        | ~70 µs        | one request to milk-cli                  |
| `y=$(@calc 1+2)`                      | ~2 ms         | `$( )` forks a subshell: avoid in loops  |

Prefer `-v`, `-a`, `-A` and inline `@…` over `$(…)`, group work with `@batch` / `@pipe`, and use
`milksleep` rather than `sleep` for short waits. On bash older than 5.3, inline forms fall back to
`$(…)` and cost a fork each.

## Using milk from Plain Bash (`source milk.bash`)

`source milk.bash` adds the `@` commands, entities and calculator to any bash shell or script; they
talk to a `milk-cli --server` started on first use. `@source file.milk` translates and runs a milk
script in that shell. Options: `MILK_QUIET=1` loads silently, `MILKCLI_SIGIL=,` uses `,` instead of
`@`, and `@unload` removes everything again.

The shell keeps its own state: `cd`, variables and functions are those of your shell, while images
and calculator variables live in the server. The prompt-only commands (`@synhl`, `@history`, ...)
exist only in milk-cli.

## Examples

Monitor a stream and log its mean once per second:

```bash
#!/usr/bin/env milk-script
@mk2Dim "@S:demo_cam" 64 64
for i in 1 2 3; do
    @calc "demo_cam = $i * 10"
    @calc -v m "imean(demo_cam)"
    printf 'frame %s mean=%s\n' "@s.demo_cam.cnt0" "$m"
    milksleep 1
done
```

Capture a cube from a live stream and save statistics:

```bash
#!/usr/bin/env milk-script
@defer echo "done"
@waitfor s.wfs -t 5 || { echo "no frames"; exit 1; }
@grab s.wfs 50 "@L:wfs_cube"
@calc -v rms "irms(wfs_cube)"
@set s.wfs.kw.RMS="$rms" "rms of the last 50 frames"
echo "rms=$rms over @l.wfs_cube.zsize frames"
```

Tune an FPS gain until a stream settles:

```bash
#!/usr/bin/env milk-script
@running f.loop || @f.loop.runstart
for g in 0.1 0.2 0.3 0.4; do
    @set f.loop.gain="$g"
    @waitfor s.dmdisp 20 -t 2
    if @test "irms(dmdisp) < 0.05"; then
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

| Syntax Element  | Highlight Group     | Example                                  |
| --------------- | ------------------- | ---------------------------------------- |
| Flow control    | `@keyword`          | `if`, `for`, `while`, `fi`               |
| Shell builtins  | `@function.builtin` | `echo`, `export`, `source`               |
| milk commands   | `@function.macro`   | `@calc`, `@waitfor`, `include_once`      |
| Module commands | `@function.call`    | `mem.listim` (builtin color, see below)  |
| Entity paths    | `@property`         | `@s.cam.xsize`, `"@f.${loop}.gain"`      |
| Calculator      | `@punctuation.special` | `@{imean(cam)}`, `@ x = 2 * 21`       |
| Variables       | `@variable.builtin` | `$VAR`, `${VAR}`                         |
| Strings         | `@string`           | `"hello"`, `'literal'`                   |
| Numbers         | `@number`           | `42`, `3.14`                             |
| Comments        | `@comment`          | `# comment`                              |

Heredocs are not parsed by the grammar: their body is highlighted as ordinary commands.

### Dynamic Module Commands

Module commands (`listim`, `saveFITS`, `imcrop`, etc.) are registered at runtime, not hardcoded in
the grammar. To highlight them, run the generator script after installing new milk modules:

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
