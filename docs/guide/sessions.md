---
tags:
  - cli
  - scripting
  - sessions
---

# Sessions

A session is the `milk-cli --server` process that holds local images (`l.`), calculator variables,
and settings (`@config`, `--default`). Each shell (`milk-cli`, `milk-script`, or plain bash after
`source milk.bash`) has its own session, started on its first milk command and ended with the
shell; `milk-cli`'s runs in its own process.

Scripts and shells started from a shell use its session: they inherit `MILKCLI_SOCK`, so a script
that sources `milk.bash` reads and changes its caller's images and variables, and loses them when
its caller exits (`@session leave` in the script gives it a session of its own). A **named session**
is shared by the shells that use it, and runs until it is ended.

See also: [Scripting](scripting.md) · [CLI Syntax](CLIcore.md) · [FIFO Command Input](fifo.md) ·
[systemd Services](../operations/systemd.md)

## Managing Sessions (`@session`)

```bash
@session new bench1           # start session bench1 and use it (-d: only start it); fails if it runs
@session new scratch --idle 2h  # ends once unused: 2 h without a command, and no shell uses it
@session use bench1           # in another shell: use it too; fails if it does not run
@session                      # status: name, state, PID, output file, shells, units
@session ls                   # this user's sessions in this MILK_SHM_DIR (--json); * marks the one in use
@session who bench1           # the shells that use it
@session leave                # back to this shell's own session
@session end bench1 --units   # end it, and stop the units started from it
@session end                  # end the session in use (a shell's own: its images and variables go)
@session end --all            # end every session that @session ls lists
@session prune                # remove what ended sessions and exited shells left
@session save run1.milk       # its state as a milk script; @session restore run1.milk rebuilds it
```

- `source milk.bash --session NAME` uses session `NAME` from the start, and starts it if it does not
  run; so does `milk-cli --session NAME` for the bash of its REPL, whose prompt then reads
  `milk-cli:NAME`. A `NAME` that cannot be used leaves the shell on its own session, with a warning.
  Names are letters, digits, `_` and `-`, at most 63 characters; a number names a shell's own
  session.
- `milk-cli` keeps a session of its own: `@session leave` returns to it, and `@history` and the REPL
  settings of `@config` (`synhl`, `prompt`, `calc`, ...) belong to `milk-cli` whichever session its
  bash uses. `@calc` mode evaluates in the session the bash uses. After `milk-cli --session NAME`
  exits, `NAME` runs on with its images and variables (`ORPHANED` if that `milk-cli` started it),
  and `milk-cli --session NAME` again resumes it with a new bash.
- `@session save FILE` writes the settings, calculator variables (numbers and strings), command
  defaults (`--default`) and local images of the session as a milk script, the images as FITS files
  in `FILE.d/`, which loses the FITS files it held. A string holding a double quote or a newline is
  not saved, with a warning. `@session restore FILE` runs it in the shell's session (`source FILE`
  does the same). Shared streams are not saved: they stay in shared memory. The units the session
  started are listed in `FILE`, as comments; restore prints them and starts none.
- A named session that was ended is not started again by the shells that used it: they get
  `milk: session NAME has ended: @session new NAME, or @session leave`.
- The shell that runs `@session new` owns the session, without being tied to it: once that shell has
  exited, `@session ls` lists the session as `ORPHANED`, and it runs on until it is ended.
- `--idle T` (`90`, `30s`, `10m`, `2h`, `1d`) ends a named session once it has had no command for `T`
  and no shell uses it. A fast `@get` or `@set` counts as a command; `@session ls` and `status` do
  not.
- A session belongs to the `MILK_SHM_DIR` it started with, where its files are: `@session ls` and
  `end --all` see the sessions of the current one only. Names are global, so `@session new` of a
  session that runs from another `MILK_SHM_DIR` fails, naming that directory.
- A unit started from a named session records it (`f.X.session`; empty for a shell's own), and
  `@session ls` counts it under `UNITS`. Units outlive their session, unless `@session end --units`
  stops them, the way `milk-fps-runstop` does (through `systemctl` for a unit that systemd runs).
- `@session end` from another shell ends a `milk-cli` session, and that `milk-cli` once its running
  command line returns; inside `milk-cli`, `exit` does.
- What a session's server prints goes to the file `@session status` names,
  `milk-srv.<uid>.<name>.out`, next to its log `milk-srv.<uid>.<name>.log.shm` in `MILK_SHM_DIR`; a
  session that ends by itself leaves both until `@session ls` or `prune` removes them. `@session log`
  (or `@log`) shows the commands it ran. Shells record the session they use in
  `${TMPDIR:-/tmp}/.milk_active_sock_<uid>_<pid>_<start>`. A named session can also run as a
  systemd user service ([systemd Services](../operations/systemd.md#7-a-named-session-as-a-service)).

## Using milk from Plain Bash

`source milk.bash` adds the module commands, keywords, entities and calculator to any bash shell or
script, without starting a process; they talk to the shell's session, a `milk-cli --server` started
on first use. `MILK_QUIET=1` loads silently, and `@unload` removes everything again.

- `@source file.milk` runs a milk script in that shell, with its `@{…}` expanded.
- `@{…}` typed at an interactive prompt is expanded once `@config inline on` is set: Enter then
  translates the line before running it (`@config inline off` restores Enter).
- The shell keeps its own state: `cd`, variables and functions are those of your shell, while
  images and calculator variables live in the server.
- `@config synhl` and the other settings of `milk-cli`'s line editor do nothing in plain bash, which
  accepts them; `@history` works in both.

## Concurrent Commands Across Shells

milk commands run one at a time in the server. `@waitfor` and `@grab` wait in the background, so
other commands keep running meanwhile, and they see live state: a local image that another command
changes, or the cube that `@grab` fills. A `SHUTDOWN` request is honoured even while a command is
running.

A command that has to run alongside another one with its own input and output runs in a copy of the
process: a later stage of a pipeline such as `@grab --stdout s.cam | @replay --stdin out`, a
`milkcall --async` command, and `@replay`. Shared-memory streams it writes are visible to everyone,
but local images and variables it creates are not kept.

A session shared by several shells runs their commands one at a time too. A command that another
shell (another process group) sends while a long one runs waits for it, and then runs with
everything the long one left; at a terminal it says `milk: waiting for '<command>'`, and Ctrl-C gives
up without running it. Reads (`@get`, `@exists`, `@test`, completion) run alongside. `milkcall
--nowait <cmd>` fails at once instead, with `milk: busy: <command> for N s` and status 1.

## Snapshots, Undo, Assertions, and Logs

While `@session save` and `@session restore` persist a session's local images, calculator variables,
and settings, `@snapshot`, `@diff`, and `@restore` capture and restore shared-memory FPS parameters
and stream FITS keywords:

```bash
@snapshot -o base.json 'f.loop*' 's.cam*'     # FPS parameters and stream keywords to JSON
@diff base.json                               # compare with the live state
@restore -n base.json                         # dry run, then without -n to apply
@restore --undo                               # set back what this shell's last @set changed
@restore --undo -n 5                          # list what its last 5 @set calls changed
@assert "istd(wfs) < 0.5" "residual diverged" # fails with the values of both sides
@assert --running --fps-min 500 s.cam "camera stalled"
@assert --max-age 0.1 --cnt-min 100 --dim 128x128 s.cam   # last frame at most 0.1 s old, ...
@assert s.dark -m "no dark"                   # an entity alone: it exists (-e, --exists)
@log -n 20                                    # recent milk commands, timing and output
@log --errors
@session                                      # the session in use, its output file
@config prompt preset std                     # live telemetry in the prompt
@config                                       # digits, precision, procinfo
```

Each shell keeps the values its `@set` calls overwrote: FPS parameters and stream keywords, not
actions such as `run=start`. `@restore --undo [N]` sets back the last `N` calls, the last first,
where a value still holds what was written; a value changed since is left as it is, with a warning
and status 1. The writes of `@set --restore` and `@defer --restore` are not recorded.

## Examples

Share local images and calculator variables across two shells with a named session:

```bash
# Shell 1: start named session "ao_bench" and load calibration matrices into session memory
source milk.bash --session ao_bench
mem.mk2Dim "@L:resp_mat" 256 64
@calc "gain_nom = 0.35; resp_mat = 1.0"
```

```bash
# Shell 2 (or a background script): attach to "ao_bench" and inspect/use its local state
#!/usr/bin/env milk-script
@session use ao_bench
@calc -v g,norm_r "gain_nom; norm(resp_mat)"
echo "gain=$g resp_norm=$norm_r"
```

Save a pre-experiment snapshot of FPS parameters and restore automatically on exit:

```bash
#!/usr/bin/env milk-script
snap="/tmp/bench_snap_$$.json"
@snapshot -o "$snap" 'f.loop*' 's.cam*'
@defer '@restore "$snap"; rm -f "$snap"'

@set f.loop.gain=0.6
@waitfor s.wfs_resid 200 -t 5
@assert "istd(wfs_resid) < 0.2" "loop unstable at high gain"
```

---

← [Scripting](scripting.md) · [Calculator & Compute Units](calculator.md) · [CLI Syntax](CLIcore.md) · [Documentation Index](../index.md)
