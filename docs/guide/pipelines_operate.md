# Operating a pipeline: `moka`

The states of a deployed pipeline, the `moka` commands that move between them, and what
happens when they fail.

**Audience:** Operator · See also: [Tour](pipelines_tour.md) · [Plans and tasks](pipelines_plans.md) ·
[Drift and resync](pipelines_provision.md) · [Glossary](pipelines_glossary.md)

## 1. Lifecycle

The states are those printed by `moka state`; the arrows are commands and tasks.

```mermaid
stateDiagram-v2
    direction LR
    cold --> configured : deploy
    configured --> running : runstart
    running --> configured : runstop
    configured --> cold : teardown
    running --> cold : teardown
    cold --> [*] : nuke
```

| State          | Rundir                                                            | FPS  | Processes                      |
| -------------- | ----------------------------------------------------------------- | ---- | ------------------------------ |
| `cold`         | absent                                                            | none | none                           |
| `deployed`     | claimed                                                           | all  | conf stopped                   |
| `configured`   | claimed                                                           | all  | conf running                   |
| `running`      | claimed                                                           | all  | every `runstart` CompU running |
| `mixed`        | claimed                                                           | all  | anything else                  |
| `inconsistent` | stale, or claimed with an FPS missing, or absent with FPS present |      |                                |

The rundir is the deploy's claim for this boot, `$MILK_SHM_DIR/pipe_rundirs/<name>/`.
`deployed` and `configured` also require that no run process is running. `running` looks only at
the CompUs listed in [`runstart`](pipelines_conf.md#3-conftoml-keys) (default: all); the others
may run or not. `nuke` is a teardown followed by `rm -rf` of the rootdir.

### 1.1. Failures and recovery

A failed deploy leaves the pipeline in a state that tells what to do next.

```mermaid
stateDiagram-v2
    direction LR
    cold --> cold : deploy fails at ClaimRundir
    cold --> deployed : deploy fails after all FPS exist
    cold --> inconsistent : deploy fails before all FPS exist
    deployed --> configured : deploy (resumes)
    deployed --> cold : teardown
    inconsistent --> cold : teardown
```

- A reboot wipes the rundir and the FPSs: the pipeline is `cold` again. A rundir left from a
  previous boot (boot id mismatch) is `inconsistent`.
- From `inconsistent`, only `teardown` is accepted. Teardown is accepted from every state.
- `mixed` counts as consistent (for a plan's `accepts`).
- A rundir claimed by another rootdir adds a reason to the state, not a new state.

## 2. Commands

```text
moka <plan>  [-p PIPE]                   run plan <plan>: deploy, teardown, ...
moka runstart|runstop [-p PIPE] [CU]...  the plan; with CU: on these CompUs only
moka state   [-p PIPE]                   state, reasons, drift summary, milk version change
moka plans   [-p PIPE]                   the plans, and which the current state accepts
moka tasks   [-p PIPE] [PLAN]            the tasks (or a plan's steps): success, can
moka link    [LINKAGE] [-p PIPE]         apply [systemlinkage.<LINKAGE>]; none: list them
moka unlink  LINKAGE [-p PIPE]           remove its symlinks; `all`: of every linkage
moka nuke    [-p PIPE] [--yes]           teardown, then rm -rf the rootdir
```

- **`-p/--pipe PIPE`** is a rootdir path, or a short name meaning `<cwd>/<PIPE>-rootdir`.
  Without `-p`, the current directory must be a rootdir. Every command first prints
  `rootdir: <path>`, the rootdir it acts on. In Python: `Pipeline(pipe)`.
- **Output.** Commands that act (plans, `link LINKAGE`, `unlink`, `nuke`) print on stderr only;
  the others print their data on stdout.
- **Plan commands** are one per plan of the pipeline, built-in or
  [local](pipelines_plans.md#4-custom-plans) (`moka -h -p PIPE` lists them; `moka plans` [also says which the state accepts](#22-plans)). `deploy` is
  idempotent: run again on a configured pipeline, every step already succeeded. `teardown` is
  best effort and keeps the rootdir and `work/`. `runstart` and `runstop` take CompU names; they
  then act on those CompUs only. `moka <plan> -h` describes a plan. The steps of every plan are in
  [Plans and tasks: built-in plans](pipelines_plans.md#1-built-in-plans).
- **`link`** is allowed only in `deployed` and `configured`. `teardown` removes every linkage.
- **`nuke`** asks you to type the short name; `--yes` skips that. It deletes the rootdir even if
  teardown fails. It runs the pipeline's `teardown` plan, a local override included; if the
  pipeline's local plans do not load, teardown does not run and the rootdir is still deleted.

### 2.1. `state`

`state` never fails because of the pipeline's state. It prints the state with its reasons, then
the drift summary, then the milk version if it changed since `init`:

```bash
moka state -p basic
```

```text title="Output"
rootdir: /home/vdeo/pipes/basic-rootdir
basic: configured
drift: 1 artifact(s); run `mokainit drift /home/vdeo/pipes/basic-rootdir`
```

The drift line is `drift: none`, `drift: <n> artifact(s); run ...` or
`drift: unavailable (<why>)`, for example when the source confdir is gone. What the drift means
is in [Drift and resync](pipelines_provision.md#2-drift).

### 2.2. `plans`

`NOW` checks the plan's `accepts` gate against the current state, not what its steps would do;
`n/a`: a task or unit the plan names does not exist.

```text title="Output"
rootdir: /home/vdeo/pipes/aoloop-rootdir
aoloop: cold
USAGE                                         NOW
deploy [-p PIPE]                              ok
  ClaimRundir -> InitFPS -> LoadData -> ConfStart
...
runstart [-p PIPE] [CU]...                    no (cold; run deploy)
  RunStart
...
teardown [-p PIPE]                            ok
  RunStop -> DeapplyAllLinkages -> ConfStop -> KillTmux -> DestroyFPS -> ReleaseRundir
```

### 2.3. `tasks`

`tasks` gives `success()` and `can()` of each task of the pipeline, built on all CompUs (`RunStart`:
on the [`runstart`](pipelines_conf.md#3-conftoml-keys) ones), or with `PLAN`, of each step of that
plan. It never runs a task.

```text title="Output"
rootdir: /home/vdeo/pipes/aoloop-rootdir
aoloop: cold
TASK         DONE  CAN  REASON
ClaimRundir  no    yes
...
ConfStart    no    no   FPS aoloop_acquWFS is missing or invalid.
```

### 2.4. Plan runs

A plan command exits 1 when a step is refused or fails. On a terminal, the running step shows its
elapsed seconds.

```text title="Output"
> moka measlinresp
rootdir: /home/vdeo/pipes/aoloop-rootdir
aoloop: configured
  [1/4] RunStart[acquWFS] ....... [ -- ] already done
...
  [4/4] WaitLinResp ............. [ ok ] done (21.3 s)
[ ok ] measlinresp done in 21.9 s
```

```text title="Output"
  [1/1] MakeCM ... [ NO ] refused: /.../calib/RMmodesWFS.fits does not exist; run `moka measlinresp`.
[FAIL] mkcm: stopped at step 1/1 (MakeCM)
```

## 3. Errors

`moka` and `mokainit` print an error as `[FAIL] <message>` on stderr (`Error: <type>: <message>`
when it is not the package's own), without a stack trace, and exit 1. A message carries its
context lines when it has some:

```text title="Output"
[FAIL] Invalid config:
  compute_units.delay0.fps.procinfo: procinfo: 1 validation error ...
path: aoloop-conf/conf.toml
```

`--errstack`, placed before the command (`mokainit --errstack resolve aoloop-conf`), lets
the exception propagate with its stack trace; `--pdb` opens an `ipdb` post-mortem session on it
instead. Log records of the package print to stderr as
`LEVEL: message`.

`--color auto|always|never`, also placed before the command, sets the colors; `auto` (default)
colors a terminal unless `NO_COLOR` is set.

A command refused because of the state names the command to run:

```text title="Output"
[FAIL] Pipeline basic is cold; run deploy.
pipeline: basic
```

A best-effort plan (`teardown`) runs every step, then exits 1 if one failed.

Anything that touches the live system needs `MILK_ROOT`, `MILK_INSTALLDIR` and `MILK_SHM_DIR` in
the environment. `mokainit` itself does not, except `resync`, which checks that no run
process is active.

## 4. Displays: `mokashow`

`mokashow mlat -p PIPE` plots the latency measurement (`hardwlatency.dat`) of a pipeline's `mlat`
CompU from its datadir under `rootdir/work/` (`-w/--workdir DIR`: under `DIR` instead); `r` in the
window re-reads the file.
`moka.show.cli.mlat_figure(datadir)` returns the figure for use in a script or a notebook.

`mokashow rm -p PIPE` and `mokashow cm -p PIPE` browse the mode pairs of `rootdir/calib/`
(`RMmodes{WFS,DM}.fits`, `CMmodes{WFS,DM}.fits`): one mode per WFS and DM panel, with its RMS, P2V,
mean and std; a slider picks the mode, Play steps through them at the rate of the fps box. When
`calib/wfsmask.fits` or `calib/dmmask.fits` exists, its check box (keys `w`, `d`) restricts the
stats to the mask. `cm` also plots the singular values over the largest one, with the `mkcm`
flavor, `-r`, `-n` and the kept count. `r` re-reads the files.
`moka.show.modes.modes_figure(calib, "rm" | "cm")` returns the figure.
