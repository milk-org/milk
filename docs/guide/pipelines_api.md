# Pipelines: API concepts

The objects of the `milk` package and how they fit together, from a user's point of view. The
entry point is [Pipelines](pipelines.md); the CLI references are
[`milk-pipeinit`](pipelines_provision.md) and [`milk-pipes`](pipelines_operate.md).

## Package layout

```
milk/
  config/      what a pipeline is: the model, {var} substitution, confresolve. Pure data.
  provision/   confdirs and rootdirs: clone, consolidate, init, drift, resync. Files only.
  operate/     the live pipeline: Pipeline, ComputeUnit, tasks, plans, state, rundir.
  show/        milk-show: displays (matplotlib) of pipeline results.
  errors.py    every exception of the package
  env.py       the only reader of the MILK_* environment variables
```

`provision` and `operate` both build on `config` and never import each other: provisioning is
file operations and never touches SHM, FPS, tmux or processes; operating reads only the rootdir.
The CLIs (`milk-pipeinit`, `milk-pipes`, `milk-show`) only parse arguments and call the API.

## Configuration: `confresolve` and `PipelineConfig`

```python
from milk.config.resolve import confresolve

config = confresolve("pipelinebasic-conf")                      # a confdir
config = confresolve("basic-rootdir")                           # a rootdir: conf/ + recorded overrides
config = confresolve("pipelinebasic-conf", {"compute_units.delay0.fps.delaysec": 0.02})
config.name, config.loop_number, config.fps_name("delay0")      # 'basic', 0, 'basic_delay0'
config.compute_units["delay0"].exec                             # 'milk-fpsexec-mem-streamdelay'
config.compute_units["delay0"].fps["delaysec"]                  # 0.02
```

`confresolve` is the only way to obtain a configuration. It merges the layers, expands the
substitutions and validates, every time it is called, with no side effects (see
[Resolving](pipelines_conf.md#resolving-includes-overrides-substitution)). The result is a
frozen `PipelineConfig` (a pydantic model): `long_name`, `name`, `loop_number`, `kind`,
`compute_units` (a dict of `ComputeUnitSpec`: `exec`, `fps`), `dataloads`, `hooks`,
`systemlinkage`, `plans`, `teardown_kill_tmux`. `fps_name(cu)` is the one place where FPS
names are made.

## Provisioning: `milk.provision`

```python
from milk.provision.clone import clone
from milk.provision.hooks import consolidate
from milk.provision.rootdir import init, drift, resync

confdir = clone(src_confdir, dest_parent)        # copies the conf files; refuses an existing target
consolidate(confdir)                             # runs the pending hooks; returns their ids
rootdir = init(confdir, overrides={"compute_units.delay0.fps.delaysec": 0.02})
report = drift(rootdir)                          # DriftReport: artifacts, resolved_diff, structural, milk_version
report = resync(rootdir, run_active=lambda: Pipeline(rootdir).any_run_active())   # ResyncReport
```

These are the functions behind `milk-pipeinit`. `init` consolidates, copies, writes
`provenance.toml` and resolves the new rootdir once to check it. `drift` only reads. `resync`
takes the "is a run process active?" check as a callable, so that provisioning stays free of
any live-system code; the CLI passes the `Pipeline` query shown above. Their reports are
plain dataclasses, the same content as the CLI output ([Drift and
resync](pipelines_conf.md#keeping-a-rootdir-in-sync-drift-and-resync)).

## Operating: `Pipeline` and `ComputeUnit`

```python
from milk.operate.pipeline import Pipeline

pp = Pipeline("basic")                 # <cwd>/basic-rootdir
pp = Pipeline("/elsewhere/basic-rootdir")
pp.config                              # the resolved PipelineConfig
pp.root_folder, pp.work_folder         # the rootdir, and rootdir/work (the FPS work folder)
pp.compute_units                       # {'delay0': ComputeUnit, 'delay1': ComputeUnit}
pp.compute_units["delay0"].fps         # the pyMilk FPS, or None while it does not exist
pp.any_run_active()
pp.link("loopback"); pp.linked(); pp.unlink("all")
```

A `Pipeline` is a view on one rootdir: it resolves the configuration once, at construction,
and derives everything else from the live system when asked. `compute_units` builds fresh
`ComputeUnit` views on each access; a `ComputeUnit` wraps one `fpsexec` executable and its FPS
(`fpsinit`, `confstart`, `confstop`, `runstart`, `runstop` run the executable's commands, in tmux
when asked). You seldom call these yourself: tasks do.

## State

```python
from milk.operate.state import state

st = state(pp)            # PipelineState(state=<State.CONFIGURED>, reasons=())
st.state.value            # 'cold' | 'inconsistent' | 'deployed' | 'configured' | 'running' | 'mixed'
```

The state is never stored. It is computed from the rundir claim (`$MILK_SHM_DIR/pipe_rundirs/
<name>/`: boot id, rootdir, loop number) and from the existence and the conf/run status of
each CompU's FPS. `cold` means nothing is deployed; `inconsistent` means the rundir and the
FPSs disagree (a stale rundir from a previous boot, an FPS destroyed by hand, a failed deploy),
and only `teardown` is accepted from there. Operations that need a given state raise a
`StateError` (`ColdSystemError`, `InconsistentStateError`) that names the command to run.
The diagram is in [Operating: lifecycle](pipelines_operate.md#lifecycle).

## Plans and tasks

A **task** is one idempotent action on the live pipeline, with three methods: `success()`
(is the goal already reached?), `can()` (may it run? sets `failure_reason` otherwise) and
`forward()` (do it). `Pipeline.task_do(task)` calls them in that order and skips the work when
the goal already holds, which is what makes every plan safe to re-run. A task acts on all
CompUs, or on a subset given at construction.

```python
from milk.operate.tasks import RunStart, RunStop, WaitRunEnd

pp.task_do(RunStart)                        # every CompU
pp.task_do(RunStart(pp, ["mlat"]))          # one CompU
pp.task_do(WaitRunEnd(pp, ["mlat"]))        # wait (up to 120 s) for a run that ends by itself
pp.task_do(RunStop(pp, ["wfscam", "dmdelay"]))
```

The built-in tasks are `ClaimRundir`, `InitFPS`, `LoadData`, `ConfStart`, `RunStart`,
`RunStop`, `WaitRunEnd`, `DeapplyAllLinkages`, `ConfStop`, `KillTmux`, `DestroyFPS` and
`ReleaseRundir`, each described in [Operating: built-in
tasks](pipelines_operate.md#built-in-tasks). `task_do` bypasses the state gate; it is the
low-level entry point.

A **plan** is data: a name, ordered `Step`s (a task name and an optional CompU subset), a
failure policy (`abort` at the first failure, or `best_effort`: run everything, then raise one
error listing the failures) and the states it accepts to start from.

```python
from milk.operate.plans import PLANS, Plan, Step, run, run_plan

run_plan(pp, "deploy")                      # a registered plan, by name: what milk-pipes deploy does

mlat = Plan("mlat", (
    Step("RunStart", ("dmdelay",)),
    Step("RunStart", ("wfscam",)),
    Step("RunStart", ("mlat",)),
    Step("WaitRunEnd", ("mlat",)),
    Step("RunStop", ("wfscam", "dmdelay")),
))
run(pp, mlat)                               # any plan, registered or not

PLANS.register("mlat", mlat)                # now also `milk-pipes mlat`
```

`run` builds every step's task first, so an unknown task or CompU name fails before anything
has changed, then checks the state once, then runs the steps. The built-in plans are `deploy`
(`ClaimRundir → InitFPS → LoadData → ConfStart`, abort, from cold or any consistent state) and
`teardown` (`RunStop → DeapplyAllLinkages → ConfStop → KillTmux → DestroyFPS → ReleaseRundir`,
best effort, from any state). Every plan in `PLANS` is a `milk-pipes` subcommand; the
`plans` key of a `conf.toml` restricts which named plans may run on that pipeline (`run`
ignores it).

A task of your own subclasses `Task` and implements the three methods; `self.compute_units`
is its CompU subset and `wait_until(self.success, self.TIMEOUT)` (from `milk.wait`) is the
idiom for asynchronous effects. Register it with `TASKS.register("MyTask", MyTask)` (tasks defined in
`milk/operate/tasks.py` itself are registered automatically) and it can be named in a `Step`.
The full contract is in [Operating: task contract](pipelines_operate.md#task-contract).

## Errors and environment

Every exception of the package lives in `milk.errors`, under `MilkError`: `EnvError`,
`ConfigError`, `ProvisionError`, `TaskError` (`TaskCannotRunError`, `TaskDidNotSucceedError`),
`ComputeUnitError`, `LinkageError`, `StateError` (`ColdSystemError`,
`InconsistentStateError`) and `BestEffortError` (the aggregate of a best-effort plan, with
`.failures`). An error carries context when it has some (`pipeline`, `compu`, `task`, `path`,
`cmd`, `stderr`), appended to its message one per line. Library code never prints or exits;
pyMilk exceptions propagate as they are.

`milk.env` reads `MILK_ROOT`, `MILK_INSTALLDIR` and `MILK_SHM_DIR` when a function is called,
not at import, and raises `EnvError` naming what is missing. `import milk` and `confresolve`
work without a milk environment; anything that touches SHM or runs an executable needs one.

## Displays: `milk-show`

`milk-show mlat -p PIPE` plots the latency measurement (`hardwlatency.dat`) of a pipeline's
`mlat` CompU from its datadir under `rootdir/work/`; `r` in the window re-reads the file.
`milk.show.cli.mlat_figure(datadir)` returns the figure for use in a script or a notebook.
