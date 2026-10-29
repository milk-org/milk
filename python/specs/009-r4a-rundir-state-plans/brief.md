# R4a brief: rundir, state, tasks, runner, generic plans

Roadmap: R4 (first half; R4b follows) · Design: design-v7 §0 (rundir, cold, inconsistent, task,
plan, state), §4.0 rule 4, §4.1, §4.2, §4.6, §4.9, D-9, D-17, D-18, D-19, D-20, D-24, D-25; A.4 ·
**Decisions:** the project doc `claude/r4-decisions.md` (Vincent, 2026-10-27/28; decision 3 was dropped on 2026-10-28, see below) · Tests: py + milk ·
Workflow: lean; prompts in `specs/pipelines/lean-prompts.md`. This brief and its tests replace
spec/plan/tasks, which overrides constitution §VIII for this slice.

## Goal
Operating a pipeline goes through **registered tasks**, composed into **plans** that are data, and
run by **one runner**. The runner checks a **derived state** first. Deploy **claims** a rundir for
this boot; teardown releases it.

**Changing the compute-unit set while the pipeline is not cold is best effort (Vincent,
2026-10-28).** Everything acts on the **current config**. The rundir records no FPS list and no
linkage pairs, and there is no "config changed since deploy" tracking. If a resync removed a
compute unit before teardown, its FPS may be left behind; that is the operator's concern.
This replaces decision 3 of `r4-decisions.md`.

## Where the code goes
New modules go directly into `milk/operate/`; R4b moves `pipeline.py` and `compute_unit.py` there
and deletes `infra/`.

```
milk/registry.py        Registry[T] (shared by tasks, plans, hook runners)
milk/wait.py            wait_until(predicate, timeout, interval=1e-3) -> bool
milk/operate/rundir.py  A.4 files, simplified: boot_id, claim.toml
milk/operate/state.py   State, PipelineState, fps_status, state, gate, require
milk/operate/tasks.py   Task base, TASKS, the 10 generic tasks
milk/operate/plans.py   Step, Plan, PLANS, run_plan
```

## Interface (the spec)

```python
# milk/registry.py
class Registry(Generic[T]):
    def __init__(self, what: str) -> None
    def register(self, name: str, item: T) -> T        # duplicate → ValueError
    def __getitem__(self, name: str) -> T              # unknown → ConfigError naming the known names
    def __contains__(self, name: str) -> bool
    def names(self) -> list[str]                       # sorted
# Instances live with their users: TASKS (operate/tasks.py), PLANS (operate/plans.py),
# HOOK_RUNNERS (provision/hooks.py, replacing R3a's RUNNERS dict).

# milk/errors.py
class StateError(MilkError); class ColdSystemError(StateError); class InconsistentStateError(StateError)
# Messages name the remedy: "...; run deploy." / "...; run teardown."

# milk/config/model.py: PipelineConfig gains (closed key set)
plans: list[str] | None = None        # restricts run_plan to these names; validated against PLANS when run
teardown_kill_tmux: bool = False      # tests/resources/pipelinebasic-conf sets it true (top of file)

# milk/operate/rundir.py: $MILK_SHM_DIR/pipe_rundirs/<name>/
current_boot_id() -> str                                     # /proc/sys/kernel/random/boot_id; tests monkeypatch it
rundir_path(name) -> Path
@dataclass(frozen=True) class Claim: name; loop_number; rootdir: str; deployed; user; pid   # no FPS list (amends A.4)
    @classmethod new(name, loop_number, rootdir) -> Claim        # now/user/pid filled
read_claim(name) -> Claim | None;  read_boot_id(name) -> str | None
write_claim(claim)                                           # writes boot_id (current) and claim.toml (TOML)
all_claims() -> list[Claim]
release(name)                                                # rm -rf the rundir

# milk/operate/state.py
class State(str, Enum): COLD, INCONSISTENT, DEPLOYED, CONFIGURED, RUNNING, MIXED
@dataclass(frozen=True) class PipelineState: state: State; reasons: tuple[str, ...] = ()
fps_status(name) -> tuple[bool, bool] | None     # (conf_running, run_running); None if no <name>.fps.shm. Tests patch it.
state(pipeline) -> PipelineState                 # never raises for these states
gate(name, st, accepts) -> None                  # raises Cold/Inconsistent per `accepts`
require(pipeline, accepts) -> PipelineState      # state() + gate()

# milk/operate/tasks.py
class Task(abc.ABC):                             # replaces infra.task_models.SimpleTask
    TIMEOUT: float = 10.0                        # bound for the wait inside forward()
    failure_reason: str
    def __init__(self, pipeline, compus: Collection[str] | None = None)   # unknown CompU → ConfigError
    compute_units -> dict[str, ComputeUnit]      # config views, restricted to the subset
    can / forward / success                      # abstract
TASKS: Registry[type[Task]]                      # registered under the class name

# milk/operate/plans.py
@dataclass(frozen=True) class Step: task: str; compus: tuple[str, ...] | None = None   # .label "Task" or "Task[a,b]"
@dataclass(frozen=True) class Plan: name; steps: tuple[Step, ...]; on_failure: "abort"|"best_effort" = "abort"
                                    accepts: "cold_or_consistent"|"consistent"|"any" = "consistent"
PLANS: Registry[Plan]
run_plan(pipeline, name) -> None

# milk/pipeline.py (stays at top level until R4b)
Pipeline.task_do(task_or_class)                  # the task rule, unchanged; accepts Task classes/instances
Pipeline.link(name) / Pipeline.unlink(name)      # replace apply/deapply_system_linkage
Pipeline.linked() -> list[str]                   # configured linkages with a symlink in place
Pipeline.nuke_unsafe()                           # run_plan("teardown"), then rm -rf, even on failure
Pipeline.any_run_active()                        # kept (resync's run check)
```

### state(pipeline): classification, first match wins

| Condition | State, reason |
|-----------|---------------|
| no claim, no config FPS present | COLD |
| no claim, some config FPS present | INCONSISTENT, "FPS … exist without a rundir" |
| claim, `boot_id` ≠ current | INCONSISTENT, "stale rundir from a previous boot" |
| claim, some **config** FPS missing | INCONSISTENT, "FPS missing: …" |
| all run | RUNNING |
| none run, all conf | CONFIGURED |
| none run, none conf | DEPLOYED |
| otherwise | MIXED |

- A claim whose `rootdir` differs from this pipeline's adds the reason "claimed by …" without
  changing the state. FPS names derive from the short name, so both rootdirs address the same
  deployment slot.

### run_plan(pipeline, name)
1. `PLANS[name]`; an unknown name is a ConfigError.
2. If `config.plans` is set: every listed name must exist in `PLANS`, and `name` must be among
   them. Otherwise raise a ConfigError whose message includes `repr(name)`.
3. Call `state()` **once**, then `gate(..., plan.accepts)`:
   - `any`: always proceeds;
   - `cold_or_consistent`: INCONSISTENT raises "run teardown";
   - `consistent`: COLD raises "run deploy", INCONSISTENT raises "run teardown".
4. For each step: `pipeline.task_do(TASKS[step.task](pipeline, step.compus))`.
   - `abort`: re-raise the first `MilkError`.
   - `best_effort`: collect `(step.label, error)` and raise one `BestEffortError` at the end.
5. Return `None`. There is no report object; library code never prints.

### Generic plans (Python, registered at import)
| Plan | Steps | on_failure | accepts |
|------|-------|-----------|---------|
| `deploy` | ClaimRundir → InitFPS → LoadData → ConfStart | abort | cold_or_consistent |
| `teardown` | RunStop → DeapplyLinkages → ConfStop → KillTmux → DestroyFPS → ReleaseRundir | best_effort | any |

The generic kind ships **no** start/stop plans; kinds may add them in R5 (decision 2).

### Tasks
All tasks act on the config's compute-unit views (`Task.compute_units`, which honours the step
subset). Teardown-side tasks are best effort and act only on compute units whose FPS exists.

| Task | can | forward | success |
|------|-----|---------|---------|
| ClaimRundir | the rundir is not claimed by another rootdir; no other claim has this `loop_number` (reason names it) | `write_claim(Claim.new(...))` | own claim, current boot |
| InitFPS | `work/` exists | fpsinit in `work/` (D-17), set the denested params | subset FPS exist and are valid |
| LoadData | sources in `initdata/` exist | as today | shape and dtype match (B5) |
| ConfStart | subset FPS exist and are valid | `confstart(tmux=True)` where not running, then `wait_until(all conf running, TIMEOUT)`; on timeout raise naming the FPS ("tmux session busy?") | subset conf running |
| RunStop | always | `compu.runstop()` where running, then `wait_until(none running, TIMEOUT)` | none running |
| DeapplyLinkages | always | `pipeline.unlink(n)` for every configured linkage | `not pipeline.linked()` |
| ConfStop | always | `compu.confstop()` where running, then `wait_until(none conf running, TIMEOUT)` | none conf running |
| KillTmux | always | `tmux kill-session -t <fps>` for live sessions, **only if `teardown_kill_tmux`** | no live session (always true when the flag is off) |
| DestroyFPS | always | `compu.fps.destroy()` | no FPS exists |
| ReleaseRundir | always | `rundir.release(name)` | rundir absent |

`TIMEOUT = 10.0` covers RunStop's wait of up to 2 × `procinfo.triggertimeout` while that timeout
is ≤ 5 s (LOOSE_ENDS item). Operational tasks in later slices may set minutes.

### Linkages (decision 6)
- **`link(name)`** calls `require(self, "consistent")`, so from COLD it raises "run deploy". It
  refuses RUNNING or MIXED with a LinkageError ("stop it first"), then applies the symlinks
  (idempotent). Nothing is recorded. The `rootdir/systemlinkage.tag` goes away.
- **`unlink(name)`** removes that linkage's symlinks that still point at the configured remote, as
  today's deapply does.
- **`linked()`** lists the configured linkages that have a symlink in place.
- **Teardown** unlinks every configured linkage.

### Also in R4a
- `tests/conftestaux/polling.py`'s `poll_until_or_fail` delegates to `milk.wait.wait_until`.
- **Minimal `milk-pipes` edits** (R4b replaces the CLI): `spool` runs `run_plan(deploy)`,
  `teardown` runs `run_plan(teardown)`, `sys` runs `link`.
- **Deleted:**
  - `infra/deploy_tasks.py` and `infra/task_models.py`, so `SimpleTask`, `ReversibleTask`,
    `CompoundTask`, `TestConfig` and `task_undo` all go;
  - `Pipeline.teardown`, `apply_system_linkage` and `deapply_system_linkage`;
  - the old `StartConfProcesses.TIMEOUT_ALL`.

## Tests
- **New** (all parked with `pytest.importorskip("milk.operate.plans")`; **P1 deletes those lines
  first**):
  - `tests/operate/operate_test.py`: 30 py tests covering the registry, claim, state, runner
    (gating, a single state check, skip, abort, best_effort, subsets, the `plans` key) and
    linkages. FPSs are simulated by the `paper` fixture.
  - `tests/operate/plans_milk_test.py`: 8 milk tests:
    - deploy from cold, idempotent, resuming a partial deploy;
    - inconsistent → teardown → deploy;
    - teardown after a reboot, and with FPSs but no rundir;
    - tmux kept by default and reused on redeploy, or killed with the flag.
  - `tests/conftestaux/operate.py`: the `shm`, `boot` and `paper` fixtures, already registered.
- **Allowed migrations of existing tests:**

| File | Change |
|------|--------|
| `tests/pipeline/compu_views_test.py` | imports become `InitFPS`, `LoadData`, `ConfStart` from `milk.operate.tasks`. B1 and B6 target ConfStart (`TIMEOUT`). B7 is replaced by: `nuke_unsafe` removes the rootdir even when `run_plan` raises (monkeypatch it). The best_effort aggregation is now covered by the runner tests |
| `tests/pipeline/deploy_pipeline_test.py` | `DeployFPS`→`InitFPS`, `StartConfProcesses`→`ConfStart`, `SimpleTask`→`Task` |
| `tests/pipeline/system_linkage_test.py` | **delete**: covered by the operate_test linkage tests (the two xfails are dropped) |
| `tests/conftestaux/pipeline.py` | `cloned_pipeline` uses `run_plan(..., "teardown")` instead of `.teardown()` |
| `tests/provision/rootdir_test.py` | `test_teardown_keeps_the_rootdir` uses `run_plan(pp, "teardown")` on a cold pipeline and asserts the rootdir, `work/` and provenance are kept (no `fake_compus`) |
| `tests/resources/pipelinebasic-conf/conf.toml` | add `teardown_kill_tmux = true` right after `loop_number` |

## Not doing (R4b or later)
- **R4b:** the `operate/` move of `pipeline.py` and `compute_unit.py`; deleting `infra/`;
  generated `milk-pipes <plan>` subcommands; `state`, `link` and `unlink` commands; `nuke` on top
  of the plan; removing `spool` and `sys`; an operator page in `docs/`.
- **Later:** start/stop plans (R5 kinds); RunStart (R5); pipeline-local plans and tasks (future
  spec); locks and concurrency (D-20); a global rundir scan.

## Done when
- The suite is green: 30 py and 8 milk tests in `tests/operate/`, plus the migrated tests. The
  milk tests must pass **3 runs in a row**.
- **Budget:** net +250 LOC in `milk/`. A throwaway reference passed the 30 py tests at about +150
  net once `infra/deploy_tasks.py` and `task_models.py` are deleted.
- ROADMAP R4 stays `in-progress` (Sub-spec = this brief); R4b sets it to `done`.
- **LOOSE_ENDS:** close the B6 busy-tmux item (covered by `teardown_kill_tmux` and ConfStart's
  timeout message), the wait-primitive item, and the triggertimeout item (TIMEOUT 10 s).
