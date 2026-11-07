# O1 brief: run any list of steps as a plan; RunStart/RunStop[cu]; milk-show mlat

Roadmap: O1 · Depends on: R4 · Input: project doc
`claude/aorun-audit.md` §4.1, §5 (row 020-mlat) · Tests: py + milk · Workflow: lean
(`specs/pipelines/lean-prompts.md`).

## Goal line

```python
from milk.operate.pipeline import Pipeline
from milk.operate.plans import Plan, Step, run, run_plan

pp = Pipeline("mlatsim")                    # from tests/resources/mlatsim-conf, after init
run_plan(pp, "deploy")
run(pp, Plan("mlat", (Step("RunStart", ("dmdelay",)), Step("RunStart", ("wfscam",)),
                      Step("RunStart", ("mlat",)), Step("WaitRunEnd", ("mlat",)),
                      Step("RunStop", ("wfscam", "dmdelay")))))
```
```console
$ milk-show mlat mlatsim-rootdir/work/fps.mlatsim_mlat.datadir     # press r to redraw
```

## Decided (Vincent, 2026-11-04)

1. The entrypoint is Python only: `run(pipeline, plan)`. The CLI is unchanged, and D-24 still
   holds.
2. `WaitRunEnd` is a third task. mlat ends by itself, and `RunStop` would kill it.
3. The conf `plans` key restricts named plans only; `run()` ignores it.
4. The display is a separate `milk-show` command in `milk/show/`, using matplotlib directly.
   Pressing `r` in the window redraws from the current file. Nothing is recorded: mlat's datadir
   under `work/` is the output.

## Interface

- **`plans.run(pipeline, plan)`**: the runner for any plan.
  - It builds every step's task first, so an unknown task or CompU raises `ConfigError` before
    any side effect.
  - It gates on `plan.accepts` once, then calls `task_do` per step, with `on_failure` as today.
- **`run_plan(pipeline, name)`** keeps its signature and its `plans` check, then calls `run`.
- **`RunStart[cus]`**:
  - `can`: each FPS exists and its conf process runs ("…; run deploy.").
  - `forward`: `runstart -tmux` for those not running, then wait up to `TIMEOUT`.
  - `success`: all of them run.
- **`RunStop[cus]`**: as it is today.
- **`WaitRunEnd[cus]`**:
  - `TIMEOUT = 120.0`.
  - `forward`: wait until no run in the subset is running. It never stops anything.
  - `success`: no run in the subset is running.
- **Timeouts**: `RunStart` and `WaitRunEnd` set `failure_reason` in `forward()`, naming the FPSs.
  They do not set it in `success()`.
- **`milk-show mlat DATADIR`** (console script `milk.show.cli:milk_show`):
  - It plots `hardwlatency.dat` (dt [µs] against value, as `cacao-aorun-021-mlatshow` does).
  - `r` re-reads the file and redraws.
  - `mlat_figure(datadir)` returns the figure, for tests.
  - matplotlib is imported lazily, and is a package dependency (Vincent, 2026-11-04).
- **`tests/resources/mlatsim-conf/`**:
  - A `command` hook (`scripts/mkdm.py`) writes a 16×16 float32 DM.
  - `[[dataloads]]` loads it as `mlatsim_dm`.
  - `dmdelay` (streamdelay, naive, triggermode 3 on the DM, `delaysec = 0.003`) feeds
    `wfscam` (streamcopy, triggermode 4, `triggerdelay = 0.001`): a camera on a fixed clock
    that DM writes never trigger. `[[dataloads]]` creates all three streams.
  - `mlat` does a single pass.

## Check first (external behaviour read from source, not yet run; stop rule applies)

- `procinfo.loopcntMax = 0` makes mlat run once and exit.
- `hardwlatency.dat` lands in `rootdir/work/fps.mlatsim_mlat.datadir/` whatever the caller's
  cwd. `fps_write_RUNoutput_file` writes to a relative datadir.
- The measured latency is close to `delaysec`. The test bounds are loose; a miss is a finding,
  not a reason to loosen them.

## Tests

- **New, parked:**
  - `tests/operate/steps_test.py` (14 py): `run()` order, subsets, validation before side
    effects, state gate, the `plans` key, and RunStart/RunStop/WaitRunEnd on subsets, timeouts
    included.
  - `tests/show/show_test.py` (3 py): what the figure plots, the `r` redraw, the CLI.
  - `tests/operate/mlat_milk_test.py` (2 milk): the measurement and its bounds, the datadir
    location, the figure on real data, and a second measurement that runs again.
- **Changed:** `tests/conftestaux/pipeline.py` `_fake_compu` gets `conf`/`run` flags, set by
  `runstart`/`runstop`. The defaults are unchanged.
- **Reference:** a throwaway reference passed the 17 new py tests and R4a/R4b's 48 (pyMilk
  stubbed), at about +80 net LOC in `milk/`. The milk tests have not been run.

## Not doing

Facts, step parameters, a CLI for `run`, recording/history, named start/stop plans, kinds.

## Done when

- The suite is green, the milk tests included.
- **Budget:** +100 net LOC in `milk/`.
