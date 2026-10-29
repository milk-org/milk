# R4b brief: the operate package and the milk-pipes CLI

Roadmap: R4 (second half; it closes R4) · Depends on: R4a (landed) · Design: design-v7 §4.2,
D-6, D-11, D-24; A.5 `milk-pipes` · Decisions: the project doc `claude/r4-decisions.md` · Tests: py ·
Workflow: lean; prompts in `specs/pipelines/lean-prompts.md`. This brief and its tests replace
spec/plan/tasks, which overrides constitution §VIII for this slice.

## Goal
`milk-pipes` is a thin CLI over the plan registry. All operate code lives in `milk/operate/`, and
the transitional `infra/` folder is gone.

## Decided (Vincent, 2026-10-28)
1. **CLI modules are composition roots.** `provision/cli.py` and `operate/cli.py` may import both
   layers; library modules keep the one-way rule (`provision` and `operate` never import each
   other). `milk-pipes state` therefore shows a drift summary.
2. **Plan commands are top-level** (`milk-pipes deploy X`, as A.5 says). A plan whose name clashes
   with a fixed command is an error.

## Step 0: pure move (outside the budget, suite green before any new code)
- `git mv milk/pipeline.py milk/operate/pipeline.py`
- `git mv milk/compute_unit.py milk/operate/compute_unit.py`
- `git mv milk/infra/mains_pipeline_mgmt.py milk/operate/cli.py`
- Delete `milk/infra/`.

Then fix the imports everywhere, tests included, with no shims:
- `milk.pipeline` → `milk.operate.pipeline`;
- `milk.compute_unit` → `milk.operate.compute_unit`;
- `milk.infra.mains_pipeline_mgmt` → `milk.operate.cli`.

`pyproject.toml`: `milk-pipes = "milk.operate.cli:milk_pipes"`. A throwaway reference needed only
those import edits (about 10 lines) and the suite stayed green.

## CLI (the spec)

```
milk-pipes <plan> TARGET [--parent DIR]          one command per PLANS name: run_plan(pipeline, plan)
milk-pipes state  TARGET [--parent DIR]          state, reasons, drift summary, milk version change
milk-pipes link   TARGET LINKAGE [--parent DIR]
milk-pipes unlink TARGET LINKAGE [--parent DIR]
milk-pipes nuke   TARGET [--parent DIR] [--yes]  teardown plan, then rm -rf the rootdir
```

- **TARGET** works as in `Pipeline(target)`: a rootdir path, or a short name meaning
  `<cwd>/<TARGET>-rootdir`. With `--parent DIR`, the CLI uses `Pipeline.from_parent(DIR, TARGET)`.
- **Plan commands are read from `PLANS` at call time.** Use a `click.Group` subclass whose
  `list_commands`/`get_command` return the fixed commands plus one generated command per plan
  name, so plans registered after import (R5 kinds) appear without import-order tricks. A name
  clash raises a `ClickException` naming it.
- **`state`** never fails because of the pipeline's state. Its output:
  ```
  basic: configured
    reason: <one line per reason>
  drift: none | drift: <n> artifact(s); run `milk-pipeinit drift <rootdir>` | drift: unavailable (<why>)
  milk: rootdir made with <recorded>, installed <installed>      # only if they differ
  ```
  `n` counts the drift rows with `source` or `local` ≠ unchanged. "unavailable" covers a missing
  source confdir (any `MilkError` from `drift()`).
- **Errors:**
  - A `BestEffortError` from a plan or `nuke` prints one `warning: <label>: <error>` line per
    failure and exits 1.
  - Any other `MilkError` becomes a `ClickException` (message shown, exit 1).
  - Other exceptions propagate. There is still no top-level handler (D-11).
- **`nuke`** prompts "Type '<short>' to delete <rootdir>"; a mismatch aborts with nothing deleted.
  `--yes` skips the prompt. `Pipeline.nuke_unsafe()` deletes the rootdir even when teardown fails.
- **Removed:** `spool` (now `deploy`), `sys` (now `link`), and the hand-written `teardown` (now
  generated). The `DMCombPipeline`, `AOSimPipeline` and `AOPipeline` stubs go too (X5).

## Examples (operator session; reuse them for the docs page)

```console
$ milk-pipeinit clone $MILK_ROOT/milk-examples/pipelines/pipelinebasic-conf ~/pipes
$ milk-pipeinit init ~/pipes/pipelinebasic-conf            # → ~/pipes/basic-rootdir
$ cd ~/pipes
$ milk-pipes state basic
basic: cold
drift: none
$ milk-pipes deploy basic                                  # ClaimRundir → InitFPS → LoadData → ConfStart
$ milk-pipes state basic
basic: configured
drift: none
$ milk-pipes link basic loopback                           # needs a deployed, non-running pipeline
$ milk-pipes deploy basic                                  # idempotent: every step already succeeded
$ vi pipelinebasic-conf/conf.toml                          # edit the source confdir
$ milk-pipes state basic
basic: configured
drift: 1 artifact(s); run `milk-pipeinit drift /home/vdeo/pipes/basic-rootdir`
$ milk-pipes teardown basic                                # best effort; keeps the rootdir and work/
$ milk-pipeinit resync basic-rootdir                       # prints the teardown → deploy banner
$ milk-pipes deploy basic
$ milk-pipes state basic --parent /elsewhere               # a pipeline outside cwd
$ milk-pipes nuke basic                                    # asks you to type 'basic'
Type 'basic' to delete /home/vdeo/pipes/basic-rootdir: basic
```

Failure examples:
```console
$ milk-pipes link basic loopback                           # cold pipeline
Error: Pipeline basic is cold; run deploy.
$ milk-pipes deploy basic                                  # an FPS was destroyed by hand
Error: Pipeline basic is inconsistent (FPS missing: basic_delay1); run teardown.
$ milk-pipes teardown basic                                # one step failed, the others ran
warning: ConfStop: <error>
$ echo $?
1
```

## Tests
- **New:** `tests/operate/cli_test.py`, 16 py tests, parked with
  `pytest.importorskip("milk.operate.cli")`. **That module already exists before step 0's move
  finishes**, so P1 deletes the importorskip line as its very first edit. It uses the `paper`,
  `shm` and `boot` fixtures. It covers:
  - the command set;
  - plans registered late becoming commands;
  - TARGET and `--parent`;
  - name clashes;
  - errors reaching the user;
  - best-effort warnings with exit 1;
  - `state` when cold, inconsistent and configured, with a drift count, with drift unavailable,
    and with a milk version change;
  - link and unlink;
  - `nuke` deleting the rootdir even when teardown fails.
- **Checked so far:** a throwaway reference (the step 0 move plus a new `operate/cli.py`) passed
  all 16 together with R4a's 30 py tests.
- **Allowed migrations of existing tests:**
  - **Import paths (step 0):** `conftestaux/pipeline.py`, `conftestaux/compute_unit.py`,
    `compute_unit/`, `infra/`, `pipeline/`, `provision/`, `operate/`.
  - **`tests/provision/rootdir_test.py`:** the nuke CLI tests import `milk.operate.cli`, and
    `test_milk_pipes_init_is_gone` stays.

## Not doing
- start/stop plans (R5).
- A top-level error handler (anticipated).
- A global rundir scan.
- Pipeline-local plans and tasks (future spec).
- Porting `tests/mains/` (not collected): delete `tests/mains/pipeline_linsim_test.py` and close
  its two LOOSE_ENDS entries. `experiment_deploy.py` stays as a scratch reference.

## Done when
- The suite is green.
- **Budget:** net +60 LOC in `milk/`, step 0 excluded. The reference came to about +20: the new
  `operate/cli.py` (~120 lines) replacing `mains_pipeline_mgmt.py` (~100).
- ROADMAP R4 is `done` (Sub-spec: both briefs).
- `docs/` gains an operator page for `milk-pipes` built from the examples above. It links to the
  design's operating-lifecycle diagram and to `milk-pipeinit` for drift and resync.
- **Stop rule:** if a failure looks external, stop and report.
