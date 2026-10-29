# R4b brief: the operate package and the milk-pipes CLI

Roadmap: R4 (second half; it closes R4) · Depends on: R4a · Design: design-v7 §4.2, D-6, D-24;
A.5 `milk-pipes` · Decisions: the project doc `claude/r4-decisions.md` · Tests: py · Workflow: lean.
**Status: spec only.** The tests are written once R4a has landed, against its real API. Review
the decisions below now.

## Goal
`milk-pipes` is a thin CLI over the plan registry. Its operate code lives in one package, and the
transitional `infra/` folder is gone.

## Step 0: pure move (outside the budget, suite green before any new code)
`git mv` the following, then fix the imports everywhere, tests included, with no shims:
- `milk/pipeline.py` → `milk/operate/pipeline.py`;
- `milk/compute_unit.py` → `milk/operate/compute_unit.py`;
- `milk/infra/mains_pipeline_mgmt.py` → `milk/operate/cli.py`.

Then delete `milk/infra/` and update the `milk-pipes` entry point in `pyproject.toml`. The
`DMCombPipeline`, `AOSimPipeline` and `AOPipeline` stubs go (X5).

## CLI (A.5, as amended)

```
milk-pipes <plan> TARGET [--parent DIR]     one subcommand per PLANS name, generated at import
milk-pipes state TARGET [--parent DIR]      state and reasons
milk-pipes link TARGET LINKAGE [--parent DIR]
milk-pipes unlink TARGET LINKAGE [--parent DIR]
milk-pipes nuke TARGET [--parent DIR] [--yes]   teardown plan, then rm -rf the rootdir; type-the-short-name prompt
```
- **TARGET resolution** follows `Pipeline(target)` / `from_parent` (R3b D2).
- **Generated plan commands** call `run_plan`. A plan that the pipeline's `plans` key excludes
  fails through `run_plan`'s ConfigError. Commands are generated for every registered plan, not
  per pipeline. Kind plans (R5) show up automatically once their module is imported before the
  CLI group is built.
- **Removed:** `spool` (now `deploy`), `sys` (now `link`), and the hand-written `teardown` (now
  generated).
- **BestEffortError** from teardown or nuke: print one warning line per failure and exit 1.
  Other errors propagate (no top-level handler, D-11).

## Decisions for Vincent
1. **Can `milk-pipes state` show drift?** A.5 says `state` shows "derived state, drift summary,
   package-version change". Drift lives in `provision`, and operate must not import provision.
   - **Recommended:** CLI modules (`provision/cli.py`, `operate/cli.py`) are *composition roots*
     and may import both layers; library modules may not. `state` then prints the state plus
     `drift()`'s summary.
   - **Alternative:** `state` prints the state only, and operators run `milk-pipeinit drift`.
2. **Where do the plan commands live?** They sit at the top level (`milk-pipes deploy X`), as
   A.5 says. The alternative is a `run` group (`milk-pipes run deploy X`), which avoids name
   clashes with fixed commands (`state`, `link`, `unlink`, `nuke`). Either way, registering a
   plan with one of those names is refused.

## Tests (to write when R4a lands)
CliRunner, py, with the `paper`, `shm` and `boot` fixtures from `tests/conftestaux/operate.py`:
- **Help:** `--help` lists `deploy`, `teardown`, `state`, `link`, `unlink`, `nuke`, and not
  `spool` or `sys`.
- **Generation:** a plan registered in a test appears as a command, and invoking it calls
  `run_plan` with that name (monkeypatched).
- **`state`** prints the state and its reasons.
- **link/unlink** through the CLI.
- **nuke:** migrate R3b's nuke tests (the prompt, `--yes`, `--parent`).
- **Teardown failures:** with a failing teardown step, a warning is printed and the exit code is 1.

## Not doing
start/stop plans (R5). A top-level error handler (anticipated). A global rundir scan.
Pipeline-local plans and tasks (future spec).

## Done when
- The suite is green.
- Budget: net +80 LOC in `milk/`, step 0 excluded.
- ROADMAP R4 is `done`. `docs/` gains an operator page for `milk-pipes`, linking to the design's
  operating-lifecycle diagram.
- LOOSE_ENDS: port or delete `tests/mains/pipeline_linsim_test.py` (it is not collected), and
  close its two entries.
