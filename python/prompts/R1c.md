# R1c: CompU views + bug fixes

Feature directory: `specs/003-r1c-compu-views-fixes` · Tests: milk

## /speckit-specify

```text
Feature directory: specs/003-r1c-compu-views-fixes
Parent roadmap: specs/pipelines/ROADMAP.md → entry R1c. Design: specs/pipelines/design-v7.md (§4.1; defects X3, B1, B4, B5, B6, B7).

Obtain compute units in exactly one way, and fix the known correctness bugs.

Why: tasks build compute-unit objects ad hoc (X3), and several tasks report success or failure wrongly (B1, B4–B7).

Requirements:
- Pipeline.compute_units is a read-only property returning dict[str, ComputeUnit], keyed by the compute-unit names in the config. Each access builds or refreshes the views: a link to an FPS that has become invalid is re-established, and a stale one is dropped. It never creates or starts anything. No other code constructs ComputeUnit objects.
- Pipeline.any_run_active() returns True if any compute unit has an active run process.
- B1: StartConfProcesses.can returns False, with a failure_reason, when an FPS is invalid.
- B4: re-running the folder setup on a partially created rootdir reports a clear failure_reason instead of crashing.
- B5: data loading counts as successful only if each target stream exists with the source's shape and dtype.
- B6: a conf-start timeout raises TaskDidNotSucceedError naming the compute units whose conf process did not start; a busy tmux session is reported through can()/failure_reason.
- B7: Pipeline.nuke runs every step best-effort, FPS destruction included, and raises one aggregate error at the end listing each failed step, instead of printing.

Out of scope: the state model, rundir and teardown (R4).

Success criteria: the suite is green; there is a test for each bug and for the property's refresh behaviour; nothing outside the property constructs ComputeUnit.
```

## /speckit-plan

```text
Python ≥3.11, no new dependencies. Touch pipeline.py, compute_unit.py (renamed in R1b) and infra/deploy_tasks.py.
The property derives views from PipelineConfig.compute_units plus the single fps_name function from R1b. A ComputeUnit is a cheap view, so caching it is optional and never authoritative.
any_run_active() replaces Pipeline.is_running. Update its callers (system linkage) and keep tests monkeypatching it the way system_linkage_test does today.
The aggregate error for nuke is a MilkError family class from milk/errors.py; add a small aggregate subclass there if none exists.
Tests: milk; reuse the existing conftest spoofing and fixtures.
```

## /speckit-implement

Two phases: (1) the property and `any_run_active`; (2) the bug fixes, one task each.

## Done when

- The suite is green and ROADMAP R1c is done.
