# Loose ends

Deliberately unfinished items: one line each, with the slice that should pick it up. Sweep this
list at each slice boundary.

- [R4] B6: detect a busy tmux session in `StartConfProcesses.can()`. R1c only hints at it in the
  timeout error. Teardown owns tmux in R4.
- [R4?] `Pipeline.compute_units` rebuilds the views (and remaps every FPS) on each access. Cache
  them only if profiling shows a cost.
- [later] `tests/cli/cli_hardening_test.py` paces pty/FIFO input with short sleeps (about 0.5 s
  in total). Left alone by R7.
- [R4] A shared `milk` wait primitive (predicate, upper bound) for the RunStart, RunStop and
  ConfStop tasks and StartConfProcesses. `tests/conftestaux/polling.py` should then delegate
  to it.
- [repo merge] The FPS structs have no version field, so drift between milk's libfps and pyMilk's
  vendored copy is only caught by the fpssynctest round-trip. The repo-merge round adds struct
  version checks (see the deferred R6 brief).
- [R4] In semaphore trigger mode a run process only notices `runstop` when its wait times out
  (`procinfo.triggertimeout`, default 2 s), so stopping takes up to about 2 timeouts. RunStop's
  wait bound must allow for it. R7 sets the timeout to 0.1 s in `test_fps_lifecycle`.
- [R4] `tests/mains/pipeline_linsim_test.py` (ignored by pytest) still calls the removed
  `Pipeline.clone_to`; port it to `milk.provision.clone` when R4 rewrites the operate layer.
- [repo merge] `specs/006-r6-pymilk-boundary/draft_pymilk_boundary_test.py` still imports
  `milk.infra.pipeline_models`, moved to `milk.config.model` by R3a step 0.
- [R4+] No global scan of rundirs to find the configured pipelines (R3b D2).
- [R4] `tests/mains/pipeline_linsim_test.py` (not collected) still imports `InitialFolderSetup`, removed by R3b.
- [future, spec only] Pipeline-local plans and tasks: Python files in a dedicated conf-set folder
  (akin to `scripts/`), copied to the rootdir as conf artifacts, then discovered, checked and
  registered into TASKS/PLANS at Pipeline load. Open points: load/sandbox policy, per-pipeline
  scoping of the registries, drift treatment of code. See the project doc `claude/r4-decisions.md`.
