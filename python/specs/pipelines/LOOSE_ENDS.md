# Loose ends

Deliberately unfinished items: one line each, with the slice that should pick it up. Sweep this
list at each slice boundary.

- [R4?] `Pipeline.compute_units` rebuilds the views (and remaps every FPS) on each access. Cache
  them only if profiling shows a cost.
- [later] `tests/cli/cli_hardening_test.py` paces pty/FIFO input with short sleeps (about 0.5 s
  in total). Left alone by R7.
- [repo merge] The FPS structs have no version field, so drift between milk's libfps and pyMilk's
  vendored copy is only caught by the fpssynctest round-trip. The repo-merge round adds struct
  version checks (see the deferred R6 brief).
- [repo merge] `specs/006-r6-pymilk-boundary/draft_pymilk_boundary_test.py` still imports
  `milk.infra.pipeline_models`, moved to `milk.config.model` by R3a step 0.
- [R4+] No global scan of rundirs to find the configured pipelines (R3b D2).
- [future, spec only] Pipeline-local plans and tasks: Python files in a dedicated conf-set folder
  (akin to `scripts/`), copied to the rootdir as conf artifacts, then discovered, checked and
  registered into TASKS/PLANS at Pipeline load. Open points: load/sandbox policy, per-pipeline
  scoping of the registries, drift treatment of code. See the project doc `claude/r4-decisions.md`.
- [milk] `md->signal` is updated with non-atomic read-modify-writes by the conf loop
  (`fps_FPCONFloopstep.c:67,73`, `fps_checkparameter.c:540`) and by `CONFstop`, so a `confstop`
  sent during the first conf iteration can still be lost (~1 in 10 with no pause after deploy).
  pyMilk's `CONFrunning` now waits for `STATUS_CONF`, which narrows the window; the R4a tests pass.

## NIT

Readability nits from the R4b review, logged at `75e5f946c9`.

- [ ] `plans.py` `run_plan`: bare `PLANS[listed]` validates the config `plans` list by side effect
  only. Make it an explicit check, or drop it.
- [ ] `Pipeline.compute_units` / `Task.compute_units` map FPSs on read, so a debugger evaluating
  them has side effects (see the `[R4?]` entry above).
- [ ] `ComputeUnit.__str__` runs a milk binary and can raise; `__repr__` mutates `self.fps`.
- [ ] `ConfStart.success()` writes `failure_reason`: a predicate with side effects.
- [ ] `TASKS` is filled from `Task.__subclasses__()`: implicit, not greppable, direct subclasses
  only.
- [ ] Plans are written as `tuple(map(Step, "A B".split()))` with positional `"abort"`/`"any"`;
  use keyword arguments and a list of names.
- [ ] `Pipeline` mirrors `config.long_name/dataloads/systemlinkage/loop_number`, and `short_name`
  duplicates `config.name`. Tests pin the mirrors.
- [ ] `Task.compus_names` vs `Step.compus`: one concept, two names (`operate_test.py` pins
  `compus_names`).
- [ ] The milk version change is shown in two formats: `provision/cli.py` (`a -> b`) and
  `operate/cli.py` (`rootdir made with a, installed b`).
- [ ] Dead code: module `fpslist()` and `ComputeUnit.fpslist()`; `ComputeUnit.procinfo` is read
  only by test fixtures.
- [ ] `_subprocess_start(wrap_in_shell=True)` builds the `bash -c` string with `" ".join(argv)`;
  use `shlex.join`.
- [ ] Background `confstart`/`runstart` (no tmux) send output to pipes nothing reads; the child
  blocks once a pipe fills.
- [ ] `_subprocess_complete` calls `wait()` and then reads the pipes, which can deadlock on large
  output; use `communicate()`.
- [ ] `ComputeUnit.runstart` checks its arguments with `assert`, which `-O` strips.
- [ ] `Pipeline.from_folder` checks `is_rootdir`, then `__init__` checks it again.
- [ ] `milk-pipes nuke` raises `ClickException` inside `_user_errors()`; it passes through only
  because it is not a `MilkError`.
- [ ] `milk-show mlat`: overlay fits on the figure (e.g. the latency step / per-iteration
  latencies from the `#` lines of `hardwlatency.dat`).
