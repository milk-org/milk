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
  `moka.infra.pipeline_models`, moved to `moka.config.model` by R3a step 0.
- [R4+] No global scan of rundirs to find the configured pipelines (R3b D2).
- [milk] `md->signal` is updated with non-atomic read-modify-writes by the conf loop
  (`fps_FPCONFloopstep.c:67,73`, `fps_checkparameter.c:540`) and by `CONFstop`, so a `confstop`
  sent during the first conf iteration can still be lost (~1 in 10 with no pause after deploy).
  pyMilk's `CONFrunning` now waits for `STATUS_CONF`, which narrows the window; the R4a tests pass.
- [later] `mokainit clone -d/--override` reads like the `--set` overrides: rename it
  `--replace`.
- [future] `RunStartAO`/`RunStopAO`, requiring a state that says the AO calibration is loaded.
  Until then `moka runstart`/`runstop` start/stop run processes without that check.
- [later] `moka nuke` on a pipeline whose `moka_plans.py` does not load deletes the rootdir
  without running teardown.
- [later] `runstart = []` is accepted; `state()` then reports `running` for any deployed pipeline.
- [future] `milk-demopipeline`: a ready-to-run demo that deploys and runs a small pipeline end to
  end from Python (also in `docs/notes_for_future.md`).
- [next] Check the code against what the docs now state (pipelines_operate, pipelines_internals
  glossary): a hand edit in `rootdir/conf/` takes effect at the next deploy; from `cold` only
  `deploy` leaves it, `teardown`/`state`/`nuke` always accepted; `mixed` counts as consistent;
  `link` refused unless `deployed` or `configured`. Pin with tests or fix the docs.
- [next] `milk-examples/pipelines/pipelinebasic-conf/conf.toml` (copy of the test confdir) keeps
  the `shared`/`standalone` linkages marked "only for tests": drop them from the example.
- [docs] `docs/developer/tooling.md`: the `../multiversion.md` link breaks `mkdocs build --strict`; the "Contributing … TODO add link" placeholder can point to `writing_docs.md`.
- [milk] Define "hardware latency", then audit the value `mlat` pushes (`.out.latencyfr`,
  `CACAO_LATENCYHARDWFR`). On `mokashow mlat` it does not sit at the middle of the
  plateau/triangle. Today it is the dt of the highest ±½-frame sliding mean of the middle third
  of sorted values, times `framerateHz` (`mlat.c` ~717-800). That argmax is ill-defined on a flat
  top. dt is the midpoint of two frames' *read* times minus `dtoffset`, the time just after
  `mlat_setdm` returns. Compare with the best-kernel knee + ½ frame from `mokashow mlat`.
- [future] Pipeline kinds (a `kind` key and a kind-defaults layer under `includes`) were removed
  by the 012 audit as unused; bring them back with the first kind other than generic.
- [later] The provenance system (`provenance.toml`, tagfiles, `drift`/`resync` records) may be
  removable: decide whether its records are worth their code.
- [later] `ComputeUnit.confstart` returns no pid and has no completion guarantee or timeout.
- [later] `Pipeline.task_do`: no partial success; a task that half succeeds raises
  `TaskDidNotSucceedError`.
- [later] `Pipeline.any_run_active` reads the live FPSs without checking the state: decide whether
  it should refuse, or say, an inconsistent pipeline.
- [later] `systemlinkage` checks names under a single `MILK_SHM_DIR`; revisit if pipelines ever use
  several.
- [milk] No Doxyfile at the repo root: Doxygen generation is removed (`writing_docs.md` §9 says it
  may return); restoring it is a build/CI item.

## NIT

Readability nits from the R4b review, logged at `75e5f946c9`.

- [ ] `Pipeline.compute_units` / `Task.compute_units` map FPSs on read, so a debugger evaluating
  them has side effects (see the `[R4?]` entry above).
- [ ] `ComputeUnit.__str__` runs a milk binary and can raise; `__repr__` mutates `self.fps`.
- [ ] `ConfStart.success()` writes `failure_reason`: a predicate with side effects.
- [ ] `TASKS` is filled from `Task.__subclasses__()`: implicit, not greppable, direct subclasses
  only.
- [ ] Plans are written as `tuple(map(Step, "A B".split()))` with positional `"abort"`/`"any"`;
  use keyword arguments and a list of names.
- [x] `Pipeline` mirrors `config.long_name/dataloads/systemlinkage/loop_number` (removed by the
  012 audit). `short_name` still duplicates `config.name`: kept, 22 readers.
- [ ] `Task.compus_names` vs `Step.compus`: one concept, two names (`operate_test.py` pins
  `compus_names`).
- [ ] The moka version change is shown in two formats: `provision/cli.py` (`a -> b`) and
  `operate/cli.py` (`rootdir made with a, installed b`).
- [x] Dead code: module `fpslist()` and `ComputeUnit.fpslist()`; `ComputeUnit.procinfo` (012 audit).
- [ ] `_subprocess_start(wrap_in_shell=True)` builds the `bash -c` string with `" ".join(argv)`;
  use `shlex.join`.
- [ ] Background `confstart`/`runstart` (no tmux) send output to pipes nothing reads; the child
  blocks once a pipe fills.
- [ ] `_subprocess_complete` calls `wait()` and then reads the pipes, which can deadlock on large
  output; use `communicate()`.
- [ ] `ComputeUnit.runstart` checks its arguments with `assert`, which `-O` strips.
- [x] `Pipeline.from_folder` checks `is_rootdir`, then `__init__` checks it again: intended, as
  `__init__` falls back to `<cwd>-rootdir` (012 audit).
- [ ] `mokashow mlat`: overlay fits on the figure (e.g. the latency step / per-iteration
  latencies from the `#` lines of `hardwlatency.dat`).
- [later] A `moka_plans.py` that raises is executed twice per invocation by the fixed
  commands that read `pp.plans` (`plans`, `tasks`, `nuke`): once by the group, once by the command,
  because the failure is not cached.
- [later] No test runs the built-in `mlat` plan on a pipeline that has `acquWFS` and `mlat` (twice:
  the second run must skip `RunStart[acquWFS]`); `fake_compus` is hardcoded to delay0/delay1.
