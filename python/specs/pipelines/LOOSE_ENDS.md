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
  013-3 e2e (`linresp_e2e_test.py`, `NBexcl = 1`): measlinresp's `dtfr = latencyfr − 0.5 + 0.5·NBexcl` equals `latencyfr` there, so the response lands on the measured frame's edge
  (~60 µs margin on mlatsim, about one nanosleep overshoot); ~half the reads show the previous
  mode. The e2e uses `NBexcl = 2` (half a frame of margin) until this is settled.
  013-review, measured on that e2e: `NBexcl = 1` fails on 256 of 256 modes and passes with the
  latency raised by 0.5 frame; at `NBexcl = 2` it passes with the latency shifted by −0.2 to +2
  frames and fails at −0.4. So the e2e only catches a latency that is too short.
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
- [x] Add a dmcomb to the simulator conf (`mlatsim-conf`, example and test resource), with
  cacao's channel map (`cacao-setDMnolink`: `aol<N>_dmRM` → `dm<DD>disp02`, `aol<N>_dmC` →
  `disp03`). Then `aoloop-conf` `[systemlinkage.delaysim]` stops pointing `aol0_dmRM` at the same
  stream as `aol0_dmC` (013 stopgap), and `measlinresp` can run while `modes2dm` runs.

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
  the second run must skip `RunStart[acquWFS]`); `fake_compus` is hardcoded to delay0/delay1, and
  `linresp_test.py` has its own fake FPS (`_FPS`, a dict with values and a procinfo) beside it.
- [later] `Plan.compus_arg` is a special case of `Plan.params` (013-1): `[CU]...` could be a
  variadic `Param`; left as is.
- [later] `mkpokes hadamard` (013-2) keeps all `next_pow2(Nact)` slices and the first `Nact` columns;
  truncating the mode set to the DM (fewer modes than the power of two) is not done.
- [x] The loop conf carries hints of the DM resolution and beam: `[dmhint]`, `Pipeline.dm_mask()`
  (013-2); `mkpokes` no longer reads a stream.
- [later] `MakePokes.success()` checks type and DM size only: a `[dmhint]` beam change keeps the
  old cube until `mkpokes` runs with another type, or the file is removed.
- [later] `[dmhint]` overlaps the `measlinresp` unit's `streamin` stream shape; nothing checks they
  agree (013-3 `SetLinResp.can()` compares the cube to the stream).
- [later] Duplicate: `SetLinResp.can()` repeats `RunStart.can()`'s "FPS missing or conf process not
  running" check and message (013-3; `SetLoopOff` reuses `RunStart.can`, 017-1).
  `LoadData.success()` keeps its own "stream exists, then its shape" beside `stream_shape()`: it
  also checks the dtype (017-1).
- [ ] 017-1 machine check not done: a `loopON = false` written through pyMilk to a deployed, idle
  mfilt is still false once its run has started (Vincent, on a live `aoloop`).
- [later] `LoadCM` without a subset takes the `MVMextract` units in declaration order, WFS side
  first; a conf declaring `modes2dm` first swaps the two cubes (`axmode` would tell them apart).
- [ ] FPS string values hold 63 characters (`FUNCTION_PARAMETER_STRMAXLEN`) and pyMilk truncates
  longer writes silently; measlinresp gets `../calib/...` paths for that (013-3).
- [ ] The procinfo of a run is gone once it exits: `WaitLinResp` holds a handle from before the
  end to read the loop counter (013-3). A generic "run completed" check would need the same.
- [ ] `logshmim_save.c:65` calls `milkrt_RTPrio(writerRTprio)` from the log writer thread, but
  `milkrt_RTPrio` targets `getpid()` (the main thread), not the writer thread. Left in place when RT
  priority was removed from `processinfo_loopstart` and the other coremods.
- [ ] Decide which existing `*_milk_test.py` files are `e2e` (013-3 added the marker; none re-marked).
- [ ] `LoadData` creates its streams with `symcode=0`, where numpy is (size[0], size[1]): a FITS
  with NAXIS1 = X, NAXIS2 = Y becomes a stream of `size[0] = Y`, transposed against milk's own
  FITS loader on a non-square image (013-review; `SetLinResp.can()` had the same swap, fixed).
  The fixed check passed a scratch e2e on a 12 × 16 DM; no test of the suite has a non-square DM.
- [ ] mlat writes `cacaovars.bash` only at the end of a measurement and never removes it
  (`mlat.c` ~817): the file of an earlier run outlives a failed `moka mlat`, and `SetLinResp.can()`
  accepts that stale latency (013-review).
- [later] `moka measlinresp` edge cases, not handled (013-review): a measlinresp run still alive
  when the plan starts (`SetLinResp` deletes its in-progress file and rewrites its FPS, `RunStart`
  is skipped); `procinfo.loopcntMax = -1` (a stopped run counts as complete); a run shorter than
  the `RunStart` to `WaitLinResp` handover is reported as "0 of N cycles".
- [x] The pywfssim mask path (`.pupmask`) is not tested: no conf sets it. The first conf that sets
  a mask adds the test (015 §6.4). Both `mlatsim-conf` set an illumination mask and the e2e model
  takes it (015 review, B2).
- [later] Duplicate: `scripts/mkdm.py` and `scripts/mkpupmask.py` are copied, identical, in `tests/resources/mlatsim-conf` and
  `milk-examples/pipelines/mlatsim-conf` (015-2 changed both).
- [later] Duplicate: the test `mlatsim-conf` owns `acquWFS` and `mlat`, and the e2e appends its own
  `measlinresp`; these belong to `aoloop` (`fpsdetails.toml`). The e2e could deploy `mlatsim` and
  `aoloop` as `sample.sh` does (015-2).
- [x] acqWFS fills `aolN_wfsmask` with ones only when it creates it: a mask left in shm by an
  earlier, broken build (uninitialized `ii`, first two pixels 0) stayed. Fixed in cacao
  `29c744cd`; the tests remove their streams after each `make_mlatsim` (015-2).
- [ ] `linresp_e2e_test.py` is flaky on a desktop since the WFScamsim camera (014-3), with or
  without RT priority: about 1 run in 2 on the pywfs simulator (015-2; 1 of 3 in the review).
  Blocks of consecutive modes read the previous poke. measlinresp flushes the WFS semaphore once
  per cycle and matches frames to pokes by iteration count, with half a frame of margin; a
  per-poke flush (tried, not kept) still failed 1 run in 2. Tolerated until a server with
  isolated cores (Vincent, 2026-11-20).
- [x] 015-1's pywfssim adds `.wlum` and scales slopes by `Nx / wlum` (modamp in λ/D). The e2e
  model follows the C (Vincent, 2026-11-20); 015 §3 corrected by the review.
- [x] With `modamp = 1.0` a 0.1 poke of measlinresp is `u = 1.6`, well into the pywfssim
  saturation, and mlat's `OPDamp = 1.0` is `u = 16`. Now `ampl = 0.005` (`u = 0.08`) and
  `OPDamp = 0.05` (015 review, B2).
- [ ] The illumination of `mlatsim`'s pywfs (`pupmask` hook, radius 7) and the DM aperture of
  `aoloop`'s `[dmhint]` (radius 8) are two things and may differ. Poke modes of `mkpokes` that are
  constant over the illumination (a piston over the DM aperture) give no response.
- [ ] Duplicate: `pycacao/calib/compute_straight_CM.py` does not run (wrong call signature, no sort,
  no truncation, no 1/σ) and is superseded by `pycacao/calib/mkcm.py` (016-1). Repair or delete.
- [ ] `moka mkcm tikhonov -r 0` on a rank-deficient RM succeeds and writes gains of `1/σ` for the
  null modes (about 1e15): `-r 0` is taken as the operator's choice (016 review, A4).
