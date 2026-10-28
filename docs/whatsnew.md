---
tags: [changelog, releases]
---

# What's New

Significant features and upgrades, newest first. Minor bugfixes are omitted — see the
[commit log](https://github.com/milk-org/milk/commits/framework-dev) for the full history.

**Filter by tag** — use your browser's find (Ctrl+F) with any `#tag` below: `#cli` `#performance`
`#streams` `#fps` `#build` `#docs` `#api` `#refactor`

---

## framework-dev

### 2026-10

- **2026-10-26** — Sessions and calculator, after an audit of their docs. An image as the condition
  of `@test`, `@assert`, `@waitfor` or `if` is an error with a hint (`imin(...)`, `imax(...)`), no
  longer always true. An image that cannot be assigned (`u, v = a, b` onto a stream of another size,
  or onto a variable) is an error, not "Cannot change name" and status 0. Parameters bound at a call
  site stay out of a unit's FPS, as documented. `@session save` keeps string variables, `@config
  sessionlog` logs again, `@session new` names the MILK_SHM_DIR of a session that runs from another
  one, and session names hold at most 63 characters. `milk-cli -h` lists the options milk-cli takes.
  The guides and the help now describe calc mode, result types, the `@assert` options, every
  function, the limits, and what child scripts and sessions share `#cli` `#docs`
- **2026-10-26** — An `@{…}` that holds only an integer is git's and is passed to bash as written, so
  `git checkout @{-1}` works in a milk script. `@get p.<proc>` and `@ls` read processinfo from
  MILK_PROC_DIR, where processes write it: they read MILK_SHM_DIR when MILK_PROC_DIR was `/dev/shm`,
  under `/milk/shm` or unset. A failed `@set` prints its error once, not twice `#cli`
- **2026-10-26** — The calculator keeps up to four temporary images (256 MiB) when a statement ends
  and gives them to the next statements of the same shape and type, so a statement run again in a
  loop or a unit's frames allocates nothing: `zC = zA * conj(zB)` on 512×512 complex images takes
  205 µs instead of 310 µs, and on 2100×2100 images 6.1 ms instead of 20.5 ms (it faulted in a fresh
  35 MB temporary each time). `mem.listim` lists the kept images as `_tmpcalcpool<k>` `#performance`
- **2026-10-26** — milk-cli, milk-script and compute units keep the memory they free for reuse: blocks
  up to 32 MiB come from the heap, which is not trimmed while less than 256 MiB of it is free. A
  buffer freed and allocated again in a loop is no longer faulted in each time: a complex calculator
  statement ran up to 2.2× slower depending on what had been allocated before. Malloc settings in
  `GLIBC_TUNABLES` take precedence (Performance Tuning §4.3) `#performance`
- **2026-10-26** — Unit properties: `f.X.ok`, `f.X.errors` and `f.X.msg` give the last parameter
  check; trigger modes are read and set by name (`@set f.X.procinfo.triggermode=delay`,
  `p.X.trigmode`); `p.X.dt` and `p.X.dtmax` give the loop interval in µs. `@get --json p.X`, `@ls -l`
  and `@ls --json` show the real loop rate (they showed 0), and `@waitfor p.X` returns once the
  process has exited cleanly instead of waiting out its timeout `#cli` `#fps`
- **2026-10-26** — `milk-demopipeline` is a milk script: each unit starts once the one before it has
  written a frame (`@waitfor`), with no fixed sleep, so its nine units start in under a second
  instead of 16 s. It is installed with the CLI, and the scripting guide shows the pattern `#cli`
  `#fps`
- **2026-10-26** — An ON/OFF parameter that a unit declares ON is ON in the FPS that `fpsinit`
  creates; it was OFF. `streamave` (`.comp.ave`) now writes its average, and `MVMextractModes`
  normalizes its modes by default (`.option.MODENORM`, off in practice since 2026-05-30) `#fps`
- **2026-10-26** — `@get --watch [T] <spec>...` shows values every T s until Ctrl-C: in place at a
  terminal, one line per refresh in a log. The scripting guide shows a parameter sweep as a loop
  with `@set --restore` and `@calc -v` `#cli`
- **2026-10-26** — Shared sessions run one command at a time: a command that another shell sends
  while a long one runs waits for it, and no longer runs in a copy whose changes were lost;
  `milkcall --nowait` fails at once with `busy: <command> for N s`. History is per session: each
  line typed at a milk-cli prompt is appended to `~/.milk_history` (no exit-time rewrite drops
  other sessions' lines), the log records its session, and `@history` shows the current session's
  lines from any front-end (`--global`: all). milk-script's session is labelled as such `#cli`
- **2026-10-26** — `milk-cli --session NAME` runs the commands of its REPL in session NAME, started
  if it does not run; the prompt reads `milk:NAME`, `@session leave` returns to milk-cli's own
  session, and `@history` and the REPL settings of `@config` stay milk-cli's. `@session save FILE`
  writes a session's settings, variables, command defaults and local images (FITS, in `FILE.d/`) as
  a milk script, and `@session restore FILE` runs it; the units the session started are listed, not
  started `#cli`
- **2026-10-26** — Session owners and lifetimes: the shell that runs `@session new` owns the session,
  and `@session ls` lists it ORPHANED once that shell has exited. `@session new NAME --idle T` ends a
  session once it has had no command for T and no shell uses it. `@on` watchers end with the shell
  that started them, even a killed one, and no longer keep its own session alive. The file in which a
  shell records the session it uses is named after its PID and start time, so a later shell with the
  same PID no longer inherits that choice `#cli`
- **2026-10-26** — Sessions: `@session` replaces `@server` (see
  [Sessions](guide/sessions.md)). `@session new NAME` starts a named session and fails if
  it runs; `@session use NAME` joins one and fails if it does not; `leave`, `who`, `ls` (with the
  units each session started), `end [--units]` and `prune` complete it. Only `new` starts a named
  session: a shell whose session was ended gets a hint instead of a new, empty one. milk.bash takes
  `--session NAME` (`-s` and `--server` are gone), `@log` takes `--session`, and `@session end` from
  another shell ends a milk-cli session. `@server` and its verbs fail with the `@session` form `#cli`
- **2026-10-26** — milk-cli servers start like units: `@server start` (and a shell's private server)
  reports why a server failed to start, with the end of its output, which now goes to a file next to
  its log (`@server status` names it); a crash message stays there. `@server stop` sends the
  shutdown command, then signals a server that ignores it. A server run by systemd (`Type=notify`)
  says when it is ready. `@server start --tmux` is removed, with a hint `#cli`
- **2026-10-26** — Units as systemd services (optional): `share/milk/systemd/milk-unit@.service` runs
  any unit through `milk-fps-run NAME`; `systemctl start` returns once the loop runs, the status
  line shows the loop rate, and `WatchdogSec=` restarts a loop that stops iterating. Once systemd
  runs a unit, `milk-fps-runstart`/`runstop`, `@set` and the TUIs go through `systemctl` until
  `milk-fps-run --release NAME`. The output stays in `run.out`; the environment file is in
  `EnvironmentFile=` format; `.procinfo.cset` is skipped under systemd (`AllowedCPUs=`). See
  [systemd Services](operations/systemd.md) `#fps` `#cli`
- **2026-10-25** — Units without tmux: an fpsexec binary and its FPS form a unit, whose conf and run
  processes start in the background, detached from whoever starts them:
  - `runstart` and `confstart` return once the loop runs (status 0), with the worker's messages
    when it fails (1), or 124 when it is not ready in time (`--timeout S`); `run` (or `--fg`) keeps
    the loop in the terminal. Stops ask the loop to exit and wake it, then signal it.
  - Output in `NAME.fps.d/run.out` and `conf.out`, rotated (`milk-fps-log [-f] NAME`); `run.info`
    records how each run started and ended; the environment is the one of the shell that ran
    `fpsinit` (`milk-fps-info --env NAME`). Processinfo is on by default (`--no-procinfo`).
  - One state for every front-end: `stopped` `starting` `running` `stopping` `exited` `failed`
    (`milk-fps-list`, `@get f.NAME.status`, `@waitfor f.NAME running`, fpsCTRL, milk-CTRL).
  - fpsCTRL and milk-CTRL start and stop units without freezing and show their output (`Ctrl+a`,
    the OUTPUT sub-tab); `@set f.NAME.run=start|stop`, `f.NAME.run.out`.
  - SIGTERM ends a loop; a crash records `CRASHED` in processinfo and ends the process.
  - Removed, with a hint: `-tmux`, the tmux API, milk-cli's `_CONFSTART_` … `_TMUXSTOP_`
    arguments, fpsCTRL `-f`/`-s`, `T` and `Ctrl+t`, `milk-fpsinit`, `milk-fpsmkcmd`,
    `milk-fpslist-addentry`, `tmuxsessionname`, `tmuxkillall`; `p.X.tmux` is `p.X.unit`. An FPS
    made by milk-cli is a parameter set, not a unit.
  - `milk-streamFITSlog` runs its logger as a unit (`milk-fpsexec-mem-streamFITSlog`). See
    [FPS](guide/fps.md) and [Unit Lifecycle](arch/unit_lifecycle.md) `#fps` `#cli` `#refactor`
- **2026-10-25** — The FPS sequencer is removed: `milk-seq`, `milk-seq-help`, the `milksequencer`
  module (`seq.*`, `USE_SEQ`), the `q.`/`fpsseq.` entity namespace and fpsCTRL's scheduler screen.
  Queued workflows are milk scripts; the old commands and `q.` lookups fail with a hint. See
  [From milk-seq](guide/scripting.md#from-milk-seq) `#cli` `#fps` `#refactor`
- **2026-10-25** — User-compiled C extensions for `milk.calc` / `@calc` (`milk_calc_ext.h`) and
  single-command `MILK_RUNNABLE` compilation (`milk-cc`):
  - **Tier 1a (`MILK_CALC_REG_BLK`, `MILK_CALC_REG_BLK2`)**: 1D SIMD block kernels fused directly
    into the L1 cache tile loop (`FUSE_BLK = 1024`) with typed parameter structs (`MILK_CALC_PARAMS`).
  - **Tier 1b (`MILK_CALC_REG_STENCIL`)**: 2D $(2R+1)\times(2R+1)$ row-strip stencil kernels
    ($R \in \{1, 2, 3\}$) receiving pre-clamped row pointers of length $N + 2R$.
  - **Tier 1c (`MILK_CALC_REG_IMG`)**: Single-function whole-image kernels (`IMAGE *out`,
    `const IMAGE *in`, `const struct par *P`) that compose inside fused expressions.
  - **Tier 2 (`MILK_CALC_OP_DECLARE`, `MILK_CALC_REG_OP`)**: Stateful multi-input / multi-output
    operators (`init` / `exec` / `free`) with named image and scalar ports, custom output dimensions
    (`milk_op_set_out_shape`), CONF-time re-initialization on shape or non-`WRITERUN` parameter
    changes, and automatic FPS exposure of scalar output ports (`FPFLAG_DEFAULT_OUTPUT`).
  - **Auto-compiler & FPS binding (`-l <file.c|file.so>`)**: Both `@calc` and
    `milk-fpsexec-milk-calc` accept `-l` / `--lib` (stored in `.unit.libs`), auto-compiling `.c`
    sources into `${TMPDIR:-/tmp}/milk-calc-cache-<uid>/ext_<hash>.so`. Unbound C parameters are
    exposed in FPS under `.<func>.<param>` (or `.<label>.<param>` for `func:label(...)`), while
    call-site bound arguments (`th = 3 * rms`) stay internal to the `.calc` program.
  - **Tier 3 (`milk-cc`)**: Single-command compiler driver that builds standalone `MILK_RUNNABLE`
    `.c` files into `<prefix>-fpsexec-<id>` executables (or Tier 1/2 `.c` files into `.so`
    plugins) without CMake boilerplate. See
    [Bringing Your Own Compiled C Code](guide/calc_c_extensions.md)
    `#cli` `#fps` `#api` `#performance`

- **2026-10-23** — Synchronised input reads and a round of small fixes:
  - Every stream a kept statement reads is covered by a torn-frame guard: a pass is re-run when an
    input's counter moved under it or a writer started a frame during it (at most three passes),
    and a writer caught mid-frame is given 100 µs to finish; `_torn` counts both in a unit.
  - `imin`, `imax` and the axis forms skip NaN pixels (NaN only when every pixel is NaN) in every
    engine; before, a NaN first pixel gave NaN for small images and the finite extreme for large.
  - A kept scalar statement types its variable as the interactive calculator does (`n = icount(a) + 1`
    stays `int`; comparisons are `int`; `/` and functions give `float`), and checks the identity of
    the variables it reads by name, not only by slot.
  - `@grab`, `@waitany` and a second `@waitfor` on the same stream take semaphore slots of their own
    instead of sharing the pinned waiter's; a `@waitfor` that finds all 64 stream mappings pinned
    reports it instead of "absent"; `s.<im>.stats` adds a `torn` flag and the attempts back off 20 µs.
  - Piped `milk-cli` input logs every line to the history (one flush per batch), and an open
    construct holds a batch back only in bash mode (`@calc` lines are not bash).
  - `randn` images no longer depend on the block size they were generated with; rows wider than
    2^31 pixels no longer overflow the coordinate kernels.
  - `milk-cli` and `milk-script` set `MILK_INSTALLDIR` from their own path when it is unset, so the
    modules they load, the `milk.bash` the engine sources and the units it launches come from the
    install the binary belongs to, whatever `PATH` holds (the tests' `milk-cli` helpers use the
    install under test too)
    `#cli` `#fps` `#performance`

- **2026-10-23** — The unit program language: conditionals, frame metadata, history, outputs:
  - `if <scalar>: <statement>` (the condition kept as a scalar program), `skip` (end the tick),
    `post(name)` (post an output this tick without writing it).
  - `_cnt0`, `_ts`, `_missed`: the trigger frame's counter, write time and skipped frames, next to
    `_n`, `_t`, `_dt`.
  - `hist(s, N)` declares the ring stream `s_hist` of the last N frames of `s`, filled after each
    tick; `prev(s[, k])` reads frame n-k of it (`diff = cam - prev(cam)`, `bg = imean(cam_hist, 2)`).
  - `zeros(nx[, ny[, nz]])` creates an output with an explicit shape (cast for the type)
    `#cli` `#fps`

- **2026-10-23** — Fused views: slices, broadcasting and axis reductions in one pass:
  - A slice operand (`a[0:63,:]`, `cube[:,:,k]`, `a[::2,-*]`) is read through a view of its image,
    no copy; an operand with extent 1 on an axis is broadcast (`a - row - col`, `cube - dark`); a
    sliced left-hand side (`out[0:63,:] = …`, `hist[:,:,_n % 8] = cam`) is written through a view;
    the axis forms of `itot imean imin imax istd ivar norm asum` are computed before the pixel loop
    and read as operands (`out = cam - imean(cam, 1)`). All of these are kept in `milk.calc`
    units; bounds with variables are resolved at each run.
  - Measured on a self-triggered unit at 128²: a sliced destination 12.8 → 2.3 µs per tick, a
    row-mean subtraction 75.6 → 7.4 µs. Results match the per-operator path bit for bit.
  - Unchanged: binned slices, `imedian`/`perc`, complex images and one-pixel slices (scalars) take
    the per-operator path; the parity test forces it with a bin-1 slice (`im[::1b]`)
    `#cli` `#performance`

- **2026-10-23** — Compute-unit performance round:
  - `.procinfo.NBthread` defaults to 0: the OpenMP team policy sizes threads from the image and
    the cores the unit may run on (a positive value caps it); OpenBLAS's pool is sized the same way
    at every BLAS call (`matmul`, `ger`, `syrk`, `symv`), so it no longer spawns one thread per
    logical CPU.
  - `-loopd` delay triggers tick on an absolute `clock_nanosleep` deadline with 1 µs timer slack:
    periods of a few tens of µs instead of a 60 µs floor.
  - Per-tick costs: FPS bindings are copied only when their `value_cnt` moved, a stream parameter
    is looked up only when it changes, the output post caches `getpid()`, and the kept-program
    checks that scan the image table run on the coarse-clock cadence, not every tick.
  - `p.<unit>.loopfreq`, `dt_median_ns`, `exectime` are computed from the timing ring by the reader
    (they read 0 before), plus `dt_max_ns`, `exec_max` and `jitter`.
  - `python/tests/perf/unit_tick_bench.py` measures self-triggered units; `test_unit_tick_ceilings`
    asserts the loop floor, a 128² kept statement and a 1024² statement under the policy
    `#cli` `#fps` `#performance`

- **2026-10-23** — Engine review fixes for compute units, the calculator and the client:
  - **Threads and affinity:** `milk_omp_team_size()` no longer changes the calling thread's CPU
    affinity (a unit launched under `taskset` keeps its mask), `OMP_NUM_THREADS` lowers the team
    but never widens it past the physical cores, forked `--async`/`@replay` children never build an
    OpenMP team, and `.procinfo.taskset` defaults to empty instead of `1-127`.
  - **Units:** no more `sched_setscheduler` attempt at RT priority 0 on every start (`RTprio`
    defaults to -1); an `@set` landing while a tick computes is no longer overwritten by the loop's
    write-back; a trigger stream re-created by its owner is remapped at the next timeout and a dead
    owner is reported (`p.<unit>.timeouts`, `p.<unit>.hz`); an output written by several statements
    is posted once per frame.
  - **Calculator:** `im /= image` reports a zero divisor again (the fused path had started to write
    `inf`); a rejected scalar divisor no longer posts an unchanged frame; a cached statement that
    fails is reported once.
  - **Client and server:** `@grab` re-checks its cube every frame whatever the input; `@waitfor`
    pins and unpins under the server command lock; a waiter's semaphore slot is released as soon as
    the wait ends, so idle shells do not exhaust a stream's slots; the shm-cache and stats mutexes
    survive `fork()`; milk-cli caps its stdin drain at 1 MiB so a fast producer's lines still run.
  - **Build and tests:** `-mno-recip` on the whole `COREMOD_arith` and `libmilkscript` targets (the
    shared inline kernels agree between entry paths); `perf_ceilings_test.py` picks an allowed CPU,
    uses a relative float32 parity tolerance, and `MILK_PERF_TESTS=0` skips the timing tests;
    `engine_fixes_test.py` guards each fix `#cli` `#fps` `#performance`

- **2026-10-23** — Calculator, linear algebra, and `COREMOD_arith` kernel consolidation:
  - **Shared arithmetic & reduction kernels (`COREMOD_arith`):** unified complex arithmetic
    (`im_complex.c`), coordinate and counter-based hash PRNG generators (`image_coord.{c,h}`),
    general-datatype/masked/axis reductions (`image_stats.{c,h}`), and matrix/BLAS kernels
    (`image_linalg.{c,h}`) across the calculator, `image_gen`, `image_basic`, and `linalgebra`,
    retiring ~4,900 lines of duplicate legacy loops. `mkrandomim` now uses the parallel counter
    hash PRNG (Gaussian sum-of-6-uniforms approximation bounded near $\pm 4.2\sigma$).
  - **Fused calculator & math library:** element-wise fused programs run in native `float32` when
    all inputs/outputs are `FLOAT` (doubling SIMD width) and `double` otherwise; reductions
    (`itot`, `imean`, `istd`/`irms`, `ivar`, `imin`, `imax`, `icount`, `asum`, `norm`, `dot`)
    fuse map+reduce in L1 blocks via Chan's recurrence without temporary images; `where()`,
    image `?:`, `&&`, `||`, coordinates (`ix`, `iy`, `ir`, `itheta`), and counter RNG (`urand`,
    `randn`) fuse in one pass; 25 scalar/image math functions (`cbrt`, `hypot`, `exp2`, `expm1`,
    `log2`, `log10`, `log1p`, `sinh`, `cosh`, `tanh`, `erf`, `erfc`, `clamp`, explicit casts
    `f32`..`i64`).
  - **Linear algebra & solvers:** BLAS level 1–3 (`dot`, `asum`, `nrm2`, `axpy`, `scal`, `gemv`,
    `symv`, `ger`, `matmul`/`gemm`, `syrk`, `trsv`, `trsm`) with algebraic `C = a*matmul(A,B)+b*C`
    lowering to a single `gemm` call, plus `gesv`, `posv`, `gels`, `solve`, `inv`, and SVD `pinv`
    with LAPACKE fast paths; `USE_OPENBLAS` and `USE_LAPACKE` now default to `ON` when detected via
    `pkg-config`.
  - **Compute units, IPC & front-end:** `milk-fpsexec-milk-calc` (`milk.calc`) and
    `milk-fpsexec-milk-script` (`milk.script`) standalone units with `.unit.*` parameters,
    `name=value` CLI arguments, and kept compilation for `/=`, scalar reductions, and scalar
    variable expressions; 32-entry LRU compiled statement cache for `@calc`; size-aware physical-core
    OpenMP team policy (`milk_omp_team_size`, `MILK_CALC_THREADS`); pinned stream mappings and
    reused semaphore slots in `@waitfor`; SIMD-accelerated `s.<im>.*` pixel stats with torn-frame
    guard and per-`(ino, cnt0)` caching; lock-free `@grab` frame copying with `.kw.DROPPED` (and
    `.k.<KEY>` keyword shorthand); batched non-TTY piped stdin in `milk-cli` (58.8 → ~2 µs/line)
    `#cli` `#performance` `#api` `#refactor`

- **2026-10-21** — The `cacao` modules (`AOloopControl*`, `computeCalib`, `pyramidWFStools`) migrated to `MILK_RUNNABLE`: each module has one registry and one header, the `cacao-fpsexec-*` executables are generated from `EXPORT_FPSEXEC`, and the `modalstatsTUI` sources are parked in `AOloopControl/tui/` `#api` `#build` `#refactor`

- **2026-10-20** — `milk-streamCTRL` is now `milk-CTRL` opened on its streams view (an installed
  link; `-h` maps the old keys), and the old streamCTRL TUI is removed. F3 gains an inspector with
  INFO, SEMS (readers and writers of each semaphore), TRACE (writer chain with timing), KEYS and
  HOLDERS (processes mapping the file, from `/proc`, no `fuser`) pages. Streams: up to 10000, mapped
  progressively; files that are not usable streams and dangling links are listed with the reason;
  symbolic links show their target; creator PID, holder count and owner sorts; Hz averaged over 1 s;
  COUNT flashes when it moves; per-stage scan timing. Libraries: libprocessinfo shares the state,
  control and trigger-mode names, computes timing medians for any reader (EXEC and DUTY no longer
  need milk-procCTRL-scan) and settles dead registry slots; libmilkdata lists streams, which fixes
  `readshmimall` and `shmimpurge` finding none in milk-cli; `processinfo_exec_end` measures across
  second boundaries ([guide](guide/milk-CTRL.md)) `#cli` `#streams` `#api` `#docs`

- **2026-10-20** — `milk-CTRL` efficiency, links and presentation round: the scan takes about
  1.6 ms instead of 5.7 ms on 2000 streams / 200 FPS and stops while paused; tmux is asked only
  while an FPS panel shows; perf counters close when RESOURCES is hidden; dead PIDs are not probed again; resident memory
  with an empty directory is down from 17 to 5 MB.
  One typed graph per scan feeds every view; CONNECTIONS rows show rate, reader state, compute
  time, missed triggers and the FPS, and mark where a pipeline stalls; stream DETAILS shows per-hop
  latency; F7 lists a loop's FPS. At 80 columns the bars keep what matters, list columns are
  dropped by priority and scroll back with the column cursor, numbers are SI-scaled, empty lists
  say why, the status line shows the keys of the current context, and states read the same
  everywhere (`ERR`, `STAL`) ([guide](guide/milk-CTRL.md)) `#cli` `#performance` `#docs`

- **2026-10-20** — `milk-CTRL` restructured so a column, view, sub-tab, key or option is one table
  entry; process rows line up with their header again, F6 shows the sub-tab it selects, F7 clicks
  select loops, ENTER edits a DETAILS parameter, F1 opens help, and the help, `-h` and status hints
  come from one key table. New options: `--view NAME`, a positional regex `PATTERN`, `--interval SEC`,
  `--log FILE` and `--debug`, `--shm-dir` ([guide](guide/milk-CTRL.md)) `#cli` `#refactor` `#docs`

- **2026-10-19** — `milk-CTRL` safety and correctness round: viewing never writes shared memory;
  actions hit the highlighted entry (identity, not row number) and check it is still the same
  stream/FPS/process before acting; `y/N` before SIGKILL, erase, cleanup and batch actions; FPS
  batch marks (`b`, `a`); no shell for inspect, tmux or snapshots; terminal restored on every
  exit; colour depth honoured (256/16/monochrome); stall detection, loop health and the
  CONNECTIONS tree fixed. Legacy TUIs: streamCTRL, procCTRL and fpsCTRL restore the terminal
  and drop shell calls; the fpsCTRL FIFO is retired (`-f`/`-s` are ignored; use `milk-fps-set`)
  `#cli` `#streams` `#fps`

- **2026-10-18** — `milk-termview`: menus, status bar, slow-link output, flow control over ssh,
  stream following, Pause `#cli` `#streams` `#performance` ([guide](guide/termview.md))

- **2026-10-16** — Fixes: extra commands of multi-command files (`mem.cp`, `mem.mv`, `extractim`,
  `imsetcnt1`, ...) read their arguments again, and images created by commands (`loadfits`,
  `breakcube`, `imexpand`, ...) are in the image table again; both broke in the 2026-03 IMGID
  migration `#cli` `#api`

- **2026-10-16** — Calculator kernels: element-wise image expressions run in one pass with no
  intermediate images (6-operator chain on 1024×1024: 10.9 → 0.2 ms; `a += 1`: 1.6 → 0.2 ms);
  in-place and slice updates are typed and OpenMP-parallel, slice copies go row by row; complex
  images work with `cplx real imag abs arg conj` and `+ - * /`. Fixes: `q = im` now creates `q`,
  mismatched image sizes are an error instead of aborting milk-cli, install paths over 100
  characters load modules `#cli` `#performance`

- **2026-10-16** — Calculator: `&&`, `||` and a scalar `?:` skip the operand they don't need; `!` is
  logical not; `+` concatenates strings; new `istd` (`irms` kept as an alias), `ivar`, `icount`,
  `iargmax`, `iargmin`, and axis reductions such as `imean(cube, 2)`; integer images read and write
  pixels as integers; string variables evaluate to their value; `>` on images is now strict (it
  was `>=`). Operator and function tables in [Scripting](guide/scripting.md#calculator) `#cli`

- **2026-10-16** — Scripting engine rework (`feat/bash-engine`): bash-aware translator, one
  command rule everywhere (`module.command` or `@name`), in-process `source`, loud errors and
  real exit status, Ctrl-C that keeps the shell, threaded `@waitfor`/`@grab`, and per-call cost
  down to a few µs (`@calc` 92→4 µs, `@get` ~6 µs). See the migration table below and
  [Scripting](guide/scripting.md) `#cli` `#performance` `#docs`

  | Old form                                     | Now                                                       |
  | -------------------------------------------- | --------------------------------------------------------- |
  | `listim`, `mk2Dim im 8 8`                    | `mem.listim`, `@mk2Dim im 8 8` (bare names need `@`)      |
  | `?`, `m?`, `cmd? x`, `synhl off`             | `@?`, `@m?`, `@cmd? x`, `@config synhl off`                      |
  | `a = 2 + 3`, `im = im * 2`                   | `@ a = 2 + 3`, `@calc "im = im * 2"`                      |
  | `s.cam.xsize` as a command                   | `@get s.cam.xsize`                                        |
  | `${s.cam.xsize}`, `${@s.cam.xsize}`          | `@s.cam.xsize` (also inside `"..."`)                      |
  | `waitfor_stream cam 10`, `waitfor_fps ...`   | `@waitfor s.cam 10`, `@waitfor f.loop.status=RUN`        |
  | `wait_any ...`                               | `@waitany s.a s.b -t 1`                                   |
  | `on_update cam { ... }`, `trap 'c' STREAM:x` | `@on s.cam 'cmd'`                                         |
  | `watch 1000 mem.listim`                      | `while :; do mem.listim; milksleep 1; done`               |
  | `vars`, `savescript`, `!cmd`, `!!`           | `declare -p`; removed; run `cmd` directly                 |
  | `loc.` namespace                             | `l.` or `local.`                                          |
  | `strlen(foo)` with an unknown `foo`          | an error now; quote strings: `strlen("foo")`              |

- **2026-10-15** — The macro-based "V2 8-section" compute unit framework is removed
  (`FPS_MAIN_STANDALONE_V2`, `INSERT_STD_PROCINFO_COMPUTEFUNC_*`, `CLIADDCMD_*`,
  `add_milk_standalone` / `add_cacao_standalone*`, `MILK_MODULE`/`init_module_CLI`): every compute
  unit is a `MILK_RUNNABLE`, built by `milk_add_runnable_module()` `#api` `#build` `#refactor`
- **2026-10-15** — Enhanced `milk.bash` & `milk-cli`: zero-subshell `-v`/`-a`/`-A` variable & array
  bindings, `/dev/shm` entity globbing & `--json` export, globbed/multi-assignment `@set`,
  `@snapshot`/`@diff`/`@restore` state management, `@assert` expression & stream/loop health
  assertions, `@config prompt` zero-subshell live telemetry prompt, `CLICMDARGDEF` type-aware tab
  completion & `@hc --sig`, throttled/multi-target `@on` watchers, single-RPC `@pipe` & `@pipe`
  execution, and `@grab --fits/--timestamps` + `@replay` stream playback `#cli` `#streams` `#fps`
  `#performance`
- **2026-10-14** — First-class GNU Bash integration (`source milk.bash`, `libmilkbash.so` loadable
  builtins, in-process `milk-cli` bash co-process, `@`-sigil command namespace, `@server`
  multi-server lifecycle/tmux management, `@log` `/dev/shm` command & output ring buffer,
  `@calc` / `@test` / `@waitfor` / `@grab` / `@defer`, and `@T<type>:` stream datatype
  modifiers) `#cli` `#streams` `#fps`
- **2026-10-14** — `milk_module_example` migrated to `MILK_RUNNABLE`: the template units `imsum1`,
  `imsum2`, `streamupdate`, `streamprocess` and `fpscli` are listed in one module registry, and the
  template executables are generated from `EXPORT_FPSEXEC` `#api` `#build` `#docs`
- **2026-10-14** — `linalgebra` migrated to `MILK_RUNNABLE`: its CLI commands are listed in one module registry, mandated (CUDA, MAGMA, LAPACKE) units are weak, and the 11 executables are generated from `EXPORT_FPSEXEC` `#api` `#build`
- **2026-10-11** — New `libcomputeunit` engine library holds the FPS runtime (standalone `main`, tmux,
  conf/run, bindings, standalone macros); `libfps` is reduced to the FPS struct and no longer
  depends on `USE_CLI`. `fpsCTRL` and `procCTRL` move to `src/tui/`, and the CLI wrapper now lives in
  `libmilkscript`. Include `computeunit.h` instead of `fps.h` for the runtime API
  `#build` `#api` `#refactor` `#fps`
- **2026-10-09** — `MILK_RUNNABLE` descriptors: several CLI commands / fpsexecs per file, sources compiled
  once, generated `main()` stubs via `EXPORT_FPSEXEC` (test module `runnabletest`) `#api` `#build`
- **2026-10-07** — Unified module `CMakeLists.txt` template, `milk_link_module()` for static-LTO standalones, and
  `INSERT_STD_CLIREGISTERFUNC_NAMED` for CLI registration `#build` `#api`
- **2026-10-06** — Reorganized `docs/` into topic subfolders
  (`quickstart/`, `guide/`, `developer/`, `arch/`, `operations/`,
  `maintenance/`) matching the site navigation `#docs`

### 2026-03

- **2026-03-29** — Complete CLI/Scripting layer decoupling; introduction of `libmilkscript` engine
  and standalone `milk-script` interpreter `#cli` `#refactor`
- **2026-03-26** — `@seq.NAME.prop` variable expansion for sequencer state in CLI scripts `#cli`
  `#fps` ([#183](https://github.com/milk-org/milk/issues/183))
- **2026-03-25** — `milk-seq` standalone sequencer, fpsCTRL and milk-cli integration `#fps` `#cli`
- **2026-03-25** — Module dependency system with `MODULE_DEPS` macro `#api` `#build`
  ([PR #169](https://github.com/milk-org/milk/pull/169))
- **2026-03-24** — Remove legacy `s>` stream modifier; consolidate to `@S:` syntax `#streams`
  `#refactor` ([PR #168](https://github.com/milk-org/milk/pull/168))
- **2026-03-24** — Split CLIcore monolithic sources into smaller modules `#cli` `#refactor`
  ([PR #165](https://github.com/milk-org/milk/pull/165))
- **2026-03-24** — CLI bitwise ops, string functions, processinfo integration `#cli`
  ([PR #164](https://github.com/milk-org/milk/pull/164))
- **2026-03-24** — Compound assignments, ternary operator, numeric literals, `until` loops `#cli`
  ([PR #163](https://github.com/milk-org/milk/pull/163))
- **2026-03-24** — Dual prompt/command history with type-tagged log `#cli`
  ([PR #161](https://github.com/milk-org/milk/pull/161))
- **2026-03-23** — FITSIO-like stream slicing syntax (`im[0:19,10:29]`) `#streams` `#cli`
  ([PR #157](https://github.com/milk-org/milk/pull/157))
- **2026-03-23** — CLI pipe operator `|>`, `on_fpschange`, `listim` glob, error context `#cli`
  ([PR #153](https://github.com/milk-org/milk/pull/153))
- **2026-03-23** — Notify user of adopted default arguments `#cli`
  ([PR #152](https://github.com/milk-org/milk/pull/152))
- **2026-03-23** — Advanced CLI introspection, regex `=~`, array splats, dynamic PS1 `#cli`
  ([PR #151](https://github.com/milk-org/milk/pull/151))
- **2026-03-22** — Native shell expansions and arithmetic evaluation in CLI scripts `#cli`
  ([PR #150](https://github.com/milk-org/milk/pull/150))
- **2026-03-22** — Advanced script template and argparse improvements `#cli`
  ([PR #147](https://github.com/milk-org/milk/pull/147))

### 2026-02

- **2026-02** — Remove function-pointer overhead from pixel loops `#performance`
  ([PR #146](https://github.com/milk-org/milk/pull/146))
- **2026-02** — Dynamic FIFO handling for CLI `#cli`
  ([PR #144](https://github.com/milk-org/milk/pull/144))
- **2026-02** — Typed CLI variables & subprocess shell pipe parsing `#cli`
  ([PR #141](https://github.com/milk-org/milk/pull/141))
- **2026-02** — Add `-c` (command) flag to milk-cli `#cli`
  ([PR #139](https://github.com/milk-org/milk/pull/139))
- **2026-02** — Bi-directional environment variable sync and subshell assignments `#cli`
  ([PR #138](https://github.com/milk-org/milk/pull/138))
- **2026-02** — Machine-readable JSON and TSV output for CLI `#cli` `#api`
  ([PR #136](https://github.com/milk-org/milk/pull/136))
- **2026-02** — CLI `wordexp()` simplification and smart fallback `#cli`
  ([PR #130](https://github.com/milk-org/milk/pull/130))
- **2026-02** — Pipe to shell commands (grep, sort, wc) `#cli`
  ([PR #126](https://github.com/milk-org/milk/pull/126))
- **2026-02** — `ghistory` / `lhistory` with time range, glob, session highlight `#cli`
  ([PR #124](https://github.com/milk-org/milk/pull/124))
- **2026-02** — Break help into topics; `help-<topic>` commands `#cli`
  ([PR #121](https://github.com/milk-org/milk/pull/121))
- **2026-02** — CLI scripting variables, aliases, bookmarks, and history `#cli`
  ([PR #116](https://github.com/milk-org/milk/pull/116))
- **2026-02** — Session-aware history commands `#cli`
  ([PR #111](https://github.com/milk-org/milk/pull/111))
- **2026-02** — Advanced scripting: env vars, conditional chaining, history, continuation `#cli`
  ([PR #107](https://github.com/milk-org/milk/pull/107),
  [PR #109](https://github.com/milk-org/milk/pull/109))

### 2026-01

- **2026-01** — Standalone `fpsexec` executables for image_gen (mkpolygon, mkdisk, mkspdisk) `#fps`
  `#build` ([PR #103](https://github.com/milk-org/milk/pull/103),
  [PR #104](https://github.com/milk-org/milk/pull/104))
- **2026-01** — Fully static musl fpsexec build support `#build` `#performance`
  ([PR #99](https://github.com/milk-org/milk/pull/99))
- **2026-01** — `USE_STATIC_LTO` cross-module LTO for fpsexec standalones `#build` `#performance`
  ([PR #55](https://github.com/milk-org/milk/pull/55))
- **2026-01** — Context-aware tab completion for streams and FPS `#cli`
  ([PR #92](https://github.com/milk-org/milk/pull/92))
- **2026-01** — Runtime perf sweeps 1–10: SIMD alignment, restrict, omp simd, BLAS MVM, powf
  specialization, `__builtin_memcpy`, UINT16 fast-path `#performance`
  ([PR #53](https://github.com/milk-org/milk/pull/53),
  [PR #91](https://github.com/milk-org/milk/pull/91))
- **2026-01** — PGO per-executable profile isolation `#build` `#performance`
- **2026-01** — GCC optimization infrastructure (`milk_compiler.h`) + codebase-wide annotations
  `#performance` `#api`
- **2026-01** — Reduce transitive includes in `CLIcore.h` and `fps.h` `#build` `#refactor`
  ([PR #54](https://github.com/milk-org/milk/pull/54))
- **2026-01** — MkDocs Material documentation site `#docs`
  ([PR #57](https://github.com/milk-org/milk/pull/57))
- **2026-01** — Codebase-wide IMGID API migration (streams, FFT, linalgebra, image_basic, …) `#api`
  `#refactor`
- **2026-01** — Gen 4 V2 CLI registration for all modules `#cli` `#refactor`
- **2026-01** — FPS parameter tab-completion + argument hints `#cli` `#fps`
  ([PR #71](https://github.com/milk-org/milk/pull/71))
