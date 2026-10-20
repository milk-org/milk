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

---

## dev

_No recent significant changes. Entries will be added here when `framework-dev` features are merged
into `dev`._
