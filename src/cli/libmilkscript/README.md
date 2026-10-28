# libmilkscript

Scripting engine, daemon server, IPC client, bash coprocess bridge, and
interactive CLI execution pipeline for the `milk` framework.

## Purpose

`libmilkscript` powers:

1. **The `milk>` interactive prompt and batch `.milk` scripts** — processing
   input through preprocessing, expansion, interception, argument checking,
   and command dispatch.
2. **The `milk-cli` Unix-domain socket daemon server (`server/`) and IPC client
   (`ipc/`)** — enabling persistent module state, stream pools, and high-speed
   shared-memory command logging across external processes and bash shells.
3. **The Bash integration engine (`bash/` and `milkbash`)** — translating
   `@{...}` inline syntax, providing loadable GNU Bash builtins, and
   synchronizing state with a persistent Bash coprocess.
4. **Shared-memory property & action access (`shm_prop/`)** — resolving
   `s.NAME.*`, `f.NAME.*`, `q.NAME.*`, and `p.NAME.*` reads and writes across
   both the daemon server and the script interpreter.

---

## Directory Organization

```
src/cli/libmilkscript/
├── shm_prop/        ← Shared-memory property & action accessor (stream/fps/proc/seq)
├── ipc/             ← Client socket protocol, server discovery, waitfor & SHM ring log
├── server/          ← Persistent Unix-domain socket server & RPC handlers
├── bash/            ← Bash coprocess lifecycle, @{...} translator & loadable bash builtins
├── calc/            ← Expression & image calculator, organized into 5 subdirectories:
│   ├── core/        ← Tokenizer, Pratt parser, slicing, scalar & image math builtins
│   ├── fuse/        ← Vectorized bytecode compiler, pre-stages, kernels & kept execution
│   ├── linalg/      ← BLAS & LAPACK matrix/vector operations & solvers
│   ├── prog/        ← Multi-statement programs, control flow, history & engine introspection
│   └── ext/         ← User C extensions (.c auto-compiler, .so loader & fuse bridge)
├── core/            ← Core CLI execution pipeline, catalog, module loader & help
├── units/           ← Standalone compute units (`milk-fpsexec-milk-calc` / `-milk-script`)
├── milkscript.h     ← Public embedding API header
├── milkscript_api.c ← Public embedding API implementation
├── milkscript_main.c ← `milk-script` standalone runner
├── milkscript_stubs.c ← Weak stubs for interactive-only functions (non-REPL builds)
├── script_includes.h ← Common includes for the script sources
└── standalone_dependencies.h ← Declarations shared with the no-CLI build
```

### 1. `shm_prop/` — Shared-Memory Property & Action Access

Single source of truth for reading and writing `stream`, `fps`, `seq`, and
`proc` shared-memory attributes, globbed entity discovery, and control actions:

- `milk_shm_prop.h` — Public API (`milk_shm_prop_get`, `milk_shm_prop_set`,
  `milk_shm_prop_exists`, `milk_shm_prop_find_or_open_stream`).
- `milk_shm_prop_internal.h` — Per-namespace handler declarations.
- `milk_shm_prop.c` — Top-level `type.name.prop` parser, top-level `/dev/shm`
  entity discovery (`s.*`, `f.*`, `p.*`), globbed property queries
  (`s.dm*.xsize`), and `--json` routing.
- `milk_shm_prop_stream.c` — `s.NAME.*` metadata queries, JSON serialization,
  FITS keywords, and stream existence checks.
- `milk_shm_prop_fps.c` — `f.NAME.*` parameter queries, JSON serialization,
  mutations, and existence/liveness checks.
- `milk_shm_prop_proc.c` — `p.NAME.*` (`PROCESSINFO`) inspection and control.
- `milk_entity_ns.h` — The one table of entity namespaces (`s`, `stream`, `l`, ...) and their kind;
  everything else (C code, `milk.bash` via `milkls --namespaces`) derives from it.
- `milk_shm_cache.h` / `milk_shm_cache.c` — Cache of mapped streams and FPS, checked against the
  file's inode on each lookup.

### 2. `ipc/` — Client IPC, Server Discovery & Shared-Memory Log

- `milk_client.h` — Public client API for `milk-call`, `milkbash`, and `milk-cli`.
- `milk_srvlog.h` — Shared-memory ring-buffer command/output log layout and API.
- `milk_ipc_common.h` / `milk_ipc_common.c` — Shared socket path formatting,
  PID liveness probing, and bounded `read_all` / `write_all` I/O helpers.
- `milk_proto.h` — Wire format: frame header, message types, `sendmsg`/`recvmsg` with fd passing.
- `milk_client_internal.h` — Internal declarations shared across client files.
- `milk_client_conn.c` — Socket connection, `MILK_SERVER` resolution, and
  auto-daemon spawning.
- `milk_client_ops.c` — Client RPC operations (`EVAL`, `CALL`, `GET`, `SET`,
  `WAITFOR`, `WAITANY`, `GRAB`, `PING`, `SHUTDOWN`) and streaming response reader.
- `milk_client_ext.c` — Client-side synchronization (`@waitfor`, `@waitany`, `@sleep`),
  stream capture (`@grab`), tab completion (`@complete`), sessions (`@session`,
  `@shutdown`, `@log`), and entity/keyword discovery (`@ls`).
- `milk_client_args.c` — Client-side argument validation and help formatting.
- `milk_waitfor.h` / `milk_waitfor.c` — Shared `@waitfor` target parser, process-state
  aliasing, and stream/property wait loops used by both client and server.
- `milk_call_main.c` — Standalone `milk-call` binary entry point (`call`, `batch`,
  `pipe`, `get`, `set`, `calc`, `test`, `exists`, `waitfor`, `waitany`, `grab`,
  `replay`, `complete`, `translate`, `session`, `log`, `shutdown`).
- `milk_srvlog_shm.c` — `milk_srvlog.*` POSIX SHM creation, attachment,
  high-resolution timestamped entry recording, and log querying (`--log`).
- `milk_srv_manager.c` — Server session discovery (`--list`), status dashboard
  (`--status`), and shutdown (`--kill`).

### 3. `server/` — Daemon Server & RPC Dispatch

- `milk_server.h` — Server lifecycle API (`milk_server_start`, `milk_server_stop`,
  `milk_server_run`).
- `milk_server_internal.h` — Internal session state and handler prototypes.
- `milk_server.c` — Accept loop, thread/blocking orchestration, and socket cleanup.
- `milk_server_session.c` — Session state, client connection handling, and stdio/cwd
  context switching.
- `milk_server_eval.c` — In-process calculator and command evaluation handlers.
- `milk_server_vars.c` — Server-side scalar variable and `IMGID` pool management.
- `milk_server_stream.c` — Stream synchronization, ring-buffer capture, and
  playback commands (`@waitfor`, `@waitany`, `@grab [--fits|--timestamps]`,
  `@replay [--fps|--count|--loop]`, `@grab --stdout`, `@replay --stdin`).
- `milk_server_io.c` — Command catalog dumping, `CLICMDARGDEF` type-aware
  argument tab-completion, and `@hc --sig` signature formatting.
- `milk_server_dispatch.c` — Request line parser and RPC verb handlers (`PING`,
  `SHUTDOWN`, `GET`, `SET`, `CALC`, `TEST`, `CALL`, including single-RPC
  `@pipe` execution).

### 4. `bash/` — Bash Coprocess, `@{...}` Translation & Loadable Builtins

- `milk_bash_engine.h` — Public API for Bash coprocess execution and `@{...}` translation.
- `milk_bash_engine_internal.h` — Shared string-buffer helpers (`sbuf_t`) and coprocess declarations.
- `milk_shell_lex.h` / `milk_shell_lex_internal.h` / `milk_shell_lex.c` / `milk_shell_lex_code.c` —
  Bash-aware scanner: context (code, quotes, here-doc, substitution, arithmetic) of every byte and
  whether input is incomplete.
- `milk_bash_translate.c` — Line-preserving translator rewriting `@{...}` and `\@` constructs into
  `milkget` / `milkcalc` substitutions while leaving quotes, comments, and here-docs intact.
- `milk_bash_cmdfile.c` — Private translated command-file directory for coprocess and `@source`.
- `milk_bash_coproc.c` — Persistent `bash` coprocess lifecycle, TTY handoff, pipe protocol, and
  `milk-cli --server` IPC servicing.
- `milk_bash_exec.c` — High-level command/file execution and interactive `@calc` REPL mode.
- `milk_bash_builtins.c` / `milk_bash_hooks.c` — Loadable GNU Bash builtins (`libmilkbash.so`) with
  zero-subshell scalar (`-v`), indexed array (`-a`), and associative array (`-A`) variable binding,
  plus `milktranslate`, `milkargs`, `milksleep`, `milkcloexec`, and `@<kw>` builtins.
- `milk_bash_defer.h` / `milk_bash_defer.c` — Scope-bound `@defer` cleanup stack for bash functions
  and scripts.

### 5. `calc/` — Expression & Image Calculator

Public headers sit at top-level `calc/` (`milk_calc_ext.h`, `cli_calc_parser.h`, `cli_calc_prog.h`,
`cli_calc_engines.h`, `cli_calc_scalar.h`, `cli_calc_tokenizer.h`, `cli_calc_internal.h`), while
implementation sources are grouped into 5 logical subdirectories:

- **`calc/core/`** — Lexer, Pratt parser, slicing, scalar & image math builtins:
  - `cli_calc_tokenizer.c` — Lexical analyzer for mathematical and image expressions.
  - `cli_calc_parser.c` / `cli_calc_parser_assign.c` / `cli_calc_parser_state.c` — Top-level entry
    points (`cli_calc_eval`, `cli_calc_eval_bash`), assignment & kept-statement dispatch, user
    definitions (`def f(...)`), and parser state.
  - `cli_calc_eval.c` — Precedence-climbing parser and primary atom evaluator.
  - `cli_calc_scalar.c` / `cli_calc_scalar_temporal.c` — Standalone scalar-only expression evaluator
    and temporal streaming estimators (`RunningStat`, `P2Quantile`) embedded in `libmilkbash.so`.
  - `cli_calc_slice.c` — Image pixel/slice indexing (`img[x,y]`, `img[x0:x1,y0:y1]`) and views.
  - `cli_calc_functions.c` / `cli_calc_functions_red_str.c` — Built-in scalar/image math functions,
    reductions, coordinate/PRNG generators, and string formatting.
  - `cli_calc_binops.c` — Element-wise and scalar-image binary operators, string `+`.
  - `cli_calc_complex.c` — Complex images (`cplx`, `real`, `imag`, `abs`, `arg`, `conj`, `expi`).
- **`calc/fuse/`** — Vectorized bytecode compiler, pre-stages, kernels & kept execution:
  - `cli_calc_fuse_internal.h` / `cli_calc_fuse_kernels.h` / `cli_calc_fuse_kernels_medfilt.h` —
    Fused bytecode structures and float32/double SIMD block kernels.
  - `cli_calc_fuse.c` / `cli_calc_fuse_views.c` — Bytecode compiler and image view/operand binding.
  - `cli_calc_fuse_ops_spatial.c` / `cli_calc_fuse_ops_axis.c` — Spatial (`shift`, `medfilt`, `lut`,
    `gather`, `interp2`) and axis/segment (`*_ax`, `seg_*`, `pixunmap`, `outer`) compilers.
  - `cli_calc_fuse_exec.c` / `cli_calc_fuse_exec_spatial.c` — OpenMP block execution and spatial
    block evaluators.
  - `cli_calc_fuse_prestage.c` / `cli_calc_fuse_prestage_seg.c` — Axis reduction, outer product, and
    CSR/segmented reduction pre-stage runners.
  - `cli_calc_fuse_reduce.c` — Fused single-pass full-image reductions and centroids.
  - `cli_calc_fuse_kept.c` / `cli_calc_fuse_merge.c` — Persistent kept-program execution, torn-read
    retry handling, and basic-block fusion across statements.
- **`calc/linalg/`** — BLAS & LAPACK matrix/vector operations & solvers:
  - `cli_calc_linalg_construct.c` / `cli_calc_linalg_blas.c` / `cli_calc_linalg_kept.c` — Tensor
    constructors (`transpose`, `reshape`, `vec`, `zeros`, `eye`, `diag`), BLAS Level 1–3 (`matmul`,
    `gemm`, `gemv`, `symv`, `syrk`, `ger`, `axpy`, `scal`), and kept BLAS compilation.
  - `cli_calc_lapack_triangular.c` / `cli_calc_lapack_solve.c` / `cli_calc_lapack_svd_qr.c` —
    Triangular (`trsv`, `trsm`), LU/Cholesky (`gesv`, `posv`, `solve`, `inv`), and QR/SVD (`gels`,
    `pinv`) linear solvers.
- **`calc/prog/`** — Multi-statement programs, control flow, history & engine introspection:
  - `cli_calc_prog_internal.h` / `cli_calc_prog.c` / `cli_calc_prog_parse.c` /
    `cli_calc_prog_history.c` — Multi-statement program compiler, `if`/`elif`/`else` and `for`
    control flow, `hist()`/`prev()` ring buffers, and basic-block merging.
  - `cli_calc_engines.c` — `@calc --list`, `--engines`, and `--tokens` bytecode inspection.
- **`calc/ext/`** — User C extensions:
  - `cli_calc_ext.c` — On-the-fly `.c` -> `.so` compiler (`milk-cc`), `.so` loader, and FPS
    parameter scanner.
  - `cli_calc_fuse_user_kern.c` / `cli_calc_fuse_user_op.c` — Tier 1 stateless block/stencil/image
    kernels and Tier 2 stateful multi-port operators (`MILK_EXT_OP`).

### 6. `core/` — Core CLI Execution Pipeline, Modules & Help

- `CLIcore_UI_execute.h` / `CLIcore_UI_execute.c` — Top-level `CLI_execute_line()`
  pipeline.
- `CLIcore_UI_execute_debug.c` — Execution tracing and timing diagnostics.
- `CLIcore_checkargs.h` / `CLIcore_checkargs.c` — CLI argument type validation.
- `CLIcore_checkargs_fps.c` — FPS-backed CLI argument binding.
- `CLIcore_fps_function.{c,h}`, `CLIcore_fps_query.{c,h}`, `CLIcore_fps_getFPSargs.h` — FPS
  CLI command helpers.
- `CLIcore_modules.h` / `CLIcore_modules.c` — Dynamic shared-object module loader.
- `CLIcore_help.h` / `CLIcore_help.c` — General help display.
- `CLIcore_help_command.c` — Per-command help formatter.
- `CLIcore_help_topics.c` — Topic-based help pages.
- `CLIcore_datainit.h` / `CLIcore_datainit.c` — Global `data` state initialization.
- `CLIcore_memory.h` / `CLIcore_memory.c` — Memory and stream table cleanup.
- `CLIcore_setSHMdir.h` / `CLIcore_setSHMdir.c` — Shared-memory directory setup.
- `CLIcore_signals.h` / `CLIcore_signals.c` — Signal handling and fault recovery.
- `CLIcore_utils.h` — CLI utility and argument macros.
- `CLIcore_inspect.c` — Inspection and process-control commands (`@ls`, `procctl`, `procwait`,
  `procstat`, `dpdigits`, JSON introspection).
- `CLIcore_script.h` / `CLIcore_vars.c` — CLI variable table and its sync with the environment.
- `milk_lexicon.h` / `milk_lexicon.c` — The one list of `@` primitives, prompt-only commands and
  bash words, for completion, highlighting and the `milk-call` symlinks.
- `milk_argval.h` / `milk_argval.c` — Argument value parsing and validation helpers.
- `milk_catalog.h` / `milk_catalog.c` / `milk_catalog_build.c` — Install-time and runtime
  module command catalog.
- `milk_cmd_args.{c,h}` / `milk_cmd_index.{c,h}` — Command table indexing and argument metadata.
- `milk_renames.h` / `milk_renames.c` — Legacy command rename mapping.
- `milk_session.h` / `milk_session.c` — CLIcore session initialization and module autoloading.

### 7. `units/` — Scripted Compute Units

- `milk_unit.h` / `milk_unit.c` — Common FPS parameter discovery (`.unit.*`), stream binding,
  and loop harness for code-free compute units.
- `milk_unit_calc.c` — `milk-fpsexec-milk-calc` (`milk.calc`): executes pre-compiled calculator
  programs per frame.
- `milk_unit_script.c` — `milk-fpsexec-milk-script` (`milk.script`): executes bash/milk script
  snippets or files (`.unit.cmd` / `.unit.file`) per frame in a persistent bash coprocess.

## Build Tier

CLI tier — requires `USE_CLI=ON`.
