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
   `@cmd` and `@entity.prop` syntax and synchronizing variables with a persistent
   Bash coprocess.
4. **Shared-memory property & action access (`shm_prop/`)** — resolving
   `stream.NAME.*`, `fps.NAME.*`, `seq.NAME.*`, and `proc.NAME.*` reads and
   writes across both the daemon server and the script interpreter.

---

## Directory Organization

```
src/cli/libmilkscript/
├── shm_prop/        ← Shared-memory property & action accessor (stream/fps/proc/seq)
├── ipc/             ← Client socket protocol, server discovery & SHM ring log
├── server/          ← Persistent Unix-domain socket server & RPC handlers
├── bash/            ← Bash coprocess lifecycle & @cmd / @shm_prop preprocessor
├── calc/            ← Mathematical expression & image-slice evaluator
├── core/            ← Core CLI execution pipeline, module loader & help
├── milkscript.h     ← Public embedding API header
├── milkscript_api.c ← Public embedding API implementation
├── milkscript_main.c ← `milk-script` standalone runner
├── milkscript_stubs.c ← Weak stubs for interactive-only functions (non-REPL builds)
├── script_includes.h ← Common includes for the script sources
└── standalone_dependencies.{c,h} ← Declarations shared with the no-CLI build
```

### 1. `shm_prop/` — Shared-Memory Property & Action Access

Single source of truth for reading and writing `stream`, `fps`, `seq`, and
`proc` shared-memory attributes, globbed entity discovery, and control actions:

- `milk_shm_prop.h` — Public API (`milk_shm_prop_get`, `milk_shm_prop_set`,
  `milk_shm_prop_exists`, `milk_shm_prop_find_or_open_stream`).
- `milk_shm_prop_internal.h` — Per-namespace handler declarations.
- `milk_shm_prop.c` — Top-level `type.name.prop` parser, top-level `/dev/shm`
  entity discovery (`s.*`, `f.*`, `p.*`, `q.*`), globbed property queries
  (`s.dm*.xsize`), and `--json` routing.
- `milk_shm_prop_stream.c` — `stream.NAME.*` metadata queries, JSON serialization,
  FITS keywords, and stream existence checks.
- `milk_shm_prop_fps.c` — `fps.NAME.*` parameter queries, JSON serialization,
  mutations, and existence/liveness checks.
- `milk_shm_prop_proc_seq.c` — `proc.NAME.*` (`PROCESSINFO`) and `seq.NAME.*`
  (`fpsseq`) inspection and control.
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
- `milk_bash_builtins.c` — Loadable GNU Bash builtins (`libmilkbash.so`) with
  zero-subshell scalar (`-v`), indexed array (`-a`), and associative array
  (`-A`) variable binding; also `milktranslate`, `milksleep` and `milkcloexec`.
- `milk_call_main.c` — Standalone `milk-call` binary entry point (`call`, `batch`,
  `pipe`, `get`, `set`, `calc`, `test`, `exists`, `waitfor`, `waitany`, `grab`,
  `replay`, `complete`, `translate`, `server`, `log`, `shutdown`).
- `milk_srvlog_shm.c` — `milk_srvlog.*` POSIX SHM creation, attachment,
  high-resolution timestamped entry recording, and log querying (`--log`).
- `milk_srv_manager.c` — Server session discovery (`--list`), status dashboard
  (`--status`), and shutdown (`--kill`).

### 3. `server/` — Daemon Server & RPC Dispatch

- `milk_server.h` — Server lifecycle API (`milk_server_start`, `milk_server_stop`,
  `milk_server_run`).
- `milk_server_internal.h` — Internal session state and handler prototypes.
- `milk_server.c` — Accept loop, thread/blocking orchestration, and socket cleanup.
- `milk_server_vars.c` — Server-side scalar variable and `IMGID` pool management.
- `milk_server_stream.c` — Stream synchronization, ring-buffer capture, and
  playback commands (`@waitfor`, `@waitany`, `@grab [--fits|--timestamps]`,
  `@replay [--fps|--count|--loop]`, `@streamcat`, `@streamwrite`).
- `milk_server_io.c` — Command catalog dumping, `CLICMDARGDEF` type-aware
  argument tab-completion, and `@hc --sig` signature formatting.
- `milk_server_dispatch.c` — Request line parser and RPC verb handlers (`PING`,
  `SHUTDOWN`, `GET`, `SET`, `CALC`, `TEST`, `CALL`, including single-RPC
  `@batch` and `@pipe` execution).

### 4. `bash/` — Bash Coprocess & Syntax Translation

- `milk_bash_engine.h` — Public API for Bash coprocess execution and translation.
- `milk_bash_engine_internal.h` — Shared internal declarations.
- `milk_shell_lex.h` / `milk_shell_lex.c` — Bash-aware scanner: the context (code, quotes, here-doc,
  substitution, ...) of every byte, and whether input is incomplete.
- `milk_bash_translate.c` — Rewrites `@cmd`, `module.command` and `@entity.prop` forms into calls
  of the milk builtins, using the scanner so quotes, comments and here-docs are left alone.
- `milk_bash_coproc.c` — Persistent `bash` coprocess lifecycle, variable/CWD
  synchronization, and `milk-cli --server` IPC servicing.

### 5. `calc/` — Expression & Image Calculator

- `cli_calc_tokenizer.h` / `cli_calc_tokenizer.c` — Lexical analyzer for
  mathematical and image expressions.
- `cli_calc_parser.h` / `cli_calc_parser.c` — Entry point (`cli_calc_eval`) and
  expression dispatcher.
- `cli_calc_internal.h` — Internal AST/value types and evaluator prototypes.
- `cli_calc_eval.c` — Recursive-descent precedence parser and primary atom
  evaluator.
- `cli_calc_slice.c` — Image pixel/slice indexing (`img[x,y]`, `img[x0:x1,y0:y1]`)
  and subarray extraction/assignment.
- `cli_calc_functions.c` — Built-in scalar and image mathematical functions
  (`sin`, `cos`, `fft`, `stats`, reductions, etc.).
- `cli_calc_binops.c` — Element-wise and scalar-image binary operators, string `+`.
- `cli_calc_fuse.c` — One-pass evaluation of element-wise image expressions, and the typed,
  OpenMP-parallel in-place updates (`im op= x`) used by whole-image and slice assignment.
- `cli_calc_complex.c` — Complex images: `cplx real imag abs arg conj`, `+ - * /`, in-place writes.

### 6. `core/` — Core CLI Execution Pipeline, Modules & Help

- `CLIcore_UI_execute.h` / `CLIcore_UI_execute.c` — Top-level `CLI_execute_line()`
  pipeline.
- `CLIcore_UI_execute_debug.c` — Execution tracing and timing diagnostics.
- `CLIcore_checkargs.h` / `CLIcore_checkargs.c` — CLI argument type validation.
- `CLIcore_checkargs_fps.c` — FPS-backed CLI argument binding.
- `CLIcore_modules.h` / `CLIcore_modules.c` — Dynamic shared-object module loader.
- `CLIcore_help.h` / `CLIcore_help.c` — General help display.
- `CLIcore_help_command.c` — Per-command help formatter.
- `CLIcore_help_topics.c` — Topic-based help pages.
- `CLIcore_datainit.h` / `CLIcore_datainit.c` — Global `data` state initialization.
- `CLIcore_memory.h` / `CLIcore_memory.c` — Memory and stream table cleanup.
- `CLIcore_setSHMdir.h` / `CLIcore_setSHMdir.c` — Shared-memory directory setup.
- `CLIcore_signals.h` / `CLIcore_signals.c` — Signal handling and fault recovery.
- `CLIcore_utils.h` / `CLIcore_utils.c` — CLI utility and argument helpers.
- `CLIcore_inspect.c` — Inspection and process-control commands (`@ls`, `procctl`, `procwait`,
  `procstat`, `dpdigits`, JSON introspection).
- `CLIcore_script.h` / `CLIcore_vars.c` — CLI variable table and its sync with the environment.
- `milk_lexicon.h` / `milk_lexicon.c` — The one list of `@` primitives, prompt-only commands and
  bash words, for completion, highlighting and the `milk-call` symlinks.

## Build Tier

CLI tier — requires `USE_CLI=ON`.
