# libmilkscript

Scripting, daemon server, IPC client, bash co-process bridge, and
interactive CLI execution pipeline for the `milk` framework.

## Purpose

`libmilkscript` powers:

1. **The `milk>` interactive prompt and batch `.milk` scripts** — processing
   input through preprocessing, expansion, interception, argument checking,
   and command dispatch.
2. **The `milk-cli` Unix-domain socket daemon server (`server/`) and IPC client
   (`ipc/`)** — enabling persistent module state, stream pools, and high-speed
   shared-memory command logging across external processes and bash shells.
3. **The bash integration (`bash/` and `milkbash`)** — translating
   `@{...}` inline syntax, providing loadable GNU Bash builtins, and
   synchronizing state with a persistent Bash coprocess.

Shared-memory property & action access (`s.NAME.*`, `l.NAME.*`, `f.NAME.*`,
`p.NAME.*`) lives in `src/prop/` (`libmilkprop`), and the expression/image
calculator, `milk_calc_ext.h` C extension SDK, and `milk-fpsexec-milk-calc`
compute unit live in `src/calc/` (`libmilkcalc`).

---

## Directory Organization

```
src/cli/libmilkscript/
├── ipc/             ← Client socket protocol, server discovery, waitfor & SHM ring log
├── server/          ← Persistent Unix-domain socket server & RPC handlers
├── bash/            ← Bash coprocess lifecycle, @{...} translator & loadable bash builtins
├── core/            ← Core CLI execution pipeline, catalog, module loader & help
├── units/           ← Standalone script compute unit (`milk-fpsexec-milk-script`)
├── milkscript.h     ← Public embedding API header
├── milkscript_api.c ← Public embedding API implementation
├── milkscript_main.c ← `milk-script` standalone runner
├── milkscript_stubs.c ← Weak stubs for interactive-only functions (non-REPL builds)
├── script_includes.h ← Common includes for the script sources
└── standalone_dependencies.h ← Declarations shared with the no-CLI build
```

### 1. `ipc/` — Client IPC, Server Discovery & Shared-Memory Log

- `milk_client.h` — Public client API for `milk-call`, `milkbash`, and `milk-cli`.
- `milk_srvlog.h` — Shared-memory ring-buffer command/output log layout and API.
- `milk_ipc_common.h` / `milk_ipc_common.c` — Shared socket path formatting,
  PID liveness probing, and bounded `read_all` / `write_all` I/O helpers.
- `milk_proto.h` — Wire format: frame header, message types, `sendmsg`/`recvmsg` with fd passing.
- `milk_client_internal.h` — Internal declarations shared across client files.
- `milk_client_conn.c` — Socket connection (one kept per process, reused by its requests),
  `MILK_SERVER` resolution, and auto-daemon spawning.
- `milk_client_ops.c` — Client RPC operations (`CALL`, `GET`, `SET`, `CALC`, `TEST`, `PING`,
  `SHUTDOWN`; the message types are in `milk_proto.h`) and streaming response reader.
- `milk_client_stream.c` — Client-side synchronization (`@waitfor`, `@waitany`, `@sleep`),
  stream capture (`@grab`), tab completion (`milkcomplete`), sessions (`@session`, `@log`,
  server shutdown), and entity/keyword discovery (`@ls`, `milkls --keywords`).
- `milk_client_args.c` — Client-side argument validation and help formatting.
- `milk_waitfor.h` / `milk_waitfor.c` — Shared `@waitfor` target parser, process-state
  aliasing, and stream/property wait loops used by both client and server.
- `milk_call_main.c` — Standalone `milk-call` binary entry point (`call`, `complete`,
  `translate`, `shutdown`, and the callable `@` keywords of the lexicon).
- `milk_srvlog.c` — `milk_srvlog.*` POSIX SHM creation, attachment,
  high-resolution timestamped entry recording, and log querying (`--log`).
- `milk_sessions.c` — Server session discovery (`--list`), status dashboard
  (`--status`), and shutdown (`--kill`).

### 2. `server/` — Daemon Server & RPC Dispatch

- `milk_server.h` — Server entry points (`milk_server_run`, `milk_server_create_listener`,
  `milk_server_accept_and_handle`) and session hooks.
- `milk_server_internal.h` — Internal session state and handler prototypes.
- `milk_server.c` — Accept loop and kept connections, client connection handling and stdio/cwd
  redirection, the output capture thread, thread/blocking orchestration,
  `PING`/`SHUTDOWN`/`GET`/`COMPLETE`/`EXISTS`, and socket cleanup.
- `milk_server_admin.c` — `@module`, `@config`, the subcommands of `@history`, and
  `@session save`.
- `milk_server_eval.c` — In-process calculator and command evaluation handlers.
- `milk_server_vars.c` — Sync of bash variables with the session's calculator variables.
- `milk_server_stream.c` — Stream synchronization, ring-buffer capture, and
  playback commands (`@waitfor`, `@waitany`, `@grab [--fits|--timestamps]`,
  `@replay [--fps|--count|--loop]`, `@grab --stdout`, `@replay --stdin`).
- `milk_server_complete.c` — Command catalog dumping, `CLICMDARGDEF` type-aware
  argument tab-completion, and `@hc --sig` signature formatting.
- `milk_server_dispatch.c` — `CALL` and `SET` handlers: command lookup, argument tokens, and
  running a command under the request guards (`@calc`, `@test`, `@assert` and `@pipe` are
  evaluated in `milk_server_eval.c`).

### 3. `bash/` — Bash Coprocess, `@{...}` Translation & Loadable Builtins

- `milk_bash.h` — Public API for Bash coprocess execution and `@{...}` translation.
- `milk_bash_internal.h` — Shared string-buffer helpers (`sbuf_t`) and coprocess
  declarations.
- `milk_shell_lex.h` / `milk_shell_lex_internal.h` / `milk_shell_lex.c` / `milk_shell_lex_code.c` —
  Bash-aware scanner: context (code, quotes, quotes, here-doc, substitution, arithmetic) of every
  byte and whether input is incomplete.
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

### 4. `core/` — Core CLI Execution Pipeline, Modules & Help

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
- `milk_keywords.h` / `milk_keywords.c` — The one list of `@` keywords (synopsis, summary, help)
  and bash words: help, usage errors, the bash builtins' help, completion, highlighting and the
  `milk-call` links.
- `milk_catalog.h` / `milk_catalog.c` / `milk_catalog_build.c` — Install-time and runtime
  module command catalog.
- `milk_cmd_args.{c,h}` / `milk_cmd_index.{c,h}` — Command table indexing and argument metadata.
- `milk_renames.h` / `milk_renames.c` — Legacy command rename mapping.
- `milk_builtin_module.h` / `milk_builtin_module.c` — CLIcore session initialization and module autoloading.

### 5. `units/` — Scripted Compute Units

- `milk_unit_script.c` — `milk-fpsexec-milk-script` (`milk.script`): executes bash/milk script
  snippets or files (`.unit.cmd` / `.unit.file`) per frame in a persistent bash coprocess
  (using `fps_unit_options` from libcomputeunit and `fps_args` from libfps, as `milk-fpsexec-milk-calc` does).

## Build Tier

CLI tier — requires `USE_CLI=ON`.
