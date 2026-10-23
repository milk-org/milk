---
trigger: always_on
---

# Compute Unit (`MILK_RUNNABLE`) Conventions

Structure, descriptors and CMake: [`docs/arch/programmers_guide.md`](../../docs/arch/programmers_guide.md)
§3. Template: `src/milk_module_example/examplefunc_fps_cli_poc.c`. Working example:
`plugins/test-src/runnabletest`.

- Sources include `milk_runnable.h`, never `CLIcore.h` directly. CLI exposure is membership in the
  `MILK_MODULE_RUNNABLES` array of the module; never call `RegisterCLIcmd()` or touch `data.cmd[]`.
- A new `.c` file or `EXPORT_FPSEXEC` line is only picked up at configure time: re-run `cmake`.
- Every fpsexec supports `-h1`: `.app.description` must be a clear one-line summary
  (listed by `milk-fpsexec-list`).
- One runnable per mode, not one runnable with a mode-selecting parameter.
- String parameters: `char var[FUNCTION_PARAMETER_STRMAXLEN]`, passed as `var` (no `&`) in the X-macro.
- Input stream driving the loop: `FPFLAG_DEFAULT_TRIGGER_STREAM`; other required inputs:
  `FPFLAG_DEFAULT_INPUT_STREAM`.
- Writing a stream: set `outimg->md->write = 1` before touching pixels;
  `processinfo_update_output_stream()` clears it and posts the semaphores.
- Default image size of units processing images: 128x128.
- Check `docs/arch/dependency_graph.md` before adding a cross-module dependency.
