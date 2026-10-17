---
name: milk-adapter-pattern
description: >-
  Architecture and checklist for integrating external C/C++ packages (algorithms, reconstructors,
  compute engines) with the Milk framework (ImageStreamIO, libprocessinfo, libfps) on framework-dev.
  Activate whenever wrapping an external package as Milk compute units (fpsexec / CLI commands).
---

# Milk Adapter Pattern: Integrating External Packages

How to wrap an external package (`mypkg`) as `MILK_RUNNABLE` units. Runnable API, descriptor and CMake:
[`docs/arch/programmers_guide.md`](../../../docs/arch/programmers_guide.md) §3. Template:
`src/milk_module_example/`.

## 1. Architectural principles

1. **Clean core.** The algorithms and data structures of `mypkg` live in a pure library. Its headers
   never include Milk headers (`ImageStreamIO.h`, `processinfo.h`, `fps.h`, `milk_runnable.h`); it stays
   buildable and testable without Milk installed. Zero-copy interfaces must never let the core free
   ImageStreamIO shared-memory buffers (e.g. an `is_borrowed` flag).
2. **Isolated adapter layer.** Everything Milk-specific (parameter tables, runnables, stream bridges)
   lives in a dedicated adapter directory (`src/mypkg-fps/` or `adapters/milk/`) that consumes the
   public C API of `libmypkg` as an ordinary client.
3. **Lean executables.** fpsexecs link `milkcomputeunit`, `milkprocessinfo` and `ImageStreamIO`; never
   `CLIcore`, readline or terminal libraries.
4. **One parameter table, two entry points.** The X-macro parameter table of a runnable powers both
   the fpsexec (`EXPORT_FPSEXEC`) and the `milk-cli` command (`MILK_MODULE_RUNNABLES`).

## 2. Adapter layout

```text
src/mypkg-fps/
├── CMakeLists.txt     # milk_add_runnable_module() + milk_add_fpsexec_stubs(), links libmypkg
├── mypkg_fps_common.h # adapter context, stream bridges, prototypes
├── mypkg_fps_common.c # stream setup, casting, output update, reset
├── mypkg_fps_unit.c   # runnable(s): parameter table, compute function, EXPORT_FPSEXEC
└── mypkg_module.c     # module registry (MILK_MODULE_RUNNABLES)
```

## 3. Real-time loop

The compute function runs the canonical semaphore-synchronized loop between
`MILK_PROCINFO_INIT/LOOPSTART/END(self)`:

```c
MILK_PROCINFO_INIT(self)
MILK_PROCINFO_LOOPSTART(self)

mypkg_compute_frame(ctx, raw_pixels, &out_result);
processinfo_update_output_stream(processinfo, &out_img, &in_img);

MILK_PROCINFO_END(self)
```

Parameters: use direct variables in the table (`char name[FUNCTION_PARAMETER_STRMAXLEN]`, `int32_t` for
`FPTYPE_ONOFF`, `double` / `uint32_t` / `int64_t`), and `FPFLAG_DEFAULT_TRIGGER_STREAM` on the stream
driving the loop so `-loops` configures the semaphore trigger.

## 4. Pre-flight checklist

- [ ] No `malloc` in the loop: buffers and stream handles pre-allocated.
- [ ] Reader semaphore obtained through `processinfo_waitoninputstream_init()`, never a hardcoded index.
- [ ] Primary input stream has `FPFLAG_DEFAULT_TRIGGER_STREAM`.
- [ ] Streams closed and engine cleaned up on normal and signal exit paths.
- [ ] Status messages through `processinfo_WriteMessage_fmt()`, rate-throttled (not at kHz).
- [ ] Coding style per `code-style-guide.md` and `parameter-alignment.md`.
