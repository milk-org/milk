---
name: milk-adapter-pattern
description: >-
  Standard architecture and boilerplate templates for integrating external C/C++ packages
  with the Milk framework (ImageStreamIO, libprocessinfo, libfps) on framework-dev. Activate
  whenever integrating any algorithm, reconstructor, or compute engine with Milk, or when
  creating an FPS standalone binary or Milk CLI module for an external project.
---

# Milk Adapter Pattern: Integrating External Packages

This skill provides the architectural specification, guidelines, and code templates for wrapping
any external scientific, algorithmic, or computational package with the **Milk** framework
(`framework-dev` branch).

---

## 1. Architectural Principles

When integrating an external package (`mypkg`) with Milk, follow these architectural rules:

1. **Strict Decoupling (Clean Core)**:
   - The core algorithms and data structures of `mypkg` must reside in a pure library (Level 2).
   - Core headers must **never** include Milk headers (`ImageStreamIO.h`, `processinfo.h`, `fps.h`,
     `CLIcore.h`).
   - The package must remain buildable, testable, and distributable without Milk installed.
   - Memory ownership: Ensure zero-copy interfaces never allow core cleanup to free ImageStreamIO
     shared-memory buffers (e.g. use an `is_borrowed` flag or avoid calling `free()` on caller data).

2. **Isolated Adapter Layer (Level 3)**:
   - All Milk-specific bindings, parameter definitions, and shared-memory stream interactions are
     confined to a dedicated adapter directory (e.g. `src/mypkg-fps/` or `adapters/milk/`).
   - The adapter consumes the public C API of `libmypkg` as an ordinary client.

3. **Lean Standalone FPS Daemons (`libmilkfpsStandalone`)**:
   - Standalone binaries (`milk-fpsexec-<funcname>`) must link `milkfpsStandalone`,
     `milkprocessinfo`, and `ImageStreamIO`.
   - Never link `CLIcore`, GNU Readline, or terminal libraries into standalone FPS binaries.

4. **Dual-Mode Delivery**:
   - A single X-Macro parameter definition (`FPS_PARAMS(X)`) powers both:
     - A standalone daemon executable (`milk-fpsexec-<funcname>`)
     - A shared Milk CLI plugin (`libmilk<funcname>.so`) for interactive `milk-cli` sessions.

---

## 2. Standard 4-File Adapter Blueprint

In the external package's adapter directory, implement the following files:

```text
src/mypkg-fps/
├── CMakeLists.txt         # Links libmypkg + milkfpsStandalone + ImageStreamIO
├── mypkg_fps_params.h     # Single source of truth: FPS_PARAMS(X) macro
├── mypkg_fps_common.h     # Adapter context, stream bridges, prototypes
├── mypkg_fps_common.c     # Stream setup, casting, output update, reset
├── mypkg_fps_main.c       # Standalone binary: FPS_MAIN_STANDALONE_V2_CONFCHECK
└── mypkg_module.c         # Milk CLI shared library: MILK_MODULE & CLIADDCMD
```

Templates for each file are located in the [`templates/`](./templates/) subdirectory.

---

## 3. Real-Time Processing Loop Pattern

Every Milk stream processor follows this canonical zero-polling, semaphore-synchronized loop:

```c
/* Set up trigger stream name and procinfo flag in CLIcmddata */
strncpy(CLIcmddata.cmdsettings->triggerstreamname, fps_in_name,
        sizeof(CLIcmddata.cmdsettings->triggerstreamname) - 1);
CLIcmddata.cmdsettings->flags |= CLICMDFLAG_PROCINFO;

/* INSERT_STD_PROCINFO_COMPUTEFUNC_START handles loop setup, signals, semaphores */
INSERT_STD_PROCINFO_COMPUTEFUNC_START
{
    /* 1. Ingest slice pointer from ImageStreamIO */
    const void *raw_pixels = (const char *)in_img.array.raw + (slice_offset * typesize);

    /* 2. Invoke core algorithm */
    mypkg_compute_frame(ctx, raw_pixels, &out_result);

    /* 3. Update output stream headers, timestamps, and trigger downstream semaphores */
    processinfo_update_output_stream(processinfo, &out_img, &in_img);

    /* 4. Rate-throttled status message for milk-procCTRL (avoid kHz string formatting) */
    static uint64_t msg_cnt = 0;
    if (++msg_cnt % 100 == 0)
    {
        processinfo_WriteMessage_fmt(processinfo, "K=%ld Frames=%lu",
                                     (long)mypkg_get_count(), (unsigned long)msg_cnt);
    }
}
INSERT_STD_PROCINFO_COMPUTEFUNC_END
```

---

## 4. Parameter Definition (`FPS_PARAMS`)

Define parameters once via the X-Macro pattern in `mypkg_fps_params.h`:

```c
#define MYPKG_FPS_PARAMS(X) \
    X(".in_name",     fps_in_name,      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT | FPFLAG_DEFAULT_TRIGGER_STREAM, "Input stream") \
    X(".out_name",    fps_out_name,     FPTYPE_STRING,     1, \
      FPFLAG_DEFAULT_INPUT, "Output stream") \
    X(".threshold",   &fps_threshold,   FPTYPE_FLOAT64,    1, \
      FPFLAG_DEFAULT_INPUT, "Threshold value") \
    X(".max_items",   &fps_max_items,   FPTYPE_UINT32,     1, \
      FPFLAG_DEFAULT_INPUT, "Max items") \
    X(".use_opt",     &fps_use_opt,     FPTYPE_ONOFF,      1, \
      FPFLAG_DEFAULT_INPUT, "Enable option") \
    X(".reset_state", &fps_reset_state, FPTYPE_ONOFF,      1, \
      FPFLAG_DEFAULT_INPUT, "Reset trigger")
```

### Critical Rules for FPS Parameters:
1. **Direct Variables, Not Pointer-to-Pointers**:
   - `FPS_CLI_BINDING` binds the variable address directly.
   - Use `char fps_in_name[FUNCTION_PARAMETER_STRMAXLEN]` for `FPTYPE_STREAMNAME` / `STRING`.
   - Use `int32_t` for `FPTYPE_ONOFF`.
   - Use `double`, `uint32_t`, `int64_t` for numerical parameters.
2. **Trigger Stream Flag**:
   - Set `FPFLAG_DEFAULT_TRIGGER_STREAM` on the primary stream entry so `-loops` automatically
     configures semaphore triggers without requiring extra flags.

---

## 5. Milk CLI Module Rules

When implementing `mypkg_module.c`:
1. Define module metadata before headers:
   ```c
   #define MODULE_SHORTNAME_DEFAULT "mypkg"
   #define MODULE_DESCRIPTION       "My Package Milk CLI module"
   #include "milk_config.h"
   ```
2. **Do NOT pass `-DMILK_MODULE` in CMake flags** (this collides with the macro definition).
3. Use `MILK_MODULE(milkmypkg, init_module_CLI, NULL);`.

---

## 6. Pre-Flight Checklist Before Finalizing

- [ ] **No Malloc in Loop**: All buffers, temporary vectors, and stream handles pre-allocated.
- [ ] **Semaphore Indexing**: Reader semaphore allocated via `processinfo_waitoninputstream_init()`
      (never hardcode `semptr[0]`).
- [ ] **Trigger Stream Configured**: Primary input has `FPFLAG_DEFAULT_TRIGGER_STREAM`.
- [ ] **Clean Exit**: Streams closed and engine cleaned up on both normal and signal exit paths.
- [ ] **Formatted Messages**: Use `processinfo_WriteMessage_fmt(processinfo, ...)`.
- [ ] **Coding Style**:
  - Allman braces.
  - Line length strictly $\le 100$ characters.
  - Multi-line function prototypes column-aligned per `parameter-alignment.md`.
  - Closing comments on scopes longer than 10 lines.
