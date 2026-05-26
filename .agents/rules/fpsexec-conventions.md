---
trigger: always_on
---

# FPS Executable (fpsexec) Conventions

## Creating a new fpsexec standalone executable

1. **Copy the template**: Use `src/milk_module_example/examplefunc_fps_cli_poc.c`
   as the starting point. It contains the standardized 8-section layout.

2. **Update identity** (section 1 — `FPS_APP_INFO`):
   - `.fps_name`    — SHM name on disk (no spaces)
   - `.cmdkey`      — milk CLI keyword
   - `.description` — one-line human-readable summary

3. **Define parameters** (sections 2–3):
   - Add local C variables in section 2.
   - Map them in the `FPS_PARAMS` X-macro in section 3.
   - Use `char*` for string-type params; pass `&ptr` in the X-macro.
   - Use `&variable` for scalars.

4. **Implement logic** (section 4 — `fpsexec()`):
   - Pure computation; parameters are already synced.

5. **CLIcmddata scoping** (section 5):
   ```c
   #ifdef FPS_STANDALONE
   CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
   #else
   static CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
   #endif
   ```

6. **Registration** (section 7):
   - Rename `CLIADDCMD_<module>__<function>` to match your module.

7. **Standalone main** (section 8):
   - Use `FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)`
   - Or `FPS_MAIN_STANDALONE_V2_CONFCHECK(...)` if you have a `customCONFcheck`.

8. **CMake targets** — add to the module's `CMakeLists.txt`:
   ```cmake
   add_executable(milk-fpsexec-<name> <source>.c)
   target_link_libraries(milk-fpsexec-<name> CLIcore <module_lib> milkfpsCLI)
   target_include_directories(milk-fpsexec-<name> PRIVATE ${PROJECT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR})
   target_compile_definitions(milk-fpsexec-<name> PRIVATE FPS_STANDALONE)
   install(TARGETS milk-fpsexec-<name> DESTINATION bin)
   ```

## Required: `-h1` one-line help option

Every fpsexec executable **must** support `-h1` (and `--help-oneline`).
This option prints only `FPS_app_info.description` (a single line) and exits.

This is handled automatically by the `FPS_MAIN_STANDALONE_V2` and
`FPS_MAIN_STANDALONE` macros in `fps.h` — no per-executable code is needed.
Just ensure that `.description` in `FPS_APP_INFO` is a clear, concise,
one-line summary of what the program does.

## Listing executables

Run `milk-fpsexec-list` to see all installed `milk-fpsexec-*` commands
with their one-line descriptions. Similarly, `cacao-fpsexec-list` lists
all `cacao-fpsexec-*` commands.

These lists are generated dynamically by invoking each executable's
`-h1` flag, so a new fpsexec only needs a proper `.description` field
in its `FPS_APP_INFO` to appear correctly in the list.

## Handling Dual-Mode `_compute` Libraries

If your module is configured to build a compute-only variant library (`_compute.so`) via CMake (which passes `-DMILK_NO_CLI`), you must guard the CLI registration code and mark the `compute_function` as unused to prevent compiler warnings.

1. **Mark `compute_function` as unused** in Section 6:
   ```c
   static MILK_HOT errno_t __attribute__((unused)) compute_function()
   ```
   *(Because it becomes orphaned when both CLI and standalone sections are excluded).*

2. **Check both `FPS_STANDALONE` and `MILK_NO_CLI`** in Section 7:
   ```c
   #if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
   static errno_t CLIfunction(void) { ... }
   errno_t CLIADDCMD_module__function() { ... }
   #endif
   ```
