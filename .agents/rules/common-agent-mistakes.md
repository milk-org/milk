---
trigger: always_on
---

# Common Agent Mistakes

Consolidated checklist of pitfalls that AI agents
frequently hit when generating milk code. Check
this list before finalizing any generated code.

## New Compute Unit (fpsexec)

1. **Forgetting to add `.c` to SOURCEFILES.**
   The new file must be added to the module's
   `CMakeLists.txt` `SOURCEFILES` list (for the
   shared library build), in addition to the
   standalone target.

2. **Forgetting to register in `initModule()`.**
   The `CLIADDCMD_<module>__<function>()` call
   must be added to the module's `initModule()`
   function, and the corresponding `.h` file
   `#include`d.

3. **Missing `FPS_CMDSETTINGS_INIT`.**
   Every V2 compute unit requires
   `FPS_CMDSETTINGS_INIT(dft, CLIcmddata,
FPS_app_info)` after the `CLIcmddata`
   declaration. Without it,
   `INSERT_STD_PROCINFO_COMPUTEFUNC_END`
   dereferences NULL.

4. **Using `FPS_MAIN_STANDALONE` instead of
   `FPS_MAIN_STANDALONE_V2`.** The V1 macro
   does not support the `FPS_PARAMS` X-macro.
   Always use `FPS_MAIN_STANDALONE_V2` (or
   `_V2_CONFCHECK` if you have a
   `customCONFcheck`).

5. **Wrong string parameter pattern.** Use
   `static char var[FUNCTION_PARAMETER_STRMAXLEN]`
   and pass `var` (no `&`) in the X-macro.
   Do not use `char *var` with `&var` in new
   code.

6. **Using `FPFLAG_DEFAULT_INPUT` for stream
   parameters.** Use `FPFLAG_DEFAULT_TRIGGER_STREAM`
   for input streams that drive the compute loop,
   or `FPFLAG_DEFAULT_INPUT_STREAM` for required
   input streams.

7. **Forgetting `outimg->md->write = 1`.**
   Must be set before modifying output stream
   pixels (inside the per-frame function).
   `processinfo_update_output_stream()` handles
   `write = 0` and semaphore posting.

## New Module / Plugin

10. **Unnecessarily editing parent directories for plugins**:
    Plugins under `plugins/` are dynamically discovered and added by
    the root `CMakeLists.txt`. Do NOT edit parent directories to
    add `add_subdirectory()`. (Only core engine modules under `src/`
    require manual registration in their parent `CMakeLists.txt`).

11. **Placing new plugins under `milk-extra-src`**:
    `milk-extra-src/` is reserved for core extra plugins included in the main
    `milk` repository. New plugins must be created in a custom folder directly
    under `plugins/` (e.g. `plugins/myplugin` or optionally a custom group folder)
    to ensure they remain untracked by Git and separate from the core repository.

12. **Missing `#ifdef MILK_NO_CLI` include guard.** Files compiled in dual mode need:

    ```c
    #ifdef MILK_NO_CLI
    #    include "CLIcore_standalone.h"
    #else
    #    include "CLIcore.h"
    #endif
    ```

13. **Accidentally tracking/committing new plugins**:
    New plugins must NEVER be added to the main `milk` repository index
    (except `plugins/milk-extra-src/`). They are ignored via
    `.gitignore`. Managing them is the user's responsibility (e.g., as
    independent repositories).

## General Code

15. **Implicit header includes.** Every `.c`
    file must include exactly the headers it
    uses. Do not rely on `CLIcore.h` pulling
    in `math.h` or `stdlib.h`.

16. **Lines > 100 characters.** The project
    enforces short lines for readability.

17. **Not compiling after edits.** Always run
    `/compile-test` after modifying C or CMake
    files.

18. **Not updating `docs/dependency_graph.md`.**
    Required when adding new cross-module
    dependencies.
