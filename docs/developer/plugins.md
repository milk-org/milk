# Adding a Plugin to Milk

Plugins in `milk` are external modules or compute units that extend its functionality without modifying the core codebase. They are automatically discovered and built thanks to `milk`'s CMake configuration.

This guide outlines exactly how to create, structure, and
link a custom plugin.

See also: [Programmer's Guide](../programmers_guide.md) ·
[Code Assist Tools](../code_assist.md) ·
[Developer Tutorial](tutorial.md) ·
[Build Tiers](../install/build_tiers.md) ·
[Loading Modules](LoadingModules.md)

## 1. Where do plugins go?

All plugins belong inside the `plugins/` directory at the root of the project.
Milk's configuration script uses `find -mindepth 2 -maxdepth 2` inside `plugins/` to discover subdirectories containing a `CMakeLists.txt`.

Typically, plugins are placed inside an intermediate group folder such as:

- `plugins/milk-extra-src/<your_plugin>`
- `plugins/cacao-src/<your_plugin>` (For `cacao` AO loop modules)

!!! note
Because plugins are decoupled, it's very common for them to be their own isolated git repositories. You can add them under `plugins/` via standard copying, as a git submodule, or even via symbolic links.

### Standalone Executables Relation to Core

When a plugin registers a standalone executable (e.g., `milk-fpsexec-myplugin`), it behaves as a native Linux process that interacts with the core engine (streams and FPS) solely via shared memory. The plugin's calculation logic should NOT depend directly on internal GUI/CLI headers from `CLIcore`. By explicitly creating plugins and linking to `_compute` libraries, you ensure strict modularity where crashing plugins do not disrupt the orchestrating daemon.

## 2. Setting up CMakeLists.txt

When `milk` detects a `CMakeLists.txt` in your plugin's directory, it will automatically process it via `add_subdirectory()`. The file is expected to define your module's shared library and register its standalone executables.

Here is a standard template for a plugin's `CMakeLists.txt`:

```cmake
# =======================================
# my_new_plugin - Short description
# =======================================

set(LIBNAME my_new_plugin)

# 1. Source files
set(SOURCEFILES
    my_plugin_func.c
)

# 2. Shared Library (used by the interactive milk-cli)
add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_link_libraries(${LIBNAME} PRIVATE CLIcore)

# 3. Compute-only variant (no CLI deps, for standalone executables linking it)
set(LIBNAME_COMPUTE ${LIBNAME}_compute)
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
target_link_libraries(${LIBNAME_COMPUTE} PRIVATE milkdata ImageStreamIO)

# 4. Standalone Executable Definition
# This automatically handles the FPS lifecycle and links COREMOD_compute libs
add_milk_standalone(my_plugin_func my_plugin_func.c)
```

## 3. Writing the Source Code

From a source code standpoint, a plugin operates exactly like any standard core `milk` module.

Specifically, it should be written as a "compute unit" implementing the V2 FPS standard architecture. You must use the canonical template `src/milk_module_example/examplefunc_fps_cli_poc.c` as your starting point.

### Dual-Mode Headers Requirement

Plugins are often compiled twice: once for the shared library (CLI) and once for the standalone executable. To stay compliant and not violate dependency chains (i.e. to ensure standalones do not depend on graphical/TUI CLI elements), your C code must use conditional includes:

```c
#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"   /* Stub types for standalone mode */
#else
#include "CLIcore.h"              /* Full CLI types for shared lib mode */
#endif
#include "fps.h"                  /* Always required for FPS structures */
```

## 4. Specialized Cacao Plugins

If you are writing a plugin specifically for the `cacao` AO loop, your standalone executable target is configured slightly differently.

Instead of `add_milk_standalone`, use `add_cacao_standalone`. This tells the build system to prefix your executable properly (e.g. `cacao-fpsexec-myfunc`).

```cmake
# Minimal cacao standalone
add_cacao_standalone(my_cacao_func my_cacao_func.c)

# If your cacao standalone depends on code from other plugins (like fft or imagegen)
add_cacao_standalone_plugins(my_cacao_func my_cacao_func.c fft imagegen)
```

## 5. Reviewing the Build

Once your `CMakeLists.txt` and C sources are in progress:

1. Navigate back to the main `milk` build directory.
2. Run `cmake ..`
3. Look for the message `ADDING SUBDIR = plugins/milk-extra-src/my_new_plugin`. If you see that, `milk` has successfully hooked your plugin into the global project.

---

← [Documentation Index](../index.md)
