---
name: plugin-creator
description: Deep reference for scaffolding a new plugin module with CMake and module registration boilerplate.
---

# Plugin Creator Guide

Plugins in milk extend its core capabilities and reside under the `plugins/` directory. Use this reference when scaffolding a new plugin.

## 1. Directory Structure

A plugin must exist inside a group folder within the `plugins/` directory:
`plugins/<group_name>/<plugin_name>/`
For example: `plugins/milk-extra-src/myplugin/`.

Inside, create the following core files:

- `<plugin_name>.c`
- `<plugin_name>.h`
- `CMakeLists.txt`
- `README.md`

## 2. CMake Integration

Your plugin's `CMakeLists.txt` must define the shared library, include directories, linking, and installation. If your plugin will provide compute functions for standalone executables, it must also build a `_compute` variant.

```cmake
# CMakeLists.txt example for "myplugin"
add_library(myplugin SHARED myplugin.c)

# Include current directory and the root source directory
target_include_directories(myplugin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

# Link against core libraries
target_link_libraries(myplugin PUBLIC CLIcore ImageStreamIO)

# Export and install
install(TARGETS myplugin
    EXPORT milkTargets
    LIBRARY DESTINATION lib
)

install(FILES myplugin.h DESTINATION include)

# Build _compute variant for standalone linking.
# This compiles the SAME source with -DMILK_NO_CLI.
add_library(myplugin_compute SHARED myplugin.c)
target_include_directories(myplugin_compute PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)
# Must NOT link CLIcore — only standalone-safe libs
target_link_libraries(
    myplugin_compute PUBLIC ImageStreamIO)
target_compile_definitions(
    myplugin_compute PRIVATE MILK_NO_CLI)

install(TARGETS myplugin_compute
    EXPORT milkTargets
    LIBRARY DESTINATION lib
)
```

# Note: Plugins are dynamically discovered by the root CMakeLists.txt

# (using find -L plugins -mindepth 2 -maxdepth 2 -type d).

# There is NO need to edit any parent CMakeLists.txt to register the plugin.

## 3. Module Registration (C Code)

Your plugin C file must register itself with the milk CLI framework.

```c
#include "myplugin.h"

// If you have commands:
// extern errno_t CLIADDCMD_myplugin__mycommand();

// Define module dependencies if any (or empty)
MODULE_DEPS() // e.g. MODULE_DEPS("milkCOREMODarith", "milkfft")

// Define the module init entry point
INIT_MODULE_LIB(myplugin)

static errno_t init_module_CLI()
{
    // Register commands here:
    // CLIADDCMD_myplugin__mycommand();

    return RETURN_SUCCESS;
}
```

## 4. Dependencies

Consult `docs/dependency_graph.md`. Plugins sit at the top of the hierarchy. If your plugin depends on another plugin, use `MODULE_DEPS("other_plugin")` and link it in CMake. Do not create circular dependencies.

## 5. Git Tracking Policy

**CRITICAL RULE**: Do NOT commit new plugins to the main `milk` repository.

- All folders under `plugins/` (except `plugins/milk-extra-src/`) are ignored by default via `.gitignore`.
- If a user creates a new plugin, it is their responsibility to initialize a new Git repository in that directory and push it to its own remote repository.
- The new plugin files must remain untracked in the `milk` repository index.
