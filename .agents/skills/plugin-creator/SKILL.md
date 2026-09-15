---
name: plugin-creator
description: Deep reference for scaffolding a new plugin module with CMake and module registration boilerplate.
---

# Plugin Creator Guide

Plugins in milk extend its core capabilities and reside under the `plugins/` directory. Use this reference when scaffolding a new plugin.

## 1. Directory Structure

A plugin should reside directly in its own directory under the `plugins/` directory:
`plugins/<plugin_name>/`
For example: `plugins/myplugin/`.

Alternatively, a plugin can reside inside an optional group folder:
`plugins/<group_name>/<plugin_name>/`
For example: `plugins/my-group/myplugin/`.

> [!IMPORTANT]
> Do NOT place new plugins under the `milk-extra-src` group folder (which is reserved
> for core extra plugins compiled directly in the main repository).

Inside, create the following core files:

- `<plugin_name>.c`
- `<plugin_name>.h`
- `CMakeLists.txt`
- `README.md`

## 2. CMake Integration

Your plugin's `CMakeLists.txt` must define the shared library, include directories, linking, and installation. If your plugin provides functions for standalone executables, they link this same regular library — no separate variant is needed. Use `add_milk_standalone()` / `add_cacao_standalone()` for the standalone executable target; those helpers apply `-DMILK_NO_CLI` to the executable itself.

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

# add_milk_standalone()/add_cacao_standalone() apply -DMILK_NO_CLI to the executable target itself.

install(FILES myplugin.h DESTINATION include)
```

Note: Plugins are dynamically discovered by the root CMakeLists.txt (using `find -L plugins -mindepth 2 -maxdepth 2 -type d` for folders that contain a `CMakeLists.txt`).
**There is NO need to edit any parent CMakeLists.txt to register the plugin.**

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
