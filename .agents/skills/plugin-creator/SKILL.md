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

Copy `src/milk_module_example/CMakeLists.txt` and rename. Layout and rules: `docs/programmers_guide.md` §6.

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
