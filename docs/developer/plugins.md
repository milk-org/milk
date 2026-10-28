# Adding a Plugin to Milk

Plugins in `milk` are external modules or compute units that extend its functionality without
modifying the core codebase. They are automatically discovered and built thanks to `milk`'s CMake
configuration.

This guide outlines exactly how to create, structure, and link a custom plugin.

See also: [Programmer's Guide](../arch/programmers_guide.md) · [Code Assist Tools](code_assist.md) ·
[Developer Tutorial](tutorial.md) · [Build Tiers](../quickstart/build_tiers.md)

## 1. Where do plugins go?

All plugins belong inside the `plugins/` directory at the root of the project. Milk's configuration
script uses `find -mindepth 1 -maxdepth 2` inside `plugins/` to discover subdirectories containing a
`CMakeLists.txt`.

Plugins can be placed directly in `plugins/<your_plugin>` or inside an intermediate group folder:

- `plugins/<your_plugin>` or `plugins/<your_group>/<your_plugin>` (for custom or third-party plugins)
- `plugins/cacao-src/<module>` (for `cacao` AO loop modules)
- `plugins/milk-extra-src/<module>` (reserved for core plugins shipped in the `milk` repository)

<!-- prettier-ignore -->
!!! note
    Because plugins are decoupled, it's very common for them to be their own isolated git repositories.
    You can add them under `plugins/` via standard copying, as a git submodule, or even via symbolic
    links.

### Standalone Executables Relation to Core

When a plugin registers a standalone executable (e.g., `milk-fpsexec-myplugin`), it is its own executable that interacts with the core engine (encompasses it and communicates via shared memory). The plugin's calculation logic should never depend on internal GUI/CLI tooling or code headers (from `CLIcore`).

## 2. Module layout

| Path             | Content                                                       |
| ---------------- | ------------------------------------------------------------- |
| `<modulename>.c` | Module registry (`MILK_MODULE_RUNNABLES`)                     |
| `<modulename>.h` | Module header and `extern` declarations of its runnables      |
| `CMakeLists.txt` | CMake build configuration (see below)                         |
| `README.md`      | Module overview, source file table, dependencies              |
| `scripts/`       | Shell scripts and utilities (optional)                        |
| `*.c` / `*.h`    | Source files, each holding one or several runnables           |

Copy `src/milk_module_example/CMakeLists.txt`; its skeleton and rules are in the [Programmer's Guide](../arch/programmers_guide.md#6-cmakeliststxt-conventions).

New plugins go in `plugins/<plugin>/` (or `plugins/<group>/<plugin>/`), never under
`milk-extra-src/`, which is reserved for the plugins shipped in the `milk` repository. Everything
under `plugins/` except `milk-extra-src/` is git-ignored: a new plugin is its own repository, and must
never be committed to `milk`. Plugins are discovered automatically, no parent `CMakeLists.txt` needs
editing.

## 3. Setting up CMakeLists.txt

The module's `CMakeLists.txt` builds the shared library with `milk_add_runnable_module()` and the
executables with `milk_add_fpsexec_stubs()`. Copy `src/milk_module_example/CMakeLists.txt`; the
conventions are in the [Programmer's Guide](../arch/programmers_guide.md#6-cmakeliststxt-conventions).

## 4. Writing the Source Code

A plugin is written like any core module, as a set of `MILK_RUNNABLE` units (see the
[Programmer's Guide](../arch/programmers_guide.md#3-writing-a-compute-unit)), starting from
`src/milk_module_example/examplefunc_fps_cli_poc.c`. Source files include `milk_runnable.h` and do
not include `CLIcore.h`. Declare module dependencies with `MODULE_DEPS("milkfft", ...)` before
`MILK_MODULE_RUNNABLES(myplugin, _module_deps, runnables)` (or pass `NULL` as the second argument if
there are no extra module dependencies), see [Managing Dependencies](dependency_system.md).

For `cacao` plugins, mark the units with `EXPORT_FPSEXEC(cacao, "id", unit);`, which builds
`cacao-fpsexec-<id>`.

## 5. Reviewing the Build

Once your `CMakeLists.txt` and C sources are in progress:

1. Navigate back to the main `milk` build directory.
2. Run `cmake ..`
3. Look for the message `ADDING SUBDIR = plugins/my_new_plugin`. If you see that,
   `milk` has successfully hooked your plugin into the global project.

---

← [Documentation Index](../index.md)
