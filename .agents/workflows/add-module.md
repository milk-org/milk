---
description: Scaffold a new core module or plugin
---

# Add a Module

Layout, CMake and plugin policy: [`docs/developer/plugins.md`](../../docs/developer/plugins.md).
Template module: `src/milk_module_example/`. **Rules**: `runnable-conventions`,
`architecture-principles`.

## 1. Gather information

Module name, library name (`milk<name>`), location (`src/` core or `plugins/<plugin>/`), one-line
description, module dependencies.

## 2. Create the module

- `<module>.c`: `MODULE_SHORTNAME_DEFAULT`, `MODULE_DESCRIPTION`, optional `MODULE_DEPS(...)`
  (listing the non-engine `milk*` libraries the module links), and the `MILK_MODULE_RUNNABLES`
  registry (initially empty).
- `<module>.h`, `README.md`, and `CMakeLists.txt` copied from `src/milk_module_example/`.

Plugins under `plugins/` are discovered automatically and are never committed to `milk` (except
`plugins/milk-extra-src/`). Core modules are listed in the root `CMakeLists.txt`.

## 3. Verify

Run [`/compile-test`](compile-test.md), then add units with [`/add-runnable`](add-runnable.md).
Update `docs/arch/dependency_graph.md` if the module adds a cross-module dependency.
