---
name: cmake-patterns
description: Pointer to the module CMakeLists.txt
  template, plus build tiers and common CMake errors
---

# CMake Patterns

## When to Use

- Creating a new module or standalone executable
- Debugging CMake build errors
- Adding new dependencies to a module

## Module CMakeLists.txt

Follow `src/milk_module_example/CMakeLists.txt`.
Skeleton, helpers and rules: `docs/programmers_guide.md` §6.
Optional dependencies: `docs/developer/dependency_system.md`.

## Build Tier Constraints

| Tier       | Available Libraries                               |
| ---------- | ------------------------------------------------- |
| Engine     | ImageStreamIO, milkfps, milkdata, milkprocessinfo |
| Core       | Engine + COREMOD\_{arith,memory,tools}            |
| Core+FITS  | Core + COREMOD_iofits, cfitsio                    |
| Full       | Everything: CLIcore, all plugins                  |
| Standalone | Engine + regular COREMOD libs, `-DMILK_NO_CLI`    |

Before adding a dependency, check
`docs/arch/dependency_graph.md` to verify the link
is allowed at your target's build tier.

## Conditional Compilation

| Variable         | Default | Controls                      |
| ---------------- | ------- | ----------------------------- |
| `USE_CLI`        | `ON`    | Whether CLI targets are built |
| `USE_CFITSIO`    | `ON`    | cfitsio-dependent modules     |
| `USE_COREMODS`   | `ON`    | COREMOD compilation           |
| `USE_STATIC_LTO` | `OFF`   | Static LTO builds             |
| `VEC_REPORT`     | `OFF`   | GCC vectorization report      |

## Common Errors and Fixes

| Error                                | Cause                            | Fix                                                     |
| ------------------------------------ | -------------------------------- | ------------------------------------------------------- |
| `undefined reference`                | Missing link dependency          | Add to `target_link_libraries`                          |
| `No such file or directory` (header) | Missing include dir              | Add `target_include_directories` or link the owning lib |
| `multiple definition`                | Non-static global in `.h`        | Make `extern` in `.h`, define in one `.c`               |
| Path doubling in install             | `CMAKE_INSTALL_PREFIX` in target | Use only at configure time                              |
