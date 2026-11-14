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

Follow `src/milk_module_example/CMakeLists.txt`. Skeleton, helpers and rules:
[programmers_guide.md §6](../../../docs/arch/programmers_guide.md#6-cmakeliststxt-conventions).
Optional dependencies: [dependency_system.md](../../../docs/developer/dependency_system.md).

## Build Tier Constraints

Libraries available at each tier, and which links are allowed:
[dependency_graph.md §5](../../../docs/arch/dependency_graph.md#5-build-tiers-at-a-glance).
Check it before adding a dependency.

## Conditional Compilation

CMake options: [build_tiers.md](../../../docs/quickstart/build_tiers.md#optional-dependencies)
(`USE_CLI`, `USE_COREMODS`, `USE_PLUGINS`, `USE_CALC` in its tier table); `VEC_REPORT`:
[performance.md §7](../../../docs/operations/performance.md#7-quick-checklist).

## Common Errors and Fixes

See the [`diagnose-build-failure`](../diagnose-build-failure/SKILL.md) skill.
