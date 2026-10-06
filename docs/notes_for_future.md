---
tags:
  - architecture
  - build-system
---

# Notes for Future Reference

Working notes on design decisions that are correct for the project's current philosophy but may need revisiting if that philosophy changes.
Not a tutorial — see [Architecture](architecture.md) and [PGO](pgo.md) for those.

---

## Milk plugination philosophy

### 1. Current philosophy: embed expansions and rebuild MILK

Right now, the only supported way to add a plugin/module to MILK is to make its source visible to the main CMake project (in-tree -- embed or symlink in `plugins/`), or via an additional `add_subdirectory` in `CMakeLists.txt`; and then let the engine's own build-time discovery pull it into the configure/build.

There is no supported path for "build milk once, install it, then separately compile a plugin against the installed static libs without rebuilding milk".

### 2. A possible future: consumers over an unchanged MILK install

A different approach, where MILK is a framework / dependency of external packages developed separately. MILK is built and installed once. Consumers would call some sort of `find_package(milk)` in their CMake build and retrieve MILK's public interface for includes and linkage.

### 3. Two separate install interfaces

- **Outer interface - the installed package.** The `SHARED` libraries (`milkfps`, `milkCOREMODarith`, `milkdata`, `milkcommon`, `ImageStreamIO`, ...) are installed via `install(TARGETS ... EXPORT milkTargets ...)` and packaged into `milkConfig.cmake`. This is what an external project touches via `find_package(milk)`.
- **Inner interface - visible during the MILK build only.**. What is not set to `EXPORT` can be use for cross-dependencing during the build of MILK and its plugins, but is not further available to consumers through a CMake interface.

### 4. When it matters and when it don't

For a normal _dynamic linkage_ build, the compiled libraries `*.so` need to be retrieved at runtime, and as such they need to be installed for fetching in `MILK_INSTALLDIR` (more or less, since they're in the `rpath` of executables, so maybe we could bypass the `LD_LIBRARY_PATH` for finding them).

In `USE_STATIC_LTO` mode, we're compiling `.a` static libraries; however, these are only internal to the build since they used for scrapping the useful portions and optimizing executables. If there's no exposing-to-consumer approach, they're neither needed at runtime, nor to be installed, nor to be exposed in a downstream CMake interface.

### 5. If the philosophy changes later

If we want to approach **out-of-tree plugin consumers linking against a previously-built, installed static `milk`** (compile once, extend later, no full rebuild), we will need to do some work: create a proper CMake interface (or, be disciplined in the CMake statements consumers need to use), and make a proper export set with just about everything.

We would also have to freeze the ABI for good, more so if we'd allow consumer LTO builds.

And we'd have to review licensing implications of shipping static libs and what's in them.
