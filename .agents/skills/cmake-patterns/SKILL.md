---
name: cmake-patterns
description: Comprehensive CMake patterns for milk
  modules, standalone executables, and build tiers
---

# CMake Patterns

This skill provides a complete reference for CMake
conventions in the milk project, covering module
setup, standalone executables, header installation,
and build tier requirements.

## When to Use

- Creating a new module or standalone executable
- Debugging CMake build errors
- Understanding `_compute` library variants
- Adding new dependencies to a module

## Module CMakeLists.txt Anatomy

A typical module `CMakeLists.txt` has this
structure:

```cmake
# Source files
set(SOURCEFILES
    module_name.c
    func1.c
    func2.c
)

# Header files to install
set(INSTALLHEADERS
    module_name.h
    func1.h
    func2.h
)

# Library name
set(LIBNAME "milkmodulename")

# Build the shared library
add_library(${LIBNAME} SHARED ${SOURCEFILES})

# Link dependencies
target_link_libraries(${LIBNAME}
    PUBLIC CLIcore milkdata)

# Include paths
target_include_directories(${LIBNAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/..)

# Install library
install(TARGETS ${LIBNAME}
    DESTINATION lib)

# Install headers
install(FILES ${INSTALLHEADERS}
    DESTINATION include/${LIBNAME})
```

## Key Conventions

### PUBLIC vs PRIVATE vs INTERFACE

| Scope       | When to Use                                                      |
| ----------- | ---------------------------------------------------------------- |
| `PUBLIC`    | Dependency used in both this module's headers AND implementation |
| `PRIVATE`   | Dependency used only in `.c` files (not exposed in headers)      |
| `INTERFACE` | Dependency used only in headers (rare)                           |

**Rule of thumb**: use `PUBLIC` for `CLIcore`,
`milkdata`, `ImageStreamIO`, `milkfps`. Use
`PRIVATE` for `m` (libm), `pthread`, BLAS.

### Header Installation

Each module is **strictly responsible** for
installing only its own headers. Never install
headers from another module.

```cmake
# Correct: install own headers
install(FILES ${INSTALLHEADERS}
    DESTINATION include/${LIBNAME})

# WRONG: installing another module's header
install(FILES ../othermod/other.h
    DESTINATION include/${LIBNAME})
```

### SOURCEFILES

- Add `.c` files only (never `.h` files).
- One file per line for clean diffs.
- Order: main module file first, then
  alphabetically.

## Standalone Executables

### Using CMake Helpers

Always use the provided helper macros:

```cmake
# milk standalone (creates milk-fpsexec-<name>)
add_milk_standalone(myname myname.c)

# cacao standalone (creates cacao-fpsexec-<name>)
add_cacao_standalone(myname myname.c)

# cacao with plugin dependencies
add_cacao_standalone_plugins(
    myname myname.c fft imagegen)
```

These helpers automatically:

- Set `FPS_STANDALONE` and `MILK_NO_CLI` defines
- Link `_compute` variants instead of full libs
- Set correct include directories
- Configure install target

### Additional Link Dependencies

If a standalone needs extra libraries beyond the
standard set:

```cmake
add_cacao_standalone(myname myname.c)
target_link_libraries(
    cacao-fpsexec-myname
    PUBLIC milkstatistic_compute)
```

### Never Do This

```cmake
# WRONG — the old 4-line manual pattern:
add_executable(milk-fpsexec-foo foo.c)
target_link_libraries(milk-fpsexec-foo
    CLIcore ...)  # NEVER link CLIcore
target_compile_definitions(...)
install(TARGETS ...)
```

## `_compute` Library Variants

Standalone executables must never link `CLIcore`.
Instead, they link `_compute` variants compiled
with `MILK_NO_CLI`:

```cmake
set(LIBNAME_COMPUTE ${LIBNAME}_compute)
add_library(
    ${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(
    ${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
target_link_libraries(
    ${LIBNAME_COMPUTE}
    PRIVATE milkdata ImageStreamIO milkfps)
```

The `_compute` variant:

- Includes `CLIcore_standalone.h` (stubs)
- Does **not** register CLI commands
- Links only engine-tier libraries
- Is safe for standalone executables

## Build Tier Constraints

| Tier       | Available Libraries                               |
| ---------- | ------------------------------------------------- |
| Engine     | ImageStreamIO, milkfps, milkdata, milkprocessinfo |
| Core       | Engine + COREMOD\_{arith,memory,tools}            |
| Core+FITS  | Core + COREMOD_iofits, cfitsio                    |
| Full       | Everything: CLIcore, all plugins                  |
| Standalone | Engine + `_compute` variants only                 |

Before adding a dependency, check
`docs/dependency_graph.md` to verify the link
is allowed at your target's build tier.

## Conditional Compilation

| Variable         | Default | Controls                      |
| ---------------- | ------- | ----------------------------- |
| `USE_CLI`        | `ON`    | Whether CLI targets are built |
| `USE_CFITSIO`    | `ON`    | cfitsio-dependent modules     |
| `USE_COREMODS`   | `ON`    | COREMOD compilation           |
| `USE_STATIC_LTO` | `OFF`   | Static LTO builds             |
| `VEC_REPORT`     | `OFF`   | GCC vectorization report      |

Guard optional dependencies in CMake:

```cmake
if(USE_CFITSIO)
    target_link_libraries(${LIBNAME}
        PRIVATE cfitsio)
endif()
```

## Common Errors and Fixes

| Error                                | Cause                            | Fix                                                     |
| ------------------------------------ | -------------------------------- | ------------------------------------------------------- |
| `undefined reference`                | Missing link dependency          | Add to `target_link_libraries`                          |
| `No such file or directory` (header) | Missing include dir              | Add `target_include_directories` or link the owning lib |
| `multiple definition`                | Non-static global in `.h`        | Make `extern` in `.h`, define in one `.c`               |
| Path doubling in install             | `CMAKE_INSTALL_PREFIX` in target | Use only at configure time                              |

## Adding a Module to the Build

Register your module in the parent
`CMakeLists.txt`:

```cmake
# For core modules: src/CMakeLists.txt
add_subdirectory(module_name)

# For plugins: plugins/milk-extra-src/CMakeLists.txt
add_subdirectory(module_name)
```

## Linking External Libraries

### BLAS (Matrix Operations)

BLAS is found by CMake at the top level. Link
it as `PRIVATE` since it's an implementation
detail:

```cmake
target_link_libraries(${LIBNAME}
    PRIVATE ${BLAS_LIBRARIES})
```

For standalone executables that need BLAS:

```cmake
add_milk_standalone(myname myname.c)
target_link_libraries(
    milk-fpsexec-myname
    PRIVATE ${BLAS_LIBRARIES})
```

### FFTW

FFTW is found via `pkg_check_modules` in the
top-level CMake. Link via the milkfft module:

```cmake
target_link_libraries(${LIBNAME}
    PRIVATE milkfft)
```

For standalone targets, use the `_compute`
variant:

```cmake
add_milk_standalone(myname myname.c)
target_link_libraries(
    milk-fpsexec-myname
    PRIVATE milkfft_compute)
```

### When to Create a `_compute` Variant

Create a `_compute` variant if:

- Your module provides functions that
  standalone executables need to link against
- Another module's standalone links your lib

Do **not** create a `_compute` variant if:

- Your module is CLI-only (TUI, interactive)
- No standalone executable needs your code
