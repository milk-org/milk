---
name: diagnose-build-failure
description: Triage CMake and GCC build errors using
  milk's tiered architecture and dependency rules
---

# Diagnose Build Failures

This skill provides structured methods for
diagnosing and fixing build failures in the milk
project. It covers common error patterns, their
root causes, and fixes mapped to the project's
tiered build system.

## When to Use

- `cmake --build` fails with compiler or linker
  errors
- A new module or standalone won't link
- Static LTO build (`-DUSE_STATIC_LTO=ON`) fails
- Cross-module dependency issues

## Quick Triage Flowchart

```
Error message
  ├─ "undefined reference to ..."
  │    → Missing library or dependency tag (§1)
  ├─ "No such file or directory" (header)
  │    → Wrong include path or owning lib (§2)
  ├─ "implicit declaration of function"
  │    → Missing #include in source file (§3)
  ├─ "multiple definition of ..."
  │    → Symbol defined in header without
  │      static/inline, or duplicate
  │      definition across .c files (§4)
  ├─ "relocation ... cannot be used"
  │    → Target built outside the helpers (§5)
  └─ CMake errors
       → See §6
```

## §1 — Undefined Reference (Linker)

**Symptom**: `undefined reference to 'function'`

**Diagnosis**:

1. Find which library provides the function:
   ```bash
   grep -r "function_name" src/ --include="*.h" -l
   ```
2. Check the module's `CMakeLists.txt` for
   `target_link_libraries`
3. Check `docs/arch/dependency_graph.md` to verify the
   dependency is allowed

**Fixes**:

- Optional external libraries (BLAS, LAPACKE, CUDA, MAGMA, cfitsio): add the
  `// MILK_CMAKE_REQUEST_<X>` or `// MILK_CMAKE_MANDATE_<X>` tag in the source
  ([dependency_system.md](../../../docs/developer/dependency_system.md)), not link rules.
- The framework libraries are linked by `milk_add_runnable_module()`; link a module-specific
  library in the template's `# CUSTOM` block:
  ```cmake
  target_link_libraries(mylib PUBLIC missinglib)
  ```
- For fpsexecs, declare the unit with `EXPORT_FPSEXEC` (the stub is linked to the module library
  by `milk_add_fpsexec_stubs()`).

**Common missing libraries**:

| Symbol             | Library                                    |
| ------------------ | ------------------------------------------ |
| `IMGID`, `imgid_make*`, `imgid_connect` | `milkfps` (inline, `libfps/IMGID.h`) |
| `imgid_create_registered`, `imcreateIMGID`, `resolveIMGID` | `milkCOREMODmemory` |
| `fps_*`, `FPS_*`   | `milkfps`                                  |
| `processinfo_*`    | `milkprocessinfo`                          |
| `RegisterCLIcmd`   | `CLIcore` (never for fpsexecs!)            |
| `data` (`DATA`)    | Raw `RegisterCLIcmd`/`data.cmd[]` in module code: list the runnable in `MILK_MODULE_RUNNABLES` instead |
| `cblas_sgemv`      | `// MILK_CMAKE_REQUEST_BLAS` in the source |
| Math functions     | `m` (libm)                                 |

## §2 — Missing Header (Compiler)

**Symptom**: `fatal error: foo/bar.h: No such
file or directory`

**Diagnosis**:

1. Find where the header lives:
   ```bash
   find src/ -name "bar.h" -type f
   ```
2. Check if the owning module sets its include
   directory as PUBLIC:
   ```bash
   grep -A3 "target_include_directories" \
     src/path/CMakeLists.txt
   ```

**Fixes**:

- Include the header by its path under `src/` or the plugin directory: the template's
  `include_directories()` provide both.
- Optional external libraries: use the `MILK_CMAKE_REQUEST_<X>` / `MANDATE_<X>` tag (§1), not
  hand-written include rules.
- Or link against the library that owns the
  header (PUBLIC includes propagate):
  ```cmake
  target_link_libraries(mylib PUBLIC ownerlib)
  ```

## §3 — Implicit Declaration (Compiler)

**Symptom**: `implicit declaration of function
'foo'`

**Diagnosis**: the `.c` file calls `foo()` but
doesn't include the header that declares it.

**Fix**: add the missing `#include`. Remember the
project rule: every `.c` file includes exactly
what it needs — no implicit transitive includes.

Common missing includes:

| Function                     | Header          |
| ---------------------------- | --------------- |
| `malloc`, `free`, `calloc`   | `<stdlib.h>`    |
| `strlen`, `strcpy`, `memcpy` | `<string.h>`    |
| `printf`, `fprintf`          | `<stdio.h>`     |
| `sqrt`, `pow`, `sin`         | `<math.h>`      |
| `clock_gettime`              | `<time.h>`      |
| `sem_post`, `sem_wait`       | `<semaphore.h>` |

## §4 — Multiple Definition (Linker)

**Symptom**: `multiple definition of 'symbol'`

**Common causes**:

1. Non-static global defined in a `.h` file
   included by multiple `.c` files → make it
   `extern` in `.h`, define in one `.c`
2. Same function defined in two `.c` files of
   the module
3. Function defined (not just declared) in `.h`
   without `static inline`

## §5 — Static LTO / PIC Issues

**Symptom**: `relocation R_X86_64_PC32 against
symbol ... can not be used; recompile with -fPIC`

**Cause**: a library or archive built outside `milk_add_runnable_module()`, which compiles the
module sources once as PIC objects for both the shared library and the `_static` archive
([programmers_guide.md §3](../../../docs/arch/programmers_guide.md#milk_runnable-units)).

**Fix**: build the module with `milk_add_runnable_module()`. For `cannot find -l<lib>_static`
after switching `USE_STATIC_LTO` off, see
[pgo.md §1.6](../../../docs/operations/pgo.md#16-build-modes).

## §6 — CMake Configuration Errors

**Common CMake issues**:

| Error                     | Fix                                                                        |
| ------------------------- | -------------------------------------------------------------------------- |
| `Could NOT find ...`      | Install the dependency or set `-DCMAKE_PREFIX_PATH`                        |
| `target ... not found`    | Check spelling; target might be created conditionally                      |
| `Cannot find source file` | File was deleted or renamed; re-run `cmake` (sources are globbed)          |
| Path doubling in install  | Use `CMAKE_INSTALL_PREFIX` only at configure time, not embedded in targets |

## Build Tier Reference

What is available at each tier:
[dependency_graph.md §5](../../../docs/arch/dependency_graph.md#5-build-tiers-at-a-glance).

## fpsexec Checklist

When an fpsexec fails to build or link:

- [ ] The unit has a single-line `EXPORT_FPSEXEC(milk|cacao, "id", unit);` in a module source
      (re-run `cmake`: the lines are scanned at configure time)
- [ ] The module is built with `milk_add_runnable_module()` + `milk_add_fpsexec_stubs()`
- [ ] No `CLIcore` in the link line (`ldd <exe> | grep CLIcore` must be empty)
- [ ] The unit's source file was not dropped by an unmet `MILK_CMAKE_MANDATE_*` (a `[SKIP]` line is
      printed at configure time)
- [ ] Follows `src/milk_module_example/CMakeLists.txt`
