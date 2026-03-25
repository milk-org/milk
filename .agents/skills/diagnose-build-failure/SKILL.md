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
  │    → Missing target_link_libraries (§1)
  ├─ "No such file or directory" (header)
  │    → Missing target_include_directories (§2)
  ├─ "implicit declaration of function"
  │    → Missing #include in source file (§3)
  ├─ "multiple definition of ..."
  │    → Symbol defined in header without
  │      static/inline, or duplicate .c in
  │      SOURCEFILES (§4)
  ├─ "relocation ... cannot be used"
  │    → Static lib not built with -fPIC (§5)
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
3. Check `docs/dependency_graph.md` to verify the
   dependency is allowed

**Fixes**:
- Add the missing library to
  `target_link_libraries`:
  ```cmake
  target_link_libraries(mylib PUBLIC missinglib)
  ```
- For standalone executables, use `_compute`
  variants:
  ```cmake
  target_link_libraries(milk-fpsexec-foo
    milkfoo_compute)  # NOT milkfoo
  ```

**Common missing libraries**:

| Symbol | Library |
|--------|---------|
| `imgid_*`, `IMGID` | `milkdata` |
| `fps_*`, `FPS_*` | `milkfps` |
| `processinfo_*` | `milkprocessinfo` |
| `RegisterCLIcmd` | `CLIcore` (never for standalone!) |
| `cblas_sgemv` | `${BLAS_LIBRARIES}` |
| Math functions | `m` (libm) |

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
- Add include directory to the target:
  ```cmake
  target_include_directories(mylib PUBLIC
    ${PROJECT_SOURCE_DIR}/..)
  ```
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

| Function | Header |
|----------|--------|
| `malloc`, `free`, `calloc` | `<stdlib.h>` |
| `strlen`, `strcpy`, `memcpy` | `<string.h>` |
| `printf`, `fprintf` | `<stdio.h>` |
| `sqrt`, `pow`, `sin` | `<math.h>` |
| `clock_gettime` | `<time.h>` |
| `sem_post`, `sem_wait` | `<semaphore.h>` |

## §4 — Multiple Definition (Linker)

**Symptom**: `multiple definition of 'symbol'`

**Common causes**:
1. Non-static global defined in a `.h` file
   included by multiple `.c` files → make it
   `extern` in `.h`, define in one `.c`
2. Same `.c` file listed twice in `SOURCEFILES`
3. Function defined (not just declared) in `.h`
   without `static inline`

## §5 — Static LTO / PIC Issues

**Symptom**: `relocation R_X86_64_PC32 against
symbol ... can not be used; recompile with -fPIC`

**Cause**: a static library was built without
position-independent code.

**Fix**: ensure the static library target has:
```cmake
set_target_properties(mylib_static PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
```

## §6 — CMake Configuration Errors

**Common CMake issues**:

| Error | Fix |
|-------|-----|
| `Could NOT find ...` | Install the dependency or set `-DCMAKE_PREFIX_PATH` |
| `target ... not found` | Check spelling; target might be created conditionally |
| `Cannot find source file` | File was moved/renamed but `CMakeLists.txt` not updated |
| Path doubling in install | Use `CMAKE_INSTALL_PREFIX` only at configure time, not embedded in targets |

## Build Tier Reference

When diagnosing, know what's available at each
tier:

| Tier | Available libraries |
|------|-------------------|
| Engine | ImageStreamIO, milkfps, milkdata, milkprocessinfo |
| Core | Engine + COREMOD_{arith,memory,tools} |
| Core+FITS | Core + COREMOD_iofits, cfitsio |
| Full | Everything: CLIcore, all plugins |
| Standalone | Engine + `_compute` variants only |

## Standalone Executable Checklist

When a standalone executable fails to build:

- [ ] Uses `CLIcore_standalone.h` not `CLIcore.h`
- [ ] Links `_compute` variants, not full libs
- [ ] Has `FPS_STANDALONE` compile definition
- [ ] Has `MILK_NO_CLI` compile definition
- [ ] Does NOT call `RegisterCLIcmd` or any CLI
      functions
- [ ] Uses `add_milk_standalone()` or
      `add_cacao_standalone()` CMake helper
