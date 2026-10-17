# Dependency System: managing extra dependencies

<!-- prettier-ignore -->
!!! info "TL;DR:"
    Do not try to modify the module's `CMakeLists.txt` into highly customized include and linkage
    rules. MILK provides magic tags `// MILK_CMAKE_REQUEST_<X>` and `// MILK_CMAKE_MANDATE_<X>`,
    with e.g. `<X> = CUDA`, from where all includes and linkage is handled automatically.

    Some care is needed with header packaging.

As discussed in [Build Tiers](../quickstart/build_tiers.md), some optional dependencies are expected by
MILK and enable additional features

| Option         | Default | Description                           |
| -------------- | ------- | ------------------------------------- |
| `USE_READLINE` | ON      | Enable readline for CLI input         |
| `USE_GSL`      | ON      | Enable GSL for plugins                |
| `USE_CUDA`     | OFF     | CUDA GPU computations                 |
| `USE_MAGMA`    | OFF     | MAGMA linear algebra on GPU routines  |
| `USE_OPENBLAS` | OFF     | BLAS: multicore linear algebra on CPU |
| `USE_MKL`      | OFF     | Intel MKL compute library             |
| `USE_LAPACKE`  | OFF     | Lapacke linear algebra library        |

## 1. User build requests and CMake symbols

When a symbol is set to `-DUSE_<X>=ON` (by the user or by default), CMake looks through the
appropriate file in the `cmake/*.cmake` directory.

If the dependency _cannot_ be located, the symbol is reverted to `USE_<X>=OFF`

If the library is located, `-DUSE_<X>=ON` is maintained and a symbol `-DHAVE_<X>=TRUE` is set (can
be `TRUE` or unset).

<!-- prettier-ignore -->
!!! Note
    In case of package `<X>` providing multiple features, or packages `<X>` and `<Y>` providing the
    same feature, several flags may be set: `-DHAVE_<a>`, `-DHAVE_<b>`, ...

## 2. Using symbols in C/C++ module code

### 2.1 Optional usage

In case the current C file can optionally use the `<X>` library, you should use preprocessor guards
that define the expected behavior everywhere it is necessary, and you must include the magic flag
`// MILK_CMAKE_REQUEST_<X>`

```C
// Top of file:
// MILK_CMAKE_REQUEST_<X>

// Computations:
#ifdef HAVE_<X>
    LIB_X_compute_something_fast();
#else
    NAIVE_compute_same_thing_slow();
#endif
```

and similarly for resources, variable (de)allocations, etc, that are necessary to work with library
<X>.

### 2.2 Mandatory usage

One may write a MILK compute session that is relevant ONLY if the `<X>` library is available, and
provides no fallback (e.g., `SVD_on_GPU_using_CUDA.c` wouldn't advertise any fallback without CUDA).

In this case, you can **omit** the `-DHAVE_<X>` guards entirely, and you should add the
`// MILK_CMAKE_MANDATE_<X>` magic flag

```C
// Top of file:
// MILK_CMAKE_MANDATE_<X>

// Computations:
LIB_X_compute_something_fast(); // We HAVE X for sure, no guards.
```

You do not need to use `HAVE_<X>` at all. It used to be common practice to guard the entire file in
a large `#ifdef HAVE_X`, but this is now unnecessary.

### 3. Build system

#### Requests

For `CMAKE_REQUEST`, this is easy: the includes and linkages from library `<X>` are dynamically
added when the tag is parsed, if `HAVE_<X>` is True. If not, the feature is disabled, but the file's
features are still available

#### Mandates

For `MILK_CMAKE_MANDATE`, the effect depends on the library `<X>` being missing. The rule applies per
source file of a module built with `milk_add_runnable_module()` (see
[Programmer's Guide](../arch/programmers_guide.md)):

- The C file is _excluded_ from the module library, and the functions it defines are not available.
  The library still exists.
- All runnables of a dropped file disappear from the module library, and their `EXPORT_FPSEXEC`
  executables are not built (CMake prints a `[SKIP]` line).
- The module registry (the file with `MILK_MODULE_RUNNABLES`) declares mandated runnables with
  `MILK_RUNNABLE_WEAK_DECL(name);` instead of including their header. The dropped entry is then `NULL`
  and the CLI skips it silently.
- The registry file itself cannot carry a mandate: CMake stops with an error.
- Requests and mandates apply to the whole module (as they already do for the library), so keep units
  with heavy optional dependencies in their own module.

Example: `plugins/test-src/compilertest`.

#### Helper functions of a mandated file

When a mandated file also exports plain helper functions, other C files may still call them while the
file is dropped. Declare the helpers in the module header as usual, and give the _registry_ file weak
definitions instead of including that header:

```C
MILK_WEAK int some_useful_computation() MILK_WEAK_FUNCDEF;
```

??? info "MILK_WEAK and MILK_WEAK_FUNCDEF"

    `MILK_WEAK` is just `__attribute__((weak))`: the function is replaced by its proper
    implementation when the MANDATE is respected and the C file is included.

    `MILK_WEAK_FUNCDEF` prints a meaningful error message and calls `abort()`. It avoids propagating
    the `MILK_CMAKE_MANDATE_<X>` to all the callers, but reports that the feature is not available.
    `MILK_WEAK` and `MILK_WEAK_FUNCDEF` are defined in `libmilkcommon`.
