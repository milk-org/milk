# Dependency System: managing extra dependencies

<!-- prettier-ignore -->
!!! info "TL;DR:"
    Do not try to modify the module's `CMakeLists.txt` into highly customized include and linkage
    rules. MILK provides magic tags `// MILK_CMAKE_REQUEST_<X>` and `// MILK_CMAKE_MANDATE_<X>`,
    with e.g. `<X> = CUDA`, from where all includes and linkage is handled automatically.

    Some care is needed with header packaging.

As discussed in [Build Tiers](../install/build_tiers.md), some optional dependencies are expected by
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

For `CMAKE_MANDATE`, it's more complicated in the case the mandated library `<X>` is missing.

- For library compilations, the C files is _excluded_, and the defined function is not available.
  The library will still exist.
- For standalone `milk-fpsexec` targets, the executable is removed from the compiled objects and
  will not exist.

#### Headers

This, however, has consequences on header files:

```c
// C file export a function, defines a FPS standalone executable, and exports a computation function
// MILK_CMAKE_MANDATE_<X>

int some_useful_computation() {
    LIB_X_some_function();
    // ...
}

/* All the FPS V2 boilerplate */

errno_t CLIADDCMD_the_useful_computation_that_uses_x_under_the_hood() {
// ...


// H file
int some_useful_computation();
errno_t CLIADDCMD_the_useful_computation_that_uses_x_under_the_hood();
```

!!! danger

    Now the problem is, when the mandate fails, the function definitions above will simply not
    exist.

    1. The module file `module.c` above will still try to execute the `CLIADDCMD` upon initializing
       the module into the CLI.
    2. Other C files may try to use `some_useful_computation()`

#### Recommended way to sort declarations and missed-MANDATE definitions

- Use one or few headers for an entire module (not mandatory)
- In the `module.h` header file, add normal function prototypes for exportable functions:

```C
int some_useful_computation();
```

- C files, internally and externally, can include computation functions normally:

```C
// Internally
#include "module.h"
// Externally
#include "module/module.h"
```

The `module.c` file _must not_ include the `module.h` file ! Instead, it should contain weak
prototypes for:

- The `CLIADDCMD` functions for the compute sessions:

```C
// Declaration is enough if the command can never miss a MILK_CMAKE_MANDATE
errno_t CLIADDCMD_the_useful_computation_that_is_always_there();
// Declaration is enough if the command can never miss a MILK_CMAKE_MANDATE
MILK_WEAK errno_t CLIADDCMD_the_useful_computation_that_uses_x_under_the_hood() {};
```

- Weak definitions for the exported helper functions (or only for those that may miss a
  `MILK_CMAKE_MANDATE`)

```C
MILK_WEAK int some_useful_computation() MILK_WEAK_FUNCDEF;
```

??? info "MILK_WEAK and MILK_WEAK_FUNCDEF"

    `MILK_WEAK` is just `__attribute__((weak))`. Which means this function will be replaced by its
    proper implementation when the MANDATE is respected and the C file actually included.

    The `CLIADDCMD` can still be invoked by `module.c`. It does nothing.

    `MILK_WEAK_FUNCDEF` prints a meaningful error message and calls `abort()`. It avoid propagating
    the `MILK_CMAKE_MANDATE_<X>` to all the callers, but reports meaningful info that the feature is
    not available. `MILK_WEAK` and `MILK_WEAK_FUNCDEF` are defined in `libmilkcommon`.
