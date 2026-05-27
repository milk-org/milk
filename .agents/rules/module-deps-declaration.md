# Module Dependency Declaration

When editing a module's main `.c` file (the file
containing `INIT_MODULE_LIB`), **always check**
whether `MODULE_DEPS` should be populated or
updated.

## When to act

- **Creating** a new module.
- **Adding** a new cross-module dependency
  (e.g. calling a function from another plugin).
- **Editing** a module's main `.c` file and
  noticing it still uses `INIT_MODULE_LIB`
  despite having real cross-module link deps.

## How to determine dependencies

1. Open the module's `CMakeLists.txt`.
2. Find the `target_link_libraries(${LIBNAME} ...)`
   line for the **shared library** target.
3. Extract every `milk*` library name that is
   **not** one of the core/engine libraries
   (CLIcore, milkdata, milkprocessinfo, milkfps,
   milkTUI, ImageStreamIO). These core libraries
   are always available and are not mload targets.
4. Each remaining `milk*` library name is a
   dependency loadname for `MODULE_DEPS`.

### Mapping library names to loadnames

The loadname used by `mload` is the CMake
`LIBNAME` of the dependency module. Typical
examples:

| CMake link target  | MODULE_DEPS loadname |
| ------------------ | -------------------- |
| `milkfft`          | `"milkfft"`          |
| `milkimagegen`     | `"milkimagegen"`     |
| `milkimagefilter`  | `"milkimagefilter"`  |
| `milklinalgebra`   | `"milklinalgebra"`   |
| `milkstatistic`    | `"milkstatistic"`    |
| `milkimagebasic`   | `"milkimagebasic"`   |
| `milkZernikePolyn` | `"milkZernikePolyn"` |

## How to apply

1. Before `#include "CLIcore.h"`, add:

   ```c
   MODULE_DEPS("milkfft", "milkimagegen")
   ```

   listing each dependency loadname.

2. Replace `INIT_MODULE_LIB(modname)` with
   `INIT_MODULE_LIB_DEPS(modname)`.

3. If the module has **zero** cross-module deps
   (only links CLIcore and engine libs), keep
   `INIT_MODULE_LIB` and omit `MODULE_DEPS`.

## Standalone builds

`MODULE_DEPS` and `INIT_MODULE_LIB_DEPS` have
matching stubs in `CLIcore_standalone.h`; no
additional changes are needed for standalone
executables.

## Verification

After adding or changing `MODULE_DEPS`, run
`/compile-test` and manually verify with:

```
milk
mload <module_loadname>
m?
```

Confirm that dependencies appear in the loaded
module list.
