---
description: Scaffold a new plugin module with full boilerplate
---

# Add a New Module

Use this workflow when creating a new plugin module
for milk or cacao. It ensures nothing is missed.

## 1. Gather Information

Ask the user for:
- **Module name** (e.g., `image_filter`)
- **Library name** (e.g., `milkimagefilter`)
- **Location**: `src/` (core) or `plugins/` (plugin)
- **One-line description** of the module
- Whether it is a **milk** or **cacao** module

## 2. Create Directory Structure

Create the module directory with these files:
```
<module_name>/
├── <module_name>.c       # Module registration
├── <module_name>.h       # Public header
├── CMakeLists.txt        # Build configuration
└── README.md             # Module documentation
```

## 3. Module Registration File

Use `src/milk_module_example/milk_module_example.c`
as the template. Update:
- Module name, short name, package
- Version numbers
- `initModule()` function with the correct
  `CLIADDCMD_` calls (initially empty)

### Dual-Mode Header Pattern

All source files in the module must use the
conditional include pattern so they compile both
as shared library and standalone executable:
```c
#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
```

## 4. Public Header

Use `src/milk_module_example/milk_module_example.h`
as the template. Update the include guard and
module name.

## 5. CMakeLists.txt

Create a `CMakeLists.txt` for the module. Use an
existing module's CMakeLists.txt as reference
(e.g., `src/coremods/COREMOD_arith/CMakeLists.txt`).

Key elements:
- `add_library(<libname> SHARED ...)`
- `target_link_libraries` with required dependencies
- `target_include_directories`
- `install(TARGETS ...)` for the library
- `install(FILES ... DESTINATION include/<module>)`
  for headers

### `_compute` Library Variant

If the module will have standalone executables,
**also** create a `_compute` variant:
```cmake
set(LIBNAME_COMPUTE ${LIBNAME}_compute)
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(
    ${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
target_link_libraries(
    ${LIBNAME_COMPUTE} PRIVATE milkdata ImageStreamIO)
```
Standalone executables must **never** link `CLIcore`.
The `_compute` variant is compiled with `MILK_NO_CLI`
and links only engine libraries.

## 6. README.md

Create a README following the standardized template:
- One-line module description
- Source file table (`| File | Description |`)
- Dependency list

## 7. Register in Parent CMake

Add `add_subdirectory(<module_name>)` in the
appropriate parent `CMakeLists.txt`:
- For core modules: `src/CMakeLists.txt`
- For plugins: the plugin's root `CMakeLists.txt`

## 8. Compile and Verify

Run the [`/compile-test`](compile-test.md) workflow to verify the new
module compiles and links correctly.

## 9. Notify

Tell the user the module boilerplate is ready and
remind them to add compute functions using the
[`/create-fpsexec`](create-fpsexec.md) workflow or by following the
`milk_module_example` templates.
