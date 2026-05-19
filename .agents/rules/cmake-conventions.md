---
trigger: always_on
---

# CMake and Header File Conventions

- **Standardize CMake Include Paths**: Use CMake `INTERFACE` or `PUBLIC` properties more strictly. When a library like `milkTUI` or `milkdata` is created, explicitly set its `target_include_directories` and `target_link_libraries` as `PUBLIC`. This ensures any target linking against it automatically inherits the include paths.
- **Header Installation Responsibility**: Each module (e.g., `libmilkdata`, `libprocessinfo`) is strictly responsible for installing its own header files. Other modules must not install headers they do not own.
- **Include Syntax**: Enforce `#include "module_name/header.h"` syntax or equivalent standard public API organization across the codebase to avoid naming collisions and make dependencies clearer.

## Header File Taxonomy

Each module should organize its headers into these
categories:

| Category | Naming | Installed? | Purpose |
|----------|--------|------------|---------|
| **Public API** | `<module>.h` | Yes | External function declarations |
| **Internal** | `<module>_internal.h` | No | Private functions, shared state across `.c` files |
| **Types/structs** | `<module>_types.h` | Yes | Struct definitions, typedefs |
| **Macros** | `<module>_macros.h` | Optional | Exported macro APIs |

**Rules:**

- Only public API and type headers should be installed.
  Internal headers must not be exported at install time.
- Per-function `.h` files (e.g., `image_crop.h`,
  `fps_connect.h`) are acceptable for fine-grained
  include control, but the module must still have a
  top-level public API header that includes them.
- Function declarations intended only for use within
  the module should go in `_internal.h`, not in the
  per-function headers.

**Good examples in the codebase:**

- `libfps`: `fps.h` (public), `fps_internal.h`
  (internal), `fps_types.h` (types)
- `ImageStreamIO`: `ImageStreamIO.h` (public),
  `ImageStruct.h` (types)
- `libprocessinfo`: `processinfo.h` (public),
  `processinfo_internal.h` (internal)
