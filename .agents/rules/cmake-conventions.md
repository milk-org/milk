---
trigger: always_on
---

# CMake and Header File Conventions

- **Standardize CMake Include Paths**: Use CMake `INTERFACE` or `PUBLIC` properties more strictly. When a library like `milkTUI` or `milkdata` is created, explicitly set its `target_include_directories` and `target_link_libraries` as `PUBLIC`. This ensures any target linking against it automatically inherits the include paths.
- **Header Installation Responsibility**: Each module (e.g., `libmilkdata`, `libprocessinfo`) is strictly responsible for installing its own header files. Other modules must not install headers they do not own.
- **Include Syntax**: Enforce `#include "module_name/header.h"` syntax or equivalent standard public API organization across the codebase to avoid naming collisions and make dependencies clearer.
