---
trigger: always_on
---

# CMake and Header File Conventions

Module `CMakeLists.txt` files follow `src/milk_module_example/CMakeLists.txt` and
[programmers_guide.md §6](../../docs/arch/programmers_guide.md#6-cmakeliststxt-conventions):
dependencies through the `MILK_CMAKE_REQUEST_<X>` / `MILK_CMAKE_MANDATE_<X>` tags, never hand-written
include or link rules, and each module installs only its own headers.

- **Engine libraries** (`milkdata`, `milkfps`, ...): set `target_include_directories` and
  `target_link_libraries` as `PUBLIC`, so targets that link them inherit the include paths.
- **Includes**: as the table of [programmers_guide.md §5](../../docs/arch/programmers_guide.md#5-dependency-architecture)
  shows (`milk_runnable.h` first in runnable sources).

## Header File Taxonomy

Each module should organize its headers into these
categories:

| Category          | Naming                | Installed? | Purpose                                           |
| ----------------- | --------------------- | ---------- | ------------------------------------------------- |
| **Public API**    | `<module>.h`          | Yes        | External function declarations                    |
| **Internal**      | `internal/*.h`, `<module>_internal.h` | No | Private functions, shared state across `.c` files |
| **Types/structs** | `<module>_types.h`    | Yes        | Struct definitions, typedefs                      |
| **Macros**        | `<module>_macros.h`   | Optional   | Exported macro APIs                               |

**Rules:**

- Only public API and type headers should be installed. The module template installs every
  top-level header: keep internal ones in `internal/`; an engine library lists its installed
  headers (libprocessinfo leaves `processinfo_internal.h` out).
- Per-function `.h` files (e.g., `image_crop.h`,
  `fps_connect.h`) are acceptable for fine-grained
  include control, but the module must still have a
  top-level public API header that includes them.
- Function declarations intended only for use within
  the module should go in `_internal.h`, not in the
  per-function headers.

**Good examples in the codebase:**

- `libfps`: `fps.h` (public), `fps_types.h` (types)
- `libcomputeunit`: `computeunit.h` (public umbrella for the FPS runtime API and standalone macros)
- `ImageStreamIO`: `ImageStreamIO.h` (public),
  `ImageStruct.h` (types)
- `libprocessinfo`: `processinfo.h` (public),
  `processinfo_internal.h` (internal)
