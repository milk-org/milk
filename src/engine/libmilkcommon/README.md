# libmilkcommon

Header-only utility library providing the lowest-level
types, macros, and constants used throughout the milk
framework. This is the foundation of the dependency
graph — every other milk library depends on it.

## Purpose

Centralizes compiler-portable definitions so that
higher-level libraries (`libmilkdata`, `ImageStreamIO`,
`libfps`, `CLIcore`) do not duplicate basic types or
error-handling patterns.

## Files

| File                  | Description                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| `milkDebugTools.h`    | Error/warning macros (`PRINT_ERROR`, `FUNC_RETURN_FAILURE`), string length constants, return codes |
| `milk_compiler.h`     | GCC performance hints (`MILK_HOT`, `MILK_RESTRICT`, `MILK_IVDEP`, `MILK_ALIGNED`, etc.)            |
| `milk_types.h`        | Aggregate include pulling in core POSIX types, `MILK_DATA`, `ImageStreamIO`, and `processtools`    |
| `multiselect_parse.h` | Inline parser for multi-selection input strings (e.g. `"1 3 5-7 all"`)                             |
| `pixel_dispatch.h`    | `FOREACH_REAL_DATATYPE` X-macro for eliminating copy-paste type dispatch                           |

## Usage

Most code includes this library transitively via
`milkDebugTools.h` (which pulls in `milk_compiler.h`).
Direct inclusion is only needed for the specialized
headers:

```c
#include "milkDebugTools.h"       /* always available */
#include "pixel_dispatch.h"       /* for X-macro dispatch */
#include "multiselect_parse.h"    /* for TUI selection parsing */
```

## Build Tier

Engine tier — built with all configurations, including
`-DUSE_CLI=OFF -DUSE_COREMODS=OFF`.
