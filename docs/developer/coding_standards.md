# Coding Standards

This document outlines the standard C coding conventions expected for the `milk` project. Consistent
conventions help ensure readability, maintainability, and seamless integration of new code or
plugins.

See also: [Programmer's Guide](../arch/programmers_guide.md) · [Documenting Code](DocumentingCode.md) ·
[Adding Plugins](plugins.md)

## General C Code Style

- **Brace Style (Allman):** Place the opening brace `{` on its own line for both function
  definitions and control-flow statements (`if`, `for`, `while`, `do`, `switch`), as enforced by the
  repository `.clang-format`.
- **Linux Kernel Conventions:** Follow Linux kernel C conventions for naming, variable scope, and
  control flow (`goto` cleanup, early exit to minimize indentation).
- **Short Lines:** Keep lines short, **no more than 100 characters**.
- **Scope Limitation:** Use code blocks (`{ ... }`) to reduce the scope of variables as much as
  possible, with a brief comment before the block explaining its purpose.

## Function Signatures

- **Multi-line, Column-Aligned Prototypes:** Function prototypes with arguments are formatted across
  multiple lines, with **one line per argument** and parameter names column-aligned:

```c
static inline errno_t example_compute_sum(
    int   a,
    int   b,
    char *output)
{
```

## Documentation Rules

- **Kernel-Doc Style:** Document functions using the standard Kernel-Doc (or Doxygen-compatible)
  style.
- **`.h` Files:** Briefly document the function's overall purpose in the header file.
- **`.c` Files:** Document the purpose and overall approach of the function _above_ the function
  code in the source file, and detail the methodology _within_ the function itself.

## Compilation and Headers

- **Compiler Warnings:** Enable and enforce compiler warnings (`-Wall`, `-Wextra`) during
  development to catch missing declarations early. Treat them as errors (`-Werror`) in CI/CD
  pipelines.
- **Explicit Includes:** Make sure every `.c` file strictly includes the exact headers it relies on.
  Do **not** implicitly rely on another header to include them (e.g., relying on `CLIcore.h` to
  provide `<math.h>` or `<stdlib.h>`).

---

← [Documentation Index](../index.md)
