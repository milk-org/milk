# Coding Standards

This document outlines the standard C coding conventions
expected for the `milk` project. Consistent conventions help
ensure readability, maintainability, and seamless integration
of new code or plugins.

See also: [Programmer's Guide](../programmers_guide.md) ·
[Documenting Code](DocumentingCode.md) ·
[Template Source Code](TemplateSourceCode.md) ·
[Module Files](ModuleFiles.md)

## General C Code Style

- **Linux Kernel Style:** Use the [Linux kernel's C coding style](https://www.kernel.org/doc/html/v4.10/process/coding-style.html) as the baseline for indentation and formatting, wherever it doesn't conflict with the rules below.
- **Short Lines:** Keep lines short, generally **no more than 80 characters**.
- **Scope limitation:** Use code blocks (`{ ... }`) to reduce the scope of variables as much as possible.

## Function Signatures

- **Multi-line Prototypes:** Function prototypes with arguments should be formatted across multiple lines, with exactly **one line per argument**. This greatly improves diff readability and annotation.

```c
inline static errno_t example_compute_sum(
    int a,
    int b,
    char *output
)
```

## Documentation Rules

- **Kernel-Doc Style:** Document functions using the standard Kernel-Doc (or Doxygen-compatible) style.
- **`.h` Files:** Briefly document the function's overall purpose in the header file.
- **`.c` Files:** Document the purpose and overall approach of the function *above* the function code in the source file, and detail the methodology *within* the function itself.

## Compilation and Headers

- **Compiler Warnings:** Enable and enforce compiler warnings (`-Wall`, `-Wextra`) during development to catch missing declarations early. Treat them as errors (`-Werror`) in CI/CD pipelines.
- **Explicit Includes:** Make sure every `.c` file strictly includes the exact headers it relies on. Do **not** implicitly rely on another header to include them (e.g., relying on `CLIcore.h` to provide `<math.h>` or `<stdlib.h>`).

---
← [Documentation Index](../index.md)
