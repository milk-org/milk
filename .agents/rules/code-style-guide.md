---
trigger: always_on
---

- Use code blocks to reduce the scope of variables as much as possible.
- Keep lines short, no more than 80 char
- Function prototypes with arguments should be multi-line, with one line per argument
- Document functions using Kernel-Doc style
- Use the Linux kernel's C coding style if it doesn't conflict with the above rules
- Document the purpose and overall approach of a function above the function code in the .c file, and the mode detailed methodoly within the function
- Document briefly the function purpose in the .h file
- Enable and enforce compiler warnings (`-Wall`, `-Wextra`) during development to catch missing declarations early. Treat them as errors (`-Werror`) in CI/CD.
- Make sure every `.c` file strictly includes the exact headers it relies on, rather than implicitly relying on another header to include them (e.g. relying on `CLIcore.h` to provide `math.h` or `stdlib.h`).
- Use `restrict` on pointer parameters to pixel/array data in compute-heavy functions where pointers are guaranteed non-aliasing. See `performance-practices.md` for full performance guidelines.
