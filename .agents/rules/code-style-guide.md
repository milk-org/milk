---
trigger: always_on
---

- Use code blocks to reduce the scope of variables as much as possible.
- Keep lines short, no more than 100 characters.
  This limit applies to both C source and agent `.md` files.
- Factorize code as much as possible. Copy-paste coding
  is a red flag — extract shared logic into helper
  functions or macros instead of duplicating it.
- Function prototypes with arguments should be multi-line,
  with one line per argument.
- Document functions using Kernel-Doc style.
- Use the Linux kernel's C coding style if it doesn't
  conflict with the above rules.
- Document the function's **purpose** in the `.h` file.
  In `.c` files, document only **implementation details**
  (algorithm notes, design rationale, non-obvious logic).
- Enable and enforce compiler warnings (`-Wall`, `-Wextra`)
  during development to catch missing declarations early.
  Treat them as errors (`-Werror`) in CI/CD.
- Make sure every `.c` file strictly includes the exact
  headers it relies on, rather than implicitly relying on
  another header to include them (e.g. relying on
  `CLIcore.h` to provide `math.h` or `stdlib.h`).
- Add a closing comment to any scope longer than ~10 lines:
  ```c
  #ifdef HAVE_CUDA
  // ... many lines ...
  #endif // HAVE_CUDA

  if (condition_met) {
      // ... many lines ...
  } // if (condition_met)
  ```
- Prefer early exit over deeply nested braces. The
  preferred control flow is the one that minimizes
  indentation:
  ```c
  /* GOOD — early exit */
  if (ptr == NULL) {
      return RETURN_FAILURE;
  }
  // main logic at low indentation ...

  /* BAD — unnecessary nesting */
  if (ptr != NULL) {
      // main logic indented ...
  }
  ```
- Use `restrict` on pointer parameters to pixel/array
  data in compute-heavy functions where pointers are
  guaranteed non-aliasing. See `performance-practices.md`
  for full performance guidelines.
