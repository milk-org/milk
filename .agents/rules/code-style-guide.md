---
trigger: always_on
---

- Use code blocks (`{ }`) to reduce the scope of variables as much as possible.
  Use this as an opportunity to add a comment immediately preceding the block
  explaining what the block is doing.
- Keep lines short, no more than 100 characters.
  This limit applies to both C source and agent `.md` files.
- Factorize code as much as possible. Copy-paste coding
  is a red flag — extract shared logic into helper
  functions or macros instead of duplicating it.
- Loop indices should always be declared within the `for` statement
  (e.g., `for (int ii = 0; ...)`), unless they are meant to be
  used outside the loop. If they are used outside the loop,
  add a comment explaining why they were declared outside.
- Function prototypes with arguments should be multi-line,
  with one line per argument.
- Document functions using Kernel-Doc style.
- Put a brief API-facing description in the `.h` file for
  declarations, and place the full Kernel-Doc comment above
  the function definition in the `.c` file.
- In `.c` files, Kernel-Doc may also include implementation
  details when useful (algorithm notes, design rationale,
  non-obvious logic), but the `.h` file should remain brief.
- Follow the Linux kernel's C coding conventions for
  naming, scope, and control flow philosophy (early
  exit, `goto fail` cleanup). See
  `error-handling-practices.md` §9 for the `goto`
  pattern.
- **Brace style: Allman.** Place the opening brace
  on its own line for both function definitions and
  control-flow statements (`if`, `for`, `while`,
  `do`, `switch`). This is the dominant style in
  the existing codebase (~77% of control blocks).
  Use the brace style already present in a file
  when editing; use Allman for new files.
  ```c
  /* Function definition */
  static errno_t compute_stream(
      const float *restrict in,
      float       *restrict out,
      uint64_t nelement)
  {
      /* Control flow — brace on next line */
      if (in == NULL)
      {
          return RETURN_FAILURE;
      }

      for (uint64_t ii = 0; ii < nelement; ii++)
      {
          out[ii] = in[ii] * 2.0f;
      }

      return RETURN_SUCCESS;
  }
  ```
- Enable and enforce compiler warnings (`-Wall`, `-Wextra`)
  during development to catch missing declarations early.
  Treat them as errors (`-Werror`) in CI/CD.
- Make sure every `.c` file strictly includes the exact
  headers it relies on, rather than implicitly relying on
  another header to include them (e.g. relying on
  `CLIcore.h` to provide `math.h` or `stdlib.h`).
- Conversely, do not include headers that the file does
  not use. Remove redundant `#include` directives —
  headers already provided transitively by a required
  include should not be listed again.
- Add a closing comment to any scope longer than
  ~10 lines:
  ```c
  #ifdef HAVE_CUDA
  // ... many lines ...
  #endif // HAVE_CUDA

  if (condition_met)
  {
      // ... many lines ...
  } // if (condition_met)
  ```
- Prefer early exit over deeply nested braces. The
  preferred control flow is the one that minimizes
  indentation:
  ```c
  /* GOOD — early exit */
  if (ptr == NULL)
  {
      return RETURN_FAILURE;
  }
  // main logic at low indentation ...

  /* BAD — unnecessary nesting */
  if (ptr != NULL)
  {
      // main logic indented ...
  }
  ```
- Use `restrict` on pointer parameters to pixel/array
  data in compute-heavy functions where pointers are
  guaranteed non-aliasing. See `performance-practices.md`
  for full performance guidelines.
- **Non-ASCII characters** are only permitted in **TUI
  display code** (e.g., `streamCTRL`, `overview`,
  `fpsCTRL`) for box-drawing symbols, status indicators,
  and progress bars. All other source files, headers,
  comments, and string literals must use ASCII only.
