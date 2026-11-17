---
trigger: always_on
---

Follow [Coding Standards](../../docs/developer/coding_standards.md); the pre-commit hook formats C
with `clang-format` and `.clang-format` ([Tooling](../../docs/developer/tooling.md) §3). In
addition:

- The 100-character line limit also applies to agent `.md` files.
- **Do not artificially split lines**: Avoid unnecessary vertical fragmentation. If an expression, variable assignment, or function call fits entirely within the 100-character limit, it must be written on a single line. Do not split short expressions across multiple lines, unless mandated by function prototype rules.
- Factorize code as much as possible. Copy-paste coding
  is a red flag — extract shared logic into helper
  functions or macros instead of duplicating it.
- Loop indices should always be declared within the `for` statement
  (e.g., `for (int ii = 0; ...)`), unless they are meant to be
  used outside the loop. If they are used outside the loop,
  add a comment explaining why they were declared outside.
- See `error-handling-practices.md` §9 for the `goto fail`
  cleanup pattern.
- Do not include headers that the file does not use.
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

- Use `restrict` on pointer parameters to pixel/array
  data in compute-heavy functions where pointers are
  guaranteed non-aliasing. See `performance-practices.md`
  for full performance guidelines.
- **Non-ASCII characters** are only permitted in **TUI
  display code** (e.g., `overview`, `procCTRL`,
  `fpsCTRL`) for box-drawing symbols, status indicators,
  and progress bars. All other source files, headers,
  comments, and string literals must use ASCII only.
