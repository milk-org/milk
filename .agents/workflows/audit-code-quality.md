---
description: Audit code for readability, simplicity, duplication, file length, and removable dependencies
---

# Audit Code Quality

Run this workflow when asked to review or audit code quality, or when you notice a file has grown too large and complex.

## Checklist

1. **Review File Length and Organization**:
   - Check if the file is excessively long. If so, consider splitting it using the `refactor-c-source` skill.
   - Check if any function is excessively long. Extract complex or duplicated logic into smaller helper functions.

2. **Review Readability, Simplicity, and Code Style**:
   - Verify general adherence to all rules in `code-style-guide.md`.
   - Look for deeply nested blocks (`if` inside `for` inside `if`). Refactor using early exits.
   - Verify variables are declared in the narrowest possible scope (using `{ }` code blocks).
   - Ensure lines are under the 100-character limit.
   - **Check closing comments**: Ensure any scope longer than ~10 lines has a closing comment (e.g., `} // if (condition)` or `#endif // MACRO`).

3. **Check Naming and Types** (see `naming-conventions.md`):
   - Verify variables are appropriately scoped in length and avoid single-letter names outside trivial loops.
   - Ensure loop indices use doubled letters (`ii`, `jj`, `kk`) and their type strictly matches the bound type (e.g., `uint32_t` for `xsize`) to prevent vectorization failures.
   - Ensure correct abbreviation use (e.g. `img`, `shm`, `fps`) and `UPPER_CASE` for macros/structs.

4. **Check for Code Duplication**:
   - Find structurally identical logic or copy-pasted code. Factorize it into shared macros or `static inline` functions.

5. **Audit Dependencies and Performance**:
   - Ensure the file strictly includes only what it uses. Check for implicit dependencies and remove unused headers.
   - Cross-check `#include` and cross-module dependencies against `docs/dependency_graph.md`.
   - Identify dependencies that can be simplified or decoupled.
   - Verify the code aligns with `performance-practices.md` (e.g., using `MILK_RESTRICT`, avoiding allocations in hot paths).

6. **Implementation**:
   - If large refactoring is needed, create an `implementation_plan.md` first.
   - Ensure all changes follow:
     - `code-style-guide.md`
     - `naming-conventions.md`
     - `performance-practices.md`
     - `architecture-principles.md`
   - Run the `/compile-test` workflow to verify your changes.
