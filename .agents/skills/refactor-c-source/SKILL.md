---
name: refactor-c-source
description: Safely split and reorganize large C source
  files into smaller, logically grouped modules
---

# Refactor C Source Files

This skill guides the agent through safely splitting
a large `.c` file into multiple smaller files while
preserving all functionality, updating the build
system, and verifying correctness.

## When to Use

- A `.c` file exceeds ~800 lines and contains
  logically distinct groups of functions
- A file mixes concerns (e.g., UI + logic + parsing)
- Functions in the file can be grouped by theme,
  data structure, or subsystem

## Phase 1 — Analyze the File

1. **Read the entire file** and list every function
   defined in it.

2. **Build a dependency map**: for each function,
   note which other functions in the same file it
   calls, and which external headers it needs.

3. **Identify logical groups**. Common boundaries:
   - Initialization / cleanup
   - Core computation / algorithm
   - I/O / serialization
   - CLI registration / help
   - UI / display
   - Parsing / tokenization
   - Utility / helper functions

4. **Propose the split** to the user as a table:

   | New file | Functions | Rationale |
   |----------|-----------|-----------|
   | `foo_init.c` | `foo_init()`, `foo_cleanup()` | Lifecycle |
   | `foo_compute.c` | `foo_run()`, `foo_step()` | Core logic |
   | `foo_display.c` | `foo_print()`, `foo_tui()` | UI rendering |

5. **Check for static functions**. Static functions
   that are only used within their new group stay
   `static`. Static functions called by another
   group must be promoted to non-static and given
   a prototype in the corresponding `.h` file.

## Phase 2 — Create the New Files

For each new `.c` file:

1. **Create the `.c` file** with:
   - The copyright/license header (copy from the
     original file)
   - Only the `#include` directives that the
     functions in this file actually use — do not
     blindly copy all includes from the original
   - The functions belonging to this group
   - Kernel-Doc comments preserved above each
     function

2. **Create a matching `.h` file** with:
   - Include guard (`#ifndef FILENAME_H` /
     `#define FILENAME_H`)
   - Brief Kernel-Doc for each public function
   - Function prototypes (multi-line, one arg per
     line, per project style)
   - Only the type includes needed for the
     prototypes (e.g., `#include <stdint.h>`)

3. **Update the original file**:
   - Remove the moved functions
   - Add `#include "new_file.h"` if the original
     file still calls any of the moved functions
   - If the original file becomes empty (all
     functions moved out), delete it entirely

## Phase 3 — Update the Build System

1. **Edit `CMakeLists.txt`** in the target
   directory:
   - Add each new `.c` file to `SOURCEFILES`
   - Remove any deleted `.c` files from
     `SOURCEFILES`
   - Do NOT add `.h` files to `SOURCEFILES`

2. **Check for install headers**: if any of the
   new `.h` files define public API used by other
   modules (not just internal helpers), add them
   to `INSTALL_HEADERS` in the `CMakeLists.txt`.

## Phase 4 — Verify

1. Run the `/compile-test` workflow.

2. Verify there are **zero new warnings** about:
   - Implicit function declarations (missing
     include)
   - Unused functions (function moved but still
     declared)
   - Multiple definitions (function not removed
     from original)

3. If the module has tests, run them.

4. If the module is part of `CLIcore`, also run
   the `/cli-robustness-test` workflow to verify
   no regressions.

## Common Pitfalls

- **Circular includes**: if `a.h` includes `b.h`
  and `b.h` includes `a.h`, factor the shared
  types into a separate `types.h`.
- **Static globals**: static file-scope variables
  accessed by functions in multiple new files must
  be refactored (passed as parameters or moved to
  a shared struct).
- **Macro definitions**: if the original file
  defines macros used by multiple groups, move them
  to the `.h` file or a shared header.
- **Include order**: project convention is:
  1. Own header (`"myfile.h"`)
  2. Project headers (`"CLIcore.h"`, `"fps.h"`)
  3. System headers (`<stdio.h>`, `<math.h>`)
