---
name: batch-kernel-doc
description: Systematically add Kernel-Doc comments
  to undocumented C functions across the codebase
---

# Batch Kernel-Doc Documentation

This skill guides the agent through efficiently
documenting large numbers of undocumented C
functions with Kernel-Doc style comments.

## When to Use

- Systematic documentation pass across a module
  or subsystem
- Post-refactoring cleanup (new files need docs)
- User asks to "document all functions in X"

## Phase 1 — Scan for Undocumented Functions

Find functions that lack a `/** ... */` comment
block immediately above them.

### Manual scan approach

For each `.c` file:

1. Read the file
2. Identify every function definition (non-static
   and static)
3. Check if there is a `/** ... */` block on the
   lines immediately above
4. Record undocumented functions in a checklist

### Prioritize files

Order by impact:

1. Public API functions (prototypes in `.h` files)
2. Recently refactored or new files
3. Complex functions with non-obvious behavior
4. Simple utility functions (lowest priority)

## Phase 2 — Write Documentation

### Kernel-Doc format

Follow [Documenting Code
§3.1](../../../docs/developer/DocumentingCode.md#31-standard-function-documentation):
`name() - summary`, one `@arg:` line per parameter, a description, then `Return:`.

### Content guidelines

- **Brief line**: verb phrase describing the
  action (`"Parse arithmetic expression"`,
  `"Build response matrix"`)
- **Parameters**: describe purpose and valid
  ranges, not just type restating
- **Body**: explain the _why_ and _how_, not
  just restate what the code literally does
- **Return**: document success/failure semantics,
  special return values
- **Don't document the obvious**: skip trivial
  getters/setters unless they have side effects

### Categories with templates

**Initialization function:**

```c
/**
 * module_init() - Initialize the FOO subsystem
 * @config:  Configuration parameters
 *
 * Allocates internal state and connects to
 * shared memory streams. Must be called before
 * any other FOO function.
 *
 * Return: RETURN_SUCCESS or RETURN_FAILURE
 */
```

**Compute function (`MILK_RUNNABLE` unit):**

```c
/**
 * bar_compute() - Execute the BAR computation
 * @self:  Runnable descriptor of the unit
 *
 * Compute function of the BAR unit. Parameters
 * are synced from shared memory before each
 * invocation.
 *
 * Algorithm: [brief description of the method]
 *
 * Performance: this function is marked MILK_HOT
 * and uses restrict-qualified pointers for
 * vectorization.
 *
 * Return: RETURN_SUCCESS or RETURN_FAILURE
 */
```

## Phase 3 — Batch Processing

Process files in batches to balance efficiency
with correctness:

1. **Batch size**: 3–5 files per iteration
2. **Within each batch**:
   a. Add docs to all undocumented functions
   b. Run `/compile-test` to verify no
   syntax errors in the doc comments
   c. Check that no functions were accidentally
   modified (only comments added)
3. **Between batches**: commit the changes on a
   feature branch, never on `dev` or
   `framework-dev` (`git-workflow` rule), so
   that failures in a later batch don't lose
   earlier work
4. **Header files**: after documenting `.c`
   functions, add brief one-line descriptions
   to matching `.h` prototypes (if not already
   present)

## Phase 4 — Verification

1. Compile successfully (docs are syntactically
   valid C comments)
2. Verify no code changes snuck in — `git diff`
   should show only comment additions
3. For large batches, spot-check that descriptions
   are accurate by reading the function body

## Common Pitfalls

- **Stale docs**: don't copy descriptions from
  neighboring functions — read each function's
  actual implementation
- **Line length**: doc comment lines must respect
  the 100-character limit
- **Missing `@param`**: every parameter must be
  documented, even obvious ones like `name`
- **Static functions**: document them too, but
  they only need the `.c` file comment (no `.h`
  entry)
- **Multi-line prototypes**: the `/**` block goes
  above the return type, not above the function
  name
