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

### Quick scan command

```bash
# List .c files with undocumented functions
# (function definitions not preceded by /**)
for f in src/path/*.c; do
  count=$(grep -cP '^\w[\w\s\*]*\(' "$f" \
    | head -1)
  documented=$(grep -cB1 '^\w[\w\s\*]*\(' "$f" \
    | grep -c '/\*\*')
  echo "$f: $count functions, $documented documented"
done
```

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

```c
/**
 * function_name - Brief one-line description
 * @param1:  Description of first parameter
 * @param2:  Description of second parameter
 *
 * Detailed description of what the function does,
 * why it exists, and any important algorithmic
 * notes. Keep lines ≤ 100 characters.
 *
 * Context: any important usage context (e.g.,
 * "called from the main loop", "must hold lock").
 *
 * Return: Description of return value, or "void"
 */
```

### Content guidelines

- **Brief line**: verb phrase describing the
  action (`"Parse arithmetic expression"`,
  `"Register CLI command for module"`)
- **Parameters**: describe purpose and valid
  ranges, not just type restating
- **Body**: explain the *why* and *how*, not
  just restate what the code literally does
- **Return**: document success/failure semantics,
  special return values
- **Don't document the obvious**: skip trivial
  getters/setters unless they have side effects

### Categories with templates

**Initialization function:**
```c
/**
 * module_init - Initialize the FOO subsystem
 * @config:  Configuration parameters
 *
 * Allocates internal state, registers CLI
 * commands, and connects to shared memory
 * streams. Must be called before any other
 * FOO function.
 *
 * Return: RETURN_SUCCESS or RETURN_FAILURE
 */
```

**Compute / FPS exec function:**
```c
/**
 * fpsexec - Execute the BAR computation
 *
 * Core computation function called by the FPS
 * framework. Parameters are synced from shared
 * memory before each invocation.
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

**CLI registration function:**
```c
/**
 * CLIADDCMD_module__funcname - Register CLI cmd
 *
 * Registers the "module.funcname" command with
 * the CLI framework. Called from initModule()
 * during module loading.
 *
 * Return: CLI registration error code
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
3. **Between batches**: commit the changes so
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
