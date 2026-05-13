---
description: Architectural error-handling boundary —
  library code returns failures; only application
  entry points may terminate the process.
---

# Library vs. Application Error Handling

This rule complements `error-handling-practices.md`
(logging macros, return codes, cleanup) and
`cli-error-help.md` (CLIcore arg-parse error display).
Where those documents cover *how* to report and
propagate errors, this one covers *who* may end the
process and *what* a function's signature must look
like.

## 1. The Boundary: Who May Call `exit()` / `abort()`

- **Library code MUST NOT call `exit()` or `abort()`.**
  This includes everything under `src/coremods/`,
  `src/engine/lib*/`, `ImageStreamIO/`, and any
  function reachable from a CLIcore command.
  Failures must propagate as return codes.
- `exit()` is permitted only inside `main()` of
  standalone binaries, or in a function called
  exclusively from `main()` whose name makes its
  terminal nature explicit (e.g. `usage_exit()`).
- `abort()` is permitted only as the body of an
  internal-invariant assertion that should crash a
  debug build. Wrap it in `#ifndef NDEBUG`, or use the
  forthcoming `MILK_BUG(fmt, ...)` macro (§5 below).

```c
/* WRONG — library code (processinfo_shm_create.c) */
SM_fd = open(SM_fname,
             O_RDWR | O_CREAT | O_TRUNC, FILEMODE);
if (SM_fd == -1)
{
    perror("Error opening file for writing");
    exit(0);    /* kills caller; reports SUCCESS */
}

/* RIGHT — propagate via the shared cleanup label */
SM_fd = open(SM_fname,
             O_RDWR | O_CREAT | O_TRUNC, FILEMODE);
if (SM_fd == -1)
{
    PRINT_ERROR("open(%s) failed: %s",
                SM_fname, strerror(errno));
    goto fail;
}
```

The `exit(0)` form is the worst possible variant: it
terminates the host process while reporting success
to whatever launched it. Library code that does this
silently breaks every supervisor and every CI runner.

## 2. Return-Type Discipline

Three approved function shapes:

1. **Status-only:** `errno_t f(...)` returning
   `RETURN_SUCCESS` / `RETURN_FAILURE`. The default
   for procedural functions.
2. **Value-or-error:** the out-parameter idiom —
   `errno_t f(int input, long *out)`. **Required**
   when the natural return type can collide with an
   error sentinel. Functions that return
   `long pindex` or `-1` from the same return slot
   (today's `processinfo_shm_list_create()` is the
   canonical counter-example) violate this rule.
3. **Pointer-returning:** MAY return `NULL` on
   failure; the function MUST call `PRINT_ERROR`
   itself before returning `NULL` (the caller has no
   other channel for context).

```c
/* GOOD — value and error are separate */
errno_t processinfo_shm_list_create(long *pindex_out);

/* DISCOURAGED — sentinel collision risk */
long processinfo_shm_list_create(void);
/* returns -1 on error */
```

`IMAGESTREAMIO_*` codes (in `ImageStreamIO/`) are
grandfathered as a richer error taxonomy. Functions
that return them must still be declared `errno_t`,
and the codes must satisfy
`code >= IMAGESTREAMIO_FAILURE`.

## 3. `EXECUTE_SYSTEM_COMMAND` — Default Is Checked

- `EXECUTE_SYSTEM_COMMAND(format, ...)` checks the
  `system()` return and logs failures via
  `PRINT_ERROR`. Use this by default.
- `EXECUTE_SYSTEM_COMMAND_NOCHECK(format, ...)` is the
  opt-in silent variant. Use it only when the exit
  status genuinely doesn't matter (e.g. a `tmux
  kill-session` whose absence is expected) and document
  the reason in a one-line comment above the call.
- New code reaches for the checked default. Existing
  `_NOCHECK` call sites should be audited
  opportunistically: convert to the checked default if
  failure is meaningful, leave the `_NOCHECK` form with
  a justifying comment otherwise.

## 4. Standalone CLI Exit Codes

For executables with their own `main()` (`milkCTRL`,
`milk-stream-graph`, `milk-stream-info`,
`milk-procinfo-info`, `milk-fps-*` wrappers,
`milk-streamCTRL`, `milk-fpsCTRL`, `milk-procCTRL`):

| Code | Meaning                                      |
|------|----------------------------------------------|
| `0`  | Success                                      |
| `1`  | Runtime error (file not found, IPC failure) |
| `2`  | Usage error (bad args, missing required)    |

Codes ≥3 may carry program-specific meaning,
documented in the binary's `--help` output.

CLIcore commands continue to use the codes from
`cli-error-help.md` (`RETURN_CLICHECKARGARRAY_*`).
This rule applies only to standalone binaries with
their own `main()`.

## 5. `MILK_BUG(fmt, ...)` — Intended Invariant Macro

- **Today:** none — `abort()` is used directly at
  ~10 sites across `ImageStreamIO.c` and
  `processinfo_exec_end.c`, with no documented
  contract about when it is allowed.
- **Documented intent (not yet implemented):**
  introduce `MILK_BUG(fmt, ...)` that expands to
  `PRINT_ERROR(...) + abort()` in debug builds and
  `PRINT_ERROR(...) + return RETURN_FAILURE` (or
  `goto fail;`-style propagation) in release builds.
  When introduced it will replace every current
  `abort()` site.
- **Rule until it lands:** new code uses neither
  `abort()` nor `assert()`. If you find yourself
  reaching for one, return through
  `FUNC_RETURN_FAILURE` instead and document the
  invariant in a code comment above the check.
