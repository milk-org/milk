---
description: Standardized error handling and logging
  via milkDebugTools.h macros.
---

# Error Handling Practices

The `milk` project standardizes its error handling and
logging via macros defined in
`src/engine/libmilkcommon/milkDebugTools.h`. To ensure
consistency, robustness, and ease of debugging across
the codebase, all C source code must adhere to the
following practices.

See also: `library-vs-application-error-handling.md`
(library/application boundary, return-type discipline,
`EXECUTE_SYSTEM_COMMAND` default, standalone CLI exit
codes); `cli-error-help.md` (CLIcore arg-parse error
display).

## 1. Unified Logging Macros

- **Do not use** raw `printf()`, `fprintf(stderr, ...)`,
  or `perror()` for error logging.
- **Always use** the standard logging macros from
  `milkDebugTools.h`:
  - `PRINT_ERROR(format, ...)`: Logs an error message.
  - `PRINT_WARNING(format, ...)`: Logs a warning message.
  - `PRINT_INFO(format, ...)`: Logs an informational
    message.
- These macros automatically append the `__FILE__`,
  `__LINE__`, and `__func__`, ensuring that the log
  outputs are highly traceable.

## 2. Standardized Return Codes

- **Standard values:** Functions that return a status
  code must use `RETURN_SUCCESS` (`0`) and
  `RETURN_FAILURE` (`1`), which are defined in
  `milkDebugTools.h`.
- **Helper macros:**
  - Opt for `FUNC_RETURN_SUCCESS()` when successfully
    returning from a function.
  - Opt for `FUNC_RETURN_FAILURE(format, ...)` when
    returning an error state from a function; this will
    internally call `PRINT_ERROR` and then return
    `RETURN_FAILURE`.

## 3. Propagation and Boilerplate Reduction

- **Avoid** manual `if (err != RETURN_SUCCESS)` blocks
  when simply propagating an error code.
- **Use:**
  - `FUNC_CHECK_RETURN(ret)`: Evaluates the return code
    and silently returns it if it is not
    `RETURN_SUCCESS`.
  - `FUNC_CHECK_RETURN_PRINT(ret, format, ...)`:
    Evaluates the return code, logs an error via
    `PRINT_ERROR` if it fails, and then returns the
    code.
- Use the non-`PRINT` form for propagation between
  internal layers; the printing form is reserved for
  the layer that adds context (see §7 below).

## 4. System Calls and `errno`

- **Avoid** calling `perror()` after a failed system or
  standard library call.
- **Use:** Instead, use
  `PRINT_ERROR("... failed: %s", strerror(errno))` (or
  your desired format string) to ensure the error log
  retains the required file/line/function context.
- **Execution:** When executing shell commands via
  `system()`, do not check the return code manually.
  Instead, use `EXECUTE_SYSTEM_COMMAND_ERRCHECK(format,
  ...)` which wraps the execution and error logging
  automatically. (The bare `EXECUTE_SYSTEM_COMMAND`
  macro silently discards the return value and is
  treated as deprecated — see
  `library-vs-application-error-handling.md` §3.)

## 5. Transition Strategy

- **New Code:** All newly written or refactored C code
  must strictly use these macros.
- **Migration:** When editing existing files that
  currently use `fprintf(stderr, ...)`, progressively
  refactor those specific lines or functions to use
  `PRINT_ERROR(...)`.

## 6. Mandatory-Check Syscalls

The following calls MUST have their return inspected.
Minimum reaction is
`PRINT_ERROR("<call> failed: %s", strerror(errno))`
(or the equivalent diagnostic) plus propagating
failure to the caller. A bare `(void)` cast is
forbidden; if the return genuinely does not matter,
write a one-line comment justifying it.

**Required-check list:**
`open`, `close`, `read`, `write`, `lseek`,
`ftruncate`, `mmap`, `munmap`, `shm_open`,
`shm_unlink`, `sem_init`, `sem_post`, `sem_wait`,
`sem_trywait`, `pthread_create`, `pthread_join`,
`pthread_mutex_*`, `ImageStreamIO_*`, `fps_connect`,
`fps_disconnect`, all `processinfo_*` lifecycle calls.

```c
/* WRONG — return discarded */
sem_post(sem);
ImageStreamIO_openIm(&img, name);

/* RIGHT */
if (sem_post(sem) != 0)
{
    PRINT_WARNING("sem_post failed: %s",
                  strerror(errno));
}
errno_t isio_ret =
    ImageStreamIO_openIm(&img, name);
FUNC_CHECK_RETURN_PRINT(isio_ret,
    "ImageStreamIO_openIm(%s) failed", name);
```

## 7. Print-Once Policy

Errors propagate up the call stack; reports do not.
Convention:

- **Innermost layer** (the syscall site): print with
  `PRINT_ERROR` — it owns `errno`, the path, and the
  parameters — then return failure.
- **Intermediate propagation layers:** use
  `FUNC_CHECK_RETURN(ret)` (silent). Do **not** use the
  `_PRINT` variant here; it produces double / triple
  log lines for the same root cause.
- **Top-level boundary** (CLIcore command function,
  `main()`, signal handler, FPS RUN entry, fpsexec
  callbacks): MAY add one user-facing `PRINT_ERROR`
  summarising what the user was trying to do, or push
  to a TUI feedback channel such as `OV_CMDLOG`.

```c
/* GOOD — single print at the layer with context */
errno_t low(void)
{
    FUNC_RETURN_FAILURE("open: %s", strerror(errno));
}

errno_t mid(void)
{
    FUNC_CHECK_RETURN(low());   /* silent */
    return RETURN_SUCCESS;
}

errno_t top(void)
{
    FUNC_CHECK_RETURN_PRINT(mid(),
        "could not initialize stream");
    return RETURN_SUCCESS;
}
```

## 8. `ERRMODE_*` for Query-Style Functions

`ERRMODE_NULL` / `ERRMODE_WARN` / `ERRMODE_FAIL` /
`ERRMODE_ABORT` (defined in `milkDebugTools.h`
lines 41–44) is the established convention for
query-style functions whose caller knows whether a
missing result is fatal — pioneered by `resolveIMGID()`
(see `src/coremods/COREMOD_memory/imageID.h`) and used
at ~223 call sites across the codebase.

- New library functions MAY adopt the `ERRMODE_*`
  parameter when caller failure tolerance is genuinely
  caller-specific.
- Do NOT mass-add it to existing `errno_t` functions —
  return codes remain the default.
- **Caller responsibility:** do not pass
  `ERRMODE_ABORT` from a long-running supervisory loop
  (FPS RUN, real-time control). `ERRMODE_ABORT` means
  what it says — the function will call `abort()`.
  Pass `ERRMODE_FAIL` instead and handle the failure
  in your loop.

## 9. Cleanup with `goto fail;`

When a function holds two or more resources (file
descriptors, memory mappings, allocations, locks) and
may fail mid-acquisition, use the single-label
`goto fail;` cleanup pattern with reverse-order
release at the bottom. Initialise each resource
handle to its sentinel value (`-1`, `NULL`,
`MAP_FAILED`) before any `goto`, so the `fail:` block
can release unconditionally.

This is the **only approved use of `goto`** in the
codebase, matching Linux kernel convention (already
cited as the project style baseline in
`code-style-guide.md`).

```c
errno_t example_function(const char *path, size_t sz)
{
    errno_t  rv  = RETURN_FAILURE;
    int      fd  = -1;
    void    *map = MAP_FAILED;
    char    *buf = NULL;

    fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd == -1)
    {
        PRINT_ERROR("open(%s) failed: %s",
                    path, strerror(errno));
        goto fail;
    }
    if (ftruncate(fd, sz) != 0)
    {
        PRINT_ERROR("ftruncate failed: %s",
                    strerror(errno));
        goto fail;
    }
    map = mmap(NULL, sz, PROT_READ | PROT_WRITE,
               MAP_SHARED, fd, 0);
    if (map == MAP_FAILED)
    {
        PRINT_ERROR("mmap failed: %s",
                    strerror(errno));
        goto fail;
    }
    buf = malloc(sz);
    if (buf == NULL)
    {
        PRINT_ERROR("malloc(%zu) failed", sz);
        goto fail;
    }

    /* ... use resources ... */
    rv = RETURN_SUCCESS;

fail:
    if (buf != NULL)       { free(buf); }
    if (map != MAP_FAILED) { munmap(map, sz); }
    if (fd  != -1)         { close(fd); }
    return rv;
}
```
