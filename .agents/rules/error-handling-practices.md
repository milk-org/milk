# Error Handling Practices

The `milk` project standardizes its error handling and logging via macros defined in `src/engine/libmilkdata/milkDebugTools.h`. To ensure consistency, robustness, and ease of debugging across the codebase, all C source code must adhere to the following practices.

## 1. Unified Logging Macros
- **Do not use** raw `printf()`, `fprintf(stderr, ...)`, or `perror()` for error logging.
- **Always use** the standard logging macros from `milkDebugTools.h`:
  - `PRINT_ERROR(format, ...)`: Logs an error message.
  - `PRINT_WARNING(format, ...)`: Logs a warning message.
  - `PRINT_INFO(format, ...)`: Logs an informational message.
- These macros automatically append the `__FILE__`, `__LINE__`, and `__func__`, ensuring that the log outputs are highly traceable.

## 2. Standardized Return Codes
- **Standard values:** Functions that return a status code must use `RETURN_SUCCESS` (`0`) and `RETURN_FAILURE` (`1`), which are defined in `milkDebugTools.h`.
- **Helper macros:**
  - Opt for `FUNC_RETURN_SUCCESS()` when successfully returning from a function.
  - Opt for `FUNC_RETURN_FAILURE(format, ...)` when returning an error state from a function; this will internally call `PRINT_ERROR` and then return `RETURN_FAILURE`.

## 3. Propagation and Boilerplate Reduction
- **Avoid** manual `if (err != RETURN_SUCCESS)` blocks when simply propagating an error code.
- **Use:**
  - `FUNC_CHECK_RETURN(ret)`: Evaluates the return code and silently returns it if it is not `RETURN_SUCCESS`.
  - `FUNC_CHECK_RETURN_PRINT(ret, format, ...)`: Evaluates the return code, logs an error via `PRINT_ERROR` if it fails, and then returns the code.

## 4. System Calls and `errno`
- **Avoid** calling `perror()` after a failed system or standard library call.
- **Use:** Instead, use `PRINT_ERROR("... failed: %s", strerror(errno))` (or your desired format string) to ensure the error log retains the required file/line/function context.
- **Execution:** When executing shell commands via `system()`, do not check the return code manually. Instead, use `EXECUTE_SYSTEM_COMMAND_ERRCHECK(format, ...)` which wraps the execution and error logging automatically.

## 5. Transition Strategy
- **New Code:** All newly written or refactored C code must strictly use these macros.
- **Migration:** When editing existing files that currently use `fprintf(stderr, ...)`, progressively refactor those specific lines or functions to use `PRINT_ERROR(...)`.
