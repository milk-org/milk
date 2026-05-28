---
name: fps-parameter-guide
description: Deep reference for FPS parameter types,
  flags, callbacks, and common configuration patterns
---

# FPS Parameter Guide

This skill provides comprehensive guidance on
configuring FPS (Function Processing System)
parameters in compute units. Essential for
correctly binding C variables to shared memory
parameters.

## When to Use

- Creating a new FPS compute unit
- Adding parameters to an existing function
- Debugging parameter sync issues
- Understanding `FPTYPE_*` and `FPFLAG_*` constants

## The FPS_PARAMS X-Macro

Parameters are declared using the `FPS_PARAMS`
X-macro in section 3 of the V2 layout:

```c
#define FPS_PARAMS(X)                              \
    X(".in_name", in_name, FPTYPE_STREAMNAME, 1,   \
      FPFLAG_DEFAULT_INPUT, "Input stream")         \
    X(".out_name", out_name, FPTYPE_STREAMNAME, 0,  \
      FPFLAG_DEFAULT_OUTPUT, "Output stream")       \
    X(".gain", &gain, FPTYPE_FLOAT64, 1,            \
      FPFLAG_DEFAULT_INPUT                          \
      | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT,         \
      "Loop gain")
```

### X-Macro Fields

| Position | Field       | Description                            |
| -------- | ----------- | -------------------------------------- |
| 1        | Keyword     | FPS parameter name (`.prefix`)         |
| 2        | Variable    | Pointer to local C variable            |
| 3        | Type        | `FPTYPE_*` constant                    |
| 4        | is_primary  | 1 if primary CLI argument, 0 otherwise |
| 5        | Flags       | `FPFLAG_*` bitfield                    |
| 6        | Description | Human-readable description             |

## Parameter Types (`FPTYPE_*`)

| Type                       | C Variable                           |
| -------------------------- | ------------------------------------ |
| `FPTYPE_INT32`             | `int32_t`                            |
| `FPTYPE_UINT32`            | `uint32_t`                           |
| `FPTYPE_INT64`             | `int64_t`                            |
| `FPTYPE_UINT64`            | `uint64_t`                           |
| `FPTYPE_FLOAT32`           | `float`                              |
| `FPTYPE_FLOAT64`           | `double`                             |
| `FPTYPE_STRING`            | `char[FUNCTION_PARAMETER_STRMAXLEN]` |
| `FPTYPE_ONOFF`             | `int32_t`                            |
| `FPTYPE_FILENAME`          | `char[FUNCTION_PARAMETER_STRMAXLEN]` |
| `FPTYPE_FITSFILENAME`      | `char[FUNCTION_PARAMETER_STRMAXLEN]` |
| `FPTYPE_EXECFILENAME`      | `char[FUNCTION_PARAMETER_STRMAXLEN]` |
| `FPTYPE_DIRNAME`           | `char[FUNCTION_PARAMETER_STRMAXLEN]` |
| `FPTYPE_STREAMNAME`        | `char[FUNCTION_PARAMETER_STRMAXLEN]` |
| `FPTYPE_FPSNAME`           | `char[FUNCTION_PARAMETER_STRMAXLEN]` |
| `FPTYPE_PID`               | `pid_t`                              |
| `FPTYPE_TIMESPEC`          | `struct timespec`                    |
| `FPTYPE_PROCESS`           | `char[FUNCTION_PARAMETER_STRMAXLEN]` |
| `FPTYPE_STRING_NOT_STREAM` | `char[FUNCTION_PARAMETER_STRMAXLEN]` |

### String-Type Parameters

For any `FPTYPE_STRING` (or string subtypes like
`FPTYPE_STREAMNAME`), use a `char` array with
default value and pass directly in the X-macro:

```c
// Section 2 — local variables (RECOMMENDED)
static char in_name[FUNCTION_PARAMETER_STRMAXLEN]
    = "default_stream";

// Section 3 — FPS_PARAMS (no & — array decays)
X(".in_name", in_name, FPTYPE_STREAMNAME, 1,
  FPFLAG_DEFAULT_INPUT, "Input stream name")
```

> [!NOTE]
> **Legacy pattern**: Older code uses `char *ptr`
> with `&ptr` in the X-macro. This works because
> `sync_fps_to_local()` overwrites the pointer to
> reference FPS shared memory. The `char[]` pattern
> above is preferred for new code because it
> provides a safe default value and works in both
> FPS and direct-CLI modes.

### Numeric Parameters

For numeric types, use the matching C type and
pass `&variable`:

```c
// Section 2
static double gain;
static int64_t nbiter;

// Section 3
X(".gain", &gain, FPTYPE_FLOAT64, 1,
  FPFLAG_DEFAULT_INPUT, "Loop gain")
X(".NBiter", &nbiter, FPTYPE_INT64, 0,
  FPFLAG_DEFAULT_INPUT,
  "Number of iterations")
```

## Parameter Flags (`FPFLAG_*`)

### CLI Input Flags

| Flag                       | Effect                                                         |
| -------------------------- | -------------------------------------------------------------- |
| `FPFLAG_DEFAULT_INPUT`     | Standard CLI input parameter                                   |
| `FPFLAG_DEFAULT_OUTPUT`    | Standard output parameter                                      |
| `FPFLAG_PRIMARY_CLI_INPUT` | **Counted in `nbarg`** — marks this as a required CLI argument |

> [!CAUTION]
> If `FPFLAG_PRIMARY_CLI_INPUT` is missing from a
> parameter that should be a CLI argument, `nbarg`
> will be wrong and may cause a segfault during
> argument parsing.

`FPFLAG_DEFAULT_INPUT` is a convenience macro that
includes `FPFLAG_PRIMARY_CLI_INPUT`. Always use
`FPFLAG_DEFAULT_INPUT` unless you have a specific
reason not to.

### Limit Flags

| Flag              | Effect                |
| ----------------- | --------------------- |
| `FPFLAG_MINLIMIT` | Enforce minimum value |
| `FPFLAG_MAXLIMIT` | Enforce maximum value |

When using limit flags, set the limits after
parameter creation in the `fpsexec()` function
or in a `customCONFcheck()`.

### Display Flags

| Flag                 | Effect                           |
| -------------------- | -------------------------------- |
| `FPFLAG_WRITERUN`    | Allow modification while running |
| `FPFLAG_WRITECONF`   | Allow modification during config |
| `FPFLAG_WRITESTATUS` | Show in status display           |

### Stream-Specific Flag Combos

For stream parameters, use these convenience
macros instead of bare `FPFLAG_DEFAULT_INPUT`:

| Macro                           | Includes                                                | Use When                                    |
| ------------------------------- | ------------------------------------------------------- | ------------------------------------------- |
| `FPFLAG_DEFAULT_INPUT_STREAM`   | `DEFAULT_INPUT` + `STREAM_RUN_REQUIRED` + `CHECKSTREAM` | Input stream that must exist at runtime     |
| `FPFLAG_DEFAULT_TRIGGER_STREAM` | `DEFAULT_INPUT_STREAM` + `TRIGGER_STREAM`               | Input stream that triggers the compute loop |
| `FPFLAG_DEFAULT_OUTPUT_STREAM`  | `DEFAULT_INPUT` + `CHECKSTREAM`                         | Output stream with runtime checking         |
| `FPFLAG_DEFAULT_STATUS`         | `ACTIVE` + `USED` + `VISIBLE`                           | Read-only status display parameter          |

For stream processors, use
`FPFLAG_DEFAULT_TRIGGER_STREAM` on the primary
input stream (the one driving the loop) and
`FPFLAG_DEFAULT_INPUT` on the output name
parameter.

## Common Patterns

### Input/output stream pair

```c
#define FPS_PARAMS(X)                          \
    X(".in_name", in_name,                     \
      FPTYPE_STREAMNAME, 1,                    \
      FPFLAG_DEFAULT_TRIGGER_STREAM,           \
      "Input stream")                          \
    X(".out_name", out_name,                   \
      FPTYPE_STREAMNAME, 1,                    \
      FPFLAG_DEFAULT_INPUT,                    \
      "Output stream")
```

### Tunable gain with limits

```c
X(".gain", &gain, FPTYPE_FLOAT64, 1,          \
  FPFLAG_DEFAULT_INPUT                         \
  | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT         \
  | FPFLAG_WRITERUN,                           \
  "Loop gain [0.0 - 1.0]")
```

### On/off toggle

```c
X(".enabled", &enabled, FPTYPE_ONOFF, 0,      \
  FPFLAG_DEFAULT_INPUT                         \
  | FPFLAG_WRITERUN,                           \
  "Enable processing")
```

## Common Mistakes

1. **Wrong variable type**: using `int` instead of
   `int64_t` for `FPTYPE_INT64` — causes undefined
   behavior on 32-bit targets.

2. **String variable type mismatch**: for
   `FPTYPE_STRING` / `FPTYPE_STREAMNAME`, always
   use `static char var[FUNCTION_PARAMETER_STRMAXLEN]`
   and pass `var` (no `&`) in the X-macro. The
   legacy `char *ptr` + `&ptr` pattern still works
   but is not recommended for new code.

3. **Missing `FPFLAG_PRIMARY_CLI_INPUT`**: the
   parameter won't count toward `nbarg` and
   CLI argument parsing will be misaligned.

4. **Forgetting `is_primary`**: set field 4 to `1`
   for required CLI positional arguments, `0` for
   optional parameters.

5. **Parameter name collision**: FPS parameter
   names must be unique within a function. Use
   descriptive prefixes (`.loop_gain` not `.gain`)
   if there could be ambiguity.
