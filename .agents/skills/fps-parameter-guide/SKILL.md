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
#define FPS_PARAMS                              \
    X(FPTYPE_STRING, ".in_name",  "input",      \
      "Input stream",                           \
      FPFLAG_DEFAULT_INPUT, in_name)            \
    X(FPTYPE_STRING, ".out_name", "output",     \
      "Output stream",                          \
      FPFLAG_DEFAULT_OUTPUT, out_name)          \
    X(FPTYPE_FLOAT64, ".gain",   "0.5",         \
      "Loop gain",                              \
      FPFLAG_DEFAULT_INPUT                      \
      | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT,     \
      &gain)
```

### X-Macro Fields

| Position | Field | Description |
|----------|-------|-------------|
| 1 | Type | `FPTYPE_*` constant |
| 2 | Keyword | FPS parameter name (`.prefix`) |
| 3 | Default | Default value as string |
| 4 | Description | Human-readable description |
| 5 | Flags | `FPFLAG_*` bitfield |
| 6 | Variable | Pointer to local C variable |

## Parameter Types (`FPTYPE_*`)

| Type | C Variable | Example Default |
|------|-----------|-----------------|
| `FPTYPE_INT64` | `int64_t` | `"100"` |
| `FPTYPE_FLOAT64` | `double` | `"0.5"` |
| `FPTYPE_FLOAT32` | `float` | `"1.0"` |
| `FPTYPE_STRING` | `char[FUNCTION_PARAMETER_STRMAXLEN]` | `"stream01"` |
| `FPTYPE_ONOFF` | `int64_t` | `"ON"` or `"OFF"` |
| `FPTYPE_FILENAME` | `char[FUNCTION_PARAMETER_STRMAXLEN]` | `"/tmp/file.fits"` |
| `FPTYPE_FITSFILENAME` | `char[FUNCTION_PARAMETER_STRMAXLEN]` | `"data.fits"` |
| `FPTYPE_EXECFILENAME` | `char[FUNCTION_PARAMETER_STRMAXLEN]` | `"/usr/bin/prog"` |
| `FPTYPE_DIRNAME` | `char[FUNCTION_PARAMETER_STRMAXLEN]` | `"/tmp/outdir"` |
| `FPTYPE_STREAMNAME` | `char[FUNCTION_PARAMETER_STRMAXLEN]` | `"wfs0"` |
| `FPTYPE_FPSNAME` | `char[FUNCTION_PARAMETER_STRMAXLEN]` | `"myfps"` |
| `FPTYPE_PROCESS_PID` | `int64_t` | `"0"` |
| `FPTYPE_TIMESPEC` | `double` | `"0.001"` |
| `FPTYPE_AUTO` | *(varies)* | Auto-detect |

### String-Type Parameters

For any `FPTYPE_STRING` (or string subtypes like
`FPTYPE_FILENAME`), the C variable must be a `char`
array of size `FUNCTION_PARAMETER_STRMAXLEN` and
passed directly (as it decays to a pointer):

```c
// Section 2 — local variables
static char in_name[FUNCTION_PARAMETER_STRMAXLEN] = "";

// Section 3 — FPS_PARAMS
X(FPTYPE_STRING, ".in_name", "stream01",
  "Input stream name",
  FPFLAG_DEFAULT_INPUT, in_name)
```

### Numeric Parameters

For numeric types, use the matching C type and
pass `&variable`:

```c
// Section 2
static double gain;
static int64_t nbiter;

// Section 3
X(FPTYPE_FLOAT64, ".gain", "0.5",
  "Loop gain", FPFLAG_DEFAULT_INPUT, &gain)
X(FPTYPE_INT64, ".NBiter", "1000",
  "Number of iterations",
  FPFLAG_DEFAULT_INPUT, &nbiter)
```

## Parameter Flags (`FPFLAG_*`)

### CLI Input Flags

| Flag | Effect |
|------|--------|
| `FPFLAG_DEFAULT_INPUT` | Standard CLI input parameter |
| `FPFLAG_DEFAULT_OUTPUT` | Standard output parameter |
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

| Flag | Effect |
|------|--------|
| `FPFLAG_MINLIMIT` | Enforce minimum value |
| `FPFLAG_MAXLIMIT` | Enforce maximum value |

When using limit flags, set the limits after
parameter creation in the `fpsexec()` function
or in a `customCONFcheck()`.

### Display Flags

| Flag | Effect |
|------|--------|
| `FPFLAG_WRITERUN` | Allow modification while running |
| `FPFLAG_WRITECONF` | Allow modification during config |
| `FPFLAG_WRITESTATUS` | Show in status display |

## Common Patterns

### Input/output stream pair

```c
#define FPS_PARAMS                            \
    X(FPTYPE_STRING, ".in_name", "in",        \
      "Input stream",                         \
      FPFLAG_DEFAULT_INPUT, in_name)          \
    X(FPTYPE_STRING, ".out_name", "out",      \
      "Output stream",                        \
      FPFLAG_DEFAULT_OUTPUT, out_name)
```

### Tunable gain with limits

```c
X(FPTYPE_FLOAT64, ".gain", "0.5",            \
  "Loop gain [0.0 - 1.0]",                   \
  FPFLAG_DEFAULT_INPUT                        \
  | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT        \
  | FPFLAG_WRITERUN, &gain)
```

### On/off toggle

```c
X(FPTYPE_ONOFF, ".enabled", "ON",            \
  "Enable processing",                       \
  FPFLAG_DEFAULT_INPUT                        \
  | FPFLAG_WRITERUN, &enabled)
```

## Common Mistakes

1. **Wrong variable type**: using `int` instead of
   `int64_t` for `FPTYPE_INT64` — causes undefined
   behavior on 32-bit targets.

2. **Using Pointers Instead of Buffers**: for
   `FPTYPE_STRING`, using `static char *var` and
   passing `&var` will cause the FPS engine to
   overwrite the pointer location itself, causing
   a severe buffer overflow and `SIGSEGV`.
   Always use `static char var[FUNCTION_PARAMETER_STRMAXLEN]`
   and pass `var` without the `&`. Use `&` only for scalar primitives.

3. **Missing `FPFLAG_PRIMARY_CLI_INPUT`**: the
   parameter won't count toward `nbarg` and
   CLI argument parsing will be misaligned.

4. **Default value type mismatch**: the default is
   always a string. Use `"100"` not `100`.

5. **Parameter name collision**: FPS parameter
   names must be unique within a function. Use
   descriptive prefixes (`.loop_gain` not `.gain`)
   if there could be ambiguity.
