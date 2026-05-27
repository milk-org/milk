---
description: Migrate legacy V1 fpsexec code to V2 template
---

# Migrate to V2 Template

Use this workflow to convert existing V1 fpsexec code
(using `FPS_MAIN_STANDALONE`) to the standardized V2
8-section layout using `FPS_MAIN_STANDALONE_V2`.

## 1. Identify V1 Patterns

Search for V1 code in the target file:

```bash
grep -n 'FPS_MAIN_STANDALONE[^_V2]' <file>
grep -n 'FPSPROCSYNC' <file>
```

V1 indicators:

- Uses `FPS_MAIN_STANDALONE` (not `_V2`)
- Manual FPS parameter setup instead of
  `FPS_PARAMS` X-macro
- Inline `CLIcmddata` initialization in the
  registration function

## 2. Read the V2 Template

Open the reference template:

```
src/milk_module_example/examplefunc_fps_cli_poc.c
```

Understand the 8-section layout before proceeding.

## 3. Convert Section by Section

### Section 1 — FPS_APP_INFO

Create the `FPS_APP_INFO` struct from existing
metadata:

```c
static FPS_APP_INFO_TYPE FPS_app_info = {
    .fps_name   = "existing_name",
    .cmdkey     = "module.func",
    .description = "One-line description"
};
```

### Section 2 — Local Variables

Extract FPS parameter variables from the existing
code into static file-scope declarations.

### Section 3 — FPS_PARAMS X-Macro

Convert manual `FPS_SETUP_INIT` / `function_parameter_*`
calls into the `FPS_PARAMS` X-macro format:

```c
#define FPS_PARAMS                              \
    X(FPTYPE_STRING, ".in_name", "in",          \
      "Input stream",                           \
      FPFLAG_DEFAULT_INPUT, &in_name)
```

See the `fps-parameter-guide` skill for type and
flag reference.

### Section 4 — fpsexec()

Move the core computation into a single `fpsexec()`
function. Parameters are already synced before
this function is called.

### Section 5 — CLIcmddata

Replace with the V2 scoping pattern:

```c
#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata =
    {"", "", CLICMD_FIELDS_DEFAULTS};
#else
static CLICMDDATA CLIcmddata =
    {"", "", CLICMD_FIELDS_DEFAULTS};
#endif
```

### Section 6 — Compute Wrapper

Use `INSERT_STD_PROCINFO_COMPUTEFUNC_*` macros:

```c
INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
{
    fpsexec();
}
INSERT_STD_PROCINFO_COMPUTEFUNC_END
```

### Section 7 — Module Registration

Rename the `CLIADDCMD_*` function and guard with
`#ifndef FPS_STANDALONE`.

### Section 8 — Standalone Main

Replace `FPS_MAIN_STANDALONE(...)` with:

```c
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
```

## 4. Update CMakeLists.txt

If the old CMake used the 4-line manual pattern,
replace with:

```cmake
add_milk_standalone(cmdkey source.c)
```

## 5. Compile and Verify

Run the [`/compile-test`](compile-test.md)
workflow.

## 6. Test

Verify the standalone works:

```bash
source ~/src/milk/local/bin/milk-setup.bash
milk-fpsexec-<name> -h
milk-fpsexec-<name> -h1
```
