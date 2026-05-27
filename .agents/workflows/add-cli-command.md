---
description: Add a CLI command to an existing module
---

# Add a CLI Command

Use this workflow to add a simple CLI command (with
or without FPS) to an existing module. This covers
adding the source file, CLI registration, and CMake
updates.

## 1. Gather Information

Ask the user for:

- **Target module directory** (e.g.,
  `~/src/milk/src/coremods/COREMOD_arith`)
- **C filename** for the new command (e.g.,
  `my_new_cmd.c`)
- **CLI keyword** (e.g., `arith.mynewcmd`)
- **One-line description**
- Whether the command uses **FPS** (if yes, use the
  [`/create-fpsexec`](create-fpsexec.md) workflow instead for standalone
  support, or continue here for CLI-only FPS)
- **Arguments**: names, types, descriptions

## 2. Create Source Files

Create `<name>.c` and `<name>.h` in the target
module directory. Use the V2 pattern from
`examplefunc_fps_cli_poc.c` (sections 1–7, skip
section 8 if no standalone is needed).

### Minimal non-FPS command structure:

```c
#include "CLIcore.h"

// CLI argument definitions
static CLICMDARGDEF farg[] = {
    {
        .type        = CLIARG_STR,
        .fptype      = FPTYPE_AUTO,
        .keyword     = ".in_name",
        .short_flag  = 'i',
        .example     = "im01",
        .descr       = "input image"
    },
};

static CLICMDDATA CLIcmddata = {
    "arith.mynewcmd",
    "description of my command",
    CLICMD_FIELDS_DEFAULTS
};

static errno_t compute_function()
{
    // Your logic here
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    CLIfunction_strict(
        farg,
        CLIcmddata.nbarg,
        compute_function);
    return RETURN_SUCCESS;
}

errno_t CLIADDCMD_COREMOD_arith__mynewcmd()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
```

## 3. Create the Header

Create `<name>.h` with the function prototype:

```c
#ifndef <MODULE>_<NAME>_H
#define <MODULE>_<NAME>_H

errno_t CLIADDCMD_<module>__<name>();

#endif
```

## 4. Update Module Registration

In the module's main `.c` file (e.g.,
`COREMOD_arith.c`):

1. `#include "<name>.h"`
2. Add `CLIADDCMD_<module>__<name>();` inside
   `initModule()`

## 5. Update CMakeLists.txt

Add the new `.c` file to `SOURCEFILES` in the
module's `CMakeLists.txt`.

## 6. Update README

Add a row to the module's `README.md` source file
table describing the new file.

## 7. Compile and Verify

Run the [`/compile-test`](compile-test.md) workflow.
