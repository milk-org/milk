---
name: module-loading-internals
description: Deep reference for milk-cli module
  registration, command dispatch, and common
  initialization bugs
---

# Module Loading Internals

This skill provides deep context on how `milk-cli`
discovers, loads, and registers modules and their
commands. Essential for diagnosing phantom commands,
metadata corruption, and load-order issues.

## When to Use

- Commands appear empty or with wrong metadata
- Module `m?` output shows ghost entries
- `sofilename` or `loadname` is wrong in module
  info
- Commands from one module appear under another
- CLI crashes during module loading
- Understanding how `__attribute__((constructor))`
  interacts with `dlopen`

## Module Loading Sequence

### High-level flow

```
main() [CLIcore.c]
  → data initialization
  → load built-in modules (COREMODs)
  → scan plugin directories for .so files
  → for each .so:
      load_module_shared(sopath)
        → dlopen(sopath, RTLD_NOW)
        → dlsym(handle, "initModule")
        → initModule()
          → CLIADDCMD_module__func1()
            → RegisterCLIcmd(&CLIcmddata1, ...)
          → CLIADDCMD_module__func2()
            → RegisterCLIcmd(&CLIcmddata2, ...)
```

### Key source files

| File                         | Role                         |
| ---------------------------- | ---------------------------- |
| `CLIcore.c`                  | `main()`, module scan loop   |
| `CLIcore/CLIcore_modules.c`  | `load_module_shared()`       |
| `CLIcore/CLIcore_modules.h`  | Module data structures       |
| `CLIcore/CLIcore_datainit.c` | `data` struct initialization |

## Data Structures

### `DATA` (global `data`)

The central global struct holding all runtime
state:

```c
// Key fields for module loading:
data.module[i].name         // module name
data.module[i].sofilename   // .so file path
data.module[i].loadname     // display name
data.module[i].nbcmd        // commands in module
data.moduleindex            // CURRENT loading idx
data.NBmodule               // total module count
```

### `CMD` (command entry)

Each registered command occupies one slot in
`data.cmd[]`:

```c
data.cmd[i].key             // CLI keyword
data.cmd[i].moduleindex     // owning module
data.cmd[i].fp              // function pointer
data.cmd[i].nbarg           // argument count
data.cmd[i].fpscliarg[]     // FPS CLI args
```

### `CLICMDDATA` (per-function static)

Each function that registers as a CLI command
has a static `CLIcmddata` struct:

```c
static CLICMDDATA CLIcmddata = {
    .key = "module.function",
    .description = "What it does",
    // ... argument definitions ...
};
```

## The `data.moduleindex` Problem

### Root cause

`data.moduleindex` is a **single global integer**
that tracks which module slot is currently being
populated. The loading code sets it before calling
`initModule()`:

```c
data.moduleindex = next_free_slot;
load_module_shared(sopath);
// initModule() uses data.moduleindex to tag
// its commands
```

### The race condition

When module A depends on library B (which is also
a module), `dlopen(A)` triggers the dynamic linker
to also load B. If B has an `__attribute__
((constructor))`, it may call `RegisterCLIcmd`
while `data.moduleindex` still points to A's slot.

Result: B's commands get tagged with A's module
index → wrong `sofilename` and `loadname`.

### Mitigation

The fix (implemented in session 1cb88742) replaces
reliance on `data.moduleindex` with a lookup:
after `dlopen` returns, scan `data.module[]` for
the slot whose `sofilename` matches the just-loaded
`.so` file. This is robust against transitive
loading.

## Constructor vs initModule Timing

### `__attribute__((constructor))`

GCC constructors run when the `.so` is loaded by
`dlopen()`, **before** `dlsym("initModule")` is
called. They are used to initialize `CLIcmddata`
static variables.

### Ordering

```
dlopen("module.so")
  → constructor1()  [sets CLIcmddata for func1]
  → constructor2()  [sets CLIcmddata for func2]
  → (dlopen returns)
dlsym("initModule")
initModule()
  → CLIADDCMD_module__func1()
    → RegisterCLIcmd(&CLIcmddata1)  [reads data]
  → CLIADDCMD_module__func2()
    → RegisterCLIcmd(&CLIcmddata2)  [reads data]
```

### Common problems

1. **Constructor not run**: if `CLIcmddata` is
   not initialized by a constructor (e.g., the
   `CLIADDCMD_*` function sets it inline), the
   initialization timing is different and usually
   correct.

2. **Constructor runs too early**: if a
   constructor depends on `data` being initialized,
   but `dlopen` happens before data init completes,
   the constructor reads garbage.

3. **Empty commands**: the `CLICMDDATA` struct is
   initialized with `CLICMD_FIELDS_DEFAULTS` which
   sets `key = ""`. If `RegisterCLIcmd` is called
   before the real key is set, an empty command is
   registered. This manifests as blank entries in
   `m?` output.

## Command Dispatch

When the user types a command:

```
Input: "module.function arg1 arg2"
  → tokenize into words
  → look up "module.function" in data.cmd[]
    → linear scan or hash lookup
  → if found:
      → validate arguments (CLIcore_checkargs.c)
      → call data.cmd[i].fp(args)
  → if not found:
      → try as calc expression
      → try as shell bypass
```

### `nbarg` calculation

The number of required arguments (`nbarg`) is
determined by counting parameters with
`FPFLAG_PRIMARY_CLI_INPUT` set. If this flag is
missing from a parameter that should be a CLI
argument, `nbarg` will be wrong and may cause a
segfault during argument parsing.

## Debugging Module Issues

### List loaded modules

```bash
echo "m?" | milk-cli 2>&1
```

### Check specific module

```bash
echo "m? modulename" | milk-cli 2>&1
```

### Verify command registration

```bash
echo "cmd? cmdkey" | milk-cli 2>&1
```

### Find duplicate registrations

```bash
echo "m?" | milk-cli 2>&1 | \
  awk '{print $1}' | sort | uniq -d
```

### Trace loading order

Set `MILK_QUIET=0` and look for module load
messages in stderr:

```bash
MILK_QUIET=0 echo "exitCLI" | milk-cli 2>&1 | \
  grep -i "module\|load"
```

## Checklist for New Modules

When a new module's commands don't appear:

- [ ] `initModule()` function exists and is
      exported (not `static`)
- [ ] `initModule()` calls all `CLIADDCMD_*`
      functions
- [ ] Each `CLIADDCMD_*` function calls
      `RegisterCLIcmd` with valid `CLIcmddata`
- [ ] `CLIcmddata.key` is non-empty and unique
- [ ] The `.so` file is installed to the plugin
      directory
- [ ] The `.so` file is listed in module scan
      output (check with `MILK_QUIET=0`)
- [ ] No `dlopen` errors (check `dlerror()` output)
