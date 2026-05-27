---
description: Add a new function to an existing module
---

# Add a Function to an Existing Module

Use this workflow when adding a single new function
(FPS compute unit or plain function) to an existing
module. This is lighter than [`/add-new-module`](add-new-module.md) but
ensures no steps are missed.

## 1. Gather Information

Ask the user for:

- **Target module directory**
- **Function name / C filename** (e.g., `my_func.c`)
- **CLI keyword** (e.g., `arith.myfunc`)
- **One-line description**
- Whether it is an **FPS compute unit** (standalone)
  or a **simple CLI command**

## 2. Choose Template

| Type             | Template                                     | Workflow                                           |
| ---------------- | -------------------------------------------- | -------------------------------------------------- |
| FPS compute unit | `examplefunc_fps_cli_poc.c`                  | [`/create-fpsexec`](create-fpsexec.md)             |
| Stream processor | `examplefunc4_streamprocess.c`               | [`/add-stream-processor`](add-stream-processor.md) |
| Simple CLI cmd   | See [`/add-cli-command`](add-cli-command.md) | [`/add-cli-command`](add-cli-command.md)           |
| Non-CLI function | `examplefunc1.c`                             | This workflow                                      |

For a non-CLI helper function (no CLI registration):

- Create `<name>.c` with the function implementation
- Create `<name>.h` with the prototype
- Use Kernel-Doc comments above the function

## 3. Add to CMakeLists.txt

Append the new `.c` file to `SOURCEFILES` in the
module's `CMakeLists.txt`.

If the function is an FPS standalone, also add:

```cmake
add_milk_standalone(cmdkey source_file.c)
```

## 4. Register in Module Init

If the function has a CLI interface, add the
`CLIADDCMD_<module>__<function>()` call inside the
module's `initModule()` function in the main module
`.c` file.

Also `#include "<name>.h"` at the top.

## 5. Update Module README

Add a row to the module's `README.md` source file
table:

```
| my_func.c | Brief description of what it does |
```

## 6. Compile and Verify

Run the [`/compile-test`](compile-test.md) workflow.
