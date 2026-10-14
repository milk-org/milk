# Template Source Code

The primary template for writing new compute units is the `MILK_RUNNABLE` template in `src/milk_module_example/`:

| Template File                    | Purpose                                                          |
| -------------------------------- | ---------------------------------------------------------------- |
| `examplefunc_fps_cli_poc.c`      | **FPS compute unit**: CLI command and `milk-fpsexec-clitest`     |
| `examplefunc1.c/h`               | CLI-only command, no FPS                                         |
| `examplefunc2_FPS.c/h`           | FPS unit with processinfo and `milk-fpsexec-imsum2`              |
| `examplefunc4_streamprocess.c/h` | Stream processing loop and `milk-fpsexec-streamprocess`          |
| `milk_module_example.c/h`        | Module registry (`MILK_MODULE_RUNNABLES`)                        |

## Getting Started

1. Copy `examplefunc_fps_cli_poc.c` to your module directory.
2. Rename the unit, update its descriptor and parameters, and list it in the module registry.
3. See the `MILK_RUNNABLE` section of the [Programmer's Guide](../arch/programmers_guide.md) and `docs/developer/tutorial.md` for a step-by-step walkthrough.

## Documenting Functions

Use Kernel-Doc style comments above each function. See `docs/developer/DocumentingCode.md` for the
full style guide.

---

← [Documentation Index](../index.md)
