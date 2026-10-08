# Template Source Code

The primary template for writing new compute units is the V2 template in `src/milk_module_example/`:

| Template File                    | Purpose                                                          |
| -------------------------------- | ---------------------------------------------------------------- |
| `examplefunc_fps_cli_poc.c`      | **V2 compute unit** — unified FPS-CLI standalone/module template |
| `examplefunc1.c/h`               | Simple function (no FPS)                                         |
| `examplefunc2_FPS.c/h`           | FPS-enabled function (V1 style)                                  |
| `examplefunc4_streamprocess.c/h` | Stream processing loop                                           |
| `milk_module_example.c/h`        | Module registration boilerplate                                  |

## Getting Started

1. Copy `examplefunc_fps_cli_poc.c` to your module directory.
2. Follow the 8-section structure documented in the file header.
3. See `docs/developer/tutorial.md` for a step-by-step walkthrough.

## Documenting Functions

Use Kernel-Doc style comments above each function. See `docs/developer/DocumentingCode.md` for the
full style guide.

---

← [Documentation Index](../index.md)
