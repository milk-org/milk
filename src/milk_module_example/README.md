# Module: milk_module_example

Template module demonstrating the `milk` compute unit architecture.

## Source Files

| File | Description |
|------|-------------|
| `examplefunc1.c` | simple function example |
| `examplefunc2_FPS.c` | simple function example with FPS and processinfo support |
| `examplefunc3_updatestreamloop.c` | simple procinfo+fps example - brief, no comments, uses macros |
| `examplefunc4_streamprocess.c` | template for simple stream processing loop |
| `examplefunc_fps_cli_poc.c` | Template for FPS V2 Compute Units |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-clitest` | `examplefunc_fps_cli_poc.c` | Template for FPS V2 Compute Units |
| `milk-fpsexec-imsum2` | `examplefunc2_FPS.c` | simple function example with FPS and processinfo support |
| `milk-fpsexec-streamprocess` | `examplefunc4_streamprocess.c` | template for simple stream processing loop |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
