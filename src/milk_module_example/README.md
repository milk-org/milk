# milk_module_example

Template module demonstrating the `milk` compute unit architecture.

## Files

| File | Purpose |
|------|---------|
| `milk_module_example.c` | Module init and CLI registration |
| `examplefunc1.c` | Basic function (no FPS) |
| `examplefunc2_FPS.c` | FPS-enabled compute unit template |
| `examplefunc4_streamprocess.c` | Stream processing loop template |
| `examplefunc_fps_cli_poc.c` | V2 standalone template (8-section layout) |

## Standalone Executables

| Executable | Source | Description |
|-----------|--------|-------------|
| `milk-fpsexec-clitest` | `examplefunc_fps_cli_poc.c` | V2 template demo |
| `milk-fpsexec-imsum2` | `examplefunc2_FPS.c` | Image sum (FPS example) |
| `milk-fpsexec-streamprocess` | `examplefunc4_streamprocess.c` | Stream loop example |

## Usage

Use this module as the starting point for new compute units:

1. Copy `examplefunc_fps_cli_poc.c` to your module
2. Follow the 8-section layout (see `docs/programmers_guide.md`)
3. Add `add_milk_standalone()` to your `CMakeLists.txt`
