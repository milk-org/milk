# Module: milk_module_example

Template module demonstrating the `milk` compute unit architecture.

## Source Files

Each file defines one or more `MILK_RUNNABLE` units. The module registry in `milk_module_example.c` lists them.

| File                              | Unit (CLI key)                                     | Description                                              |
| --------------------------------- | -------------------------------------------------- | -------------------------------------------------------- |
| `examplefunc1.c`                  | `runnable_modex_imsum1` (`imsum1`)                 | CLI-only command, simple function example                |
| `examplefunc2_FPS.c`              | `runnable_modex_imsum2` (`imsum2`)                 | FPS unit with processinfo support                        |
| `examplefunc3_updatestreamloop.c` | `runnable_modex_streamupdate` (`streamupdate`)     | FPS unit that updates a stream, CLI only                 |
| `examplefunc4_streamprocess.c`    | `runnable_modex_streamprocess` (`streamprocess`)   | template for simple stream processing loop               |
| `examplefunc_fps_cli_poc.c`       | `runnable_modex_fpscli` (`fpsclitest`)             | Template for FPS compute units                           |

## Standalone Executables

Generated from the `EXPORT_FPSEXEC` lines of the sources (see `cmake/milk_runnable.cmake`).

| Executable                   | Source File                    | Description                                              |
| ---------------------------- | ------------------------------ | -------------------------------------------------------- |
| `milk-fpsexec-clitest`       | `examplefunc_fps_cli_poc.c`    | Template for FPS compute units                           |
| `milk-fpsexec-imsum2`        | `examplefunc2_FPS.c`           | simple function example with FPS and processinfo support |
| `milk-fpsexec-streamprocess` | `examplefunc4_streamprocess.c` | template for simple stream processing loop               |

## Dependencies

- `milk_runnable.h` (`libcomputeunit`), which provides the FPS and processinfo macros
- `COREMOD_memory` (image connection)
