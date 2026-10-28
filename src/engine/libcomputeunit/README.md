# Module: libcomputeunit

FPS runtime: everything needed to set up, run and tear down a compute unit. Depends on
[`libfps`](../libfps/README.md) (struct and parameters); `libfps` does not depend on it. No CLI
dependency. Umbrella header: `computeunit.h`.

## Source Files

| File                                    | Description                                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------- |
| `core/fps_standalone_main.c`            | `fpsexec_main_function`: command-line driver of every fpsexec                     |
| `core/fps_lifecycle.c`                  | generic init / conf / run / stop                                                  |
| `cli/fps_cli_init.c`                    | create FPS entries from bindings; fill CLI examples                               |
| `cli/fps_cli_sync.c`                    | standalone argv to FPS to module variables                                        |
| `params/fps_modulevars_resync.c`        | FPS to module-local variable sync via bindings                                    |
| `core/fps_unit.c`                       | unit lifecycle (`fps_unit.h`): start, stop, environment, front-end tools, scripts |
| `state/fps_CONF*.c`, `fps_FPCONF*.c`    | conf process start/stop/setup/loop/exit                                           |
| `state/fps_RUN*.c`                      | run process start/stop/exit                                                       |
| `procinfo/fps_processinfo*.c`           | processinfo entries and signalling for FPS                                        |
| `procinfo/fps_apply_process_settings.c` | apply cset / taskset / RT priority from FPS                                       |
| `tools/milk-fps-cmd.c`                  | `milk-fps-{confstart,confstop,confstep,runstart,runstop}`                         |
| `tools/milk-fps-log.c`                  | `milk-fps-log`: a unit's output, optionally followed                              |
| `tools/milk-fps-rm.c`                   | `milk-fps-rm`: stop a unit, remove its FPS and unit directory                     |

The milk-cli side of the FPS wrapper (`fps_generic_CLIfunction`, the `?` query) lives in
`libmilkscript`.

## Dependencies

- `milkfps`, `milkdata`, `milkprocessinfo`, `milkcommon`, `ImageStreamIO`
