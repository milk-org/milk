# Module: libcomputeunit

FPS runtime: everything needed to set up, run and tear down a compute unit. Depends on
[`libfps`](../libfps/README.md) (struct and parameters); `libfps` does not depend on it. No CLI
dependency. Umbrella header: `computeunit.h`.

## Source Files

| File                                   | Description                                               |
| -------------------------------------- | --------------------------------------------------------- |
| `core/fps_standalone_main.c`           | `fpsexec_main_function`: command-line driver of every fpsexec |
| `core/fps_lifecycle.c`                 | generic init / conf / run / stop                          |
| `cli/fps_cli_init.c`                   | create FPS entries from bindings; fill CLI examples       |
| `cli/fps_cli_sync.c`                   | standalone argv to FPS to module variables                |
| `cli/fps_cli_function_registry.c`      | pointer through which modules reach the milk-cli wrapper  |
| `params/fps_modulevars_resync.c`       | FPS to module-local variable sync via bindings            |
| `cmd/fps_tmux.c`                       | tmux session management                                   |
| `cmd/fps_execFPScmd.c`                 | execute FPS command                                       |
| `cmd/fps_processcmdline_*.c`           | FPS command line processing                               |
| `cmd/fps_process_fpsCMDarray.c`        | find the next task to execute                             |
| `io/fps_read_fpsCMD_fifo.c`            | fill up task list from fifo submissions                   |
| `state/fps_CONF*.c`, `fps_FPCONF*.c`   | conf process start/stop/setup/loop/exit                   |
| `state/fps_RUN*.c`                     | run process start/stop/exit                               |
| `procinfo/fps_processinfo*.c`          | processinfo entries and signalling for FPS                |
| `procinfo/fps_apply_process_settings.c`| apply cset / taskset / RT priority from FPS               |
| `tools/milk-fps-cmd.c`                 | `milk-fps-{confstart,confstop,confstep,runstart,runstop}` |

The milk-cli side of the FPS wrapper (`fps_generic_CLIfunction`, the `?` query) lives in
`libmilkscript`.

## Dependencies

- `milkfps`, `milkdata`, `milkprocessinfo`, `milkcommon`, `ImageStreamIO`
