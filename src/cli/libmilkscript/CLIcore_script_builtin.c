/**
 * @file CLIcore_script_builtin.c
 *
 * @brief [STUB] — all implementations have been
 *        split into focused sub-modules.
 *
 * This file intentionally contains no code.
 * The implementations previously here have moved
 * to:
 *
 *   CLIcore_script_traps.c
 *       Signal traps (cli_trap_run,
 *       cli_trap_run_exit) and non-blocking
 *       engine event traps (STREAM:/FPS:/PROC:)
 *
 *   CLIcore_script_cmd_io.c
 *       I/O and variable commands:
 *       echo, fpsset, read, export, shift, printf
 *       Also: cli_fps_set_param() helper
 *
 *   CLIcore_script_cmd_inspect.c
 *       System inspection commands:
 *       fpslist, fpsdump, streamlist,
 *       proclist, milkquery
 *
 *   CLIcore_script_cmd_defer.c
 *       Defer LIFO stack:
 *       cli_cmd_defer, cli_defer_run
 *
 * All public symbols remain declared in
 * CLIcore_script.h — no other file needs
 * to change its includes.
 */
