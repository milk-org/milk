/**
 * @file CLIcore_script_expand.c
 *
 * @brief [STUB] — all implementations split into
 *        focused sub-modules.
 *
 * This file intentionally contains no code.
 * Implementations have moved to:
 *
 *   CLIcore_script_expand_arith.c
 *       ArithParser, arith_*() helpers,
 *       cli_expand_arith()
 *
 *   CLIcore_script_expand_fps.c
 *       expand_fpsvar_write(), expand_fpsvar_seq(),
 *       expand_fpsvar_procinfo(),
 *       expand_fpsvar_stream(),
 *       expand_fpsvar_fps_strict(),
 *       cli_expand_fpsvar()
 *
 *   CLIcore_script_expand_test.c
 *       test_unary_file(), test_unary_shm(),
 *       test_binary_op(), cli_eval_test()
 *
 *   CLIcore_script_expand_env.c
 *       emit_str_local(), cli_expand_env()
 *
 * Shared internal types (ArithParser):
 *   CLIcore_script_expand_internal.h
 *
 * All public symbols remain declared in
 * CLIcore_script.h — no other file needs updating.
 */
