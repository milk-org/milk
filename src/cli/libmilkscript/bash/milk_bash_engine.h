// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_engine.h
 * @brief Supervised bash co-process and milk-cli shorthand translator.
 *
 * Role & Goal:
 *   Provides the public interface for milk-cli's bash execution engine:
 *   1. `milk_translate_shorthand_to_bash()`: Translates `.milk` scripts and interactive
 *      REPL lines (`milk-cli` `@` shorthands, `@{expr}`, `@s.im.prop`, and interactive
 *      REPL commands) into `milk.bash` syntax while preserving 1:1 line numbers.
 *   2. `milk_bash_engine_*()`: Manages the persistent supervised bash co-process
 *      and in-process UNIX domain socket server used by interactive `milk-cli`
 *      and `milk-cli <script.milk>`.
 */

#ifndef MILK_BASH_ENGINE_H
#define MILK_BASH_ENGINE_H

#include <stddef.h>

/**
 * @brief Check whether the bash execution engine is enabled for milk-cli.
 *
 * @return Always 1 (the bash engine is the sole execution engine)
 */
int milk_is_bash_engine_enabled(void);

/**
 * @brief Translate milk-cli shorthands into standard milk.bash syntax.
 *
 * Preserves 1:1 line numbering across multi-line scripts so bash `$LINENO`
 * matches the source file line numbers.
 *
 * @param src          Input script or command line
 * @param allow_legacy Non-zero to also translate un-sigiled interactive milk-cli commands
 * @return Newly heap-allocated translated string (caller must `free()`), or NULL
 */
char *milk_translate_shorthand_to_bash(const char *src, int allow_legacy);

/**
 * @brief Initialize the in-process milk socket server and supervised bash co-process.
 *
 * @return 0 on success, -1 on error
 */
int milk_bash_engine_init(void);

/**
 * @brief Translate and execute a command or multi-line block in the bash co-process.
 *
 * While bash executes the command, the parent milk-cli process services all
 * incoming socket requests (`milkcall`, `milkcalc`, `milkget`, `milkset`, etc.)
 * in-process.
 *
 * @param cmdline      Command line or multi-line script text
 * @param out_exit_req Optional: set to 1 if script invoked `exit`/`quit`
 * @return Exit status of the bash command (0 = success)
 */
int milk_bash_engine_exec(const char *cmdline, int *out_exit_req);

/**
 * @brief Read, translate, and execute a script file in the bash co-process.
 *
 * @param filename     Path to script file
 * @param out_exit_req Optional: set to 1 if script invoked `exit`/`quit`
 * @return Exit status of the script (0 = success)
 */
int milk_bash_engine_exec_file(const char *filename, int *out_exit_req);

/**
 * @brief Return 1 while a command is executing in the bash co-process (milk commands called back
 *        from bash run in this state and must not re-enter the engine).
 */
int milk_bash_engine_busy(void);

/**
 * @brief Shut down the supervised bash co-process (running EXIT/defer traps)
 *        and close the in-process socket server.
 */
void milk_bash_engine_shutdown(void);

/**
 * @brief Abort an in-flight bash co-process command after a signal interrupt
 *        or fault-isolation `siglongjmp`.
 */
void milk_bash_engine_abort(void);

/**
 * @brief Return 1 if @src needs more input to be a complete bash construct (open quote,
 *        substitution, here-document, compound command, trailing `\`, `|`, `&&`, `||`).
 */
int milk_shell_incomplete(const char *src);

#endif /* MILK_BASH_ENGINE_H */
