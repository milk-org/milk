// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_engine.h
 * @brief Supervised bash co-process and Tier-2 shorthand translator for milk-cli.
 */

#ifndef MILK_BASH_ENGINE_H
#define MILK_BASH_ENGINE_H

#include <stddef.h>

/**
 * @brief Check whether the bash execution engine is enabled for milk-cli.
 *
 * Inspects MILKCLI_ENGINE ("bash" or "native"). Defaults to 1 (bash engine).
 *
 * @return 1 if bash engine is enabled, 0 if legacy native engine is requested
 */
int milk_is_bash_engine_enabled(void);

/**
 * @brief Translate Tier-2 milk-cli shorthands into Tier-1 plain-bash syntax.
 *
 * Preserves 1:1 line numbering across multi-line scripts so bash $LINENO
 * matches the source file line numbers.
 *
 * @param src          Input script or command line
 * @param allow_legacy Non-zero to also translate un-sigiled milk-cli commands
 *                     and spaced calculator assignments
 * @return Newly heap-allocated translated string (caller must free()), or NULL
 */
char *milk_translate_tier2_to_tier1(const char *src, int allow_legacy);

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
 * incoming socket requests (milkcall, milkcalc, milkget, milkset, etc.) in-process.
 *
 * @param cmdline       Command line or multi-line script text
 * @param out_exit_req  Optional: set to 1 if script invoked exit/quit
 * @return Exit status of the bash command (0 = success)
 */
int milk_bash_engine_exec(const char *cmdline, int *out_exit_req);

/**
 * @brief Read, translate, and execute a script file in the bash co-process.
 *
 * @param filename      Path to script file
 * @param out_exit_req  Optional: set to 1 if script invoked exit/quit
 * @return Exit status of the script (0 = success)
 */
int milk_bash_engine_exec_file(const char *filename, int *out_exit_req);

/**
 * @brief Shut down the supervised bash co-process (running EXIT/defer traps)
 *        and close the in-process socket server.
 */
void milk_bash_engine_shutdown(void);

#endif /* MILK_BASH_ENGINE_H */
