// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_sync.h
 * @brief   Sync CLI arguments to FPS and local variables
 */

#ifndef FPS_CLI_SYNC_H
#define FPS_CLI_SYNC_H

#include "fps_types.h"
#include "fps_cli_binding.h"
#include "fps_modulevars_resync.h"

/**
 * @brief Set standalone argc/argv for CLI sync.
 *
 * Must be called from main() before any lifecycle
 * function when running as a standalone executable.
 *
 * @param argc  Argument count
 * @param argv  Argument vector
 */
void fps_cli_set_standalone_args(int argc, char **argv);


/**
 * fps_cli_option_span() - Number of argv words (1 or 2) taken by the fpsexec option at
 * @argv[@i]; 0 if it is not one (a value such as "-0.5" is not an option).
 */
int fps_cli_option_span(int argc, char *const *argv, int i);


/**
 * fps_cli_set_unit_help() - Describe the own syntax of a standalone unit in its -h: @usage
 * replaces "[values...]" after "<command>", @options ("  -x ARG   text\n" lines) is printed as
 * "Unit Options". NULL keeps the default. Call before fpsexec_main_function().
 */
void fps_cli_set_unit_help(const char *usage, const char *options);

/** fps_cli_unit_usage(), fps_cli_unit_options() - what fps_cli_set_unit_help() set, or NULL. */
const char *fps_cli_unit_usage(void);
const char *fps_cli_unit_options(void);


/**
 * fps_cli_check_values() - Check the values after the command word of the standalone argv
 * against @bindings: `.name=value` names a binding, positional values fill the primary
 * bindings not set by name, in order, and each value parses for its type.
 *
 * Return: 0, or 2 (usage error, reported on stderr).
 */
int fps_cli_check_values(const FPS_CLI_BINDING *bindings, int nb_b);


/**
 * @brief Sync standalone argv to FPS and local variables.
 *
 * Performs two-step sync:
 * 1. argv values -> FPS (`.name=value`, then positional values
 *    on the primary bindings not set by name)
 * 2. FPS values -> local C variables via bindings
 *
 * @param fps       FPS structure to sync
 * @param bindings  Binding array
 * @param nb_b      Number of bindings
 * @return          RETURN_SUCCESS on success
 */
errno_t fps_process_argv_and_sync(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b);


/**
 * @brief Copy a single FPS parameter value back into
 *        the module-local C variable via the binding.
 *
 * Defined in params/fps_modulevars_resync.c.
 */
void sync_fps_to_local(FPS *fps, long pindex, FPS_CLI_BINDING *b);


/**
 * @brief Copy the module-local C variable value back into
 *        the FPS parameter slot via the binding, if it
 *        actually differs from the current FPS value.
 *
 * @return 1 if the local value differed and was written
 *         back, 0 if it already matched (nothing done).
 *
 * Defined in params/fps_modulevars_resync.c.
 */
int sync_local_to_fps(FPS *fps, long pindex, FPS_CLI_BINDING *b);


/**
 * @brief Refresh module-local C variables from FPS.
 *
 * Resolves and caches each binding's parameter index on
 * first use, then copies the current FPS value into the
 * local variable via sync_fps_to_local().
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter binding array
 * @param nb_b      Number of bindings
 * @param force_cache_build  Force pindex cache rebuild
 * @return          RETURN_SUCCESS on success
 *
 * Defined in params/fps_modulevars_resync.c
 */
errno_t fpsresync_fps_to_modvar(FPS             *fps,
                                FPS_CLI_BINDING *bindings,
                                int              nb_b,
                                int              force_cache_build);


/**
 * @brief Push module-local C variable values into FPS
 *        parameter slots for every binding.
 *
 * Calls sync_local_to_fps() for each binding, then updates
 * the binding's cached generation counter so a concurrent
 * external update does not get superseded on the next call.
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter binding array
 * @param nb_b      Number of bindings
 * @return          RETURN_SUCCESS on success
 *
 * Defined in params/fps_modulevars_resync.c
 */
errno_t fpsresync_modvar_to_fps(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b);


#endif /* FPS_CLI_SYNC_H */
