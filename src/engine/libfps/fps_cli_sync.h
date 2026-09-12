// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_sync.h
 * @brief   Sync CLI arguments to FPS and local variables
 */

#ifndef FPS_CLI_SYNC_H
#define FPS_CLI_SYNC_H

#include "fps.h"
#include "fps_cli_binding.h"
#include "libmilkdata/milkdata_clicmd.h"

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
 * @brief Sync CLI arguments to FPS and local variables.
 *
 * Performs two-step sync:
 * 1. CLI tokens -> FPS values (from standalone args
 *    or milk CLI argdata)
 * 2. FPS values -> local C variables via bindings
 *
 * @param fps       FPS structure to sync
 * @param farg      CLICMDARGDEF array (for milk CLI mode)
 * @param bindings  Binding array
 * @param nb_b      Number of bindings
 * @return          RETURN_SUCCESS on success
 */
errno_t fps_process_cli_and_sync(FPS *fps, CLICMDARGDEF *farg, FPS_CLI_BINDING *bindings, int nb_b);


/**
 * @brief Copy a single FPS parameter value back into
 *        the module-local C variable via the binding.
 *
 * Defined in params/fps_modulevars_resync.c (base milkfps library, no CLI dependency).
 */
void sync_fps_to_local(FPS *fps, long pindex, FPS_CLI_BINDING *b);


/**
 * @brief Refresh module-local C variables from FPS
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter binding array
 * @param nb_b      Number of bindings
 * @return          RETURN_SUCCESS on success
 */
errno_t fps_to_modulevars_resync_bindings(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b);


#endif /* FPS_CLI_SYNC_H */
