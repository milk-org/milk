// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_modulevars_resync.h
 * @brief   Copy FPS parameter values into module-local C
 *          variables via FPS_CLI_BINDING
 *
 * Declares the base-milkfps-tier functions defined in
 * fps_modulevars_resync.c. See that file for why these two
 * functions live in the base library rather than in
 * fps_cli_sync.c (milkfpsStandalone/milkfpsCLI).
 */

#ifndef FPS_MODULEVARS_RESYNC_H
#define FPS_MODULEVARS_RESYNC_H

#include "fps_types.h"
#include "fps_cli_binding.h"

/**
 * @brief Copy a single FPS parameter value back into
 *        the module-local C variable via the binding.
 *
 * @param fps     Connected FPS
 * @param pindex  Parameter index in fps->parray
 * @param b       Binding naming the local C variable
 */
void sync_fps_to_local(FPS *fps, long pindex, FPS_CLI_BINDING *b);

/**
 * @brief Refresh module-local C variables from FPS shared memory
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter binding array
 * @param nb_b      Number of bindings
 * @return          RETURN_SUCCESS on success
 */
errno_t fps_to_modulevars_resync_bindings(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b);

#endif /* FPS_MODULEVARS_RESYNC_H */
