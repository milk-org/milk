// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    CLIcore_fps_function.h
 * @brief   Generic CLI lifecycle of FPS-based MILK_RUNNABLE units
 */

#ifndef CLICORE_FPS_FUNCTION_H
#define CLICORE_FPS_FUNCTION_H

#include "computeunit.h"

/**
 * @brief Run an FPS-based command from the milk CLI.
 *
 * Connects (or creates) the FPS, syncs the CLI arguments into it, sets up processinfo if
 * requested and calls @p compute_fn. Also handles fpsinit and the "?" query.
 *
 * @param app_info    FPS application identity
 * @param farg        CLI argument definitions
 * @param cmdata      CLI command metadata
 * @param bindings    Parameter bindings (C var <-> FPS)
 * @param nb_b        Number of bindings
 * @param compute_fn  Compute function to invoke
 * @return RETURN_SUCCESS on completion
 */
errno_t fps_generic_CLIfunction(FPS_APP_INFO    *app_info,
                                CLICMDARGDEF    *farg,
                                CLICMDDATA      *cmdata,
                                FPS_CLI_BINDING *bindings,
                                int              nb_b,
                                fps_compute_fn   compute_fn);

#endif /* CLICORE_FPS_FUNCTION_H */
