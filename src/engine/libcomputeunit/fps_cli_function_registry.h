// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_function_registry.h
 * @brief   Function pointer registry for FPS-CLI
 *
 * Provides a global function pointer that the CLI fills via fps_cli_register().
 *
 * Lives in libcomputeunit.
 *
 * Uses void* function pointer typedefs for the
 * registry storage to avoid circular includes.
 * The typed safe_fps_* wrappers (defined when the
 * full type headers are available) perform the cast.
 */

#ifndef FPS_CLI_FUNCTION_REGISTRY_H
#define FPS_CLI_FUNCTION_REGISTRY_H

/**
 * @brief Global function pointer for
 *        fps_generic_CLIfunction.
 *
 * Set by fps_cli_register(). NULL until the CLI has called it.
 */
extern void *fps_generic_CLIfunction_ptr;


#endif /* FPS_CLI_FUNCTION_REGISTRY_H */
