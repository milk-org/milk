// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_function_registry.h
 * @brief   Function pointer registry for FPS-CLI
 *
 * Provides global function pointers that milkfpsCLI
 * fills at load time. Module libraries call through
 * these pointers instead of weak symbols, so there
 * is no link-time dependency on milkfpsCLI.
 *
 * Lives in milkfps (no CLI dependency).
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
 * Set by milkfpsCLI constructor.
 * NULL when milkfpsCLI is not loaded.
 */
extern void *fps_generic_CLIfunction_ptr;

/**
 * @brief Global function pointer for
 *        fps_fill_farg_examples.
 *
 * Set by milkfpsCLI constructor.
 * NULL when milkfpsCLI is not loaded.
 */
extern void *fps_fill_farg_examples_ptr;

/**
 * @brief Name of the last FPS used by a CLI
 *        command (local or shared).
 *
 * Set after each successful V2 CLI execution.
 * Used by the ? query to show parameters.
 */
extern char fps_last_used_name[200];

/**
 * @brief fps_name of the command that last set
 *        fps_last_used_name.
 *
 * Used to scope the ? query per compute unit.
 */
extern char fps_last_used_cmdkey[200];


#endif /* FPS_CLI_FUNCTION_REGISTRY_H */
