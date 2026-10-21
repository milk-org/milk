// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    CLIcore_fps_query.h
 * @brief   FPS "?" query handler and last-used FPS state
 */

#ifndef CLICORE_FPS_QUERY_H
#define CLICORE_FPS_QUERY_H

#include "fps_cli_binding.h"

/** @brief Name of the last FPS used by a CLI command (local or shared). */
extern char fps_last_used_name[200];

/** @brief fps_name of the command that last set fps_last_used_name. */
extern char fps_last_used_cmdkey[200];

/**
 * @brief Print FPS info when user types "command ?"
 *
 * Lists local and shared FPS instances, then prints
 * parameter values from the most recent FPS or defaults.
 *
 * @param app_info  Application identity
 * @param bindings  Parameter bindings
 * @param nb_b      Number of bindings
 */
void fps_print_query_info(FPS_APP_INFO *app_info, FPS_CLI_BINDING *bindings, int nb_b);


#endif /* CLICORE_FPS_QUERY_H */
