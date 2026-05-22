/**
 * @file    fps_cli_query.h
 * @brief   FPS "?" query handler
 */

#ifndef FPS_CLI_QUERY_H
#define FPS_CLI_QUERY_H

#include "fps_cli_binding.h"

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


#endif /* FPS_CLI_QUERY_H */
