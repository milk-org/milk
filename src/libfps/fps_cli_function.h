/**
 * @file    fps_cli_function.h
 * @brief   Generic CLIfunction and CLIADDCMD for FPS modules
 */

#ifndef FPS_CLI_FUNCTION_H
#define FPS_CLI_FUNCTION_H

#include "fps.h"
#include "fps_cli_binding.h"
#include "CLIcore/CLIcore_checkargs.h"

/** Compute function signature */
typedef errno_t (*fps_compute_fn)(void);


/**
 * @brief Generic CLIfunction for FPS modules.
 *
 * Handles the full milk CLI lifecycle:
 *   connect/create FPS → check args → sync → compute
 *
 * @param app_info    Application identity
 * @param farg        CLICMDARGDEF array
 * @param CLIcmddata  CLI command data
 * @param bindings    Parameter bindings
 * @param nb_b        Number of bindings
 * @param compute_fn  Module computation function
 * @return            RETURN_SUCCESS on success
 */
errno_t fps_generic_CLIfunction(
    FPS_APP_INFO    *app_info,
    CLICMDARGDEF    *farg,
    CLICMDDATA      *CLIcmddata,
    FPS_CLI_BINDING *bindings,
    int              nb_b,
    fps_compute_fn   compute_fn
);


/**
 * @brief Fill CLIcmddata and farg example fields
 *        from bindings' current values.
 *
 * @param farg      CLICMDARGDEF array
 * @param bindings  Parameter bindings
 * @param nb_b      Number of bindings
 */
void fps_fill_farg_examples(
    CLICMDARGDEF    *farg,
    FPS_CLI_BINDING *bindings,
    int              nb_b
);


#endif /* FPS_CLI_FUNCTION_H */
