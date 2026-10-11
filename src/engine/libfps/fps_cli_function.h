// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_function.h
 * @brief   Generic CLIfunction and CLIADDCMD for FPS modules
 *
 * Modules include this header to get the safe_fps_*
 * wrappers.  The generic CLIfunction pointer is stored
 * as void* in fps_cli_function_registry.h (part of
 * milkfps, no CLI dependency) and set by the CLI via
 * fps_cli_register().
 *
 * This header casts void* to the proper function type.
 */

#ifndef FPS_CLI_FUNCTION_H
#define FPS_CLI_FUNCTION_H

#include <stdio.h>

#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function_registry.h"
#include "fps_cli_init.h"
#include "libmilkdata/milkdata_clicmd.h"

/** Compute function signature */
typedef errno_t (*fps_compute_fn)(void);

/**
 * @brief Typed function pointer for
 *        fps_generic_CLIfunction.
 */
typedef errno_t (*fps_generic_CLIfunction_fn)(FPS_APP_INFO    *app_info,
                                              CLICMDARGDEF    *farg,
                                              CLICMDDATA      *CLIcmddata,
                                              FPS_CLI_BINDING *bindings,
                                              int              nb_b,
                                              fps_compute_fn   compute_fn);

/**
 * @brief Publish fps_generic_CLIfunction to modules.
 *
 * Defined in milkfpsCLI; called by the CLI startup.
 */
void fps_cli_register(void);


/* ---- Wrappers ---- */

/**
 * @brief Call fps_generic_CLIfunction; fails loudly if fps_cli_register() was never called.
 */
static inline errno_t safe_fps_generic_CLIfunction(FPS_APP_INFO    *app_info,
                                                   CLICMDARGDEF    *farg,
                                                   CLICMDDATA      *cd,
                                                   FPS_CLI_BINDING *bindings,
                                                   int              nb_b,
                                                   fps_compute_fn   compute_fn)
{
    if (fps_generic_CLIfunction_ptr)
    {
        fps_generic_CLIfunction_fn fn = (fps_generic_CLIfunction_fn) fps_generic_CLIfunction_ptr;
        return fn(app_info, farg, cd, bindings, nb_b, compute_fn);
    }
    fprintf(stderr, "ERROR: FPS CLI wrapper not registered (fps_cli_register() not called)\n");
    return RETURN_FAILURE;
}

/**
 * @brief Call fps_fill_farg_examples (kept for existing call sites).
 */
static inline void safe_fps_fill_farg_examples(CLICMDARGDEF    *farg,
                                               FPS_CLI_BINDING *bindings,
                                               int              nb_b)
{
    fps_fill_farg_examples(farg, bindings, nb_b);
}


#endif /* FPS_CLI_FUNCTION_H */
