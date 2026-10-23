// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_runnable_main.c
 * @brief   fpsexec entry point shared by all generated main() stubs
 *
 * Compiled into libcomputeunit; has no CLI dependency.
 */

#include "milk_runnable.h"

int milk_runnable_main(int argc, char *argv[], const MILK_RUNNABLE *r)
{
    if (r->kind != MILK_RUNNABLE_FPS)
    {
        fprintf(stderr, "Error: runnable '%s' is not an FPS runnable, it cannot be an fpsexec.\n",
                r->app.cmdkey);
        return 1;
    }

    milk_runnable_prepare(r);

    const MILK_RUNNABLE *prev = milk_runnable_set_current(r);

    int retval =
        fpsexec_main_function(argc, argv, (FPS_APP_INFO *) &r->app, r->bindings, r->nb_bindings,
                              r->farg, milk_runnable_compute_thunk, r->confcheck);

    milk_runnable_set_current(prev);
    return retval;
}
