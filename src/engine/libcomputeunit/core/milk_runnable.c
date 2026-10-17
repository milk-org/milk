// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_runnable.c
 * @brief   Runtime support for MILK_RUNNABLE descriptors (no CLI dependency)
 */

#include "milk_runnable.h"

/* Runnable whose compute() the fps_compute_fn thunk forwards to; set around each invocation. */
static const MILK_RUNNABLE *current_runnable;

void milk_runnable_prepare(const MILK_RUNNABLE *r)
{
    MILK_RUNNABLE_RT *rt = r->rt;
    CLICMDDATA       *cd = &rt->cmddata;

    strncpy(cd->key, r->app.cmdkey, sizeof(cd->key) - 1);
    strncpy(cd->description, r->app.description, sizeof(cd->description) - 1);
    strncpy(cd->sourcefilename, r->srcfile, sizeof(cd->sourcefilename) - 1);

    cd->nbarg               = r->nbarg;
    cd->funcfpscliarg       = r->farg;
    cd->flags               = r->flags | (r->kind == MILK_RUNNABLE_FPS ? CLICMDFLAG_FPS : 0);
    cd->FPS_customCONFcheck = r->confcheck;

    if (cd->cmdsettings == NULL)
    {
        cd->cmdsettings = &rt->cmdsettings_default;
    }
}

const MILK_RUNNABLE *milk_runnable_set_current(const MILK_RUNNABLE *r)
{
    const MILK_RUNNABLE *prev = current_runnable;
    current_runnable          = r;
    return prev;
}

errno_t milk_runnable_compute_thunk(void)
{
    return current_runnable->compute(current_runnable);
}
