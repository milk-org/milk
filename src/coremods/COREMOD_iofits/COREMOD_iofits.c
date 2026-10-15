// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file COREMOD_iofits.c
 * @brief Coremod iofits module
 */

/**
 * @file    COREMOD_iofits.c
 */

#define MODULE_SHORTNAME_DEFAULT "iofits"
#define MODULE_DESCRIPTION "Read/Write FITS files"

#include "milk_runnable.h"

#include "COREMOD_iofits_common.h"
#include "breakcube.h"
#include "images2cube.h"
#include "savefits.h"

COREMOD_IOFITS_DATA COREMOD_iofits_data = { .FITSIO_status = 0 };

// Mandated units are weak: NULL when their file is dropped by an unmet MILK_CMAKE_MANDATE_*.
MILK_RUNNABLE_WEAK_DECL(runnable_COREMOD_iofits_loadfits);

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_COREMOD_iofits_loadfits,
    &runnable_COREMOD_iofits_savefits,
    &runnable_COREMOD_iofits_breakcube,
    &runnable_COREMOD_iofits_images2cube,
};

MILK_MODULE_RUNNABLES(COREMOD_iofits, NULL, runnables);
