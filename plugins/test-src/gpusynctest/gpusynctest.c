// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    gpusynctest.c
 * @brief   Registry of the GPU-resident stream synchronization test unit
 */

#define MODULE_SHORTNAME_DEFAULT "gpusynctest"
#define MODULE_DESCRIPTION "Test module: GPU-resident stream synchronization"

#include "milk_runnable.h"

// Mandated unit: NULL when gst_addone.c is dropped by an unmet MILK_CMAKE_MANDATE_CUDA.
MILK_RUNNABLE_WEAK_DECL(runnable_gst_addone);

static const MILK_RUNNABLE *const runnables[] = {
    &runnable_gst_addone,
};

MILK_MODULE_RUNNABLES(gpusynctest, NULL, runnables);
