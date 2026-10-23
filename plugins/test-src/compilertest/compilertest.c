// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    compilertest.c
 * @brief   Registry of the CMake dependency-system test units (REQUEST / MANDATE)
 */

#define MODULE_SHORTNAME_DEFAULT "compilertest"
#define MODULE_DESCRIPTION "Cmake stack test"

#include "milk_runnable.h"
#include "compilertest.h"

// Mandated units: NULL when their file is dropped by an unmet MILK_CMAKE_MANDATE_*.
MILK_RUNNABLE_WEAK_DECL(runnable_ct_mandatecuda);
MILK_RUNNABLE_WEAK_DECL(runnable_ct_mandatelapacke);

static const MILK_RUNNABLE *const runnables[] = {
    &runnable_ct_requestcuda,
    &runnable_ct_mandatecuda,
    &runnable_ct_mandatelapacke,
};

MILK_MODULE_RUNNABLES(compilertest, NULL, runnables);
