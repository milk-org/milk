// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    compilertest.c
 * @brief   Linear Algebra functions wrapper
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "compilertest"
#define MODULE_DESCRIPTION "Cmake stack test"

#include "milk_runnable.h"
#include "fpstest.h"

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_fpstest_fps_sync_test,
};

MILK_MODULE_RUNNABLES(fpstest, NULL, runnables);


// Additional weak function definitions for things that need exported and miss a MANDATE
