// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    COREMOD_tools.c
 * @brief   Frequently used tools
 *
 * Includes basic file I/O
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "tools"
#define MODULE_DESCRIPTION "misc tools"

#include "milk_runnable.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_COREMOD_tools_csetandprioext, &runnable_COREMOD_tools_rtprio,
    &runnable_COREMOD_tools_tsetpmove,      &runnable_COREMOD_tools_tsetpmoveext,
    &runnable_COREMOD_tools_csetpmove,      &runnable_COREMOD_tools_fileutils,
    &runnable_COREMOD_tools_imdisplay3d,    &runnable_COREMOD_tools_statusstat,
};

MILK_MODULE_RUNNABLES(COREMOD_tools, NULL, runnables);
