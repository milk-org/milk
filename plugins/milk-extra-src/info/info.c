// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    info.c
 * @brief   Information about images
 *
 * Computes information about images
 *
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "info"
#define MODULE_DESCRIPTION "Image information and statistics"

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "info.h"

int infoscreen_wcol;
int infoscreen_wrow; // window size

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_info_cubeMatchMatrix, &runnable_info_cubestats,     &runnable_info_stream_monproc,
    &runnable_info_image_stats_1,   &runnable_info_image_stats_2, &runnable_info_improfile,
};

MILK_MODULE_RUNNABLES(info, NULL, runnables);
