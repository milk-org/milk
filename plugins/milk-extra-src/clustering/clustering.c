// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    clustering.c
 * @brief   cluster imates and data
 *
 *
 */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "clustering"

// Module short description
#define MODULE_DESCRIPTION "Cluster images and data"

#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "clustering.h"

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_clustering_cubecluster,
    &runnable_clustering_mindiffscan,
};

MILK_MODULE_RUNNABLES(clustering, NULL, runnables);
