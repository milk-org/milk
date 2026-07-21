// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file update_level.c
 * @brief Update level module
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"
#include "clustering_defs.h"


errno_t update_level(CLUSTERTREE *ctree, long CFindex)
{
    DEBUG_TRACE_FSTART();

    for (int cfi = 0; cfi < ctree->CFarray[CFindex].NBchild; cfi++)
    {
        long cfic                  = ctree->CFarray[CFindex].childindex[cfi];
        ctree->CFarray[cfic].level = ctree->CFarray[CFindex].level + 1;
        update_level(ctree, cfic);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
