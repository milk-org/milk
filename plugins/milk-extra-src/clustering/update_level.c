// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "CommandLineInterface/CLIcore.h"
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
