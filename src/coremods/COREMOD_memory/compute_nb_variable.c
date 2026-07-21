// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file compute_nb_variable.c
 * @brief Compute nb variable module
 */

/**
 * @file    compute_nb_variable.c
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#endif

/**
 * @brief Count the number of active variables.
 */
long compute_nb_variable()
{
    long NBvar = 0;

    for (variableID i = 0; i < dcnvar; i++)
    {
        if (dcvar[i].used == 1)
        {
            NBvar += 1;
        }
    }

    return NBvar;
}
