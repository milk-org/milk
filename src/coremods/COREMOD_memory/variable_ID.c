// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    variable_ID.c
 * @brief   find variable ID(s) from name
 */

#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#endif

/* One past the highest slot ever marked used (only grows): slots at or above it are all unused, so
 * a lookup stops there instead of scanning the whole table */
static variableID var_hwm = 0;

void variable_ID_note_used(variableID ID)
{
    if (ID >= var_hwm)
    {
        var_hwm = ID + 1;
    }
}

/* ID number corresponding to a name */
variableID variable_ID(const char *name)
{
    if (name == NULL || dcnvar <= 0)
    {
        return -1;
    }

    size_t     namelen = strlen(name);
    variableID nscan   = (var_hwm < dcnvar) ? var_hwm : dcnvar;
    for (variableID i = 0; i < nscan; i++)
    {
        if (dcvar[i].used == 1)
        {
            if ((strncmp(name, dcvar[i].name, namelen) == 0) && (dcvar[i].name[namelen] == '\0'))
            {
                return i;
            }
        }
    }

    return -1;
}

/* next available ID number, or -1 if full */
variableID next_avail_variable_ID()
{
    for (variableID i = 0; i < dcnvar; i++)
    {
        if (dcvar[i].used == 0)
        {
            return i;
        }
    }

    return -1;
}

/**
 * @brief Compute total memory used by variables.
 *
 * Sums the storage of all active variable entries.
 */
long compute_variable_memory()
{
    long totalvmem = 0;

    for (variableID i = 0; i < dcnvar; i++)
    {
        totalvmem += sizeof(VARIABLE);
        if (dcvar[i].used == 1)
        {
            totalvmem += 0;
        }
    }
    return totalvmem;
}
