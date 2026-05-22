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

/* ID number corresponding to a name */
variableID variable_ID(const char *name)
{
    variableID i;
    variableID tmpID;
    int        loopOK;

    i      = 0;
    loopOK = 1;
    while (loopOK == 1)
    {
        if (dcvar[i].used == 1)
        {
            if ((strncmp(name, dcvar[i].name, strlen(name)) == 0) &&
                (dcvar[i].name[strlen(name)] == '\0'))
            {
                loopOK = 0;
                tmpID  = i;
            }
        }

        i++;
        if (i == dcnvar)
        {
            loopOK = 0;
            tmpID  = -1;
        }
    }

    return tmpID;
}

/* next available ID number */
variableID next_avail_variable_ID()
{
    variableID i;
    variableID ID    = -1;
    int        found = 0;

    for (i = 0; i < dcnvar; i++)
    {
        if ((dcvar[i].used == 0) && (found == 0))
        {
            ID    = i;
            found = 1;
        }
    }

    if (ID == -1)
    {
        ID = dcnvar;
    }

    return ID;
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
