/**
 * @file    variable_ID.c
 * @brief   find variable ID(s) from name
 */

#include "CommandLineInterface/CLIcore.h"

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
            if (data.variable[i].used == 1)
                {
                    if ((strncmp(name, data.variable[i].name, strlen(name)) ==
                         0) &&
                        (data.variable[i].name[strlen(name)] == '\0'))
                        {
                            loopOK = 0;
                            tmpID  = i;
                        }
                }

            i++;
            if (i == data.NB_MAX_VARIABLE)
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

    for (i = 0; i < data.NB_MAX_VARIABLE; i++)
        {
            if ((data.variable[i].used == 0) && (found == 0))
                {
                    ID    = i;
                    found = 1;
                }
        }

    if (ID == -1)
        {
            ID = data.NB_MAX_VARIABLE;
        }

    return ID;
}

long compute_variable_memory()
{
    long totalvmem = 0;

    for (variableID i = 0; i < data.NB_MAX_VARIABLE; i++)
        {
            totalvmem += sizeof(VARIABLE);
            if (data.variable[i].used == 1)
                {
                    totalvmem += 0;
                }
        }
    return totalvmem;
}
