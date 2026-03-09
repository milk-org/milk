/**
 * @file    compute_nb_variable.c
 */

#include "CLIcore.h"

long compute_nb_variable()
{
    long NBvar = 0;

    for(variableID i = 0; i < dcnvar; i++)
    {
        if(dcvar[i].used == 1)
        {
            NBvar += 1;
        }
    }

    return NBvar;
}
