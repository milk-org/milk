// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    compute_nb_variable.c
 */

#include "CommandLineInterface/CLIcore.h"

long compute_nb_variable()
{
    long NBvar = 0;

    for (variableID i = 0; i < data.NB_MAX_VARIABLE; i++)
    {
        if (data.variable[i].used == 1)
        {
            NBvar += 1;
        }
    }

    return NBvar;
}
