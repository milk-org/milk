// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    CLIcore_utils.h
 * @brief   Util functions and macros for coding convenience
 *
 */

#ifndef CLICORE_UTILS_H
#define CLICORE_UTILS_H

#include <string.h>

#include "CLIcore.h"

#include "libfps/IMGID.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fps_procinfo_macros.h" // Will define the macro that we undef later in this file.

#undef INSERT_STD_CLIREGISTERFUNC_NAMED
#define INSERT_STD_CLIREGISTERFUNC_NAMED(varCLIcmddata, varCLIfunction)            \
    {                                                                              \
        if (getenv("MILK_FPSPROCINFO"))                                            \
        {                                                                          \
            varCLIcmddata.flags |= CLICMDFLAG_PROCINFO;                            \
        }                                                                          \
        int cmdi                  = RegisterCLIcmd(varCLIcmddata, varCLIfunction); \
        varCLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;                   \
    }

#undef INSERT_STD_CLIREGISTERFUNC
#define INSERT_STD_CLIREGISTERFUNC INSERT_STD_CLIREGISTERFUNC_NAMED(CLIcmddata, CLIfunction)

/*
 * INSERT_STD_CLIREGISTERFUNC_ARGS() - register an extra command of a file with argument table @varfarg.
 * Its CLICMDDATA is declared with CLICMD_FIELDS_NOPARAM (before the table exists); without the table
 * the command's arguments are never parsed and it runs with the default parameter values.
 */
#undef INSERT_STD_CLIREGISTERFUNC_ARGS
#define INSERT_STD_CLIREGISTERFUNC_ARGS(varCLIcmddata, varCLIfunction, varfarg)       \
    {                                                                                 \
        varCLIcmddata.nbarg         = (int) (sizeof(varfarg) / sizeof(CLICMDARGDEF)); \
        varCLIcmddata.funcfpscliarg = varfarg;                                        \
        INSERT_STD_CLIREGISTERFUNC_NAMED(varCLIcmddata, varCLIfunction);              \
    }


#endif // #ifndef CLICORE_UTILS_H
