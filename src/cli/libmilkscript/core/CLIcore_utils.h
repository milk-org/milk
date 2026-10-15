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


#endif // #ifndef CLICORE_UTILS_H
