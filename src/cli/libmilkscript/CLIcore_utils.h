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

#include "fps_procinfo_macros.h"

/** @brief Standard Function call wrapper
 *
 * CLI argument(s) is(are) parsed and checked with CLI_checkarray(), then
 * passed to the compute function call.
 *
 * Custom code may be added for more complex processing of function arguments.
 *
 * If CLI call arguments check out, go ahead with computation.
 * Arguments not contained in CLI call line are extracted from the
 * command argument list
 */
#undef INSERT_STD_CLIfunction
#define INSERT_STD_CLIfunction                                       \
    static errno_t CLIfunction(void)                                 \
    {                                                                \
        errno_t retval = CLI_checkarg_array(farg, CLIcmddata.nbarg); \
        if (retval == RETURN_SUCCESS)                                \
        {                                                            \
            STD_FARG_LINKfunction return compute_function();         \
        }                                                            \
        if (retval == RETURN_CLICHECKARGARRAY_HELP)                  \
        {                                                            \
            return RETURN_SUCCESS;                                   \
        }                                                            \
        if (retval == RETURN_CLICHECKARGARRAY_FUNCPARAMSET)          \
        {                                                            \
            return RETURN_SUCCESS;                                   \
        }                                                            \
        return retval;                                               \
    }


#undef INSERT_STD_CLIREGISTERFUNC
#define INSERT_STD_CLIREGISTERFUNC                                        \
    {                                                                     \
        if (getenv("MILK_FPSPROCINFO"))                                   \
        {                                                                 \
            CLIcmddata.flags |= CLICMDFLAG_PROCINFO;                      \
        }                                                                 \
        int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction); \
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;             \
    }

#endif
