// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    compilertest.c
 * @brief   Linear Algebra functions wrapper
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "compilertest"
#define MODULE_DESCRIPTION "Cmake stack test"

#include "CLIcore.h"
// #include "COREMOD_memory/COREMOD_memory.h"

// clang-format off

// Weak and non-weak CLIADDCMD declarations.

          errno_t FPSTEST_CLIADDCMD_FPSSYNCTEST();
// clang-format on

static errno_t init_module_CLI()
{
    // Add non-mandating initializers here.
    FPSTEST_CLIADDCMD_FPSSYNCTEST();

    return RETURN_SUCCESS;
}

MILK_MODULE(fpstest, init_module_CLI, NULL);

// Additional weak function definitions for things that need exported and miss a MANDATE
