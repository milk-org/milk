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

// These function NEED a week declaration: they MUST exist somewhere
// if the corresponding C file missed the mandate.
// so they exist here and do nothing.
MILK_WEAK errno_t COMPILERTEST_CLIADDCMD_MANDATE_CUDA() {return 0;};
MILK_WEAK errno_t COMPILERTEST_CLIADDCMD_MANDATE_LAPACKE() {return 0;};


// This function doesn't _need_ a weak definition, just a forward declaration
// so that it can be invoked in init_module_CLI.
// Since this function doesn't MANDATE anything, its definition in the C file always
// exists.
          errno_t COMPILERTEST_CLIADDCMD_REQUEST_CUDA();
// clang-format on

static errno_t init_module_CLI()
{
#ifdef HAVE_CUDA
    // Add mandating initializers here ?
#endif

    // Add non-mandating initializers here ?
    COMPILERTEST_CLIADDCMD_MANDATE_CUDA();
    COMPILERTEST_CLIADDCMD_REQUEST_CUDA();
    COMPILERTEST_CLIADDCMD_MANDATE_LAPACKE();

    return RETURN_SUCCESS;
}

MILK_MODULE(compilertest, init_module_CLI, NULL);

// Additional weak function definitions for things that need exported and miss a MANDATE
