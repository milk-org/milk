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

MILK_WEAK errno_t COMPILERTEST_CLIADDCMD_MANDATE_CUDA() {};
MILK_WEAK errno_t COMPILERTEST_CLIADDCMD_MANDATE_LAPACKE() {};

static errno_t init_module_CLI()
{
#ifdef HAVE_CUDA
    // Add mandating initializers here ?
#endif

    // Add non-mandating initializers here ?
    COMPILERTEST_CLIADDCMD_MANDATE_CUDA();
    COMPILERTEST_CLIADDCMD_REQUEST_CUDA();

    return RETURN_SUCCESS;
}

MILK_MODULE(compilertest, init_module_CLI, NULL);

// Additional weak function definitions for things that need exported and miss a MANDATE
