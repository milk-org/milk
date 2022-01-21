/**
 * @file    COREMOD_tools.c
 * @brief   Frequently used tools
 *
 * Includes basic file I/O
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT ""
#define MODULE_DESCRIPTION       "misc tools"

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "imdisplay3d.h"
#include "mvprocCPUset.h"
#include "statusstat.h"

INIT_MODULE_LIB(COREMOD_tools)

static errno_t init_module_CLI()
{
    mvprocTset_addCLIcmd();
    mvprocTsetExt_addCLIcmd();
    mvprocCPUset_addCLIcmd();
    mvprocCPUsetExt_addCLIcmd();
    fileutils_addCLIcmd();
    imdisplay3d_addCLIcmd();
    statusstat_addCLIcmd();

    return RETURN_SUCCESS;
}
