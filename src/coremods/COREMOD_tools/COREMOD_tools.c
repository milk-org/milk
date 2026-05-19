/**
 * @file    COREMOD_tools.c
 * @brief   Frequently used tools
 *
 * Includes basic file I/O
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "tools"
#define MODULE_DESCRIPTION       "misc tools"

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "imdisplay3d.h"
#include "mvprocCPUset.h"
#include "statusstat.h"


INIT_MODULE_LIB(COREMOD_tools)

/**
 * @brief Register COREMOD_tools CLI commands.
 */
static errno_t init_module_CLI()
{
    CLIADDCMD_COREMOD_tools__mvprocCPUset();

    CLIADDCMD_COREMOD_tools__fileutils();
    CLIADDCMD_COREMOD_tools__imdisplay3d();
    CLIADDCMD_COREMOD_tools__statusstat();



    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */
