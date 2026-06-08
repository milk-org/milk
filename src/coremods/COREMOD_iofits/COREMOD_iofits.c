/**
 * @file COREMOD_iofits.c
 * @brief Coremod iofits module
 */

/**
 * @file    COREMOD_iofits.c
 */

#define MODULE_SHORTNAME_DEFAULT "iofits"
#define MODULE_DESCRIPTION "Read/Write FITS files"

#include "COREMOD_iofits_common.h"

COREMOD_IOFITS_DATA COREMOD_iofits_data = { .FITSIO_status = 0 };

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "CLIcore.h"

#    include "breakcube.h"
#    include "images2cube.h"
#    include "loadfits.h"
#    include "read_keyword.h"
#    include "savefits.h"

// External function from savefits.c
//extern errno_t save_fl_fits(const char *inputimname, const char *outputFITSname);

/**
 * @brief Register COREMOD_iofits CLI commands.
 */
static errno_t init_module_CLI()
{
    COREMOD_iofits_data.FITSIO_status = 0;

    CLIADDCMD_COREMOD_iofits__loadfits();
    CLIADDCMD_COREMOD_iofits__saveFITS();

    CLIADDCMD_COREMOD_iofits__breakcube();
    CLIADDCMD_COREMOD_iofits__images2cube();

    // add atexit functions here

    return RETURN_SUCCESS;
}

// Dummy function to ensure symbols are exported
/**
 * @brief Export symbol to ensure library is linked.
 *
 * Empty function referenced externally to prevent
 * the linker from discarding this module.
 */
MILK_MODULE(COREMOD_iofits, init_module_CLI, NULL);

#endif /* ELSE MILK_NO_CLI */
