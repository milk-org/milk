/**
 * @file    COREMOD_iofits.c
 * @brief   I/O for FITS files
 *
 * Uses CFITSIO library
 */

#define MODULE_SHORTNAME_DEFAULT ""
#define MODULE_DESCRIPTION       "Read/Write FITS files"

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_iofits_common.h"

#include "breakcube.h"
#include "images2cube.h"
#include "loadfits.h"
#include "read_keyword.h"
#include "savefits.h"

COREMOD_IOFITS_DATA COREMOD_iofits_data;


INIT_MODULE_LIB(COREMOD_iofits)


static errno_t init_module_CLI()
{
	COREMOD_iofits_data.FITSIO_status = 0;

	CLIADDCMD_COREMOD_iofits__loadfits();
	CLIADDCMD_COREMOD_iofits__saveFITS();

	breakcube_addCLIcmd();
	images2cube_addCLIcmd();


    // add atexit functions here

    return RETURN_SUCCESS;
}
