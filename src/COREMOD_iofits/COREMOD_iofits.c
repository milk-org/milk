/**
 * @file    COREMOD_iofits.c
 * @brief   I/O for FITS files
 *
 * Uses CFITSIO library heavily
 *
 *
 */




/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT ""

// Module short description
#define MODULE_DESCRIPTION       "Read/Write FITS files"





#include <stdint.h>
#include <fitsio.h> /* required by every program that uses CFITSIO  */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <pthread.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "COREMOD_memory/COREMOD_memory.h"


#include "COREMOD_iofits_common.h"

#include "breakcube.h"
#include "images2cube.h"
#include "loadfits.h"
#include "read_keyword.h"
#include "savefits.h"



COREMOD_IOFITS_DATA COREMOD_iofits_data;











/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(COREMOD_iofits)



/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */














static errno_t init_module_CLI()
{
	COREMOD_iofits_data.FITSIO_status = 0;


	breakcube_addCLIcmd();
	images2cube_addCLIcmd();

	loadfits_addCLIcmd();
	savefits_addCLIcmd();
	
	
    // add atexit functions here

    return RETURN_SUCCESS;
}


























