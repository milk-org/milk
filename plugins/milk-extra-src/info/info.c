/**
 * @file    info.c
 * @brief   Information about images
 *
 * Computes information about images
 *
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "info"
#define MODULE_DESCRIPTION "Image information and statistics"

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cubeMatchMatrix.h"
#include "cubestats.h"
#include "image_stats.h"
#include "improfile.h"
#include "stream_monproc.h"

int infoscreen_wcol;
int infoscreen_wrow; // window size

static errno_t init_module_CLI()
{
    CLIADDCMD_info__cubeMatchMatrix();
    CLIADDCMD_info__cubestats();

    CLIADDCMD_info__stream_monproc();

    CLIADDCMD_info__image_stats();
    CLIADDCMD_info__improfile();

    return RETURN_SUCCESS;
}

MILK_MODULE(info, init_module_CLI, NULL);
