// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    read_shmimall.c
 * @brief   read all shared memory streams
 *
 * Uses FPS V2 framework.
 */

#include <fcntl.h>    // open
#include <sys/mman.h> // mmap
#include <sys/stat.h>
#include <unistd.h> // close

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"

#ifndef MILK_NO_CLI
#    include "streamCTRL_find_streams.h"
#endif


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "readshmimall",
    .cmdkey           = "readshmimall",
    .description      = "read all shared memory images",
    .description_long = "Connect to all shared memory image streams currently present in /dev/shm. "
                        "Maps every stream into the process address space."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char strfilter[FUNCTION_PARAMETER_STRMAXLEN] = "aol_";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".strfilter", strfilter, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "string filter")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t read_sharedmem_image_all(const char *strfilter)
{
#ifdef MILK_NO_CLI
    (void) strfilter;
    return RETURN_SUCCESS;
#else
    int         NBstreamMAX = 10000;
    STREAMINFO *streaminfo;

    streaminfo = (STREAMINFO *) calloc(NBstreamMAX, sizeof(STREAMINFO));

    int NBstream = find_streams(streaminfo, 1, strfilter);

    for (int sindex = 0; sindex < NBstream; sindex++)
    {
        if (!imgid_exists(streaminfo[sindex].sname))
        {
            read_sharedmem_image(streaminfo[sindex].sname, dcimg, dcnimg);
        }
    }

    free(streaminfo);

    return RETURN_SUCCESS;
#endif
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START FUNC_CHECK_RETURN(read_sharedmem_image_all(strfilter));

    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_COREMOD_memory__read_shmimall()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif
