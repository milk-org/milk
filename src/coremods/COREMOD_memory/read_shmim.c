// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    read_shmim.c
 * @brief   read shared memory stream
 *
 * Uses FPS V2 framework.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "image_keyword_list.h"
#include "list_image.h"
#include "imageID.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "readshmim",
    .cmdkey      = "readshmim",
    .description = "read shared memory image",
    .description_long =
        "Connect to an existing shared memory image stream by name. Maps the stream into the "
        "current process address space for reading. Returns failure if the stream does not exist."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char insname[FUNCTION_PARAMETER_STRMAXLEN] = "stream";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_sname", insname, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT, "input stream")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

imageID read_sharedmem_image(const char *restrict sname, IMAGE *imagearray, long NB_images)
{
    IMGID img = imgid_make_from_name(sname);
    resolveIMGID(&img, ERRMODE_NULL, imagearray, NB_images);
    imgid_connect(&img, IMGID_CONNECT_NOCHECK);
    if (img.ID == -1)
    {
        return -1;
    }

    return RegisterIMGID(&img, imagearray, NB_images);
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

    INSERT_STD_PROCINFO_COMPUTEFUNC_START read_sharedmem_image(insname, dcimg, dcnimg);

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

errno_t CLIADDCMD_COREMOD_memory__read_sharedmem_image()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
