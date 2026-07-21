// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    delete_sharedmem_image.c
 * @brief   delete shared memory image and files
 *
 * Uses FPS V2 framework.
 */

#include <malloc.h>
#include <sys/mman.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "list_image.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "rmshmim",
    .cmdkey           = "rmshmim",
    .description      = "remove shared image and files",
    .description_long = "Destroy a shared memory image stream and its associated files in "
                        "/dev/shm. Removes the stream from all connected processes."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char imname[FUNCTION_PARAMETER_STRMAXLEN] = "im";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) X(".imname", imname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "image name")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t destroy_shared_image_ID(const char *__restrict imname)
{
    IMGID img = imgid_make_from_name(imname);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);

    if ((img.ID != -1) && (img.im->md[0].shared == 1))
    {
        ImageStreamIO_destroyIm(img.im);
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm WARNING: shared image"
                " %s does not exist [ %s  "
                "%s  %d ] %c[%d;m\n",
                (char) 27, 1, 31, imname, __FILE__, __func__, __LINE__, (char) 27, 0);
    }

    return RETURN_SUCCESS;
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

    INSERT_STD_PROCINFO_COMPUTEFUNC_START FUNC_CHECK_RETURN(destroy_shared_image_ID(imname));

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

errno_t CLIADDCMD_COREMOD_memory__delete_sharedmem_image()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
