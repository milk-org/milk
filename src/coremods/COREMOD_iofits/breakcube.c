// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    breakcube.c
 * @brief   break cube into individual 2D images
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "breakcube",
    .cmdkey           = "breakcube",
    .description      = "break cube into individual images",
    .description_long = "Split a 3D FITS cube into individual 2D FITS files, one per slice along "
                        "the z-axis. Output files are numbered sequentially."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inname[FUNCTION_PARAMETER_STRMAXLEN] = "imc";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", inname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input cube image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

imageID break_cube(const char *restrict ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t xsize = imgin.md->size[0];
    uint32_t ysize = imgin.md->size[1];
    uint32_t nz    = imgin.md->size[2];

    for (uint32_t kk = 0; kk < nz; kk++)
    {
        char framename[STRINGMAXLEN_IMGNAME];
        CREATE_IMAGENAME(framename, "%s_%5u", ID_name, kk);
        for (long i = 0; i < (long) strlen(framename); i++)
        {
            if (framename[i] == ' ')
            {
                framename[i] = '0';
            }
        }

        IMGID imgfr       = imgid_make_from_name_2D(framename, xsize, ysize);
        imgfr.mdt->shared = 0;
        imgfr.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgfr);

        for (uint32_t ii = 0; ii < xsize; ii++)
        {
            for (uint32_t jj = 0; jj < ysize; jj++)
            {
                imgfr.im->array.F[jj * xsize + ii] =
                    imgin.im->array.F[kk * xsize * ysize + jj * xsize + ii];
            }
        }
    }

    return imgin.ID;
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

    INSERT_STD_PROCINFO_COMPUTEFUNC_START break_cube(inname);

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

errno_t CLIADDCMD_COREMOD_iofits__breakcube()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif
