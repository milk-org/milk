// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "image_make3D.h"
#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char     *outimname;
static uint32_t *imxsize;
static uint32_t *imysize;
static uint32_t *imzsize;

static CLICMDARGDEF farg[] = {
    { CLIARG_STR, ".out_name", "output image", "out1", CLIARG_VISIBLE_DEFAULT, (void **) &outimname,
      NULL },
    { CLIARG_INT64, ".xsize", "x size", "512", CLIARG_VISIBLE_DEFAULT, (void **) &imxsize, NULL },
    { CLIARG_INT64, ".ysize", "y size", "512", CLIARG_VISIBLE_DEFAULT, (void **) &imysize, NULL },
    { CLIARG_INT64, ".zsize", "z size", "512", CLIARG_VISIBLE_DEFAULT, (void **) &imzsize, NULL }
};

static CLICMDDATA CLIcmddata = { "mk3Dim", "make 3D image", CLICMD_FIELDS_DEFAULTS };

// detailed help
static errno_t help_function()
{
    printf("attributes :\n"
           " s> shared\n"
           "k20> : 20 keywords");

    return RETURN_SUCCESS;
}

imageID make_3Dimage_IMGID(IMGID *img)
{
    // Create image if needed
    imcreateIMGID(img);

    return (img->ID);
}

imageID make_3Dimage(const char *name, uint32_t xsize, uint32_t ysize, uint32_t zsize)
{
    IMGID img = makeIMGID_3D(name, xsize, ysize, zsize);
    return make_3Dimage_IMGID(&img);
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID img = makeIMGID_3D(outimname, *imxsize, *imysize, *imzsize);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    make_3Dimage_IMGID(&img);

    processinfo_update_output_stream(processinfo, img.ID);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

    // Register function in CLI
    errno_t CLIADDCMD_COREMOD_memory__mk3Dim()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
