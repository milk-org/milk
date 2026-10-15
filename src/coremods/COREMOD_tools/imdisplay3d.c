// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imdisplay3d.c
 * @brief Display 2D image as 3D surface
 *
 * Uses FPS V2 framework.
 */

#include "milk_runnable.h"

#include "COREMOD_memory/COREMOD_memory.h"


/* forward decl */
errno_t COREMOD_TOOLS_imgdisplay3D(const char *IDname, long step);


/* ================================================================
 *  PARAMS
 * ============================================================= */

static char      p_imname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static long long p_step                                 = 5;

#define FPS_PARAMS(X)                                                                \
    X(".imname", p_imname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "image name") \
    X(".step", &p_step, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "pixel step")

MILK_PARAMS_DEFINE(imdisplay3d, FPS_PARAMS);

static MILK_HOT errno_t imdisplay3d_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    COREMOD_TOOLS_imgdisplay3D(p_imname, p_step);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_tools_imdisplay3d = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "dispim3d",
                 .cmdkey      = "dispim3d",
                 .description = "display 2D image as 3D surface using gnuplot",
                 .description_long =
                     "Display slices of a 3D image cube interactively. Provides a TUI-based "
                     "viewer for browsing through cube frames." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imdisplay3d),
    .compute = imdisplay3d_compute,
};

// displays 2D image in 3D using gnuplot
//
errno_t COREMOD_TOOLS_imgdisplay3D(const char *IDname, long step)
{
    imageID ID;
    long    xsize, ysize;
    long    ii;
    char    cmd[512];
    FILE   *fp;
    FILE   *fpgnuplot;

    ID = image_ID(IDname, dcimg, dcnimg);
    if (ID == -1)
    {
        PRINT_ERROR("image \"%s\" not found", IDname);
        return RETURN_FAILURE;
    }
    xsize = dcimg[ID].md[0].size[0];
    ysize = dcimg[ID].md[0].size[1];

    snprintf(cmd, 512, "gnuplot");

    if ((fpgnuplot = popen(cmd, "w")) == NULL)
    {
        PRINT_ERROR("could not connect to gnuplot");
        return RETURN_FAILURE;
    }

    printf("image: %s [%ld x %ld], step = %ld\n", IDname, xsize, ysize, step);

    fprintf(fpgnuplot, "set pm3d\n");
    fprintf(fpgnuplot, "set hidden3d\n");
    fprintf(fpgnuplot, "set palette\n");
    fflush(fpgnuplot);

    fp = fopen("pts.dat", "w");
    if (fp == NULL)
    {
        PRINT_ERROR("cannot open file \"pts.dat\"");
        pclose(fpgnuplot);
        return RETURN_FAILURE;
    }
    fprintf(fpgnuplot, "splot \"-\" w d notitle\n");
    for (ii = 0; ii < xsize; ii += step)
    {
        for (long jj = 0; jj < ysize; jj += step)
        {
            fprintf(fpgnuplot, "%ld %ld %f\n", ii, jj, dcimg[ID].array.F[jj * xsize + ii]);
            fprintf(fp, "%ld %ld %f\n", ii, jj, dcimg[ID].array.F[jj * xsize + ii]);
        }
        fprintf(fpgnuplot, "\n");
        fprintf(fp, "\n");
    }
    fprintf(fpgnuplot, "e\n");
    fflush(fpgnuplot);
    fclose(fp);
    pclose(fpgnuplot);

    return RETURN_SUCCESS;
}
