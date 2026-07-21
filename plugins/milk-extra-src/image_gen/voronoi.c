// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file voronoi.c
 * @brief Voronoi module
 */

#include "ImageStreamIO/ImageStruct.h"
#include <math.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "CLIcore.h"
#endif


// input points positions, ASCII file

/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "voronoi",
    .cmdkey           = "voronoi",
    .description      = "make Voronoi map from points file",
    .description_long = "Compute a Voronoi tessellation from a set of seed points. Each pixel is "
                        "assigned to the nearest seed, creating a segmented map."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char           *inpos = NULL;
static LOCVAR_OUTIMG2D outim;
static float          *radius  = NULL;
static float          *gapsize = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                             \
    X(".inpos", &inpos, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "points positions, filename") \
    X(".radius", &radius, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "radius")                      \
    X(".gapsize", &gapsize, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "gap size")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

/**
 * Create Voronoi map
 *
 * imgptspos holds ponts posistions
 * size coords x nbpt
 * for 100 points, 2D, this will be 2*100
 *
 *
 * Each following line is a point, with following format:
 * index x y
 *
 * (x,y) coordinates in range [0:1]
 *
 */
imageID image_gen_make_voronoi_map(IMGID *imgpos,
                                   IMGID *imgout,
                                   float  radius, // maximum radius of each Voronoi zone
                                   float  maxsep  // gap between Voronoi zones
)
{
    // resolve imgpos
    resolveIMGID(imgpos, ERRMODE_WARN, dcimg, dcnimg);

    // Create output image if needed
    imcreateIMGID(imgout);
    if (imgpos->ID == -1)
    {
        return RETURN_FAILURE;
    }


    uint32_t xsize  = imgout->md->size[0];
    uint32_t ysize  = imgout->md->size[1];
    uint64_t xysize = xsize * ysize;
    uint32_t NBpt   = imgpos->md->size[1];


    //printf("%u points\n", NBpt);

    int64_t *__restrict nearest_index;
    float *__restrict nearest_distance2;
    int64_t *__restrict nextnearest_index;
    float *__restrict nextnearest_distance2;
    int *__restrict gapim;

    nearest_index = (int64_t *) malloc(sizeof(int64_t) * xysize);
    if (nearest_index == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nearest_distance2 = (float *) malloc(sizeof(float) * xysize);
    if (nearest_distance2 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nextnearest_index = (int64_t *) malloc(sizeof(int64_t) * xysize);
    if (nextnearest_index == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nextnearest_distance2 = (float *) malloc(sizeof(float) * xysize);
    if (nextnearest_distance2 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    gapim = (int *) malloc(sizeof(int) * xysize);
    if (gapim == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    // initialize arrays
    float bigval = 1.0e20;
    for (uint64_t ii = 0; ii < xysize; ii++)
    {
        nearest_index[ii]          = -1;
        nearest_distance2[ii]      = bigval;
        nextnearest_index[ii]      = -1;
        nextnearest_distance2[ii]  = bigval;
        imgout->im->array.SI32[ii] = -1;
    }

    for (uint32_t ii = 0; ii < xsize; ii++)
    {
        for (uint32_t jj = 0; jj < ysize; jj++)
        {
            int   pindex = jj * xsize + ii;
            float x      = 2.0 * ii / xsize - 1.0;
            float y      = 2.0 * jj / ysize - 1.0;

            for (int pt = 0; pt < NBpt; pt++)
            {
                float dx = x - imgpos->im->array.F[2 * pt];
                float dy = y - imgpos->im->array.F[2 * pt + 1];

                float dist2 = dx * dx + dy * dy;

                if (dist2 < nearest_distance2[pindex])
                {
                    nextnearest_index[pindex]     = nearest_index[pindex];
                    nextnearest_distance2[pindex] = nearest_distance2[pindex];

                    nearest_index[pindex]     = pt;
                    nearest_distance2[pindex] = dist2;
                }
                else if (dist2 < nextnearest_distance2[pindex])
                {
                    nextnearest_index[pindex]     = pt;
                    nextnearest_distance2[pindex] = dist2;
                }
            }
            if ((nearest_distance2[pindex] < radius * radius))
            {
                imgout->im->array.SI32[pindex] = nearest_index[pindex];
            }
        }
    }


    // add gap
    int gapsizepix = (int) (maxsep * xsize);
    // int gapsizepix2 = (int) (maxsep*xsize/sqrt(2.0));

    for (uint32_t ii = 0; ii < xsize; ii++)
    {
        for (uint32_t jj = 0; jj < ysize; jj++)
        {
            gapim[jj * xsize + ii] = 0;
        }
    }

    for (uint32_t ii = gapsizepix; ii < xsize - gapsizepix; ii++)
    {
        for (uint32_t jj = gapsizepix; jj < ysize - gapsizepix; jj++)
        {
            int pindex0  = jj * xsize + ii;
            int pindex0p = jj * xsize + ii + gapsizepix;
            int pindex0m = jj * xsize + ii - gapsizepix;
            int pindexp0 = (jj + gapsizepix) * xsize + ii;
            int pindexm0 = (jj - gapsizepix) * xsize + ii;
            int pindexpp = (jj + gapsizepix) * xsize + ii + gapsizepix;
            int pindexpm = (jj + gapsizepix) * xsize + ii - gapsizepix;
            int pindexmp = (jj - gapsizepix) * xsize + ii + gapsizepix;
            int pindexmm = (jj - gapsizepix) * xsize + ii - gapsizepix;

            int32_t pv0p = imgout->im->array.SI32[pindex0p];
            int32_t pv0m = imgout->im->array.SI32[pindex0m];
            int32_t pvp0 = imgout->im->array.SI32[pindexp0];
            int32_t pvm0 = imgout->im->array.SI32[pindexm0];
            int32_t pvpp = imgout->im->array.SI32[pindexpp];
            int32_t pvpm = imgout->im->array.SI32[pindexpm];
            int32_t pvmp = imgout->im->array.SI32[pindexmp];
            int32_t pvmm = imgout->im->array.SI32[pindexmm];

            gapim[pindex0] = 1;

            if ((pv0p != pv0m) || (pvp0 != pvm0) || (pvpp != pvmm) || (pvpm != pvmp))
            {
                gapim[pindex0] = 0;
            }
        }
    }

    for (uint32_t ii = 0; ii < xsize; ii++)
    {
        for (uint32_t jj = 0; jj < ysize; jj++)
        {
            int pindex = jj * xsize + ii;
            if (gapim[pindex] == 0)
            {
                imgout->im->array.SI32[pindex] = -1;
            }
        }
    }


    free(nearest_index);
    free(nearest_distance2);
    free(nextnearest_index);
    free(nextnearest_distance2);

    free(gapim);


    return (imgout->ID);
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgpos = imgid_make_from_name(inpos);
    resolveIMGID(&imgpos, ERRMODE_WARN, dcimg, dcnimg);

    // link/create output image/stream
    FARG_OUTIM2DCREATE(outim, imgout, _DATATYPE_INT32);
    if (imgpos.ID == -1)
    {
        return RETURN_FAILURE;
    }


    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    {
        image_gen_make_voronoi_map(&imgpos, &imgout,
                                   *radius, // maximum radius of each Voronoi zone
                                   *gapsize // gap between Voronoi zones
        );


        processinfo_update_output_stream(processinfo, imgout.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgpos);
    imgid_free(&imgout);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_gen__voronoi()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
