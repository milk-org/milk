/**
 * @file mkzercube.c
 * @brief Mkzercube module
 */

/**
 * @file mkzercube.c
 *
 */


#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include <math.h>

#include "zernike_value.h"


// zonal WFS response
//

/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "mkzerc",
    .cmdkey           = "mkzerc",
    .description      = "make Zernike modes cube",
    .description_long = "Generate a 3D cube of Zernike polynomial mode images. Each slice is one "
                        "Zernike mode evaluated on a circular aperture."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *outzcubename     = NULL;
static uint32_t *xsize            = NULL;
static uint32_t *ysize            = NULL;
static float    *xcent            = NULL;
static float    *ycent            = NULL;
static float    *radius           = NULL;
static float    *radiusmaskfactor = NULL;
static float    *TTfactor         = NULL;
static uint32_t *NBzermode        = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                        \
    X(".xsize", &xsize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "X size")    \
    X(".ysize", &ysize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "Y size")    \
    X(".xcent", &xcent, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "X center") \
    X(".ycent", &ycent, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "Y center") \
    X(".rad", &radius, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "radius")    \
    X(".NBzermode", &NBzermode, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "Number modes")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    zernike_init();

    IMGID imgout = imgid_make_from_name_3D(outzcubename, *xsize, *ysize, *NBzermode);
    createimagefromIMGID(&imgout);

    uint64_t xysize = *xsize;
    xysize *= *ysize;

    double *polar_r;
    double *polar_theta;


    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {
        polar_r = (double *) malloc(xysize * sizeof(double));
        if (polar_r == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        polar_theta = (double *) malloc(xysize * sizeof(double));
        if (polar_theta == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        if ((polar_r == NULL) || (polar_theta == NULL))
        {
            printf("error in memory allocation !!!\n");
        }

        // polar coordinates
        //
        for (uint32_t ii = 0; ii < *xsize; ii++)
        {
            float x = (*xcent) - ii;
            for (uint32_t jj = 0; jj < *ysize; jj++)
            {
                float y                     = (*ycent) - jj;
                polar_r[jj * (*xsize) + ii] = sqrt(x * x + y * y) / (*radius);

                polar_theta[jj * (*xsize) + ii] = atan2(y, x);
            }
        }


        // Make Zernikes
        //
        for (uint32_t zi = 0; zi < (*NBzermode); zi++)
        {
            float ampl = 1.0;
            if ((zi == 0) || (zi == 1))
            {
                ampl = *TTfactor;
            }
            else
            {
                ampl = 1.0;
            }
            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                float r = polar_r[ii];
                if (r < (*radiusmaskfactor))
                {
                    imgout.im->array.F[zi * xysize + ii] =
                        ampl * Zernike_value(zi + 1, r, polar_theta[ii]);
                }
                else
                {
                    imgout.im->array.F[ii] = 0.0;
                }
            }
        }
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(polar_r);
    free(polar_theta);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_ZernikePolyn__mkzercube()
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
