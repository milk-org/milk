/**
 * @file    imcube_product.c
 * @brief   Compute product between two image cubes
 *
 *
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers

static char *inimc0;
static long  fpi_inimc0;

static char *inimc1;
static long  fpi_inimc1;

static char *inimmask;
static long  fpi_inimmask;

static char *imout;
static long  fpi_imout;


static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".imcube0",
        "input image cube 0",
        "imc0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimc0,
        &fpi_inimc0
    },
    {
        CLIARG_IMG,
        ".imcube1",
        "input image cube 1",
        "imc1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimc1,
        &fpi_inimc1
    },
    {
        CLIARG_IMG,
        ".immask",
        "pixel mask",
        "immask",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimmask,
        &fpi_inimmask
    },
    {
        CLIARG_STR,
        ".outim",
        "output matrix",
        "outm",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imout,
        &fpi_imout
    }
};

static CLICMDDATA CLIcmddata =
{
    "imcubeXprod", "cross product of two image cubes", CLICMD_FIELDS_DEFAULTS
};




// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




static errno_t imcube_crossproduct(IMGID imgcube0,
                                   IMGID imgcube1,
                                   IMGID imgmask,
                                   char *imoutname)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&imgcube0, ERRMODE_ABORT);
    resolveIMGID(&imgcube1, ERRMODE_ABORT);
    resolveIMGID(&imgmask, ERRMODE_ABORT);

    uint32_t xsize  = imgcube0.size[0];
    uint32_t ysize  = imgcube0.size[1];
    uint32_t zsize0 = imgcube0.size[2];
    uint32_t zsize1 = imgcube1.size[2];

    uint64_t xysize = xsize * ysize;

    IMGID imgout = makeIMGID_2D(imoutname, zsize0, zsize1);
    createimagefromIMGID(&imgout);

    // compute mask sum
    double masksum = 0.0;
    for(uint64_t pixi = 0; pixi < xysize; pixi++)
    {
        masksum += imgmask.im->array.F[pixi];
    }

    for(uint32_t kk0 = 0; kk0 < zsize0; kk0++)
    {
        for(uint32_t kk1 = kk0; kk1 < zsize1; kk1++)
        {
            double   tmpv     = 0.0;
            uint64_t z0offset = xysize * kk0;
            uint64_t z1offset = xysize * kk1;
            for(uint64_t pixi = 0; pixi < xysize; pixi++)
            {
                tmpv += imgmask.im->array.F[pixi] *
                        (imgcube0.im->array.F[z0offset + pixi] *
                         imgcube1.im->array.F[z1offset + pixi]);
            }
            imgout.im->array.F[kk1 * zsize0 + kk0] = tmpv / masksum;
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




/**
 * @brief Cross product of 2 image cubes
 *
 *
 * @return errno_t
 */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // connect to input mode values array and get number of modes
    //
    IMGID imginc0 = mkIMGID_from_name(inimc0);
    IMGID imginc1 = mkIMGID_from_name(inimc1);
    IMGID imgmask = mkIMGID_from_name(inimmask);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    imcube_crossproduct(imginc0, imginc1, imgmask, imout);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__imcube_crossproduct()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
