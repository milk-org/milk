/**
 * @file mkzercube.c
 *
 */



#include "CommandLineInterface/CLIcore.h"

#include <math.h>

#include "zernike_value.h"


// zonal WFS response
//
static char *outzcubename;
static long  fpi_outzcubename;


static uint32_t *xsize;
static long  fpi_xsize;

static uint32_t *ysize;
static long  fpi_ysize;

static float *xcent;
static long  fpi_xcent;

static float *ycent;
static long  fpi_ycent;

static float *radius;
static long  fpi_radius;

static float *radiusmaskfactor;
static long  fpi_radiusmaskfactor;


static float *TTfactor;
static long fpi_TTfactor;

static uint32_t *NBzermode;
static long  fpi_NBzermode;



static CLICMDARGDEF farg[] =
{
    {
        // zonal RM WFS
        CLIARG_STR,
        ".outimg",
        "output image name",
        "zerc",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outzcubename,
        &fpi_outzcubename
    },
    {
        CLIARG_UINT32,
        ".xsize",
        "X size",
        "50",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &xsize,
        &fpi_xsize
    },
    {
        CLIARG_UINT32,
        ".ysize",
        "Y size",
        "50",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &ysize,
        &fpi_ysize
    },
    {
        CLIARG_FLOAT32,
        ".xcent",
        "X center",
        "24.5",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &xcent,
        &fpi_xcent
    },
    {
        CLIARG_FLOAT32,
        ".ycent",
        "Y center",
        "24.5",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &ycent,
        &fpi_ycent
    },
    {
        CLIARG_FLOAT32,
        ".rad",
        "radius",
        "24.5",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &radius,
        &fpi_radius
    },
    {
        CLIARG_FLOAT32,
        ".radmaskfact",
        "masking radius factor",
        "1.2",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &radiusmaskfactor,
        &fpi_radiusmaskfactor
    },
     {
        CLIARG_FLOAT32,
        ".TTfactor",
        "amplitude factor on TTr",
        "1.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &TTfactor,
        &fpi_TTfactor
    },
    {
        CLIARG_UINT32,
        ".NBzermode",
        "Number modes",
        "5",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &NBzermode,
        &fpi_NBzermode
    }
};




// Optional custom configuration setup. comptbuff
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {

    }

    return RETURN_SUCCESS;
}



// Optional custom configuration checks.
// Runs at every configuration check loop iteration
//
static errno_t customCONFcheck()
{

    if(data.fpsptr != NULL)
    {
    }

    return RETURN_SUCCESS;
}

static CLICMDDATA CLIcmddata =
{
    "mkzerc", "make Zernike modes cube", CLICMD_FIELDS_DEFAULTS
};




// detailed help
static errno_t help_function()
{


    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    zernike_init();

    IMGID imgout = makeIMGID_3D(outzcubename, *xsize, *ysize, *NBzermode);
    createimagefromIMGID(&imgout);

    uint64_t xysize = *xsize;
    xysize *= *ysize;

    double *polar_r;
    double *polar_theta;


    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {
        polar_r = (double *) malloc(xysize * sizeof(double));
        if(polar_r == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        polar_theta = (double *) malloc(xysize * sizeof(double));
        if(polar_theta == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        if((polar_r == NULL) || (polar_theta == NULL))
        {
            printf("error in memory allocation !!!\n");
        }

        // polar coordinates
        //
        for(uint32_t ii = 0; ii < *xsize; ii++)
        {
            float x = (*xcent) - ii;
            for(uint32_t jj = 0; jj < *ysize; jj++)
            {
                float y = (*ycent) - jj;
                polar_r[jj * (*xsize) + ii] = sqrt(x*x+y*y) / (*radius);

                polar_theta[jj * (*xsize) + ii] = atan2(y, x);
            }
        }


        // Make Zernikes
        //
        for(uint32_t zi = 0; zi < (*NBzermode); zi++)
        {
            float ampl = 1.0;
            if((zi == 0) || (zi == 1))
            {
                ampl = *TTfactor;
            }
            else
            {
                ampl = 1.0;
            }
            for(uint32_t ii = 0; ii < xysize; ii++)
            {
                float r = polar_r[ii];
                if(r < (*radiusmaskfactor))
                {
                    imgout.im->array.F[zi*xysize + ii] = ampl * Zernike_value(zi+1, r, polar_theta[ii]);
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




INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_ZernikePolyn__mkzercube()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
