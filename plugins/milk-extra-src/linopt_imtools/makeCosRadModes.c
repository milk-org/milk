#include <math.h>

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char   *outimname;
static long   *sizeout;
static long   *kmaxval;
static double *radiusval;
static double *radfactorlimval;

static CLICMDARGDEF farg[] = {{
        CLIARG_STR,
        ".outim",
        "output image",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_INT64,
        ".size",
        "size",
        "512",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &sizeout,
        NULL
    },
    {
        CLIARG_INT64,
        ".kmax",
        "k max",
        "100",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &kmaxval,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".radius",
        "radius [pix]",
        "160.0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &radiusval,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".rfactlim",
        "radius factor limit",
        "2.0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &radfactorlimval,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "mkcosrmodes", "make basis of cosine radial modes", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

//
// make cosine radial modes
//
errno_t linopt_imtools_makeCosRadModes(const char *ID_name,
                                       long        size,
                                       long        kmax,
                                       float       radius,
                                       float       radfactlim,
                                       imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID ID;
    long    size2;
    imageID IDr;
    FILE   *fp;

    size2 = size * size;
    create_2Dimage_ID("linopt_tmpr", size, size, &IDr);

    fp = fopen("ModesExpr_CosRad.txt", "w");
    fprintf(fp, "# unit for r = %f pix\n", radius);
    fprintf(fp, "\n");
    for(long k = 0; k < kmax; k++)
    {
        fprintf(fp, "%5ld   cos(r*M_PI*%ld)\n", k, k);
    }

    fclose(fp);

    for(long ii = 0; ii < size; ii++)
    {
        float x = (1.0 * ii - 0.5 * size) / radius;
        for(long jj = 0; jj < size; jj++)
        {
            float y = (1.0 * jj - 0.5 * size) / radius;
            float r = sqrt(x * x + y * y);
            data.image[IDr].array.F[jj * size + ii] = r;
        }
    }

    FUNC_CHECK_RETURN(create_3Dimage_ID(ID_name, size, size, kmax, &ID));

    for(long k = 0; k < kmax; k++)
        for(long ii = 0; ii < size2; ii++)
        {
            float r = data.image[IDr].array.F[ii];
            if(r < radfactlim)
            {
                data.image[ID].array.F[k * size2 + ii] = cos(r * M_PI * k);
            }
        }

    FUNC_CHECK_RETURN(
        delete_image_ID("linopt_tmpr", DELETE_IMAGE_ERRMODE_WARNING));

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_imtools_makeCosRadModes(outimname,
                                   *sizeout,
                                   *kmaxval,
                                   *radiusval,
                                   *radfactorlimval,
                                   NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__makeCosRadModes()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
