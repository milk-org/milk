#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "image_fitModes.h"



/*
static void **vclean;
static size_t nclean;

void milk_atexit_free_add(void *data)
{
    vclean = realloc(vclean, sizeof(void *) * (nclean + 1));
    vclean[nclean++] = data;
}

void milk_memclean(void)
{
    size_t i;

    for (i = 0; i < nclean; i++) {
        free(vclean[i]);
    }
    free(vclean);
}
*/




// Local variables pointers
static char *infname;
static long *NBptval;
static long *maxorderval;
static char *outfname;
static long *modeval;

static CLICMDARGDEF farg[] = {{
        CLIARG_STR,
        ".indat",
        "input file",
        "data.txt",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &infname,
        NULL
    },
    {
        CLIARG_INT64,
        ".NBpt",
        "number of sample points",
        "1000",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &NBptval,
        NULL
    },
    {
        CLIARG_INT64,
        ".maxorder",
        "maximum polynomial order",
        "8",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &maxorderval,
        NULL
    },
    {
        CLIARG_STR,
        ".outdat",
        "output file",
        "fitsol.txt",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outfname,
        NULL
    },
    {
        CLIARG_INT64,
        ".mode",
        "fit mode",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &modeval,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "linopt1Dfit", "least-square 1D fit", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

// MODE :
// 0 : polynomial
//
errno_t linopt_compute_1Dfit(const char *fnamein,
                             long        NBpt,
                             long        MaxOrder,
                             const char *fnameout,
                             int         MODE,
                             imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID IDin, IDin0;
    imageID IDmask;
    imageID IDmodes;
    long    NBmodes;

    float SVDeps = 0.0000001;

    long  NBiter = 100;
    float gain   = 1.0;
    long  iter;

    //atexit(milk_memclean);

    float * __restrict xarray;
    xarray = (float *) malloc(sizeof(float) * NBpt);
    if(xarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }
    //milk_atexit_free_add(xarray);

    float * __restrict valarray;
    valarray = (float *) malloc(sizeof(float) * NBpt);
    if(valarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }
    //milk_atexit_free_add(valarray);


    {
        FILE *fp;
        fp = fopen(fnamein, "r");
        for(int_fast32_t ii = 0; ii < NBpt; ii++)
        {
            int fscanfcnt = fscanf(fp, "%f %f\n", &xarray[ii], &valarray[ii]);

            if(fscanfcnt == EOF)
            {
                if(ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr,
                            "Error: fscanf reached end of file, no matching "
                            "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if(fscanfcnt != 2)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i input "
                        "items, 2 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }
        }
        fclose(fp);
    }

    FUNC_CHECK_RETURN(create_2Dimage_ID("invect", NBpt, 1, &IDin));

    FUNC_CHECK_RETURN(create_2Dimage_ID("invect0", NBpt, 1, &IDin0));

    FUNC_CHECK_RETURN(create_2Dimage_ID("inmask", NBpt, 1, &IDmask));

    for(uint_fast32_t ii = 0; ii < NBpt; ii++)
    {
        //			printf("%18.16f  %+18.16f\n", xarray[ii], valarray[ii]);
        data.image[IDin].array.F[ii]   = valarray[ii];
        data.image[IDin0].array.F[ii]  = valarray[ii];
        data.image[IDmask].array.F[ii] = 1.0;
    }

    NBmodes = MaxOrder;
    FUNC_CHECK_RETURN(
        create_3Dimage_ID("fitmodes", NBpt, 1, NBmodes, &IDmodes));

    imageID IDout;
    FUNC_CHECK_RETURN(create_2Dimage_ID("outcoeff", NBmodes, 1, &IDout));

    switch(MODE)
    {
    case 0:
        for(uint_fast32_t m = 0; m < NBmodes; m++)
        {
            for(uint_fast32_t ii = 0; ii < NBpt; ii++)
            {
                data.image[IDmodes].array.F[m * NBpt + ii] =
                    pow(xarray[ii], 1.0 * m);
            }
        }
        break;
    case 1:
        for(uint_fast32_t m = 0; m < NBmodes; m++)
        {
            for(uint_fast32_t ii = 0; ii < NBpt; ii++)
            {
                data.image[IDmodes].array.F[m * NBpt + ii] =
                    cos(xarray[ii] * M_PI * m);
            }
        }
        break;
    default:
        printf("ERROR: MODE = %d not supported\n", MODE);
        exit(0);
        break;
    }


    for(iter = 0; iter < NBiter; iter++)
    {
        FUNC_CHECK_RETURN(linopt_imtools_image_fitModes("invect0",
                          "fitmodes",
                          "inmask",
                          SVDeps,
                          "outcoeffim0",
                          1,
                          NULL));
        imageID IDout0 = image_ID("outcoeffim0");

        for(uint_fast32_t m = 0; m < NBmodes; m++)
        {
            data.image[IDout].array.F[m] +=
                gain * data.image[IDout0].array.F[m];
        }

        double err = 0.0;
        for(int_fast32_t ii = 0; ii < NBpt; ii++)
        {
            double val = 0.0;
            for(int_fast32_t m = 0; m < NBmodes; m++)
            {
                val += data.image[IDout].array.F[m] *
                       data.image[IDmodes].array.F[m * NBpt + ii];
            }
            data.image[IDin0].array.F[ii] = data.image[IDin].array.F[ii] - val;
            err +=
                data.image[IDin0].array.F[ii] * data.image[IDin0].array.F[ii];
        }
        err = sqrt(err / NBpt);
        printf("ITERATION %4ld   residual = %20g   [gain = %20g]\n",
               iter,
               err,
               gain);
        gain *= 0.95;
    }

    {
        FILE * fp;
        fp = fopen(fnameout, "w");
        for(long m = 0; m < NBmodes; m++)
        {
            fprintf(fp, "%4ld %+.8g\n", m, data.image[IDout].array.F[m]);
        }
        fclose(fp);
    }

    {
        FILE * fp;
        fp  = fopen("testout.txt", "w");
        double err = 0.0;
        for(int_fast32_t ii = 0; ii < NBpt; ii++)
        {
            double val = 0.0;
            for(int_fast32_t m = 0; m < NBmodes; m++)
            {
                val += data.image[IDout].array.F[m] *
                       data.image[IDmodes].array.F[m * NBpt + ii];
            }
            double vale = valarray[ii] - val;
            err += vale * vale;
            fprintf(fp,
                    "%05ld  %18.16f  %18.16f   %18.16f\n",
                    ii,
                    xarray[ii],
                    valarray[ii],
                    val);
        }
        fclose(fp);

        err = sqrt(err / NBpt);
        printf("FIT error = %g m\n", err);
    }

    free(xarray);
    free(valarray);

    if(outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_compute_1Dfit(infname,
                         *NBptval,
                         *maxorderval,
                         outfname,
                         *modeval,
                         NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__lin1Dfits()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
