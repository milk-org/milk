#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/savefits.h"

#include "compute_SVDpseudoInverse.h"
#include "linalgebra/magma_compute_SVDpseudoInverse.h"

// Local variables pointers
static char *inputimname;
static char *inmaskname;
static char *mrespimname;
static char *outRMimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".inimname",
        "input image",
        "im",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inputimname,
        NULL
    },
    {
        CLIARG_IMG,
        ".inmaskname",
        "mask image",
        "mask",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmaskname,
        NULL
    },
    {
        CLIARG_IMG,
        ".mrespimname",
        "measured response images",
        "mresp",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &mrespimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outRM",
        "output RM image",
        "ourRM",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outRMimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "lincRMiter",
    "estimate response matrix from input and output",
    CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

//
// solve for response matrix given a series of input and output
// initial value of RM should be best guess
// inmask = 0 over input that are known to produce no response
//
errno_t linopt_compute_linRM_from_inout(const char *IDinput_name,
                                        const char *IDinmask_name,
                                        const char *IDoutput_name,
                                        const char *IDRM_name,
                                        imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID IDRM;
    imageID IDin;
    imageID IDinmask;
    imageID IDout;
    long    insize; // number of input
    long    xsizein, ysizein, xsizeout, ysizeout;
    double  fitval;
    long    kk, ii_in, jj_in, ii_out, jj_out;
    //double tot;
    imageID IDtmp;
    double  tmpv1;
    //long iter;
    imageID IDout1;
    //double alpha = 0.001;

    uint32_t *sizearray;
    imageID   IDpokeM; // poke matrix (input)
    //imageID IDoutM; // outputX
    double SVDeps = 1.0e-4;

    long    NBact, act;
    long   *inpixarray;
    long    spl; // sample measurement
    long    ii;
    imageID ID_rm;
    int     autoMask_MODE =
        0; // if 1, automatically measure input mask based on IDinput_name image
    imageID IDpinv;
    //int use_magma = 0;

    //int ngpu;

    //ngpu = 0;
    setenv("CUDA_VISIBLE_DEVICES", "3,4", 1);

    IDin  = image_ID(IDinput_name);
    IDout = image_ID(IDoutput_name);
    IDRM  = image_ID(IDRM_name);

    insize   = data.image[IDin].md[0].size[2];
    xsizeout = data.image[IDRM].md[0].size[0];
    ysizeout = data.image[IDRM].md[0].size[1];
    xsizein  = data.image[IDin].md[0].size[0];
    ysizein  = data.image[IDin].md[0].size[1];

    if(autoMask_MODE == 0)
    {
        IDinmask = image_ID(IDinmask_name);
    }
    else
    {
        create_2Dimage_ID("_RMmask", xsizein, ysizein, &IDinmask);
        for(spl = 0; spl < insize; spl++)
            for(ii = 0; ii < xsizein * ysizein; ii++)
                if(data.image[IDin].array.F[spl * xsizein * ysizein + ii] >
                        0.5)
                {
                    data.image[IDinmask].array.F[ii] = 1.0;
                }
    }

    // create pokeM
    NBact = 0;
    for(ii = 0; ii < xsizein * ysizein; ii++)
        if(data.image[IDinmask].array.F[ii] > 0.5)
        {
            NBact++;
        }

    printf("NBact = %ld\n", NBact);

    inpixarray = (long *) malloc(sizeof(long) * NBact);
    if(inpixarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    act = 0;
    for(ii = 0; ii < xsizein * ysizein; ii++)
        if(data.image[IDinmask].array.F[ii] > 0.5)
        {
            inpixarray[act] = ii;
            act++;
        }

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(sizearray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    sizearray[0] = NBact;
    sizearray[1] = insize; // number of measurements

    printf("NBact = %ld\n", NBact);
    for(act = 0; act < 10; act++)
    {
        printf("act %5ld -> pix %5ld\n", act, inpixarray[act]);
    }

    create_2Dimage_ID("pokeM", NBact, insize, &IDpokeM);

    for(spl = 0; spl < insize; spl++)
        for(act = 0; act < NBact; act++)
        {
            data.image[IDpokeM].array.F[NBact * spl + act] =
                data.image[IDin]
                .array.F[spl * xsizein * ysizein + inpixarray[act]];
        }
    save_fits("pokeM", "_test_pokeM.fits");

    // compute pokeM pseudo-inverse
#ifdef HAVE_MAGMA
    LINALGEBRA_magma_compute_SVDpseudoInverse("pokeM",
                                            "pokeMinv",
                                            SVDeps,
                                            insize,
                                            "VTmat",
                                            0,
                                            0,
                                            64,
                                            0, // GPU device
                                            NULL);
#else
    linopt_compute_SVDpseudoInverse("pokeM",
                                    "pokeMinv",
                                    SVDeps,
                                    insize,
                                    "VTmat",
                                    NULL);
#endif

    list_image_ID();
    save_fits("pokeMinv", "pokeMinv.fits");
    IDpinv = image_ID("pokeMinv");

    // multiply measurements by pokeMinv
    create_3Dimage_ID("_respmat",
                      xsizeout,
                      ysizeout,
                      xsizein * ysizein,
                      &ID_rm);

    for(act = 0; act < NBact; act++)
    {
        for(kk = 0; kk < insize; kk++)
            for(ii = 0; ii < xsizeout * ysizeout; ii++)
            {
                data.image[ID_rm]
                .array.F[inpixarray[act] * xsizeout * ysizeout + ii] +=
                    data.image[IDout].array.F[kk * xsizeout * ysizeout + ii] *
                    data.image[IDpinv].array.F[kk * NBact + act];
            }
    }
    save_fits("_respmat", "_test_RM.fits");
    //exit(0);

    // COMPUTE SOLUTION QUALITY

    IDRM = image_ID("_respmat");

    create_2Dimage_ID("_tmplicli", xsizeout, ysizeout, &IDtmp);
    create_3Dimage_ID("testout", xsizeout, ysizeout, insize, &IDout1);

    printf("IDin  = %ld\n", IDin);
    printf("IDout = %ld\n", IDout);
    printf("IDinmask = %ld\n", IDinmask);

    // on iteration 0, compute initial fit value
    fitval = 0.0;

    for(kk = 0; kk < insize; kk++)
    {
        printf("\r kk = %5ld / %5ld    ", kk, insize);
        fflush(stdout);

        for(ii_out = 0; ii_out < xsizeout; ii_out++)
            for(jj_out = 0; jj_out < ysizeout; jj_out++)
            {
                data.image[IDtmp].array.F[jj_out * xsizeout + ii_out] = 0.0;
            }

        for(ii_in = 0; ii_in < xsizein; ii_in++)
            for(jj_in = 0; jj_in < ysizein; jj_in++)
            {

                //printf("%ld  pix %ld %ld active\n", kk, ii_in, jj_in);
                for(ii_out = 0; ii_out < xsizeout; ii_out++)
                    for(jj_out = 0; jj_out < ysizeout; jj_out++)
                    {
                        data.image[IDtmp].array.F[jj_out * xsizeout + ii_out] +=
                            data.image[IDin].array.F[kk * xsizein * ysizein +
                                                     jj_in * xsizein + ii_in] *
                            data.image[IDRM]
                            .array.F[(jj_in * xsizein + ii_in) * xsizeout *
                                                               ysizeout +
                                                               jj_out * xsizeout + ii_out];
                    }
            }
        for(ii_out = 0; ii_out < xsizeout; ii_out++)
            for(jj_out = 0; jj_out < ysizeout; jj_out++)
            {
                tmpv1 = data.image[IDtmp].array.F[jj_out * xsizeout + ii_out] -
                        data.image[IDout].array.F[kk * xsizeout * ysizeout +
                                                  jj_out * xsizeout + ii_out];
                fitval += tmpv1 * tmpv1;
                data.image[IDout1].array.F[kk * xsizeout * ysizeout +
                                           jj_out * xsizeout + ii_out] =
                                               tmpv1; //data.image[IDtmp].array.F[jj_out*xsizeout+ii_out];
            }
    }
    printf("\n");
    printf("  %5ld    fitval = %.20f\n",
           kk,
           sqrt(fitval / xsizeout / ysizeout));

    delete_image_ID("_tmplicli", DELETE_IMAGE_ERRMODE_WARNING);

    free(sizearray);
    free(inpixarray);

    if(outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static char *inputimname;
static char *inmaskname;
static char *mrespimname;
static char *outRMimname;

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_compute_linRM_from_inout(inputimname,
                                    inmaskname,
                                    mrespimname,
                                    outRMimname,
                                    NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__linRM_from_inout()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
