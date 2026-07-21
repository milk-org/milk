// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file applyPF.c
 * @brief Applypf module
 */


#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "linalgebra/linalgebra.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "applyPF",
    .cmdkey           = "applyPF",
    .description      = "apply predictive filter",
    .description_long = "Apply a linear predictive filter to a stream. Uses pre-computed filter "
                        "coefficients to predict future frames from past ones."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static uint64_t *AOloopindex        = NULL;
static char     *indata             = NULL;
static char     *inmask             = NULL;
static char     *PFmat              = NULL;
static char     *outdata            = NULL;
static char     *outmask            = NULL;
static char     *outPFstat          = NULL;
static char     *GPUsetstr          = NULL;
static uint64_t *compOLresidual     = NULL;
static uint32_t *compOLresidualNBpt = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                            \
    X(".AOloopindex", &AOloopindex, FPTYPE_UINT64, 1, FPFLAG_DEFAULT_INPUT, "AO loop index")     \
    X(".indata", &indata, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input data stream")       \
    X(".inmask", &inmask, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input data mask")         \
    X(".PFmat", &PFmat, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "predictive filter matrix")  \
    X(".outdata", &outdata, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output data stream")    \
    X(".outmask", &outmask, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output data mask")      \
    X(".outPFstat", &outPFstat, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output PF stats image") \
    X(".GPUset", &GPUsetstr, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                             \
      "column-separated list of GPUs")                                                           \
    X(".comp.residual", &compOLresidual, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                  \
      "compute residual mismatch")                                                               \
    X(".comp.OLresidualNBpt", &compOLresidualNBpt, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,       \
      "sampling size for OL residual")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


#ifdef HAVE_CUDA
    int status;
    int GPUstatus[100];
    int GPUMATMULTCONFindex = 2;
#endif


    // Connect to 2D input stream
    //
    IMGID imgin = imgid_make_from_name(indata);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    long NBmodeINmax = imgin.md->size[0] * imgin.md->size[1];
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    // connect to 2D predictive filter (PF) matrix
    //
    IMGID imgPFmat = imgid_make_from_name(PFmat);
    resolveIMGID(&imgPFmat, ERRMODE_WARN, dcimg, dcnimg);
    long NBmodeOUT = imgPFmat.md->size[1];
    if (imgPFmat.ID == -1)
    {
        return RETURN_FAILURE;
    }

    list_image_ID();


    // Input mask
    // 0: inactive input
    // 1: active input
    //
    IMGID imginmask = imgid_make_from_name(inmask);
    resolveIMGID(&imginmask, ERRMODE_WARN, dcimg, dcnimg);

    long  NBinmaskpix = 0;
    long *inmaskindex;
    if (imginmask.ID != -1)
    {
        NBinmaskpix = 0;
        for (uint32_t ii = 0; ii < imginmask.md->size[0] * imginmask.md->size[1]; ii++)
        {
            if (imginmask.im->array.SI8[ii] == 1)
            {
                NBinmaskpix++;
            }
        }

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);
        if (inmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        NBinmaskpix = 0;
        for (uint32_t ii = 0; ii < imginmask.md->size[0] * imginmask.md->size[1]; ii++)
        {
            if (imginmask.im->array.SI8[ii] == 1)
            {
                inmaskindex[NBinmaskpix] = ii;
                NBinmaskpix++;
            }
        }
        //printf("Number of active input modes  = %ld\n", NBinmaskpix);
    }
    else
    {
        NBinmaskpix = NBmodeINmax;
        printf("no input mask -> assuming NBinmaskpix = %ld\n", NBinmaskpix);

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);

        for (uint32_t ii = 0; ii < imginmask.md->size[0] * imginmask.md->size[1]; ii++)
        {
            inmaskindex[NBinmaskpix] = ii;
        }
    }
    long NBmodeIN = NBinmaskpix;


    long NBPFstep = imgPFmat.md->size[0] / NBmodeIN;

    printf("Number of active input modes  = %ld  / %ld\n", NBmodeIN, NBmodeINmax);
    printf("Number of output modes        = %ld\n", NBmodeOUT);
    printf("Number of time steps          = %ld\n", NBPFstep);


    // create input buffer holding recent input values
    //
    printf("Creating input buffer\n");
    IMGID imginbuff = imgid_make_from_name_2D("iminbuff", NBmodeIN, NBPFstep);
    createimagefromIMGID(&imginbuff);


    // create input buffer holding recent input values
    //
    printf("Creating output buffer\n");
    IMGID imgoutbuff = imgid_make_from_name_2D("imoutbuff", NBmodeOUT, 1);
    createimagefromIMGID(&imgoutbuff);


    // Create output buffer holding recent output values
    // The buffer is used to measure residual OL error as a function of latency
    //
    printf("Creating output time buffer\n");
    IMGID imgoutTbuff = imgid_make_from_name_2D("imoutTbuff", NBmodeOUT, NBPFstep);
    createimagefromIMGID(&imgoutTbuff);


    // OUTPUT

    // Connect to output mask and data stream
    //
    IMGID imgout = imgid_make_from_name(outdata);
    resolveIMGID(&imgout, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imgoutmask = imgid_make_from_name(outmask);
    resolveIMGID(&imgoutmask, ERRMODE_WARN, dcimg, dcnimg);


    // output update
    // set values to 1 when updated
    //
    IMGID imgoutPFstat;
    {
        imgoutPFstat = stream_connect_create_2Df32(outPFstat, NBmodeINmax, 1);
    }


    // If both outdata and outmask exist, check they are consistent
    if ((imgout.ID != -1) && (imgoutmask.ID != -1))
    {
        // compate image sizes (not type)
        int compOK = 1;
        if (imgout.md->naxis != imgoutmask.md->naxis)
        {
            printf("ERROR: naxis %d %d values don't match\n", imgout.md->naxis,
                   imgoutmask.md->naxis);
            compOK = 0;
        }
        for (int dim = 0; dim < imgout.md->naxis; dim++)
        {
            if (imgout.md->size[dim] != imgoutmask.md->size[dim])
            {
                printf("ERROR: size[%d] %d %d values don't match\n", dim, imgout.md->size[dim],
                       imgoutmask.md->size[dim]);
                compOK = 0;
            }
        }


        if (compOK == 0)
        {
            PRINT_ERROR("images %s and %s are incompatible\n", outdata, outmask);
            DEBUG_TRACE_FEXIT();
            return (EXIT_FAILURE);
        }
    }
    else
    {
        if (imgout.ID != -1)
        {
            // outdata exists, but outmask does not
            //
            // Check that outdata is big enough
            //
            if (imgout.md->nelement < (uint64_t) NBmodeOUT)
            {
                PRINT_ERROR("images %s too small to contain %ld output modes\n", outdata,
                            NBmodeOUT);
                DEBUG_TRACE_FEXIT();
                return (EXIT_FAILURE);
            }
            imcreatelikewiseIMGID(&imgoutmask, &imgout);
            for (uint32_t ii = 0; ii < NBmodeOUT; ii++)
            {
                imgoutmask.im->array.SI8[ii] = 1;
            }
        }
        else if (imgoutmask.ID != -1)
        {
            // outmask exists, but outdata does not
            // create outdata according to outmask
            //
            imgid_copy(&imgoutmask, &imgout);
            imgout.mdt->datatype = _DATATYPE_FLOAT;
            createimagefromIMGID(&imgout);
        }
        else
        {
            // Neither outdata nor outmask exist
            // 2D array
            //
            imgout     = stream_connect_create_2Df32(outdata, NBmodeOUT, 1);
            imgoutmask = stream_connect_create_2Df32(outmask, NBmodeOUT, 1);
            for (uint32_t ii = 0; ii < NBmodeOUT; ii++)
            {
                imgoutmask.im->array.SI8[ii] = 1;
            }
        }
    }

    // output mask index
    //
    long  NBoutmaskpix = 0;
    long *outmaskindex;
    if (imgoutmask.ID != -1)
    {
        NBoutmaskpix = 0;
        for (uint32_t ii = 0; ii < imginmask.md->size[0] * imginmask.md->size[1]; ii++)
        {
            if (imginmask.im->array.SI8[ii] == 1)
            {
                NBoutmaskpix++;
            }
        }

        outmaskindex = (long *) malloc(sizeof(long) * NBoutmaskpix);
        if (outmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        NBoutmaskpix = 0;
        for (uint32_t ii = 0; ii < imgoutmask.md->size[0] * imgoutmask.md->size[1]; ii++)
        {
            if (imgoutmask.im->array.SI8[ii] == 1)
            {
                outmaskindex[NBoutmaskpix] = ii;
                NBoutmaskpix++;
            }
        }
        //printf("Number of active input modes  = %ld\n", NBinmaskpix);
    }
    else
    {
        NBoutmaskpix = NBmodeOUT;
        printf("no output mask -> assuming NBoutmaskpix = %ld\n", NBoutmaskpix);

        outmaskindex = (long *) malloc(sizeof(long) * NBoutmaskpix);

        for (uint32_t ii = 0; ii < imgoutmask.md->size[0] * imgoutmask.md->size[1]; ii++)
        {
            outmaskindex[NBoutmaskpix] = ii;
        }
    }
    if (NBmodeOUT != NBoutmaskpix)
    {
        PRINT_ERROR("output mask active pix (%ld) not matching output dim %ld\n", NBoutmaskpix,
                    NBmodeOUT);
        DEBUG_TRACE_FEXIT();
        return (EXIT_FAILURE);
    }


    // Identify GPUs
    //
    int  NBGPUmax = 20;
    int  NBGPU    = 0;
    int *GPUset   = (int *) malloc(sizeof(int) * NBGPUmax);
    for (int gpui = 0; gpui < NBGPUmax; gpui++)
    {
        char gpuistr[5];
        snprintf(gpuistr, sizeof(gpuistr), ":%d:", gpui);
        if (strstr(GPUsetstr, gpuistr) != NULL)
        {
            GPUset[NBGPU] = gpui;
            printf("Using GPU device %d\n", gpui);
            NBGPU++;
        }
    }
    if (NBGPU > 0)
    {
        printf("Using %d GPUs\n", NBGPU);
    }
    else
    {
        printf("Using CPU\n");
    }

    list_image_ID();

    printf("MVM  %s %s -> %s\n", imginbuff.name, imgPFmat.name, imgoutbuff.name);


    //sprocessinfo_WriteMessage("MVM %d -> %d", NBmodeIN*NBPFstep, NBmodeOUT);

    // initialize OL residual measurement counter
    uint32_t OLrescnt  = 0;
    double  *OLRMS2res = (double *) malloc(sizeof(double) * NBPFstep);

    // average and time delay array on input OL buffer
    double *OLRMS2avedt = (double *) malloc(sizeof(double) * NBPFstep * NBPFstep);


    struct timespec t0, t1;

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    clock_gettime(CLOCK_MILK, &t0);

    // Fill in input buffer most recent measurement
    // At this point, the older measurements have already been moved down
    //
    for (long mi = 0; mi < NBmodeIN; mi++)
    {
        imginbuff.im->array.F[mi] = imgin.im->array.F[inmaskindex[mi]];
    }


    if (NBGPU > 0) // if using GPU
    {
#ifdef HAVE_CUDA
        if (processinfo->loopcnt == 0)
        {
            printf("INITIALIZE GPU(s)\n\n");
            fflush(stdout);

            GPU_loop_MultMat_setup(GPUMATMULTCONFindex, imgPFmat.name, imginbuff.name,
                                   imgoutbuff.name, NBGPU, GPUset, 0, 1, 1, *AOloopindex);

            printf("INITIALIZATION DONE\n\n");
            fflush(stdout);
        }
        GPU_loop_MultMat_execute(GPUMATMULTCONFindex, &status, &GPUstatus[100], 1.0, 0.0, 0, 0);
#endif
    }
    else // if using CPU
    {
        // compute output : matrix vector mult with a CPU-based loop
        imgout.md->write = 1;
        for (long mi = 0; mi < NBmodeOUT; mi++)
        {
            imgout.im->array.F[mi] = 0.0;
            for (uint32_t ii = 0; ii < NBmodeIN * NBPFstep; ii++)
            {
                imgout.im->array.F[mi] +=
                    imginbuff.im->array.F[ii] * imgPFmat.im->array.F[mi * NBmodeIN * NBPFstep + ii];
            }
        }
        COREMOD_MEMORY_image_set_sempost_byID(imgout.ID, -1);
        imgout.md->write = 0;
        imgout.md->cnt0++;
    }


    // Place output block in main output
    //
    for (long mi = 0; mi < NBmodeOUT; mi++)
    {
        imgout.im->array.F[outmaskindex[mi]]       = imgoutbuff.im->array.F[mi];
        imgoutPFstat.im->array.F[outmaskindex[mi]] = 1.0;
    }
    processinfo_update_output_stream(processinfo, imgoutPFstat.im, NULL);
    processinfo_update_output_stream(processinfo, imgout.im, NULL);


    clock_gettime(CLOCK_MILK, &t1);
    struct timespec tdiff;
    tdiff       = timespec_diff(t0, t1);
    double t01d = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    processinfo_WriteMessage_fmt(processinfo, "%dx%d->%d MVM %.3f us", NBmodeIN, NBPFstep,
                                 NBmodeOUT, t01d * 1e6);

    if (*compOLresidual == 1)
    {
        // Update time buffer output
        // shift down by 1 unit time
        //
        for (long tstep = NBPFstep - 1; tstep > 0; tstep--)
        {
            // shift down by 1 unit time
            for (long mi = 0; mi < NBmodeOUT; mi++)
            {
                imgoutTbuff.im->array.F[NBmodeOUT * tstep + mi] =
                    imgoutTbuff.im->array.F[NBmodeOUT * (tstep - 1) + mi];
            }
        }
        // update top entry
        for (long mi = 0; mi < NBmodeOUT; mi++)
        {
            imgoutTbuff.im->array.F[mi] = imgoutbuff.im->array.F[mi];
        }


        for (long tstep = 0; tstep < NBPFstep; tstep++)
        {
            // Compute OL residual as a function of latency
            // Evaluated for integer frame latency
            //
            double val2 = 0.0;
            for (long mi = 0; mi < NBmodeOUT; mi++)
            {
                double vdiff =
                    imginbuff.im->array.F[mi] - imgoutTbuff.im->array.F[NBmodeOUT * tstep + mi];
                val2 += vdiff * vdiff;
            }
            OLRMS2res[tstep] += val2;
        }

        for (long tstep = 1; tstep < NBPFstep; tstep++)
        {
            // Residual across time delay and ave on input OL
            //
            for (long tave = 1; tave < NBPFstep - tstep; tave++)
            {
                double val2 = 0.0;
                for (long mi = 0; mi < NBmodeOUT; mi++)
                {
                    double vave = 0.0;
                    for (long tstep1 = tstep; tstep1 < tstep + tave; tstep1++)
                    {
                        vave += imginbuff.im->array.F[NBmodeOUT * tstep1 + mi];
                    }
                    vave /= tave;
                    double vdiff = imginbuff.im->array.F[mi] - vave;
                    val2 += vdiff * vdiff;
                }
                OLRMS2avedt[tave * NBPFstep + tstep] += val2;
            }
        }


        if (OLrescnt == *compOLresidualNBpt)
        {
            long NBPFstep_display = NBPFstep;
            if (NBPFstep_display > 5)
            {
                NBPFstep_display = 5;
            }
            for (long tstep = 1; tstep < NBPFstep_display; tstep++)
            {
                printf("%ld-frame delay  ", tstep);

                // PREDICTION
                OLRMS2res[tstep] /= (*compOLresidualNBpt);
                printf("   %7.03f", 1000.0 * sqrt(OLRMS2res[tstep]));
                OLRMS2res[tstep] = 0.0;

                // PURE DELAY + AVE
                long tavemax_display = NBPFstep - tstep;
                if (tavemax_display > 5)
                {
                    tavemax_display = 5;
                }
                for (long tave = 1; tave < tavemax_display; tave++)
                {
                    OLRMS2avedt[tave * NBPFstep + tstep] /= (*compOLresidualNBpt);
                    printf(" [ ave %ld %7.03f ] ", tave,
                           1000.0 * sqrt(OLRMS2avedt[tave * NBPFstep + tstep]));
                    OLRMS2avedt[tave * NBPFstep + tstep] = 0.0;
                }
                printf("\n");
            }
            printf("\n");
            OLrescnt = 0;
        }
        OLrescnt++;
    }

    // Update time buffer input
    // do this now to save time when semaphore is posted
    //
    for (long tstep = NBPFstep - 1; tstep > 0; tstep--)
    {
        // tstep-1 -> tstep
        for (long mi = 0; mi < NBmodeIN; mi++)
        {
            imginbuff.im->array.F[NBmodeIN * tstep + mi] =
                imginbuff.im->array.F[NBmodeIN * (tstep - 1) + mi];
        }
    }


    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(GPUset);
    free(inmaskindex);
    free(OLRMS2res);
    free(OLRMS2avedt);

    imgid_free(&imgin);
    imgid_free(&imgPFmat);
    imgid_free(&imginmask);
    imgid_free(&imginbuff);
    imgid_free(&imgoutbuff);
    imgid_free(&imgoutTbuff);
    imgid_free(&imgout);
    imgid_free(&imgoutmask);
    imgid_free(&imgoutPFstat);

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

errno_t CLIADDCMD_LinARfilterPred__applyPF()
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
