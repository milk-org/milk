/**
 * @file    applyPF.c
 * @brief   Apply predictive filter
 *
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"



#ifdef HAVE_CUDA
#include "linalgebra/linalgebra.h"
#endif



static uint64_t *AOloopindex;

static char *indata;
static char *inmask;

static char *PFmat;

static char *outdata;
static char *outmask;

// shared by muplitple processes to keep track
static char *outPFstat;


static char *GPUsetstr;
static long  fpi_GPUsetstr;

static uint64_t *compOLresidual;
static long      fpi_compOLresidual;

static uint32_t *compOLresidualNBpt;
static long      fpi_compOLresidualNBpt;



static CLICMDARGDEF farg[] =
{
    {
        // AO loop index - used for automatic naming of streams aolX_
        CLIARG_UINT64,
        ".AOloopindex",
        "AO loop index",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &AOloopindex,
        NULL
    },
    {
        // Input stream
        CLIARG_STREAM,
        ".indata",
        "input data stream",
        "inim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &indata,
        NULL
    },
    {
        // Input stream active mask
        CLIARG_STREAM,
        ".inmask",
        "input data mask",
        "inmask",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmask,
        NULL
    },
    {
        // Prediction filter matrix
        CLIARG_STREAM,
        ".PFmat",
        "predictive filter matrix",
        "PFmat",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &PFmat,
        NULL
    },
    {
        // Output stream
        CLIARG_STREAM,
        ".outdata",
        "output data stream",
        "outPF",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outdata,
        NULL
    },
    {
        // Output mask
        CLIARG_STREAM,
        ".outmask",
        "output data mask",
        "outmask",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outmask,
        NULL
    },
    {
        // Output update
        CLIARG_STR,
        ".outPFstat",
        "output PF stats image",
        "outPFstat",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outPFstat,
        NULL
    },
    {
        // Set of GPU(s) for computation
        CLIARG_STR,
        ".GPUset",
        "column-separated list of GPUs",
        ":0:",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUsetstr,
        &fpi_GPUsetstr
    },
    {
        // compute residual mismatch
        CLIARG_ONOFF,
        ".comp.residual",
        "compute residual mismatch",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &compOLresidual,
        &fpi_compOLresidual
    },
    {
        // Set of GPU(s) for computation
        CLIARG_UINT32,
        ".comp.OLresidualNBpt",
        "sampling size for OL residual",
        "1000",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &compOLresidualNBpt,
        &fpi_compOLresidualNBpt
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
    "applyPF", "apply predictive filter", CLICMD_FIELDS_DEFAULTS
};




// detailed help
static errno_t help_function()
{


    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


#ifdef HAVE_CUDA
    int status;
    int GPUstatus[100];
    int GPUMATMULTCONFindex = 2;
#endif


    // Connect to 2D input stream
    //
    IMGID imgin = mkIMGID_from_name(indata);
    resolveIMGID(&imgin, ERRMODE_ABORT);
    long NBmodeINmax = imgin.md->size[0] * imgin.md->size[1];

    // connect to 2D predictive filter (PF) matrix
    //
    IMGID imgPFmat = mkIMGID_from_name(PFmat);
    resolveIMGID(&imgPFmat, ERRMODE_ABORT);
    long NBmodeOUT = imgPFmat.md->size[1];

    list_image_ID();



    // Input mask
    // 0: inactive input
    // 1: active input
    //
    IMGID imginmask = mkIMGID_from_name(inmask);
    resolveIMGID(&imginmask, ERRMODE_WARN);

    long  NBinmaskpix = 0;
    long *inmaskindex;
    if(imginmask.ID != -1)
    {
        NBinmaskpix = 0;
        for(uint32_t ii = 0;
                ii < imginmask.md->size[0] * imginmask.md->size[1];
                ii++)
            if(imginmask.im->array.SI8[ii] == 1)
            {
                NBinmaskpix++;
            }

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);
        if(inmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        NBinmaskpix = 0;
        for(uint32_t ii = 0;
                ii < imginmask.md->size[0] * imginmask.md->size[1];
                ii++)
            if(imginmask.im->array.SI8[ii] == 1)
            {
                inmaskindex[NBinmaskpix] = ii;
                NBinmaskpix++;
            }
        //printf("Number of active input modes  = %ld\n", NBinmaskpix);
    }
    else
    {
        NBinmaskpix = NBmodeINmax;
        printf("no input mask -> assuming NBinmaskpix = %ld\n", NBinmaskpix);

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);

        for(uint32_t ii = 0;
                ii < imginmask.md->size[0] * imginmask.md->size[1];
                ii++)
        {
            inmaskindex[NBinmaskpix] = ii;
        }
    }
    long NBmodeIN = NBinmaskpix;




    long NBPFstep = imgPFmat.md->size[0] / NBmodeIN;

    printf("Number of active input modes  = %ld  / %ld\n",
           NBmodeIN,
           NBmodeINmax);
    printf("Number of output modes        = %ld\n", NBmodeOUT);
    printf("Number of time steps          = %ld\n", NBPFstep);




    // create input buffer holding recent input values
    //
    printf("Creating input buffer\n");
    IMGID imginbuff = makeIMGID_2D("iminbuff", NBmodeIN, NBPFstep);
    createimagefromIMGID(&imginbuff);



    // create input buffer holding recent input values
    //
    printf("Creating output buffer\n");
    IMGID imgoutbuff = makeIMGID_2D("imoutbuff", NBmodeOUT, 1);
    createimagefromIMGID(&imgoutbuff);


    // Create output buffer holding recent output values
    // The buffer is used to measure residual OL error as a function of latency
    //
    printf("Creating output time buffer\n");
    IMGID imgoutTbuff = makeIMGID_2D("imoutTbuff", NBmodeOUT, NBPFstep);
    createimagefromIMGID(&imgoutTbuff);



    // OUTPUT

    // Connect to output mask and data stream
    //
    IMGID imgout = mkIMGID_from_name(outdata);
    resolveIMGID(&imgout, ERRMODE_WARN);

    IMGID imgoutmask = mkIMGID_from_name(outmask);
    resolveIMGID(&imgoutmask, ERRMODE_WARN);



    // output update
    // set values to 1 when updated
    //
    IMGID imgoutPFstat;
    {
        imgoutPFstat = stream_connect_create_2Df32(outPFstat, NBmodeINmax, 1);
    }



    // If both outdata and outmask exist, check they are consistent
    if((imgout.ID != -1) && (imgoutmask.ID != -1))
    {
        // compate image sizes (not type)
        int compOK = 1;
        if(imgout.naxis != imgoutmask.naxis)
        {
            printf("ERROR: naxis %d %d values don't match\n",
                   imgout.naxis,
                   imgoutmask.naxis);
            compOK = 0;
        }
        for(int dim = 0; dim < imgout.naxis; dim++)
        {
            if(imgout.size[dim] != imgoutmask.size[dim])
            {
                printf("ERROR: size[%d] %d %d values don't match\n",
                       dim,
                       imgout.size[dim],
                       imgoutmask.size[dim]);
                compOK = 0;
            }
        }


        if(compOK == 0)
        {
            PRINT_ERROR("images %s and %s are incompatible\n",
                        outdata,
                        outmask);
            DEBUG_TRACE_FEXIT();
            return (EXIT_FAILURE);
        }
    }
    else
    {
        if(imgout.ID != -1)
        {
            // outdata exists, but outmask does not
            //
            // Check that outdata is big enough
            //
            if(imgout.md->nelement < (uint64_t) NBmodeOUT)
            {
                PRINT_ERROR("images %s too small to contain %ld output modes\n",
                            outdata,
                            NBmodeOUT);
                DEBUG_TRACE_FEXIT();
                return (EXIT_FAILURE);
            }
            imcreatelikewiseIMGID(&imgoutmask, &imgout);
            for(uint32_t ii = 0; ii < NBmodeOUT; ii++)
            {
                imgoutmask.im->array.SI8[ii] = 1;
            }
        }
        else if(imgoutmask.ID != -1)
        {
            // outmask exists, but outdata does not
            // create outdata according to outmask
            //
            copyIMGID(&imgoutmask, &imgout);
            imgout.datatype = _DATATYPE_FLOAT;
            createimagefromIMGID(&imgout);
        }
        else
        {
            // Neither outdata nor outmask exist
            // 2D array
            //
            imgout = stream_connect_create_2Df32(outdata, NBmodeOUT, 1);
            imgout = stream_connect_create_2Df32(outmask, NBmodeOUT, 1);
            for(uint32_t ii = 0; ii < NBmodeOUT; ii++)
            {
                imgoutmask.im->array.SI8[ii] = 1;
            }
        }
    }

    // output mask index
    //
    long  NBoutmaskpix = 0;
    long *outmaskindex;
    if(imgoutmask.ID != -1)
    {
        NBoutmaskpix = 0;
        for(uint32_t ii = 0;
                ii < imginmask.md->size[0] * imginmask.md->size[1];
                ii++)
            if(imginmask.im->array.SI8[ii] == 1)
            {
                NBoutmaskpix++;
            }

        outmaskindex = (long *) malloc(sizeof(long) * NBoutmaskpix);
        if(outmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        NBoutmaskpix = 0;
        for(uint32_t ii = 0;
                ii < imgoutmask.md->size[0] * imgoutmask.md->size[1];
                ii++)
            if(imgoutmask.im->array.SI8[ii] == 1)
            {
                outmaskindex[NBoutmaskpix] = ii;
                NBoutmaskpix++;
            }
        //printf("Number of active input modes  = %ld\n", NBinmaskpix);
    }
    else
    {
        NBoutmaskpix = NBmodeOUT;
        printf("no output mask -> assuming NBoutmaskpix = %ld\n", NBoutmaskpix);

        outmaskindex = (long *) malloc(sizeof(long) * NBoutmaskpix);

        for(uint32_t ii = 0;
                ii < imgoutmask.md->size[0] * imgoutmask.md->size[1];
                ii++)
        {
            outmaskindex[NBoutmaskpix] = ii;
        }
    }
    if(NBmodeOUT != NBoutmaskpix)
    {
        PRINT_ERROR(
            "output mask active pix (%ld) not matching output dim %ld\n",
            NBoutmaskpix,
            NBmodeOUT);
        DEBUG_TRACE_FEXIT();
        return (EXIT_FAILURE);
    }




    // Identify GPUs
    //
    int  NBGPUmax = 20;
    int  NBGPU    = 0;
    int *GPUset   = (int *) malloc(sizeof(int) * NBGPUmax);
    for(int gpui = 0; gpui < NBGPUmax; gpui++)
    {
        char gpuistr[5];
        sprintf(gpuistr, ":%d:", gpui);
        if(strstr(GPUsetstr, gpuistr) != NULL)
        {
            GPUset[NBGPU] = gpui;
            printf("Using GPU device %d\n", gpui);
            NBGPU++;
        }
    }
    if(NBGPU > 0)
    {
        printf("Using %d GPUs\n", NBGPU);
    }
    else
    {
        printf("Using CPU\n");
    }

    list_image_ID();

    printf("MVM  %s %s -> %s\n",
           imginbuff.name,
           imgPFmat.name,
           imgoutbuff.name);


    //sprocessinfo_WriteMessage("MVM %d -> %d", NBmodeIN*NBPFstep, NBmodeOUT);

    // initialize OL residual measurement counter
    uint32_t OLrescnt  = 0;
    double  *OLRMS2res = (double *) malloc(sizeof(double) * NBPFstep);

    // average and time delay array on input OL buffer
    double *OLRMS2avedt =
        (double *) malloc(sizeof(double) * NBPFstep * NBPFstep);


    struct timespec t0, t1;

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    clock_gettime(CLOCK_MILK, &t0);

    // Fill in input buffer most recent measurement
    // At this point, the older measurements have already been moved down
    //
    for(long mi = 0; mi < NBmodeIN; mi++)
    {
        imginbuff.im->array.F[mi] = imgin.im->array.F[inmaskindex[mi]];
    }


    if(NBGPU > 0)  // if using GPU
    {


#ifdef HAVE_CUDA
        if(processinfo->loopcnt == 0)
        {
            printf("INITIALIZE GPU(s)\n\n");
            fflush(stdout);

            GPU_loop_MultMat_setup(GPUMATMULTCONFindex,
                                   imgPFmat.name,
                                   imginbuff.name,
                                   imgoutbuff.name,
                                   NBGPU,
                                   GPUset,
                                   0,
                                   1,
                                   1,
                                   *AOloopindex);

            printf("INITIALIZATION DONE\n\n");
            fflush(stdout);
        }
        GPU_loop_MultMat_execute(GPUMATMULTCONFindex,
                                 &status,
                                 &GPUstatus[100],
                                 1.0,
                                 0.0,
                                 0,
                                 0);
#endif
    }
    else // if using CPU
    {
        // compute output : matrix vector mult with a CPU-based loop
        imgout.md->write = 1;
        for(long mi = 0; mi < NBmodeOUT; mi++)
        {
            imgout.im->array.F[mi] = 0.0;
            for(uint32_t ii = 0; ii < NBmodeIN * NBPFstep; ii++)
            {
                imgout.im->array.F[mi] +=
                    imginbuff.im->array.F[ii] *
                    imgPFmat.im->array.F[mi * NBmodeIN * NBPFstep + ii];
            }
        }
        COREMOD_MEMORY_image_set_sempost_byID(imgout.ID, -1);
        imgout.md->write = 0;
        imgout.md->cnt0++;
    }


    // Place output block in main output
    //
    for(long mi = 0; mi < NBmodeOUT; mi++)
    {
        imgout.im->array.F[outmaskindex[mi]] = imgoutbuff.im->array.F[mi];
        imgoutPFstat.im->array.F[outmaskindex[mi]] = 1.0;
    }
    processinfo_update_output_stream(processinfo, imgoutPFstat.ID);
    processinfo_update_output_stream(processinfo, imgout.ID);



    clock_gettime(CLOCK_MILK, &t1);
    struct timespec tdiff;
    tdiff = timespec_diff(t0, t1);
    double t01d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    processinfo_WriteMessage_fmt(processinfo, "%dx%d->%d MVM %.3f us",
                                 NBmodeIN, NBPFstep, NBmodeOUT, t01d * 1e6);

    if(*compOLresidual == 1)
    {
        // Update time buffer output
        // shift down by 1 unit time
        //
        for(long tstep = NBPFstep - 1; tstep > 0; tstep--)
        {
            // shift down by 1 unit time
            for(long mi = 0; mi < NBmodeOUT; mi++)
            {
                imgoutTbuff.im->array.F[NBmodeOUT * tstep + mi] =
                    imgoutTbuff.im->array.F[NBmodeOUT * (tstep - 1) + mi];
            }
        }
        // update top entry
        for(long mi = 0; mi < NBmodeOUT; mi++)
        {
            imgoutTbuff.im->array.F[mi] = imgoutbuff.im->array.F[mi];
        }




        for(long tstep = 0; tstep < NBPFstep; tstep++)
        {

            // Compute OL residual as a function of latency
            // Evaluated for integer frame latency
            //
            double val2 = 0.0;
            for(long mi = 0; mi < NBmodeOUT; mi++)
            {
                double vdiff = imginbuff.im->array.F[mi] -
                               imgoutTbuff.im->array.F[NBmodeOUT * tstep + mi];
                val2 += vdiff * vdiff;
            }
            OLRMS2res[tstep] += val2;
        }

        for(long tstep = 1; tstep < NBPFstep; tstep++)
        {
            // Residual across time delay and ave on input OL
            //
            for(long tave = 1; tave < NBPFstep - tstep; tave++)
            {
                double val2 = 0.0;
                for(long mi = 0; mi < NBmodeOUT; mi++)
                {
                    double vave = 0.0;
                    for(long tstep1 = tstep; tstep1 < tstep + tave; tstep1++)
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


        if(OLrescnt == *compOLresidualNBpt)
        {

            long NBPFstep_display = NBPFstep;
            if(NBPFstep_display > 5)
            {
                NBPFstep_display = 5;
            }
            for(long tstep = 1; tstep < NBPFstep_display; tstep++)
            {
                printf("%ld-frame delay  ", tstep);

                // PREDICTION
                OLRMS2res[tstep] /= (*compOLresidualNBpt);
                printf("   %7.03f", 1000.0 * sqrt(OLRMS2res[tstep]));
                OLRMS2res[tstep] = 0.0;

                // PURE DELAY + AVE
                long tavemax_display = NBPFstep - tstep;
                if(tavemax_display > 5)
                {
                    tavemax_display = 5;
                }
                for(long tave = 1; tave < tavemax_display; tave++)
                {
                    OLRMS2avedt[tave * NBPFstep + tstep] /=
                        (*compOLresidualNBpt);
                    printf(" [ ave %ld %7.03f ] ",
                           tave,
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
    for(long tstep = NBPFstep - 1; tstep > 0; tstep--)
    {
        // tstep-1 -> tstep
        for(long mi = 0; mi < NBmodeIN; mi++)
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

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_LinARfilterPred__applyPF()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
