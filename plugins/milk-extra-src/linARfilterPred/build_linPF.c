/**
 * @file build_linPF.c
 *
 *
 */


#include <math.h>
#include <time.h>

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"
#include "COREMOD_iofits/COREMOD_iofits.h"


#ifdef HAVE_CUDA
#include "cudacomp/cudacomp.h"
#endif


static char *inname;

static uint32_t *PForder;
static long      fpi_PForder;

static float *PFlatency;
static long   fpi_PFlatency;

static double *SVDeps;
static long    fpi_SVDeps;

static double *reglambda;
static long    fpi_reglambda;

static char *outPFname;

static float *loopgain;
static long   fpi_loopgain;

static uint64_t *out3Dwrite;
static long      fpi_out3Dwrite;

static int32_t *GPUdevice;
static long     fpi_GPUdevice;




static CLICMDARGDEF farg[] =
{
    {
        // input telemetry
        CLIARG_STREAM,
        ".inname",
        "input telemetry",
        "indata",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inname,
        NULL
    },
    {
        // temporal order of filter: number of time steps in state
        CLIARG_UINT32,
        ".PForder",
        "predictive filter order",
        "10",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &PForder,
        &fpi_PForder
    },
    {
        // latency: how far ahead to predict
        CLIARG_FLOAT32,
        ".PFlatency",
        "time latency [frame]",
        "2.7",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &PFlatency,
        &fpi_PFlatency
    },
    {
        // SVD limit
        CLIARG_FLOAT64,
        ".SVDeps",
        "SVD cutoff",
        "0.001",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &SVDeps,
        &fpi_SVDeps
    },
    {
        // Regularization
        CLIARG_FLOAT64,
        ".reglambda",
        "regularization coefficient",
        "0.001",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &reglambda,
        &fpi_reglambda
    },
    {
        CLIARG_STR,
        ".outPFname",
        "output filter",
        "outPF",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outPFname,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".loopgain",
        "loop gain",
        "0.2",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &loopgain,
        &fpi_loopgain
    },
    {
        CLIARG_ONOFF,
        ".out3Dfilt",
        "write output 3D filter",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &out3Dwrite,
        &fpi_out3Dwrite
    },
    {
        CLIARG_INT32,
        ".GPUdevice",
        "GPU device",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUdevice,
        &fpi_GPUdevice
    }
};




// Optional custom configuration setup. comptbuff
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
        data.fpsptr->parray[fpi_PFlatency].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_SVDeps].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_reglambda].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_loopgain].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_out3Dwrite].fpflag |= FPFLAG_WRITERUN;
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
    "mkPF", "make linear predictive filter", CLICMD_FIELDS_DEFAULTS
};




// detailed help
static errno_t help_function()
{


    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


    int DC_MODE = 0; // 1 if average value of each mode is removed


    // connect to input telemetry
    //
    IMGID imgin = mkIMGID_from_name(inname);
    resolveIMGID(&imgin, ERRMODE_ABORT);




    /// ## Selecting input values

    /// The goal of this function is to build a linear link between
    /// input and output variables. \n
    /// Input variables values are provided by the input telemetry image
    /// which is first read to measure dimensions, and allocate memory.\n
    /// Note that an optional variable selection step allows only a
    /// subset of the telemetry variables to be considered.

    uint32_t nbspl    = 0;
    uint32_t xsize    = 0;
    uint32_t ysize    = 0;
    uint32_t inNBelem = 0;
    imageID  IDincp;

    switch(imgin.md->naxis)
    {

        case 2:
            /// If 2D image:
            /// - xysize <- size[0] is number of variables
            /// - nbspl <- size[1] is number of samples
            nbspl = imgin.md->size[1];
            xsize = imgin.md->size[0];
            ysize = 1;
            // copy of image to avoid input change during computation
            create_2Dimage_ID("PFin_cp",
                              imgin.md->size[0],
                              imgin.md->size[1],
                              &IDincp);
            inNBelem = imgin.md->size[0] * imgin.md->size[1];
            break;

        case 3:
            /// If 3D image
            /// - xysize <- size[0] * size[1] is number of variables
            /// - nbspl <- size[2] is number of samples
            nbspl = imgin.md->size[2];
            xsize = imgin.md->size[0];
            ysize = imgin.md->size[1];
            create_3Dimage_ID("PFin_copy",
                              imgin.md->size[0],
                              imgin.md->size[1],
                              imgin.md->size[2],
                              &IDincp);

            inNBelem = imgin.md->size[0] * imgin.md->size[1] * imgin.md->size[2];
            break;

        default:
            printf("Invalid image size\n");
            break;
    }
    uint64_t xysize = (uint64_t) xsize * ysize;
    printf("xysize = %lu\n", xysize);


    /// Once input telemetry size measured, arrays are created:
    /// - pixarray_x  : x coordinate of each variable (useful to keep track of spatial coordinates)
    /// - pixarray_y  : y coordinate of each variable (useful to keep track of spatial coordinates)
    /// - pixarray_xy : combined index (avoids re-computing index frequently)
    /// - ave_inarray : time averaged value, useful because the predictive filter often needs average to be zero, so we will remove it

    long *pixarray_x = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_x == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    long *pixarray_y = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_y == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    long *pixarray_xy = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_xy == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    double *ave_inarray = (double *) malloc(sizeof(double) * xsize * ysize);
    if(ave_inarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }


    /// ### Select input variables from mask (optional)
    /// If image "inmask" exists, use it to select which variables are active.
    /// Otherwise, all variables are active\n
    /// The number of active input variables is stored in NBpixin.

    imageID IDinmask = image_ID("inmask");
    long    NBpixin  = 0;
    if(IDinmask == -1)
    {
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                pixarray_x[NBpixin]  = ii;
                pixarray_y[NBpixin]  = jj;
                pixarray_xy[NBpixin] = jj * xsize + ii;
                NBpixin++;
            }
    }
    else
    {
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
                if(data.image[IDinmask].array.F[jj * xsize + ii] > 0.5)
                {
                    pixarray_x[NBpixin]  = ii;
                    pixarray_y[NBpixin]  = jj;
                    pixarray_xy[NBpixin] = jj * xsize + ii;
                    NBpixin++;
                }
    }
    printf("NBpixin = %ld\n", NBpixin);



    /// ## Selecting Output Variables
    /// By default, the output variables are the same as the input variables,
    /// so the prediction is performed on the same variables as the input.\n
    ///
    /// With inmask and outmask, input AND output variables can be
    /// selected amond the telemetry.

    /// Arrays are created:
    /// - outpixarray_x  : x coordinate of each output variable (useful to keep track of spatial coordinates)
    /// - outpixarray_y  : y coordinate of each output variable (useful to keep track of spatial coordinates)
    /// - outpixarray_xy : combined output index (avoids re-computing index frequently)

    long *outpixarray_x = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_x == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    long *outpixarray_y = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_y == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    long *outpixarray_xy = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_xy == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    imageID IDoutmask = image_ID("outmask");
    long    NBpixout  = 0;
    if(IDoutmask == -1)
    {
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                outpixarray_x[NBpixout]  = ii;
                outpixarray_y[NBpixout]  = jj;
                outpixarray_xy[NBpixout] = jj * xsize + ii;
                NBpixout++;
            }
    }
    else
    {
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
                if(data.image[IDoutmask].array.F[jj * xsize + ii] > 0.5)
                {
                    outpixarray_x[NBpixout]  = ii;
                    outpixarray_y[NBpixout]  = jj;
                    outpixarray_xy[NBpixout] = jj * xsize + ii;
                    NBpixout++;
                }
    }




    /// ## Build Empty Data Matrix
    ///
    /// Note: column / row description follows FITS file viewing conventions.\n
    /// The data matrix is build from the telemetry. Each column (= time sample) of the
    /// data matrix consists of consecutives columns (= time sample) of the input telemetry.\n
    ///
    /// Variable naming:
    /// - NBmvec is the number of telemetry vectors (each corresponding to a different time) in the data matrix.
    /// - mvecsize is the size of each vector, equal to NBpixin times PForder
    ///
    /// Data matrix is stored as image of size NBmvec x mvecsize, to be fed to routine compute_SVDpseudoInverse in linopt_imtools (CPU mode) or in cudacomp (GPU mode)\n
    ///
    long NBmvec =
        nbspl - *PForder - (int)(*PFlatency) -
        2; // could put "-1", but "-2" allows user to change PFlag_run by up to 1 frame without reading out of array
    long mvecsize =
        NBpixin *
        *PForder; // size of each sample vector for AR filter, excluding regularization

    /// Regularization can be added to penalize strong coefficients in the predictive filter.
    /// It is optionally implemented by adding extra columns at the end of the data matrix.\n
    long    NBmvec1 = 0;
    imageID IDmatA  = -1;
    int     REG     = 0;
    if(REG == 0)  // no regularization
    {
        printf("NBmvec   = %ld  -> %ld \n", NBmvec, NBmvec);
        NBmvec1 = NBmvec;
        create_2Dimage_ID("PFmatD", NBmvec, mvecsize, &IDmatA);
    }
    else // with regularization
    {
        printf("NBmvec   = %ld  -> %ld \n", NBmvec, NBmvec + mvecsize);
        NBmvec1 = NBmvec + mvecsize;
        create_2Dimage_ID("PFmatD", NBmvec + mvecsize, mvecsize, &IDmatA);
    }



    /// Data matrix conventions :
    /// - each column (ii = cst) is a measurement
    /// - m index is measurement
    /// - dt*NBpixin+pix index is pixel

    printf("mvecsize = %ld  (%u x %ld)\n", mvecsize, *PForder, NBpixin);
    printf("NBpixin = %ld\n", NBpixin);
    printf("NBpixout = %ld\n", NBpixout);
    printf("NBmvec1 = %ld\n", NBmvec1);
    printf("PForder = %u\n", *PForder);

    printf("xysize = %ld\n", xysize);
    printf("IDin = %ld\n\n", imgin.ID);
    list_image_ID();




    // Allocate future measured data matrix
    imageID IDfm;
    create_2Dimage_ID("PFfmdat", NBmvec, NBpixout, &IDfm);


    // Prepare output filter images
    //
    // 3D FILTER MATRIX - contains all pixels
    // axis 0 [ii] : input mode
    // axis 1 [jj] : reconstructed mode
    // axis 2 [kk] : time step

    // 2D Filter - contains only used input and output
    // axis 0 [ii1] : input mode x time step
    // axis 1 [jj1] : output mode

    imageID IDoutPF2Draw;
    imageID IDoutPF2D;
    {
        uint32_t *imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
        if(imsizearray == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        imsizearray[0] = NBpixin * (*PForder);
        imsizearray[1] = NBpixout;
        char IDoutPF_name_raw[STRINGMAXLEN_IMGNAME];
        WRITE_IMAGENAME(IDoutPF_name_raw, "%s_raw", outPFname);

        create_image_ID(outPFname,
                        2,
                        imsizearray,
                        _DATATYPE_FLOAT,
                        1,
                        1,
                        0,
                        &IDoutPF2D);
        create_image_ID(IDoutPF_name_raw,
                        2,
                        imsizearray,
                        _DATATYPE_FLOAT,
                        1,
                        1,
                        0,
                        &IDoutPF2Draw);
        free(imsizearray);
        COREMOD_MEMORY_image_set_semflush(outPFname, -1);
        COREMOD_MEMORY_image_set_semflush(IDoutPF_name_raw, -1);
    }




    struct timespec t0;
    struct timespec t1;




    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    clock_gettime(CLOCK_MILK, &t0);

    printf("=========== LOOP ITERATION %6ld =======\n", processinfo->loopcnt);
    printf("  PFlag     = %20f      ", *PFlatency);
    printf("  SVDeps    = %20f\n", *SVDeps);
    printf("  RegLambda = %20f      ", *reglambda);
    printf("  LOOPgain  = %20f\n", *loopgain);
    printf("\n");

    /// *STEP: Copy IDin to IDincp*
    ///
    /// Necessary as input may be continuously changing between consecutive loop iterations.
    ///
    IDincp = image_ID("PFin_copy");
    memcpy(data.image[IDincp].array.F,
           imgin.im->array.F,
           sizeof(float) * inNBelem);




    /// *STEP: if DC_MODE==1, compute average value from each variable*
    if(DC_MODE == 1)  // remove average
    {
        for(long pix = 0; pix < NBpixin; pix++)
        {
            ave_inarray[pix] = 0.0;
            for(uint32_t m = 0; m < nbspl; m++)
            {
                ave_inarray[pix] +=
                    data.image[IDincp].array.F[m * xysize + pixarray_xy[pix]];
            }
            ave_inarray[pix] /= nbspl;
        }
    }
    else
    {
        for(uint32_t pix = 0; pix < NBpixin; pix++)
        {
            ave_inarray[pix] = 0.0;
        }
    }



    /// *STEP: Fill up data matrix PFmatD from input telemetry*
    ///
    for(long m = 0; m < NBmvec1; m++)
    {
        long k0 = m + *PForder - 1; // dt=0 index
        for(long pix = 0; pix < NBpixin; pix++)
            for(long dt = 0; dt < *PForder; dt++)
            {
                data.image[IDmatA].array.F[(NBpixin * dt + pix) * NBmvec1 + m] =
                    data.image[IDincp]
                    .array.F[(k0 - dt) * xysize + pixarray_xy[pix]] -
                    ave_inarray[pix];
            }
    }



    /// *STEP: Write regularization coefficients (optional)*
    ///
    if(REG == 1)
    {
        for(long m = 0; m < mvecsize; m++)
        {
            //m1 = NBmvec + m;
            data.image[IDmatA].array.F[(m) *NBmvec1 + (NBmvec + m)] =
                *reglambda;
        }
    }

    // int Save = 1;
    // if (Save == 1)
    // {
    //save_fits("PFmatD", "PFmatD.fits");
    // }


    /// ### Compute pseudo-inverse of PFmatD
    ///
    /// *STEP: Compute Pseudo-Inverse of PFmatD*
    ///

    // Assemble future measured data matrix
    float alpha = *PFlatency - ((long)(*PFlatency));
    for(long PFpix = 0; PFpix < NBpixout; PFpix++)
        for(long m = 0; m < NBmvec; m++)
        {
            long k0 = m + *PForder - 1;
            k0 += (long) * PFlatency;

            data.image[IDfm].array.F[PFpix * NBmvec + m] =
                (1.0 - alpha) *
                data.image[IDincp]
                .array.F[(k0) * xysize + outpixarray_xy[PFpix]] +
                alpha * data.image[IDincp]
                .array.F[(k0 + 1) * xysize + outpixarray_xy[PFpix]];
        }
    //save_fits("PFfmdat", "PFfmdat.fits");

    /// If using MAGMA, call function CUDACOMP_magma_compute_SVDpseudoInverse()\n
    /// Otherwise, call function linopt_compute_SVDpseudoInverse()\n

    long NB_SVD_Modes = 10000;
    int  LOOPmode     = 0; // 1 if re-use arrays

    // TEST
    //save_fl_fits("PFmatD", "PFmatD.fits");

#ifdef HAVE_MAGMA
    printf("Using magma ...\n");
    CUDACOMP_magma_compute_SVDpseudoInverse("PFmatD",
                                            "PFmatC",
                                            *SVDeps,
                                            NB_SVD_Modes,
                                            "PF_VTmat",
                                            LOOPmode,
                                            0, // testmode
                                            32,
                                            *GPUdevice,
                                            NULL);
#else
    printf("Not using magma ...\n");
    linopt_compute_SVDpseudoInverse("PFmatD",
                                    "PFmatC",
                                    *SVDeps,
                                    NB_SVD_Modes,
                                    "PF_VTmat",
                                    NULL);
#endif

    // TEST
    //save_fl_fits("PFmatC", "PFmatC.fits");

    // Result (pseudoinverse) is stored in image PFmatC\n

    //if (Save == 1)
    // {
    //    save_fits("PF_VTmat", "PF_VTmat.fits");
    //    save_fits("PFmatC", "PFmatC.fits");
    // }
    imageID IDmatC = image_ID("PFmatC");

    ///
    /// ### Assemble Predictive Filter
    ///
    //printf("Compute filters\n");
    //fflush(stdout);

    if(system("mkdir -p pixfilters") != 0)
    {
        PRINT_ERROR("system() returns non-zero value");
    }


    /*
    printf("===========================================================\n");
    printf("ASSEMBLING OUTPUT\n");
    printf("  NBpixout = %ld\n", NBpixout);
    printf("  NBmvec   = %ld\n", NBmvec);
    printf("  NBmvec1  = %ld\n", NBmvec1);
    printf("  NBpixin  = %ld\n", NBpixin);
    printf("  PForder  = %u\n", *PForder);
    printf("===========================================================\n");
    */

    long IDoutPF2Dn = image_ID("psinvPFmat");
    if(IDoutPF2Dn == -1)
    {
        printf("------------------- CPU computing PF matrix\n");

        create_2Dimage_ID("psinvPFmat",
                          NBpixin * *PForder,
                          NBpixout,
                          &IDoutPF2Dn);
        for(long PFpix = 0; PFpix < NBpixout; PFpix++)
        {
            // PFpix is the pixel for which the filter is created (axis 1 in cube, jj)
            // loop on input values
            for(long pix = 0; pix < NBpixin; pix++)
            {
                for(long dt = 0; dt < *PForder; dt++)
                {
                    float val  = 0.0;
                    long  ind1 = (NBpixin * dt + pix) * NBmvec1;
                    for(long m = 0; m < NBmvec; m++)
                    {
                        val += data.image[IDmatC].array.F[ind1 + m] *
                               data.image[IDfm].array.F[PFpix * NBmvec + m];
                    }

                    data.image[IDoutPF2Dn].array.F[PFpix * (*PForder * NBpixin) + dt * NBpixin +
                                                   pix] =
                                                       val;
                }
            }
        }
    }
    else
    {
        printf("------------------- Using GPU-computed PF matrix\n");
    }
    // delete_image_ID("PFfmdat", DELETE_IMAGE_ERRMODE_WARNING);

    //printf("IDoutPF2Draw = %ld\n", IDoutPF2Draw);
    data.image[IDoutPF2Draw].md[0].write = 1;
    memcpy(data.image[IDoutPF2Draw].array.F,
           data.image[IDoutPF2Dn].array.F,
           sizeof(float) * NBpixout * NBpixin * *PForder);
    COREMOD_MEMORY_image_set_sempost_byID(IDoutPF2Draw, -1);
    data.image[IDoutPF2Draw].md[0].cnt0++;
    data.image[IDoutPF2Draw].md[0].write = 0;



    //printf("IDoutPF2D = %ld\n", IDoutPF2D);
    // Mix current PF with last one
    data.image[IDoutPF2D].md[0].write = 1;


    // on first iteration, set loopgain to 1 to initalize content
    float loopgainval = 0.0;
    if(processinfo->loopcnt == 0)
    {
        loopgainval = 1.0;
    }
    else
    {
        loopgainval = *loopgain;
    }
    printf("Mixing PF matrix with gain = %f / %f ....", loopgainval, *loopgain);
    fflush(stdout);
    for(long PFpix = 0; PFpix < NBpixout; PFpix++)
        for(long pix = 0; pix < NBpixin; pix++)
            for(long dt = 0; dt < *PForder; dt++)
            {
                float val0 = data.image[IDoutPF2D]
                             .array.F[PFpix * (*PForder * NBpixin) +
                                            dt * NBpixin + pix]; // Previous
                float val = data.image[IDoutPF2Dn]
                            .array.F[PFpix * (*PForder * NBpixin) +
                                           dt * NBpixin + pix]; // New
                data.image[IDoutPF2D].array.F[PFpix * (*PForder * NBpixin) +
                                              dt * NBpixin + pix] =
                                                  (1.0 - *loopgain) * val0 + *loopgain * val;
            }
    printf(" done\n");
    fflush(stdout);

    delete_image_ID("psinvPFmat", DELETE_IMAGE_ERRMODE_ERROR);


    COREMOD_MEMORY_image_set_sempost_byID(IDoutPF2D, -1);
    data.image[IDoutPF2D].md[0].cnt0++;
    data.image[IDoutPF2D].md[0].write = 0;

    if(*out3Dwrite == 1)
    {
        printf("Prepare 3D output \n");

        imageID IDoutPF3D;
        create_3Dimage_ID("outPF3D", NBpixin, NBpixout, *PForder, &IDoutPF3D);

        for(long pix = 0; pix < NBpixin; pix++)
            for(long PFpix = 0; PFpix < NBpixout; PFpix++)
                for(long dt = 0; dt < *PForder; dt++)
                {
                    float val = data.image[IDoutPF2D]
                                .array.F[PFpix * (*PForder * NBpixin) +
                                               dt * NBpixin + pix];
                    data.image[IDoutPF3D].array.F[NBpixout * NBpixin * dt +
                                                  NBpixin * PFpix + pix] = val;
                }
        save_fits("outPF3D", "_outPF3D.fits");
        delete_image_ID("outPF3D", DELETE_IMAGE_ERRMODE_WARNING);
    }


    struct timespec t2;
    clock_gettime(CLOCK_MILK, &t2);

    struct timespec tdiff = timespec_diff(t0, t2);
    double          texec = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    tdiff        = timespec_diff(t1, t2);
    double tloop = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    t1.tv_sec  = t2.tv_sec;
    t1.tv_nsec = t2.tv_nsec;

    printf("Computing time = %5.3f s / %5.3f s -> fraction = %8.6f\n",
           texec,
           tloop,
           texec / tloop);



    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(pixarray_x);
    free(pixarray_y);
    free(pixarray_xy);

    free(outpixarray_x);
    free(outpixarray_y);
    free(outpixarray_xy);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_LinARfilterPred__build_linPF()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
