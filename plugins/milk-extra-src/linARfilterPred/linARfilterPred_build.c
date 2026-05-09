#include "linARfilterPred_internal.h"

/** ## Purpose
 *
 * Build predictive filter from real-time AO telemetry
 *
 *
 * ## Masking
 *
 *  Optional input and output pixel masks select active input & output
 *
 *
 * ## Loop mode
 *
 * If LOOPmode = 1, operate in a loop, and re-run filter computation everytime IDin_name changes
 *
 *
 * ## Input parameters: dynamic mode
 *
 * if <IFoutPF_name>_PFparam image exist, read parameters from it: PFlag, SVDeps, RegLambda, LOOPgain
 * create it in shared memory by default
 *
 *
 * @return If testmode=2, write 3D output filter
 * @return output filter image indentifier
 *
   */

imageID LINARFILTERPRED_Build_LinPredictor(const char *IDin_name,
        long        PForder,
        float       PFlag,
        double      SVDeps,
        double      RegLambda,
        const char *IDoutPF_name,
        __attribute__((unused)) int outMode,
        int                         LOOPmode,
        float                       LOOPgain,
        int                         testmode)
{
    /// ---
    /// # Code Description

    imageID IDin;
    imageID IDmatA;
    //imageID IDout;
    imageID IDinmask;
    imageID IDoutmask;
    long    nbspl; // Number of samples
    long    NBpixin, NBpixout;
    long    NBmvec, NBmvec1;
    long    mvecsize;
    long    xsize, ysize;
    long   *pixarray_x;
    long   *pixarray_y;
    long   *pixarray_xy;

    long *outpixarray_x;
    long *outpixarray_y;
    long *outpixarray_xy;

    double *ave_inarray;
    int     REG = 0; // 1 if regularization
    long    m, pix, k0, dt;
    int     Save = 0;
    long    xysize;
    long    IDmatC;
    //int use_magma = 1;                         // use MAGMA library if available
    //int magmacomp = 0;

    //imageID IDfiltC;
    // float *valfarray;
    float alpha;
    long  PFpix;
    //char filtname[200];
    //char filtfname[200];
    //imageID ID_Pfilt;
    float   val, val0;
    long    ind1;
    imageID IDoutPF2D;    // averaged with previous filters
    imageID IDoutPF2Draw; // individual filter
    char    IDoutPF_name_raw[200];
    //  long IDoutPF3D;
    //  char IDoutPF_name3D[500];

    long NB_SVD_Modes;

    int DC_MODE = 0; // 1 if average value of each mode is removed

    long      NBiter, iter;
    long      semtrig = 2;
    uint32_t *imsizearray;

    //char fname[200];

    //time_t t;
    //struct tm *uttime;
    //struct timespec timenow;

    struct timespec t0;
    struct timespec t1;
    struct timespec t2;
    struct timespec tdiff;
    double          tdiffv01; // waiting time
    double          tdiffv12; // computing time

    imageID IDPFparam; // parameters in shared memory (optional)
    char    imname[200];
    int     ExternalPFparam;

    float PFlag_run;
    float SVDeps_run;
    float RegLambda_run;
    float LOOPgain_run;
    float gain;

    uint32_t *imsize;
    long      IDincp;
    long      inNBelem;

    list_variable_ID(NULL);

    int  PSINV_MODE = 0;
    long IDv;
    if((IDv = variable_ID("_SVD_PSINV")) != -1)
    {
        PSINV_MODE = (int)(dcvar[IDv].value.f + 0.1);
        printf("PSINV_MODE = %d\n", PSINV_MODE);
    }

    float PSINV_s = 1.0e-6;
    if((IDv = variable_ID("_SVD_s")) != -1)
    {
        PSINV_s = dcvar[IDv].value.f;
        printf("PSINV_s = %f\n", PSINV_s);
    }

    float PSINV_tol = 1.0;
    if((IDv = variable_ID("_SVD_tol")) != -1)
    {
        PSINV_tol = dcvar[IDv].value.f;
        printf("PSINV_tol = %f\n", PSINV_tol);
    }

    /// ## Reading Parameters from Image

    /// If image named <IDoutPF_name>_PFparam exists, the predictive filter
    /// parameters are read from it instead of the function arguments. \n
    /// This mode is particularly useful in LOOP mode if the user needs
    /// to change the parameters between LOOP iterations.\n

    snprintf(imname, sizeof(imname),
             "%s_PFparam", IDoutPF_name);
    imsize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(imsize == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }
    imsize[0] = 4;
    imsize[1] = 1;
    {
        IMGID imgparam =
            imgid_make_from_name(imname);
        imgparam.mdt->naxis = 2;
        imgparam.mdt->size[0] =
            imsize[0];
        imgparam.mdt->size[1] =
            imsize[1];
        imgparam.mdt->datatype =
            _DATATYPE_FLOAT;
        imgparam.mdt->shared = 1;
        imgparam.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgparam);
        IDPFparam = imgparam.ID;
    }
    free(imsize);

    if((IDPFparam = image_ID(imname, dcimg, dcnimg)) != -1)
    {
        ExternalPFparam                  = 1;
        dcimg[IDPFparam].array.F[0] = PFlag;
        dcimg[IDPFparam].array.F[1] = SVDeps;
        dcimg[IDPFparam].array.F[2] = RegLambda;
        dcimg[IDPFparam].array.F[3] = LOOPgain;
    }
    else
    {
        ExternalPFparam = 0;
    }

    LOOPgain_run = LOOPgain;
    if(LOOPmode == 0)
    {
        LOOPgain_run = 1.0;
        NBiter       = 1;
    }
    else
    {
        NBiter = 100000000;
    }

    //sprintf(IDoutPF_name3D, "%s_3D", IDoutPF_name);

    /// ## Selecting input values

    /// The goal of this function is to build a linear link between
    /// input and output variables. \n
    /// Input variables values are provided by the input telemetry image
    /// which is first read to measure dimensions, and allocate memory.\n
    /// Note that an optional variable selection step allows only a
    /// subset of the telemetry variables to be considered.

    /// ### Read input telemetry image IDin_name to measure xsize, ysize and number of samples
    IDin = image_ID(IDin_name, dcimg, dcnimg);

    switch(dcimg[IDin].md[0].naxis)
    {

        case 2:
            /// If 2D image:
            /// - xysize <- size[0] is number of variables
            /// - nbspl <- size[1] is number of samples
            nbspl = dcimg[IDin].md[0].size[1];
            xsize = dcimg[IDin].md[0].size[0];
            ysize = 1;
            // copy of image to avoid input change during computation
            create_2Dimage_ID("PFin_cp",
                              dcimg[IDin].md[0].size[0],
                              dcimg[IDin].md[0].size[1],
                              &IDincp);
            inNBelem =
                dcimg[IDin].md[0].size[0] * dcimg[IDin].md[0].size[1];
            break;

        case 3:
            /// If 3D image
            /// - xysize <- size[0] * size[1] is number of variables
            /// - nbspl <- size[2] is number of samples
            nbspl = dcimg[IDin].md[0].size[2];
            xsize = dcimg[IDin].md[0].size[0];
            ysize = dcimg[IDin].md[0].size[1];
            create_3Dimage_ID("PFin_copy",
                              dcimg[IDin].md[0].size[0],
                              dcimg[IDin].md[0].size[1],
                              dcimg[IDin].md[0].size[2],
                              &IDincp);

            inNBelem = dcimg[IDin].md[0].size[0] *
                       dcimg[IDin].md[0].size[1] *
                       dcimg[IDin].md[0].size[2];
            break;

        default:
            printf("Invalid image size\n");
            break;
    }
    xysize = xsize * ysize;
    printf("xysize = %ld\n", xysize);

    /// Once input telemetry size measured, arrays are created:
    /// - pixarray_x  : x coordinate of each variable (useful to keep track of spatial coordinates)
    /// - pixarray_y  : y coordinate of each variable (useful to keep track of spatial coordinates)
    /// - pixarray_xy : combined index (avoids re-computing index frequently)
    /// - ave_inarray : time averaged value, useful because the predictive filter often needs average to be zero, so we will remove it

    pixarray_x = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_x == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    pixarray_y = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_y == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    pixarray_xy = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_xy == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ave_inarray = (double *) malloc(sizeof(double) * xsize * ysize);
    if(ave_inarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    /// ### Select input variables from mask (optional)
    /// If image "inmask" exists, use it to select which variables are active.
    /// Otherwise, all variables are active\n
    /// The number of active input variables is stored in NBpixin.

    IDinmask = image_ID("inmask", dcimg, dcnimg);
    if(IDinmask == -1)
    {
        NBpixin = 0; //xsize*ysize;

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
        NBpixin = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
                if(dcimg[IDinmask].array.F[jj * xsize + ii] > 0.5f)
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

    outpixarray_x = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_x == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outpixarray_y = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_y == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outpixarray_xy = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_xy == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IDoutmask = image_ID("outmask", dcimg, dcnimg);
    if(IDoutmask == -1)
    {
        NBpixout = 0; //xsize*ysize;

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
        NBpixout = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
                if(dcimg[IDoutmask].array.F[jj * xsize + ii] > 0.5f)
                {
                    outpixarray_x[NBpixout]  = ii;
                    outpixarray_y[NBpixout]  = jj;
                    outpixarray_xy[NBpixout] = jj * xsize + ii;
                    NBpixout++;
                }
    }

    /// ## Reading PFlag from image (optional)
    /// PFlag_run needs to be read before entering the loop as some
    /// array sizes depend on its value.
    if(ExternalPFparam == 1)
    {
        PFlag_run = dcimg[IDPFparam].array.F[0];
    }
    else
    {
        PFlag_run = PFlag;
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
    /// Data matrix is stored as image of size NBmvec x mvecsize, to be fed to routine compute_SVDpseudoInverse 
    // in linopt_imtools (CPU mode) or in linalgebra (GPU mode)\n
    ///
    NBmvec =
        nbspl - PForder -
        (int)(PFlag_run) -
        2; // could put "-1", but "-2" allows user to change PFlag_run by up to 1 frame without reading out of array
    mvecsize =
        NBpixin *
        PForder; // size of each sample vector for AR filter, excluding regularization

    /// Regularization can be added to penalize strong coefficients in the predictive filter.
    /// It is optionally implemented by adding extra columns at the end of the data matrix.\n
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

    IDmatA = image_ID("PFmatD", dcimg, dcnimg);

    /// Data matrix conventions :
    /// - each column (ii = cst) is a measurement
    /// - m index is measurement
    /// - dt*NBpixin+pix index is pixel

    printf("mvecsize = %ld  (%ld x %ld)\n", mvecsize, PForder, NBpixin);
    printf("NBpixin = %ld\n", NBpixin);
    printf("NBpixout = %ld\n", NBpixout);
    printf("NBmvec1 = %ld\n", NBmvec1);
    printf("PForder = %ld\n", PForder);

    printf("xysize = %ld\n", xysize);
    printf("IDin = %ld\n\n", IDin);
    list_image_ID();

    /// ## Predictive Filter Computation
    ///
    /// In LOOP mode, LOOP STARTS HERE \n
    ///

    if(LOOPmode == 1)
    {
        COREMOD_MEMORY_image_set_semflush(IDin_name, semtrig);
    }

    for(iter = 0; iter < NBiter; iter++)
    {

        /// ### Prepare data matrix PFmatD

        /// *STEP: Read parameters from external image (optional)*\n
        if(ExternalPFparam == 1)
        {
            PFlag_run     = dcimg[IDPFparam].array.F[0];
            SVDeps_run    = dcimg[IDPFparam].array.F[1];
            RegLambda_run = dcimg[IDPFparam].array.F[2];
            LOOPgain_run  = dcimg[IDPFparam].array.F[3];
        }
        else
        {
            PFlag_run     = PFlag;
            SVDeps_run    = SVDeps;
            RegLambda_run = RegLambda;
            LOOPgain_run  = LOOPgain;
        }

        printf(
            "=========== LOOP ITERATION %6ld ======= [ExternalPFparam = %d ]\n",
            iter,
            ExternalPFparam);
        printf(" parameters read from %s\n", dcimg[IDPFparam].md[0].name);
        printf("  PFlag     = %20f      ", PFlag_run);
        printf("  SVDeps    = %20f\n", SVDeps_run);
        printf("  RegLambda = %20f      ", RegLambda_run);
        printf("  LOOPgain  = %20f\n", LOOPgain_run);
        printf("\n");

        gain = 1.0 / (iter + 1);
        if(gain < LOOPgain_run)
        {
            gain = LOOPgain_run;
        }

        /// *STEP: In loop mode, wait for input data to arrive*

        printf("WAITING FOR INPUT DATA ...... \n");
        clock_gettime(CLOCK_MILK, &t0);
        if(LOOPmode == 1)
        {
            ImageStreamIO_semwait(dcimg+IDin, semtrig);
        }

        /// *STEP: Copy IDin to IDincp*
        ///
        /// Necessary as input may be continuously changing between consecutive loop iterations.
        ///
        IDincp = image_ID("PFin_copy", dcimg, dcnimg);
        memcpy(dcimg[IDincp].array.F,
               dcimg[IDin].array.F,
               sizeof(float) * inNBelem);

        //save_fits("PFin_copy", "test_PFin_copy.fits");
        //save_fits(IDin_name, "test_PFin.fits");

        clock_gettime(CLOCK_MILK, &t1);

        /// *STEP: if DC_MODE==1, compute average value from each variable*
        if(DC_MODE == 1)  // remove average
        {
            for(pix = 0; pix < NBpixin; pix++)
            {
                ave_inarray[pix] = 0.0;
                for(m = 0; m < nbspl; m++)
                {
                    ave_inarray[pix] +=
                        dcimg[IDincp]
                        .array.F[m * xysize + pixarray_xy[pix]];
                }
                ave_inarray[pix] /= nbspl;
            }
        }
        else
        {
            for(pix = 0; pix < NBpixin; pix++)
            {
                ave_inarray[pix] = 0.0;
            }
        }

        ///
        /// *STEP: Fill up data matrix PFmatD from input telemetry*
        ///
        for(m = 0; m < NBmvec1; m++)
        {
            k0 = m + PForder - 1; // dt=0 index
            for(pix = 0; pix < NBpixin; pix++)
                for(dt = 0; dt < PForder; dt++)
                {
                    dcimg[IDmatA]
                    .array.F[(NBpixin * dt + pix) * NBmvec1 + m] =
                        dcimg[IDincp]
                        .array.F[(k0 - dt) * xysize + pixarray_xy[pix]] -
                        ave_inarray[pix];
                }
        }

        if(LOOPmode == 0)
        {
            free(ave_inarray); // No need to hold on to array
        }

        ///
        /// *STEP: Write regularization coefficients (optional)*
        ///
        if(REG == 1)
        {
            for(m = 0; m < mvecsize; m++)
            {
                //m1 = NBmvec + m;
                dcimg[IDmatA].array.F[(m) *NBmvec1 + (NBmvec + m)] =
                    RegLambda_run;
            }
        }

        if(Save == 1)
        {
            save_fits("PFmatD", "PFmatD.fits");
        }
        //list_image_ID();

        /// ### Compute pseudo-inverse of PFmatD
        ///
        /// *STEP: Compute Pseudo-Inverse of PFmatD*
        ///
        printf("Assembling pseudoinverse\n");
        fflush(stdout);

        // Assemble future measured data matrix
        imageID IDfm;
        create_2Dimage_ID("PFfmdat", NBmvec, NBpixout, &IDfm);

        alpha = PFlag_run - ((long) PFlag_run);
        for(PFpix = 0; PFpix < NBpixout; PFpix++)
            for(m = 0; m < NBmvec; m++)
            {
                k0 = m + PForder - 1;
                k0 += (long) PFlag_run;

                dcimg[IDfm].array.F[PFpix * NBmvec + m] =
                    (1.0 - alpha) *
                    dcimg[IDincp]
                    .array.F[(k0) * xysize + outpixarray_xy[PFpix]] +
                    alpha *
                    dcimg[IDincp]
                    .array.F[(k0 + 1) * xysize + outpixarray_xy[PFpix]];
            }
        save_fits("PFfmdat", "PFfmdat.fits");

        /// If using MAGMA, call function LINALGEBRA_magma_compute_SVDpseudoInverse()\n
        /// Otherwise, call function linopt_compute_SVDpseudoInverse()\n

        NB_SVD_Modes = 10000;

#ifdef HAVE_MAGMA
        printf("Using magma ...\n");
        LINALGEBRA_magma_compute_SVDpseudoInverse("PFmatD",
                                                "PFmatC",
                                                SVDeps_run,
                                                NB_SVD_Modes,
                                                "PF_VTmat",
                                                LOOPmode,
                                                testmode,
                                                64,
                                                0, // GPU device
                                                NULL);
#else
        printf("Not using magma ...\n");
        linopt_compute_SVDpseudoInverse("PFmatD",
                                        "PFmatC",
                                        SVDeps_run,
                                        NB_SVD_Modes,
                                        "PF_VTmat",
                                        NULL);
#endif

        /// Result (pseudoinverse) is stored in image PFmatC\n
        printf("Done assembling pseudoinverse\n");
        fflush(stdout);

        if(Save == 1)
        {
            save_fits("PF_VTmat", "PF_VTmat.fits");
            save_fits("PFmatC", "PFmatC.fits");
        }
        IDmatC = image_ID("PFmatC", dcimg, dcnimg);

        ///
        /// ### Assemble Predictive Filter
        ///
        printf("Compute filters\n");
        fflush(stdout);

        if(system("mkdir -p pixfilters") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }

        // 3D FILTER MATRIX - contains all pixels
        // axis 0 [ii] : input mode
        // axis 1 [jj] : reconstructed mode
        // axis 2 [kk] : time step

        // 2D Filter - contains only used input and output
        // axis 0 [ii1] : input mode x time step
        // axis 1 [jj1] : output mode

        if(LOOPmode == 0)
        {
            create_2Dimage_ID(IDoutPF_name,
                              NBpixin * PForder,
                              NBpixout,
                              &IDoutPF2D);
        }

        else
        {
            if(iter == 0)  // create 2D and 3D filters as shared memory
            {
                imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
                if(imsizearray == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                imsizearray[0] = NBpixin * PForder;
                imsizearray[1] = NBpixout;
                snprintf(IDoutPF_name_raw,
                         sizeof(IDoutPF_name_raw),
                         "%s_raw", IDoutPF_name);

                {
                    IMGID imgpf =
                        imgid_make_from_name(
                            IDoutPF_name);
                    imgpf.mdt->naxis = 2;
                    imgpf.mdt->size[0] =
                        imsizearray[0];
                    imgpf.mdt->size[1] =
                        imsizearray[1];
                    imgpf.mdt->datatype =
                        _DATATYPE_FLOAT;
                    imgpf.mdt->shared = 1;
                    imgpf.mdt->NBkw = 1;
                    imgpf.im =
                        (IMAGE *) calloc(
                            1,
                            sizeof(IMAGE));
                    imgid_mkimage(&imgpf);
                    IDoutPF2D = imgpf.ID;
                }
                {
                    IMGID imgpfr =
                        imgid_make_from_name(
                            IDoutPF_name_raw);
                    imgpfr.mdt->naxis = 2;
                    imgpfr.mdt->size[0] =
                        imsizearray[0];
                    imgpfr.mdt->size[1] =
                        imsizearray[1];
                    imgpfr.mdt->datatype =
                        _DATATYPE_FLOAT;
                    imgpfr.mdt->shared = 1;
                    imgpfr.mdt->NBkw = 1;
                    imgpfr.im =
                        (IMAGE *) calloc(
                            1,
                            sizeof(IMAGE));
                    imgid_mkimage(&imgpfr);
                    IDoutPF2Draw =
                        imgpfr.ID;
                }
                free(imsizearray);
                COREMOD_MEMORY_image_set_semflush(IDoutPF_name, -1);
                COREMOD_MEMORY_image_set_semflush(IDoutPF_name_raw, -1);
            }
            else
            {
                IDoutPF2D = image_ID(IDoutPF_name, dcimg, dcnimg);
            }
        }

        IDoutmask = image_ID("outmask", dcimg, dcnimg);

        printf("===========================================================\n");
        printf("ASSEMBLING OUTPUT\n");
        printf("  NBpixout = %ld\n", NBpixout);
        printf("  NBmvec   = %ld\n", NBmvec);
        printf("  NBmvec1  = %ld\n", NBmvec1);
        printf("  NBpixin  = %ld\n", NBpixin);
        printf("  PForder  = %ld\n", PForder);
        printf("===========================================================\n");

        long IDoutPF2Dn = image_ID("psinvPFmat", dcimg, dcnimg);
        if(IDoutPF2Dn == -1)
        {
            printf("------------------- CPU computing PF matrix\n");

            create_2Dimage_ID("psinvPFmat",
                              NBpixin * PForder,
                              NBpixout,
                              &IDoutPF2Dn);
            for(
                PFpix = 0; PFpix < NBpixout;
                PFpix++) // PFpix is the pixel for which the filter is created (axis 1 in cube, jj)
            {

                // loop on input values
                for(pix = 0; pix < NBpixin; pix++)
                {
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val  = 0.0;
                        ind1 = (NBpixin * dt + pix) * NBmvec1;
                        for(m = 0; m < NBmvec; m++)
                        {
                            val += dcimg[IDmatC].array.F[ind1 + m] *
                                   dcimg[IDfm].array.F[PFpix * NBmvec + m];
                        }

                        dcimg[IDoutPF2Dn]
                        .array.F[PFpix * (PForder * NBpixin) +
                                       dt * NBpixin + pix] = val;
                    }
                }
            }
        }
        else
        {
            printf("------------------- Using GPU-computed PF matrix\n");
        }
        delete_image_ID("PFfmdat", DELETE_IMAGE_ERRMODE_WARNING);

        if(LOOPmode == 1)
        {
            dcimg[IDoutPF2Draw].md[0].write = 1;
            memcpy(dcimg[IDoutPF2Draw].array.F,
                   dcimg[IDoutPF2Dn].array.F,
                   sizeof(float) * NBpixout * NBpixin * PForder);
            COREMOD_MEMORY_image_set_sempost_byID(IDoutPF2Draw, -1);
            dcimg[IDoutPF2Draw].md[0].cnt0++;
            dcimg[IDoutPF2Draw].md[0].write = 0;
        }

        // Mix current PF with last one
        dcimg[IDoutPF2D].md[0].write = 1;
        if(LOOPmode == 0)
        {
            memcpy(dcimg[IDoutPF2D].array.F,
                   dcimg[IDoutPF2Dn].array.F,
                   sizeof(float) * NBpixout * NBpixin * PForder);
            save_fits(IDoutPF_name, "_outPF.fits");
        }
        else
        {
            printf("Mixing PF matrix with gain = %f ....", gain);
            fflush(stdout);
            for(PFpix = 0; PFpix < NBpixout; PFpix++)
                for(pix = 0; pix < NBpixin; pix++)
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val0 = dcimg[IDoutPF2D]
                               .array.F[PFpix * (PForder * NBpixin) +
                                              dt * NBpixin + pix]; // Previous
                        val = dcimg[IDoutPF2Dn]
                              .array.F[PFpix * (PForder * NBpixin) +
                                             dt * NBpixin + pix]; // New
                        dcimg[IDoutPF2D]
                        .array.F[PFpix * (PForder * NBpixin) +
                                       dt * NBpixin + pix] =
                                     (1.0 - gain) * val0 + gain * val;
                    }
            printf(" done\n");
            fflush(stdout);
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDoutPF2D, -1);
        dcimg[IDoutPF2D].md[0].cnt0++;
        dcimg[IDoutPF2D].md[0].write = 0;

        if(testmode == 2)
        {
            printf("Prepare 3D output \n");

            imageID IDoutPF3D;
            create_3Dimage_ID("outPF3D",
                              NBpixin,
                              NBpixout,
                              PForder,
                              &IDoutPF3D);

            for(pix = 0; pix < NBpixin; pix++)
                for(PFpix = 0; PFpix < NBpixout; PFpix++)
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val = dcimg[IDoutPF2D]
                              .array.F[PFpix * (PForder * NBpixin) +
                                             dt * NBpixin + pix];
                        dcimg[IDoutPF3D].array.F[NBpixout * NBpixin * dt +
                                                      NBpixin * PFpix + pix] =
                                                          val;
                    }
            save_fits("outPF3D", "_outPF3D.fits");
        }

        printf("DONE\n");
        fflush(stdout);
        clock_gettime(CLOCK_MILK, &t2);

        tdiff    = timespec_diff(t0, t1);
        tdiffv01 = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff    = timespec_diff(t1, t2);
        tdiffv12 = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        printf("Computing time = %5.3f s / %5.3f s -> fraction = %8.6f\n",
               tdiffv12,
               tdiffv01 + tdiffv12,
               tdiffv12 / (tdiffv01 + tdiffv12));
    }
    ///
    /// In LOOP mode, LOOP ENDS HERE \n
    ///

    // free(valfarray);

    free(pixarray_x);
    free(pixarray_y);
    free(pixarray_xy);

    free(outpixarray_x);
    free(outpixarray_y);
    free(outpixarray_xy);

    ///
    /// ---
    ///

    return IDoutPF2D;
}


/* =============================================================================================== */
/*                                                                                                 */
/* 5. MISC TOOLS, DIAGNOSTICS                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

//
// IDin_name is a 2 or 3D image, open-loop disturbance
// last axis is time (step)
// this optimization asssumes no correlation in noise
//
float LINARFILTERPRED_ScanGain(char *IDin_name, float multfact, float framelag)
{
    float   gain;
    float   gainmax = 1.1;
    float   optgainblock;
    float   residualblock;
    float   residualblock0;
    float   gainstep = 0.01;
    imageID IDin;

    long nbstep;
    long step, step0, step1;

    long  framelag0;
    long  framelag1;
    float alpha;

    float *actval_array; // actuator value
    float  actval;

    long nbvar;
    long axis, naxis;

    double *errval;
    double  errvaltot;
    long    cnt;

    FILE *fp;
    char  fname[200];
    float mval;
    long  ii;
    float tmpv;

    int   TEST       = 0;
    float TESTperiod = 20.0;

    // results
    float *optgain;
    float *optres;
    float *res0;
    int    optinit = 0;

    if(framelag < 1.00000001)
    {
        printf("ERROR: framelag should be be > 1\n");
        exit(0);
    }

    IDin  = image_ID(IDin_name, dcimg, dcnimg);
    naxis = dcimg[IDin].md[0].naxis;

    nbvar = 1;
    for(axis = 0; axis < naxis - 1; axis++)
    {
        nbvar *= dcimg[IDin].md[0].size[axis];
    }

    errval = (double *) malloc(sizeof(double) * nbvar);
    if(errval == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nbstep = dcimg[IDin].md[0].size[naxis - 1];

    framelag0 = (long) framelag;
    framelag1 = framelag0 + 1;
    alpha     = framelag - framelag0;

    printf("alpha = %f    nbvar = %ld\n", alpha, nbvar);

    list_image_ID();
    if(TEST == 1)
    {
        for(ii = 0; ii < nbvar; ii++)
            for(step = 0; step < nbstep; step++)
            {
                dcimg[IDin].array.F[step * nbvar + ii] =
                    1.0 * sin(2.0 * M_PI * step / TESTperiod);
            }
    }

    actval_array = (float *) malloc(sizeof(float) * nbstep);
    if(actval_array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    optgain = (float *) malloc(sizeof(float) * nbvar);
    if(optgain == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    optres = (float *) malloc(sizeof(float) * nbvar);
    if(optres == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    res0 = (float *) malloc(sizeof(float) * nbvar);
    if(res0 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    snprintf(fname, sizeof(fname), "gainscan.txt");

    gain          = 0.2;
    ii            = 0;
    fp            = fopen(fname, "w");
    residualblock = 1.0e20;
    optgainblock  = 0.0;
    for(gain = 0; gain < gainmax; gain += gainstep)
    {
        fprintf(fp, "%5.3f", gain);

        errvaltot = 0.0;
        for(ii = 0; ii < nbvar; ii++)
        {
            errval[ii] = 0.0;
            cnt        = 0.0;
            for(step = 0; step < framelag1 + 2; step++)
            {
                actval_array[step] = 0.0;
            }
            for(step = framelag1; step < nbstep; step++)
            {
                step0 = step - framelag0;
                step1 = step - framelag1;

                actval = (1.0 - alpha) * actval_array[step0] +
                         alpha * actval_array[step1];
                mval = ((1.0 - alpha) *
                        dcimg[IDin].array.F[step0 * nbvar + ii] +
                        alpha * dcimg[IDin].array.F[step1 * nbvar + ii]) -
                       actval;
                actval_array[step] =
                    multfact * (actval_array[step - 1] + gain * mval);
                tmpv = dcimg[IDin].array.F[step * nbvar + ii] -
                       actval_array[step];
                errval[ii] += tmpv * tmpv;
                cnt++;
            }
            errval[ii] = sqrt(errval[ii] / cnt);
            fprintf(fp, " %10f", errval[ii]);
            errvaltot += errval[ii] * errval[ii];

            if(optinit == 0)
            {
                optgain[ii] = gain;
                optres[ii]  = errval[ii];
                res0[ii]    = errval[ii];
            }
            else
            {
                if(errval[ii] < optres[ii])
                {
                    optres[ii]  = errval[ii];
                    optgain[ii] = gain;
                }
            }
        }

        if(optinit == 0)
        {
            residualblock0 = errvaltot;
        }

        optinit = 1;
        fprintf(fp, "%10f\n", errvaltot);

        if(errvaltot < residualblock)
        {
            residualblock = errvaltot;
            optgainblock  = gain;
        }
    }
    fclose(fp);

    free(actval_array);
    free(errval);

    for(ii = 0; ii < nbvar; ii++)
    {
        printf(
            "MODE %4ld    optimal gain = %5.2f     residual = %.6f -> %.6f \n",
            ii,
            optgain[ii],
            res0[ii],
            optres[ii]);
    }

    printf("\noptimal block gain = %f     residual = %.6f -> %.6f\n\n",
           optgainblock,
           sqrt(residualblock0),
           sqrt(residualblock));
    printf("RMS per mode = %f -> %f\n",
           sqrt(residualblock0 / nbvar),
           sqrt(residualblock / nbvar));

    free(optgain);
    free(optres);
    free(res0);

    return (optgainblock);
}
