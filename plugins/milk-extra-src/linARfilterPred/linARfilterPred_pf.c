// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "linARfilterPred_internal.h"

//
// IDPF_name and IDPFM_name should be pre-loaded
//
imageID LINARFILTERPRED_PF_updatePFmatrix(const char *IDPF_name,
                                          const char *IDPFM_name,
                                          float       alpha)
{
    imageID IDPF;
    imageID IDPFM;
    long    inmode, NBmode, outmode, NBmode2;
    long    tstep, NBtstep;

    uint32_t *sizearray;
    uint8_t   naxis;

    // IDPF should be square
    IDPF    = image_ID(IDPF_name, dcimg, dcnimg);
    NBmode  = dcimg[IDPF].md[0].size[0];
    NBmode2 = NBmode * NBmode;
    assert(dcimg[IDPF].md[0].size[0] == dcimg[IDPF].md[0].size[1]);
    NBtstep = dcimg[IDPF].md[0].size[2];

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if (sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sizearray[0] = NBmode * NBtstep;
    sizearray[1] = NBmode;
    naxis        = 2;

    IDPFM = image_ID(IDPFM_name, dcimg, dcnimg);

    if (IDPFM == -1)
    {
        printf("Creating shared mem image %s  [ %ld  x  %ld ]\n", IDPFM_name, (long) sizearray[0],
               (long) sizearray[1]);
        fflush(stdout);
        {
            IMGID imgpfm         = imgid_make_from_name(IDPFM_name);
            imgpfm.mdt->naxis    = naxis;
            imgpfm.mdt->size[0]  = sizearray[0];
            imgpfm.mdt->size[1]  = sizearray[1];
            imgpfm.mdt->datatype = _DATATYPE_FLOAT;
            imgpfm.mdt->shared   = 1;
            imgpfm.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
            imgid_mkimage(&imgpfm);
            IDPFM = imgpfm.ID;
        }
    }
    free(sizearray);

    dcimg[IDPFM].md[0].write = 1;
    for (outmode = 0; outmode < NBmode; outmode++)
    {
        for (tstep = 0; tstep < NBtstep; tstep++)
        {
            for (inmode = 0; inmode < NBmode; inmode++)
            {
                dcimg[IDPFM].array.F[outmode * (NBmode * NBtstep) + tstep * NBmode + inmode] =
                    (1.0 - alpha) *
                        dcimg[IDPFM]
                            .array.F[outmode * (NBmode * NBtstep) + tstep * NBmode + inmode] +
                    alpha * dcimg[IDPF].array.F[tstep * NBmode2 + outmode * NBmode + inmode];
            }
        }
    }
    COREMOD_MEMORY_image_set_sempost_byID(IDPFM, -1);
    dcimg[IDPFM].md[0].write = 0;
    dcimg[IDPFM].md[0].cnt0++;

    return IDPFM;
}


//
// IDmodevalIN_name : open loop modal coefficients
// IndexOffset      : predicted mode start at this input index
// semtrig          : semaphore trigger index in input input
// IDPFM_name       : predictive filter matrix
// IDPFout_name     : prediction
//
//  NBiter: run for fixed number of iteration
//  SAVEMODE:   0 no file output
//  			1	write txt and FITS output
//				2	write FITS telemetry with prediction: replace output measurements with predictions
//
//	tlag is only used if SAVEMODE = 2
//  used outmask to identify outputs
//
imageID LINARFILTERPRED_PF_RealTimeApply(const char *IDmodevalIN_name,
                                         long        IndexOffset,
                                         int         semtrig,
                                         const char *IDPFM_name,
                                         long        NBPFstep,
                                         const char *IDPFout_name,
                                         int         nbGPU,
                                         long        loop,
                                         long        NBiter,
                                         int         SAVEMODE,
                                         float       tlag,
                                         long        PFindex)
{
    imageID IDmodevalIN;
    long    NBmodeIN, NBmodeIN0, NBmodeOUT, mode;
    imageID IDPFM;

    imageID   IDINbuff;
    long      tstep;
    uint32_t *sizearray;
    uint8_t   naxis;

    imageID IDPFout;

    int *GPUsetPF;
    char GPUsetfname[200];
    int  gpuindex;

#ifdef HAVE_CUDA
    int status;
    int GPUstatus[100];
    int GPUMATMULTCONFindex = 2;
#endif

    FILE *fp;

    //time_t t;
    //struct tm *uttime;
    struct timespec timenow;
    double          timesec, timesec0;
    long            IDsave;

    FILE *fpout;
    long  iter;
    long  kk;

    imageID IDinmask;
    long   *inmaskindex;
    long    NBinmaskpix;

    long  tlag0;
    float tlagalpha = 0.0;

    imageID IDoutmask;
    long   *outmaskindex;
    long    NBoutmaskpix;
    long    kk0, kk1;
    float   val, val0, val1;
    long    ii0, ii1;

    long IDmasterout;
    char imname[200];

    IDmodevalIN = image_ID(IDmodevalIN_name, dcimg, dcnimg);
    NBmodeIN0   = dcimg[IDmodevalIN].md[0].size[0];

    IDPFM     = image_ID(IDPFM_name, dcimg, dcnimg);
    NBmodeOUT = dcimg[IDPFM].md[0].size[1];

    snprintf(imname, sizeof(imname), "aol%ld_modevalPF", loop);
    IDmasterout = image_ID(imname, dcimg, dcnimg);

    IDinmask = image_ID("inmask", dcimg, dcnimg);
    if (IDinmask != -1)
    {
        NBinmaskpix = 0;
        for (uint32_t ii = 0; ii < dcimg[IDinmask].md[0].size[0]; ii++)
        {
            if (dcimg[IDinmask].array.F[ii] > 0.5f)
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
        for (uint32_t ii = 0; ii < dcimg[IDinmask].md[0].size[0]; ii++)
        {
            if (dcimg[IDinmask].array.F[ii] > 0.5f)
            {
                inmaskindex[NBinmaskpix] = ii;
                NBinmaskpix++;
            }
        }
        //printf("Number of active input modes  = %ld\n", NBinmaskpix);
    }
    else
    {
        NBinmaskpix = NBmodeIN0;
        printf("no input mask -> assuming NBinmaskpix = %ld\n", NBinmaskpix);
        create_2Dimage_ID("inmask", NBinmaskpix, 1, &IDinmask);
        for (uint32_t ii = 0; ii < dcimg[IDinmask].md[0].size[0]; ii++)
        {
            dcimg[IDinmask].array.F[ii] = 1.0f;
        }

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);
        if (inmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for (uint32_t ii = 0; ii < dcimg[IDinmask].md[0].size[0]; ii++)
        {
            inmaskindex[NBinmaskpix] = ii;
        }
    }
    NBmodeIN = NBinmaskpix;

    NBPFstep = dcimg[IDPFM].md[0].size[0] / NBmodeIN;

    printf("Number of input modes         = %ld\n", NBmodeIN0);
    printf("Number of active input modes  = %ld\n", NBmodeIN);
    printf("Number of output modes        = %ld\n", NBmodeOUT);
    printf("Number of time steps          = %ld\n", NBPFstep);
    if (IDmasterout != -1)
    {
        printf("Writing result in master output stream %s  (%ld)\n", imname, IDmasterout);
    }

    if ((SAVEMODE > 0) || (IDmasterout != -1))
    {
        IDoutmask = image_ID("outmask", dcimg, dcnimg);
        if (IDoutmask == -1)
        {
            printf("ERROR: outmask image required\n");
            exit(0);
        }
        NBoutmaskpix = 0;
        for (uint32_t ii = 0; ii < dcimg[IDoutmask].md[0].size[0]; ii++)
        {
            if (dcimg[IDoutmask].array.F[ii] > 0.5f)
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
        for (uint32_t ii = 0; ii < dcimg[IDoutmask].md[0].size[0]; ii++)
        {
            if (dcimg[IDoutmask].array.F[ii] > 0.5f)
            {
                outmaskindex[NBoutmaskpix] = ii;
                NBoutmaskpix++;
            }
        }
        if (NBoutmaskpix != NBmodeOUT)
        {
            printf("ERROR: NBoutmaskpix (%ld)   !=   NBmodeOUT (%ld)\n", NBoutmaskpix, NBmodeOUT);
            list_image_ID();
            exit(0);
        }
    }

    create_2Dimage_ID("INbuffer", NBmodeIN, NBPFstep, &IDINbuff);

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if (sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sizearray[0] = NBmodeOUT;
    sizearray[1] = 1;
    naxis        = 2;
    IDPFout      = image_ID(IDPFout_name, dcimg, dcnimg);

    if (IDPFout == -1)
    {
        {
            IMGID imgpfout         = imgid_make_from_name(IDPFout_name);
            imgpfout.mdt->naxis    = naxis;
            imgpfout.mdt->size[0]  = sizearray[0];
            imgpfout.mdt->size[1]  = sizearray[1];
            imgpfout.mdt->datatype = _DATATYPE_FLOAT;
            imgpfout.mdt->shared   = 1;
            imgpfout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
            imgid_mkimage(&imgpfout);
            IDPFout = imgpfout.ID;
        }
    }
    free(sizearray);

    if (nbGPU > 0)
    {
        GPUsetPF = (int *) malloc(sizeof(int) * nbGPU);
        if (GPUsetPF == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for (gpuindex = 0; gpuindex < nbGPU; gpuindex++)
        {
            snprintf(GPUsetfname, sizeof(GPUsetfname), "./conf/param_PFb%ldGPU%ddevice.txt",
                     PFindex, gpuindex);
            fp = fopen(GPUsetfname, "r");
            if (fp == NULL)
            {
                printf("ERROR: file %s not found\n", GPUsetfname);
                exit(0);
            }
            if (fscanf(fp, "%d", &GPUsetPF[gpuindex]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
            fclose(fp);
        }
        printf("USING %d GPUs: ", nbGPU);
        for (gpuindex = 0; gpuindex < nbGPU; gpuindex++)
        {
            printf(" %d", GPUsetPF[gpuindex]);
        }
        printf("\n\n");
    }
    else
    {
        printf("Using CPU\n");
    }

    iter = 0;
    if (SAVEMODE > 0)
    {
        if (NBiter > 50000)
        {
            NBiter = 50000;
        }
    }

    if (SAVEMODE == 1)
    {
        create_2Dimage_ID("testPFsave", 1 + NBmodeIN0 + NBmodeOUT, NBiter, &IDsave);
    }
    if (SAVEMODE == 2)
    {
        create_3Dimage_ID("testPFTout", NBmodeIN0, 1, NBiter, &IDsave);
    }

    //	t = time(NULL);
    //    uttime = gmtime(&t);
    //	clock_gettime(CLOCK_MILK, &timenow);
    //	timesec0 = 3600.0*uttime->tm_hour  + 60.0*uttime->tm_min + 1.0*(timenow.tv_sec % 60) + 1.0e-9*timenow.tv_nsec;

    printf("Running on semaphore trigger %d of image %s\n", semtrig, dcimg[IDmodevalIN].md[0].name);

    while (iter != NBiter)
    {
        //	printf("iter %5ld / %5ld", iter, NBiter);
        //	fflush(stdout);

        ImageStreamIO_semwait(dcimg + IDmodevalIN, semtrig);
        //	printf("\n");
        //	fflush(stdout);

        // fill in buffer
        for (mode = 0; mode < NBmodeIN; mode++)
        {
            dcimg[IDINbuff].array.F[mode] =
                dcimg[IDmodevalIN].array.F[IndexOffset + inmaskindex[mode]];
        }

        //
        // Main matrix multiplication is done here
        // input vector contains recent history of mode coefficients
        // output vector contains the predicted mode coefficients
        //
        if (nbGPU > 0) // if using GPU
        {
#ifdef HAVE_CUDA
            if (iter == 0)
            {
                printf("INITIALIZE GPU(s)\n\n");
                fflush(stdout);

                GPU_loop_MultMat_setup(GPUMATMULTCONFindex, IDPFM_name, "INbuffer", IDPFout_name,
                                       nbGPU, GPUsetPF, 0, 1, 1, loop);

                printf("INITIALIZATION DONE\n\n");
                fflush(stdout);
            }
            GPU_loop_MultMat_execute(GPUMATMULTCONFindex, &status, &GPUstatus[100], 1.0, 0.0, 0, 0);
#endif
        }
        else // if using CPU
        {
            // compute output : matrix vector mult with a CPU-based loop
            dcimg[IDPFout].md[0].write = 1;
            for (mode = 0; mode < NBmodeOUT; mode++)
            {
                dcimg[IDPFout].array.F[mode] = 0.0f;
                for (uint32_t ii = 0; ii < NBmodeIN * NBPFstep; ii++)
                {
                    dcimg[IDPFout].array.F[mode] +=
                        dcimg[IDINbuff].array.F[ii] *
                        dcimg[IDPFM].array.F[mode * dcimg[IDPFM].md[0].size[0] + ii];
                }
            }
            COREMOD_MEMORY_image_set_sempost_byID(IDPFout, -1);
            dcimg[IDPFout].md[0].write = 0;
            dcimg[IDPFout].md[0].cnt0++;
        }

        if (iter == 0)
        {
            /// measure time
            //t = time(NULL);
            //uttime = gmtime(&t);
            clock_gettime(CLOCK_MILK, &timenow);
            timesec0 = 1.0 * timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;

            // fprintf(fp, "%02d:%02d:%02ld.%09ld ", uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
        }

        if (SAVEMODE == 1)
        {
            //		printf("	Saving step (mode = 1) ...");
            //		fflush(stdout);

            //t = time(NULL);
            //uttime = gmtime(&t);
            clock_gettime(CLOCK_MILK, &timenow);
            timesec = 1.0 * timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;

            kk = 0;
            dcimg[IDsave].array.F[iter * (1 + NBmodeIN0 + NBmodeOUT)] =
                (float) (timesec - timesec0);
            //printf(" [%f] ", dcimg[IDsave].array.F[iter*(1+NBmodeIN0+NBmodeOUT)]);
            kk++;
            for (mode = 0; mode < NBmodeIN0; mode++)
            {
                dcimg[IDsave].array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk] =
                    dcimg[IDmodevalIN].array.F[IndexOffset + mode];
                kk++;
            }
            for (mode = 0; mode < NBmodeOUT; mode++)
            {
                dcimg[IDsave].array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk] =
                    dcimg[IDPFout].array.F[mode];
                kk++;
            }
            //	printf(" done\n");
            //	fflush(stdout);
        }
        if (SAVEMODE == 2)
        {
            //	printf("	Saving step (mode = 2) ...");
            //	fflush(stdout);

            for (mode = 0; mode < NBmodeIN0; mode++)
            {
                dcimg[IDsave].array.F[iter * NBmodeIN0 + mode] =
                    dcimg[IDmodevalIN].array.F[IndexOffset + mode];
            }
            for (mode = 0; mode < NBmodeOUT; mode++)
            {
                dcimg[IDsave].array.F[iter * NBmodeIN0 + outmaskindex[mode]] =
                    dcimg[IDPFout].array.F[mode];
            }
            //	printf(" done\n");
            //	fflush(stdout);
        }

        if (IDmasterout != -1)
        {
            dcimg[IDmasterout].md[0].write = 1;
            for (mode = 0; mode < NBmodeOUT; mode++)
            {
                dcimg[IDmasterout].array.F[outmaskindex[mode]] = dcimg[IDPFout].array.F[mode];
            }
            COREMOD_MEMORY_image_set_sempost_byID(IDmasterout, -1);
            dcimg[IDmasterout].md[0].write = 0;
            dcimg[IDmasterout].md[0].cnt0++;
        }

        iter++;

        if (iter != NBiter)
        {
            // do this now to save time when semaphore is posted
            for (tstep = NBPFstep - 1; tstep > 0; tstep--)
            {
                // tstep-1 -> tstep
                for (mode = 0; mode < NBmodeIN; mode++)
                {
                    dcimg[IDINbuff].array.F[NBmodeIN * tstep + mode] =
                        dcimg[IDINbuff].array.F[NBmodeIN * (tstep - 1) + mode];
                }
            }
        }
    }
    printf("LOOP done\n");
    fflush(stdout);

    // output ASCII file
    if (SAVEMODE == 1)
    {
        printf("SAVING DATA [1] ...");
        fflush(stdout);

        printf("IDsave = %ld     %ld  %ld\n", IDsave, 1 + NBmodeIN0 + NBmodeOUT, NBmodeOUT);
        list_image_ID();

        //	for(mode=0;mode<NBmodeOUT;mode++)
        //	printf("output %4ld -> %5ld\n", outmaskindex[mode]);

        fpout = fopen("testPFsave.dat", "w");
        for (iter = 0; iter < NBiter; iter++)
        {
            fprintf(fpout, "%5ld ", iter);
            for (kk = 0; kk < (1 + NBmodeIN0 + NBmodeOUT); kk++)
            {
                fprintf(fpout, "%10f ",
                        dcimg[IDsave].array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk]);
            }

            tlag0     = (long) tlag;
            tlagalpha = tlag - tlag0;

            ii0 = iter - (tlag0 + 1);
            ii1 = iter - (tlag0);

            for (mode = 0; mode < NBmodeOUT; mode++)
            {
                if (ii0 > -1)
                {
                    val0 = dcimg[IDsave]
                               .array.F[ii0 * (1 + NBmodeIN0 + NBmodeOUT) + 1 + NBmodeIN0 + mode];
                    val1 = dcimg[IDsave]
                               .array.F[ii1 * (1 + NBmodeIN0 + NBmodeOUT) + 1 + NBmodeIN0 + mode];
                }
                val = tlagalpha * val0 + (1.0 - tlagalpha) * val1;
                fprintf(fpout, "%10f ", val);
            }
            fprintf(fpout, "\n");
        }
        fclose(fpout);

        printf(" done\n");
        fflush(stdout);
    }

    free(inmaskindex);

    if (SAVEMODE == 2) // time shift predicted output into FITS output
    {
        tlag0     = (long) tlag;
        tlagalpha = tlag - tlag0;
        for (kk = NBiter - 1; kk > tlag0; kk--)
        {
            kk0 = kk - (tlag0 + 1);
            kk1 = kk - (tlag0);

            for (mode = 0; mode < NBmodeOUT; mode++)
            {
                val0 = dcimg[IDmodevalIN].array.F[kk0 * NBmodeIN0 + outmaskindex[mode]];
                val1 = dcimg[IDmodevalIN].array.F[kk1 * NBmodeIN0 + outmaskindex[mode]];
                val  = tlagalpha * val0 + (1.0 - tlagalpha) * val1;

                dcimg[IDsave].array.F[kk * NBmodeIN0 + outmaskindex[mode]] = val;
            }
        }

        save_fits("testPFTout", "testPFTout.fits");
    }

    if (SAVEMODE > 0)
    {
        free(outmaskindex);
    }

    return IDPFout;
}
