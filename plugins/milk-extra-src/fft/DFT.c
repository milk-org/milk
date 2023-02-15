/** @file DFT.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ----------------- CUSTOM DFT ------------- */

//
// Zfactor is zoom factor
// dir = -1 for FT, 1 for inverse FT
// kin in selects slice in IDin_name if this is a cube
//
errno_t fft_DFT(const char *IDin_name,
                const char *IDinmask_name,
                const char *IDout_name,
                const char *IDoutmask_name,
                double      Zfactor,
                int         dir,
                long        kin,
                imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID IDin;
    imageID IDout;
    imageID IDinmask;
    imageID IDoutmask;

    //uint32_t NBptsin;
    //uint32_t NBptsout;

    imageID IDcosXX, IDcosYY, IDsinXX, IDsinYY;

    IDin = image_ID(IDin_name);

    IDinmask        = image_ID(IDinmask_name);
    uint32_t xsize  = data.image[IDinmask].md[0].size[0];
    uint32_t ysize  = data.image[IDinmask].md[0].size[1];
    uint64_t xysize = xsize;
    xysize *= ysize;

    // list of active coordinates
    uint32_t *iiinarrayActive  = (uint32_t *) malloc(sizeof(uint32_t) * xsize);
    uint32_t *jjinarrayActive  = (uint32_t *) malloc(sizeof(uint32_t) * ysize);
    uint32_t *iioutarrayActive = (uint32_t *) malloc(sizeof(uint32_t) * xsize);
    uint32_t *jjoutarrayActive = (uint32_t *) malloc(sizeof(uint32_t) * ysize);

    uint32_t NBptsin       = 0;
    uint32_t NBpixact_iiin = 0;
    for(uint32_t ii = 0; ii < xsize; ii++)
    {
        int pixact = 0;
        for(uint32_t jj = 0; jj < ysize; jj++)
        {
            float val = data.image[IDinmask].array.F[jj * xsize + ii];
            if(val > 0.5)
            {
                pixact = 1;
                NBptsin++;
            }
        }
        if(pixact == 1)
        {
            iiinarrayActive[NBpixact_iiin] = ii;
            NBpixact_iiin++;
        }
    }

    uint32_t NBpixact_jjin = 0;
    for(uint32_t jj = 0; jj < ysize; jj++)
    {
        int pixact = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
        {
            float val = data.image[IDinmask].array.F[jj * xsize + ii];
            if(val > 0.5)
            {
                pixact = 1;
            }
        }
        if(pixact == 1)
        {
            jjinarrayActive[NBpixact_jjin] = jj;
            NBpixact_jjin++;
        }
    }

    float *XinarrayActive = (float *) malloc(sizeof(float) * NBpixact_iiin);
    float *YinarrayActive = (float *) malloc(sizeof(float) * NBpixact_jjin);

    for(uint32_t pixiiin = 0; pixiiin < NBpixact_iiin; pixiiin++)
    {
        uint32_t iiin           = iiinarrayActive[pixiiin];
        XinarrayActive[pixiiin] = (1.0 * iiin / xsize - 0.5);
    }
    for(uint32_t pixjjin = 0; pixjjin < NBpixact_jjin; pixjjin++)
    {
        uint32_t jjin           = jjinarrayActive[pixjjin];
        YinarrayActive[pixjjin] = (1.0 * jjin / ysize - 0.5);
    }

    printf("DFT (factor %f, slice %ld):  %u input points (%u %u)-> ",
           Zfactor,
           kin,
           NBptsin,
           NBpixact_iiin,
           NBpixact_jjin);

    uint32_t *iiinarray   = (uint32_t *) malloc(sizeof(uint32_t) * NBptsin);
    uint32_t *jjinarray   = (uint32_t *) malloc(sizeof(uint32_t) * NBptsin);
    double   *xinarray    = (double *) malloc(sizeof(double) * NBptsin);
    double   *yinarray    = (double *) malloc(sizeof(double) * NBptsin);
    double   *valinamp    = (double *) malloc(sizeof(double) * NBptsin);
    double   *valinpha    = (double *) malloc(sizeof(double) * NBptsin);
    float    *cosvalinpha = (float *) malloc(sizeof(float) * NBptsin);
    float    *sinvalinpha = (float *) malloc(sizeof(float) * NBptsin);

    {
        uint32_t k = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                float val = data.image[IDinmask].array.F[jj * xsize + ii];
                if(val > 0.5)
                {
                    iiinarray[k] = ii;
                    jjinarray[k] = jj;
                    xinarray[k]  = 1.0 * ii / xsize - 0.5;
                    yinarray[k]  = 1.0 * jj / xsize - 0.5;
                    float re =
                        data.image[IDin]
                        .array.CF[kin * xsize * ysize + jj * xsize + ii]
                        .re;
                    float im =
                        data.image[IDin]
                        .array.CF[kin * xsize * ysize + jj * xsize + ii]
                        .im;
                    valinamp[k]    = sqrt(re * re + im * im);
                    valinpha[k]    = atan2(im, re);
                    cosvalinpha[k] = cosf(valinpha[k]);
                    sinvalinpha[k] = sinf(valinpha[k]);
                    k++;
                }
            }
    }

    IDoutmask = image_ID(IDoutmask_name);

    uint32_t NBptsout       = 0;
    uint32_t NBpixact_iiout = 0;
    for(uint32_t ii = 0; ii < xsize; ii++)
    {
        int pixact = 0;
        for(uint32_t jj = 0; jj < ysize; jj++)
        {
            float val = data.image[IDoutmask].array.F[jj * xsize + ii];
            if(val > 0.5)
            {
                pixact = 1;
                NBptsout++;
            }
        }
        if(pixact == 1)
        {
            iioutarrayActive[NBpixact_iiout] = ii;
            NBpixact_iiout++;
        }
    }

    uint32_t NBpixact_jjout = 0;
    for(uint32_t jj = 0; jj < ysize; jj++)
    {
        int pixact = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
        {
            float val = data.image[IDoutmask].array.F[jj * xsize + ii];
            if(val > 0.5)
            {
                pixact = 1;
            }
        }
        if(pixact == 1)
        {
            jjoutarrayActive[NBpixact_jjout] = jj;
            NBpixact_jjout++;
        }
    }
    float *XoutarrayActive = (float *) malloc(sizeof(float) * NBpixact_iiout);
    float *YoutarrayActive = (float *) malloc(sizeof(float) * NBpixact_jjout);

    for(uint32_t pixiiout = 0; pixiiout < NBpixact_iiout; pixiiout++)
    {
        uint32_t iiout = iioutarrayActive[pixiiout];
        XoutarrayActive[pixiiout] =
            (1.0 / Zfactor) * (1.0 * iiout / xsize - 0.5) * xsize;
    }

    for(uint32_t pixjjout = 0; pixjjout < NBpixact_jjout; pixjjout++)
    {
        uint32_t jjout = jjoutarrayActive[pixjjout];
        YoutarrayActive[pixjjout] =
            (1.0 / Zfactor) * (1.0 * jjout / ysize - 0.5) * ysize;
    }

    printf("%u output points (%u %u) \n",
           NBptsout,
           NBpixact_iiout,
           NBpixact_jjout);

    uint32_t *iioutarray = (uint32_t *) malloc(sizeof(uint32_t) * NBptsout);
    uint32_t *jjoutarray = (uint32_t *) malloc(sizeof(uint32_t) * NBptsout);
    double   *xoutarray  = (double *) malloc(sizeof(double) * NBptsout);
    double   *youtarray  = (double *) malloc(sizeof(double) * NBptsout);

    {
        uint32_t kout = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                float val = data.image[IDoutmask].array.F[jj * xsize + ii];
                if(val > 0.5)
                {
                    iioutarray[kout] = ii;
                    jjoutarray[kout] = jj;
                    xoutarray[kout] =
                        (1.0 / Zfactor) * (1.0 * ii / xsize - 0.5) * xsize;
                    youtarray[kout] =
                        (1.0 / Zfactor) * (1.0 * jj / ysize - 0.5) * ysize;
                    kout++;
                }
            }
    }

    FUNC_CHECK_RETURN(create_2DCimage_ID(IDout_name, xsize, ysize, &IDout));

    FUNC_CHECK_RETURN(create_2Dimage_ID("_cosXX", xsize, xsize, &IDcosXX));

    FUNC_CHECK_RETURN(create_2Dimage_ID("_sinXX", xsize, xsize, &IDsinXX));

    FUNC_CHECK_RETURN(create_2Dimage_ID("_cosYY", ysize, ysize, &IDcosYY));

    FUNC_CHECK_RETURN(create_2Dimage_ID("_sinYY", ysize, ysize, &IDsinYY));

    DEBUG_TRACEPOINT(" ");

    printf(" <");
    fflush(stdout);

#ifdef _OPENMP
    printf("Using openMP %d", omp_get_max_threads());
    #pragma omp parallel
    {
        #pragma omp for
#endif

        for(uint32_t pixiiout = 0; pixiiout < NBpixact_iiout; pixiiout++)
        {
            uint32_t iiout = iioutarrayActive[pixiiout];
            for(uint32_t pixiiin = 0; pixiiin < NBpixact_iiin; pixiiin++)
            {
                uint32_t iiin = iiinarrayActive[pixiiin];
                float    pha =
                    2.0 * dir * M_PI *
                    (XinarrayActive[pixiiin] * XoutarrayActive[pixiiout]);
                float cospha = cosf(pha);
                float sinpha = sinf(pha);

                data.image[IDcosXX].array.F[iiout * xsize + iiin] = cospha;
                data.image[IDsinXX].array.F[iiout * xsize + iiin] = sinpha;
            }
        }
#ifdef _OPENMP
    }
#endif

    printf("> ");
    fflush(stdout);

    DEBUG_TRACEPOINT(" ");

    printf(" <");
    fflush(stdout);

#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for
#endif
        for(uint32_t pixjjout = 0; pixjjout < NBpixact_jjout; pixjjout++)
        {
            uint32_t jjout = jjoutarrayActive[pixjjout];
            for(uint32_t pixjjin = 0; pixjjin < NBpixact_jjin; pixjjin++)
            {
                uint32_t jjin = jjinarrayActive[pixjjin];
                float    pha =
                    2.0 * dir * M_PI *
                    (YinarrayActive[pixjjin] * YoutarrayActive[pixjjout]);
                float cospha = cosf(pha);
                float sinpha = sinf(pha);

                data.image[IDcosYY].array.F[jjout * ysize + jjin] = cospha;
                data.image[IDsinYY].array.F[jjout * ysize + jjin] = sinpha;
            }
        }
#ifdef _OPENMP
    }
#endif
    printf("> ");
    fflush(stdout);

    // DFT
    DEBUG_TRACEPOINT(" ");

    printf(" <");
    fflush(stdout);

#ifdef _OPENMP
    printf(" -omp- %d ", omp_get_max_threads());
    fflush(stdout);
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf(" [%d thread(s)] ", omp_get_num_threads());
            fflush(stdout);
        }

        #pragma omp for
#endif

        for(uint32_t kout = 0; kout < NBptsout; kout++)
        {
            uint32_t iiout = iioutarray[kout];
            uint32_t jjout = jjoutarray[kout];

            float re = 0.0;
            float im = 0.0;
            for(uint32_t k = 0; k < NBptsin; k++)
            {
                uint32_t iiin = iiinarray[k];
                uint32_t jjin = jjinarray[k];

                float cosXX = data.image[IDcosXX].array.F[iiout * xsize + iiin];
                float cosYY = data.image[IDcosYY].array.F[jjout * ysize + jjin];

                float sinXX = data.image[IDsinXX].array.F[iiout * xsize + iiin];
                float sinYY = data.image[IDsinYY].array.F[jjout * ysize + jjin];

                float cosXY = cosXX * cosYY - sinXX * sinYY;
                float sinXY = sinXX * cosYY + cosXX * sinYY;

                float cospha = cosvalinpha[k] * cosXY - sinvalinpha[k] * sinXY;
                float sinpha = sinvalinpha[k] * cosXY + cosvalinpha[k] * sinXY;

                re += valinamp[k] * cospha;
                im += valinamp[k] * sinpha;
            }
            data.image[IDout]
            .array.CF[jjoutarray[kout] * xsize + iioutarray[kout]]
            .re = re / Zfactor;
            data.image[IDout]
            .array.CF[jjoutarray[kout] * xsize + iioutarray[kout]]
            .im = im / Zfactor;
        }

#ifdef _OPENMP
    }
#endif
    printf("> ");
    fflush(stdout);

    DEBUG_TRACEPOINT(" ");

    free(cosvalinpha);
    free(sinvalinpha);

    FUNC_CHECK_RETURN(delete_image_ID("_cosXX", DELETE_IMAGE_ERRMODE_WARNING));

    FUNC_CHECK_RETURN(delete_image_ID("_sinXX", DELETE_IMAGE_ERRMODE_WARNING));

    FUNC_CHECK_RETURN(delete_image_ID("_cosYY", DELETE_IMAGE_ERRMODE_WARNING));

    FUNC_CHECK_RETURN(delete_image_ID("_sinYY", DELETE_IMAGE_ERRMODE_WARNING));

    free(XinarrayActive);
    free(YinarrayActive);
    free(XoutarrayActive);
    free(YoutarrayActive);

    free(iiinarrayActive);
    free(jjinarrayActive);
    free(iioutarrayActive);
    free(jjoutarrayActive);

    free(iiinarray);
    free(jjinarray);
    free(xinarray);
    free(yinarray);
    free(valinamp);
    free(valinpha);

    free(iioutarray);
    free(jjoutarray);
    free(xoutarray);
    free(youtarray);

    DEBUG_TRACEPOINT("IDout = %ld", IDout);

    if(outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * @brief Use DFT to insert Focal Plane Mask
 *
 *  Pupil convolution by complex focal plane mask of limited support
 *  typically used with fpmz = zoomed copy of 1-fpm
 *
 * High resolution focal plane mask using DFT
 *
 * Forces computation over pixels >0.5 in _DFTmask00 if it exists
 *
 */
errno_t fft_DFTinsertFPM(const char *pupin_name,
                         const char *fpmz_name,
                         double      zfactor,
                         const char *pupout_name,
                         imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    double   eps = 1.0e-16;
    imageID  ID;
    imageID  IDpupin_mask;
    imageID  IDfpmz;
    imageID  IDfpmz_mask;
    uint32_t xsize, ysize, zsize;
    imageID  IDin, IDout;
    double   total = 0;
    imageID  IDout2D;
    int      FORCE_IMZERO = 0;
    //double imresidual = 0.0;

    imageID ID_DFTmask00;

    if(variable_ID("_FORCE_IMZERO") != -1)
    {
        FORCE_IMZERO = 1;
        printf("---------------FORCING IMAGINARY PART TO ZERO-------------\n");
    }

    ID_DFTmask00 = image_ID("_DFTmask00");

    printf("zfactor = %f\n", zfactor);

    IDin            = image_ID(pupin_name);
    xsize           = data.image[IDin].md[0].size[0];
    ysize           = data.image[IDin].md[0].size[1];
    uint64_t xysize = (uint64_t) xsize;
    xysize *= ysize;

    if(data.image[IDin].md[0].naxis > 2)
    {
        zsize = data.image[IDin].md[0].size[2];
    }
    else
    {
        zsize = 1;
    }
    printf("zsize = %ld\n", (long) zsize);

    FUNC_CHECK_RETURN(
        create_3DCimage_ID(pupout_name, xsize, ysize, zsize, &IDout));

    for(uint32_t k = 0; k < zsize; k++)  // increment slice (= wavelength)
    {
        //
        // Create default input mask for DFT
        // if amplitude above threshold value, turn pixel "on"
        //
        FUNC_CHECK_RETURN(
            create_2Dimage_ID("_DFTpupmask", xsize, ysize, &IDpupin_mask));

        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            double re   = data.image[IDin].array.CF[k * xysize + ii].re;
            double im   = data.image[IDin].array.CF[k * xysize + ii].im;
            double amp2 = re * re + im * im;
            if(amp2 > eps)
            {
                data.image[IDpupin_mask].array.F[ii] = 1.0;
            }
            else
            {
                data.image[IDpupin_mask].array.F[ii] = 0.0;
            }
        }
        //
        // If _DFTmask00 exists, make corresponding pixels = 1
        //
        if(ID_DFTmask00 != -1)
            for(uint64_t ii = 0; ii < xsize * ysize; ii++)
            {
                if(data.image[ID_DFTmask00].array.F[ii] > 0.5)
                {
                    data.image[IDpupin_mask].array.F[ii] = 1.0;
                }
            }

        //
        // Construct focal plane mask for DFT
        // If amplitude >eps, turn pixel ON, save result in _fpmzmask
        //
        IDfpmz = image_ID(fpmz_name);
        FUNC_CHECK_RETURN(
            create_2Dimage_ID("_fpmzmask", xsize, ysize, &IDfpmz_mask));

        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            double re   = data.image[IDfpmz].array.CF[k * xysize + ii].re;
            double im   = data.image[IDfpmz].array.CF[k * xysize + ii].im;
            double amp2 = re * re + im * im;
            if(amp2 > eps)
            {
                data.image[IDfpmz_mask].array.F[ii] = 1.0;
            }
            else
            {
                data.image[IDfpmz_mask].array.F[ii] = 0.0;
            }
        }

        //	save_fits("_DFTpupmask", "_DFTpupmask.fits");

        FUNC_CHECK_RETURN(fft_DFT(pupin_name,
                                  "_DFTpupmask",
                                  "_foc0",
                                  "_fpmzmask",
                                  zfactor,
                                  -1,
                                  k,
                                  &ID));

        total      = 0.0;
        double tx  = 0.0;
        double ty  = 0.0;
        double tcx = 0.0;
        double tcy = 0.0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                double x  = 1.0 * ii - 0.5 * xsize;
                double y  = 1.0 * jj - 0.5 * ysize;
                double re = data.image[IDfpmz]
                            .array.CF[k * xysize + jj * xsize + ii]
                            .re;
                double im = data.image[IDfpmz]
                            .array.CF[k * xysize + jj * xsize + ii]
                            .im;
                double amp = sqrt(re * re + im * im);
                double pha = atan2(im, re);

                double rein  = data.image[ID].array.CF[jj * xsize + ii].re;
                double imin  = data.image[ID].array.CF[jj * xsize + ii].im;
                double ampin = sqrt(rein * rein + imin * imin);
                double phain = atan2(imin, rein);

                ampin *= amp;
                total += ampin * ampin;
                phain += pha;

                data.image[ID].array.CF[jj * xsize + ii].re =
                    ampin * cos(phain);
                data.image[ID].array.CF[jj * xsize + ii].im =
                    ampin * sin(phain);

                tx += x * ampin * sin(phain) * ampin;
                ty += y * ampin * sin(phain) * ampin;
                tcx += x * x * ampin * ampin;
                tcy += y * y * ampin * ampin;
            }
        printf("TX TY = %.18lf %.18lf", tx / tcx, ty / tcy);
        if(FORCE_IMZERO ==
                1) // Remove tip-tilt in focal plane mask imaginary part
        {
            tx = 0.0;
            ty = 0.0;
            for(uint32_t ii = 0; ii < xsize; ii++)
                for(uint32_t jj = 0; jj < ysize; jj++)
                {
                    double x = 1.0 * ii - 0.5 * xsize;
                    double y = 1.0 * jj - 0.5 * ysize;

                    double re  = data.image[ID].array.CF[jj * xsize + ii].re;
                    double im  = data.image[ID].array.CF[jj * xsize + ii].im;
                    double amp = sqrt(re * re + im * im);

                    data.image[ID].array.CF[jj * xsize + ii].im -=
                        amp * (x * tx / tcx + y * ty / tcy);
                    tx += x * data.image[ID].array.CF[jj * xsize + ii].im * amp;
                    ty += y * data.image[ID].array.CF[jj * xsize + ii].im * amp;
                }
            printf("  ->   %.18lf %.18lf", tx / tcx, ty / tcy);

            FUNC_CHECK_RETURN(
                mk_amph_from_complex("_foc0", "_foc0_amp", "_foc0_pha", 0));

            FUNC_CHECK_RETURN(save_fl_fits("_foc0_amp", "_foc_amp.fits"));

            FUNC_CHECK_RETURN(save_fl_fits("_foc0_pha", "_foc_pha.fits"));

            FUNC_CHECK_RETURN(
                delete_image_ID("_foc0_amp", DELETE_IMAGE_ERRMODE_WARNING));

            FUNC_CHECK_RETURN(
                delete_image_ID("_foc0_pha", DELETE_IMAGE_ERRMODE_WARNING));
        }
        printf("\n");

        data.FLOATARRAY[0] = (float) total;

        /*  if(FORCE_IMZERO==1) // Remove tip-tilt in focal plane mask imaginary part
        {
        imresidual = 0.0;
        ID = image_ID("_foc0");
        ID1 = create_2Dimage_ID("imresidual", xsize, ysize);
        for(ii=0; ii<xsize*ysize; ii++)
        {
        data.image[ID1].array.F[ii] = data.image[ID].array.CF[ii].im;
        imresidual += data.image[ID].array.CF[ii].im*data.image[ID].array.CF[ii].im;
        data.image[ID].array.CF[ii].im = 0.0;
        }
        printf("IM RESIDUAL = %lf\n", imresidual);
        save_fl_fits("imresidual", "imresidual.fits");
        delete_image_ID("imresidual");
        }
        */

        if(0)  // TEST
        {
            /// @warning This internal test could crash the process as multiple write operations to the same filename may occurr: leave option OFF for production

            FUNC_CHECK_RETURN(
                mk_amph_from_complex("_foc0", "tmp_foc0_a", "tmp_foc0_p", 0));

            FUNC_CHECK_RETURN(save_fl_fits("tmp_foc0_a", "_DFT_foca.fits"));

            FUNC_CHECK_RETURN(save_fl_fits("tmp_foc0_p", "_DFT_focp.fits"));

            FUNC_CHECK_RETURN(
                delete_image_ID("tmp_foc0_a", DELETE_IMAGE_ERRMODE_WARNING););

            FUNC_CHECK_RETURN(
                delete_image_ID("tmp_foc0_p", DELETE_IMAGE_ERRMODE_WARNING));
        }

        /* for(ii=0; ii<xsize; ii++)
        for(jj=0; jj<ysize; jj++)
         {
         x = 1.0*ii-xsize/2;
         y = 1.0*jj-ysize/2;
         r = sqrt(x*x+y*y);
         if(r<150.0)
         data.image[IDpupin_mask].array.F[jj*xsize+ii] = 1.0;
         }*/

        FUNC_CHECK_RETURN(fft_DFT("_foc0",
                                  "_fpmzmask",
                                  "_pupout2D",
                                  "_DFTpupmask",
                                  zfactor,
                                  1,
                                  0,
                                  &IDout2D));

        DEBUG_TRACEPOINT("k %u / %u  IDout = %ld  IDout2D = %ld",
                         k,
                         zsize,
                         IDout,
                         IDout2D);
        DEBUG_TRACEPOINT("xysize = %lu", xysize);

        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            data.image[IDout].array.CF[k * xysize + ii].re =
                data.image[IDout2D].array.CF[ii].re / (xysize);
            data.image[IDout].array.CF[k * xysize + ii].im =
                data.image[IDout2D].array.CF[ii].im / (xysize);
        }

        DEBUG_TRACEPOINT("IDout image content written");

        FUNC_CHECK_RETURN(
            delete_image_ID("_pupout2D", DELETE_IMAGE_ERRMODE_WARNING));

        FUNC_CHECK_RETURN(
            delete_image_ID("_foc0", DELETE_IMAGE_ERRMODE_WARNING));

        FUNC_CHECK_RETURN(
            delete_image_ID("_DFTpupmask", DELETE_IMAGE_ERRMODE_WARNING));

        FUNC_CHECK_RETURN(
            delete_image_ID("_fpmzmask", DELETE_IMAGE_ERRMODE_WARNING));
    }

    if(outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

//
// pupil convolution by real focal plane mask of limited support
// typically used with fpmz = zoomed copy of 1-fpm
// high resolution focal plane mask using DFT
// zoom factor
//
//
//
errno_t fft_DFTinsertFPM_re(const char *pupin_name,
                            const char *fpmz_name,
                            double      zfactor,
                            const char *pupout_name,
                            imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    double  eps = 1.0e-10;
    imageID ID;
    imageID IDpupin_mask;
    imageID IDfpmz;
    imageID IDfpmz_mask;
    imageID IDout;

    imageID ID_DFTmask00;

    imageID  IDin   = image_ID(pupin_name);
    uint32_t xsize  = data.image[IDin].md[0].size[0];
    uint32_t ysize  = data.image[IDin].md[0].size[1];
    uint64_t xysize = xsize;
    xysize *= ysize;

    ID_DFTmask00 = image_ID("_DFTmask00");

    printf("zfactor = %f\n", zfactor);

    FUNC_CHECK_RETURN(
        create_2Dimage_ID("_DFTpupmask", xsize, ysize, &IDpupin_mask));
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        double re   = data.image[IDin].array.CF[ii].re;
        double im   = data.image[IDin].array.CF[ii].im;
        double amp2 = re * re + im * im;
        if(amp2 > eps)
        {
            data.image[IDpupin_mask].array.F[ii] = 1.0;
        }
        else
        {
            data.image[IDpupin_mask].array.F[ii] = 0.0;
        }
    }
    //  save_fl_fits("_DFTpupmask", "_DFTpupmask.fits");

    if(ID_DFTmask00 != -1)
        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            if(data.image[ID_DFTmask00].array.F[ii] > 0.5)
            {
                data.image[IDpupin_mask].array.F[ii] = 1.0;
            }
        }

    // ! Why read and re-create ?
    IDfpmz = image_ID(fpmz_name);
    FUNC_CHECK_RETURN(
        create_2Dimage_ID("_fpmzmask", xsize, ysize, &IDfpmz_mask));

    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        double amp = fabs(data.image[IDfpmz].array.F[ii]);
        if(amp > eps)
        {
            data.image[IDfpmz_mask].array.F[ii] = 1.0;
        }
        else
        {
            data.image[IDfpmz_mask].array.F[ii] = 0.0;
        }
    }

    FUNC_CHECK_RETURN(fft_DFT(pupin_name,
                              "_DFTpupmask",
                              "_foc0",
                              "_fpmzmask",
                              zfactor,
                              -1,
                              0,
                              &ID));

    {
        double total = 0.0;
        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            double amp = data.image[IDfpmz].array.F[ii];

            double rein  = data.image[ID].array.CF[ii].re;
            double imin  = data.image[ID].array.CF[ii].im;
            double ampin = sqrt(rein * rein + imin * imin);
            double phain = atan2(imin, rein);

            ampin *= amp;
            total += ampin * ampin;

            data.image[ID].array.CF[ii].re = ampin * cos(phain);
            data.image[ID].array.CF[ii].im = ampin * sin(phain);
        }
        data.FLOATARRAY[0] = (float) total;
    }

    if(0)  // TEST
    {
        char fname[STRINGMAXLEN_FULLFILENAME];

        mk_amph_from_complex("_foc0", "tmp_foc0_a", "tmp_foc0_p", 0);

        WRITE_FULLFILENAME(fname, "%s/_DFT_foca", data.SAVEDIR);
        FUNC_CHECK_RETURN(save_fl_fits("tmp_foc0_a", fname));

        WRITE_FULLFILENAME(fname, "%s/_DFT_focp", data.SAVEDIR);
        FUNC_CHECK_RETURN(save_fl_fits("tmp_foc0_p", fname));

        FUNC_CHECK_RETURN(
            delete_image_ID("tmp_foc0_a", DELETE_IMAGE_ERRMODE_WARNING));

        FUNC_CHECK_RETURN(
            delete_image_ID("tmp_foc0_p", DELETE_IMAGE_ERRMODE_WARNING));
    }

    FUNC_CHECK_RETURN(fft_DFT("_foc0",
                              "_fpmzmask",
                              pupout_name,
                              "_DFTpupmask",
                              zfactor,
                              1,
                              0,
                              &IDout));

    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        data.image[IDout].array.CF[ii].re /= xsize * ysize;
        data.image[IDout].array.CF[ii].im /= xsize * ysize;
    }

    FUNC_CHECK_RETURN(delete_image_ID("_foc0", DELETE_IMAGE_ERRMODE_WARNING));

    FUNC_CHECK_RETURN(
        delete_image_ID("_DFTpupmask", DELETE_IMAGE_ERRMODE_WARNING));

    FUNC_CHECK_RETURN(
        delete_image_ID("_fpmzmask", DELETE_IMAGE_ERRMODE_WARNING));

    if(outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
