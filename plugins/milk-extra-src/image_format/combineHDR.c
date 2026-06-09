/**
 * @file combineHDR.c
 * @brief Combinehdr module
 */

/** @file combineHDR.c
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#ifdef USE_CFITSIO
#include "COREMOD_iofits/COREMOD_iofits.h"
#endif

#include "image_filter/image_filter.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "combineHDR",
    .cmdkey           = "combineHDR",
    .description      = "combine HDR image",
    .description_long = "Combine multiple exposures into a high dynamic range (HDR) image. Merges "
                        "short and long exposures weighted by signal-to-noise."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   *flistname = NULL;
static double *satlevel  = NULL;
static double *biaslevel = NULL;
static char   *outimname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                      \
    X(".flistname", &flistname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "file list name")  \
    X(".satlevel", &satlevel, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "Saturation level") \
    X(".biaslevel", &biaslevel, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "Bias level")     \
    X(".outimname", &outimname, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT, "output image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

errno_t combine_HDR_image(const char *__restrict flistname,
                          float satvalue,
                          float biasvalue,
                          char *__restrict outimname)
{
    int HDRmaxindex = 100;

    // Read input files and exposure times
    int     NB_HDRindex = 0;
    float   etimearray[HDRmaxindex];
    imageID IDarray[HDRmaxindex];

    {
        FILE *fpin;

        fpin = fopen(flistname, "r");
        char FITSfname[200];

        float etime;
        int   HDRindex = 0;
        char  imHDRin[200];
        char  timestring[200];
        while (fscanf(fpin, "%s %f %s\n", FITSfname, &etime, timestring) == 3)
        {
            imageID ID;
            printf("Input file [%11.6f] : %s\n", etime, FITSfname);
            etimearray[HDRindex] = etime;
            snprintf(imHDRin, sizeof(imHDRin), "imHRDin_%03d", HDRindex);
#ifdef USE_CFITSIO
            load_fits(FITSfname, imHDRin, 2, &ID);
#else
            printf("Compiled without CFITSIO\n");
            exit(1);
#endif
            IDarray[HDRindex] = ID;
            HDRindex++;
        }
        fclose(fpin);
        NB_HDRindex = HDRindex;
    }

    printf("PARAMS : %20f %20f\n", biasvalue, 1.0 * satvalue);

    uint32_t xsize = dcimg[IDarray[0]].md->size[0];
    uint32_t ysize = dcimg[IDarray[0]].md->size[1];
    uint32_t zsize = NB_HDRindex;

    int      binstep = 5;
    uint32_t xsize1  = (uint32_t) (xsize / binstep);
    uint32_t ysize1  = (uint32_t) (ysize / binstep);
    //
    // Assemble cube and subsampled cube
    //
    imageID IDimHDRc;
    create_3Dimage_ID("imHDRc", xsize, ysize, zsize, &IDimHDRc);

    imageID IDimHDRc1;
    create_3Dimage_ID("imHDRc1", xsize1, ysize1, zsize, &IDimHDRc1);

    imageID IDimHDRc1w;
    create_3Dimage_ID("imHDRc1w", xsize1, ysize1, zsize, &IDimHDRc1w);

    for (uint32_t kk = 0; kk < zsize; kk++)
    {
        for (uint32_t jj = 0; jj < ysize; jj++)
        {
            float    y   = 1.0 * jj / ysize;
            uint32_t jj1 = (uint32_t) (y * ysize1);

            for (uint32_t ii = 0; ii < xsize; ii++)
            {
                float    x   = 1.0 * ii / xsize;
                uint32_t ii1 = (uint32_t) (x * xsize1);

                float pval = 1.0f * dcimg[IDarray[kk]].array.F[jj * xsize + ii] - biasvalue;

                dcimg[IDimHDRc].array.F[kk * xsize * ysize + jj * xsize + ii] = pval;

                dcimg[IDimHDRc1].array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] += pval;
                dcimg[IDimHDRc1w].array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] += 1.0f;
            }
        }
    }

    for (uint32_t kk = 0; kk < zsize; kk++)
    {
        for (uint32_t jj1 = 0; jj1 < ysize1; jj1++)
        {
            for (uint32_t ii1 = 0; ii1 < xsize1; ii1++)
            {
                dcimg[IDimHDRc1].array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] /=
                    dcimg[IDimHDRc1w].array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1];
            }
        }
    }

    {
        printf("---------------- Convolve binned image ------------\n");
        fflush(stdout);
        int    NBfiter = 5;
        float *pixcol  = (float *) malloc(sizeof(float) * ysize1);
        float *pixline = (float *) malloc(sizeof(float) * xsize1);
        for (int fiter = 0; fiter < NBfiter; fiter++)
        {
            printf(".");
            fflush(stdout);

            for (uint32_t kk = 0; kk < zsize; kk++)
            {
                for (uint32_t jj1 = 0; jj1 < ysize1; jj1++)
                {
                    for (uint32_t ii1 = 1; ii1 < xsize1 - 1; ii1++)
                    {
                        pixline[ii1] =
                            0.3 * dcimg[IDimHDRc1]
                                      .array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1 - 1] +
                            0.4 * dcimg[IDimHDRc1]
                                      .array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] +
                            0.3 * dcimg[IDimHDRc1]
                                      .array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1 + 1];
                    }
                    for (uint32_t ii1 = 1; ii1 < xsize1 - 1; ii1++)
                    {
                        dcimg[IDimHDRc1].array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] =
                            pixline[ii1];
                    }
                }

                for (uint32_t ii1 = 0; ii1 < xsize1; ii1++)
                {
                    for (uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                    {
                        pixcol[jj1] =
                            0.3 * dcimg[IDimHDRc1]
                                      .array.F[kk * xsize1 * ysize1 + (jj1 - 1) * xsize1 + ii1] +
                            0.4 * dcimg[IDimHDRc1]
                                      .array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] +
                            0.3 * dcimg[IDimHDRc1]
                                      .array.F[kk * xsize1 * ysize1 + (jj1 + 1) * xsize1 + ii1];
                    }
                    for (uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                    {
                        dcimg[IDimHDRc1].array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] =
                            pixcol[jj1];
                    }
                }
            }
        }
        free(pixcol);
        free(pixline);
        printf("\n");
        printf("---------------- DONE ------------\n");
        fflush(stdout);
    }

    //double alpha1 = 1.0;

    // contruct layer image
    imageID IDlayer;
    create_2Dimage_ID("imlayer", xsize1, ysize1, &IDlayer);

    imageID IDlayermin;
    create_2Dimage_ID("imlayermin", xsize1, ysize1, &IDlayermin);

    for (uint32_t ij1 = 0; ij1 < xsize1 * ysize1; ij1++)
    {
        //float layer = 0.0;
        uint32_t layer0 = 0;
        uint32_t layer1 = 0;
        uint32_t kk     = 0;
        while ((kk < zsize) && (dcimg[IDimHDRc1].array.F[kk * xsize1 * ysize1 + ij1] > satvalue))
        {
            layer0 = kk;
            kk++;
        }

        layer1 = layer0 + 1;
        if (layer1 == zsize)
        {
            layer1 = zsize - 1;
        }

        float valmax = dcimg[IDimHDRc1].array.F[layer0 * xsize1 * ysize1 + ij1];
        if ((valmax > satvalue) && (layer1 < zsize - 1))
        {
            // increment layers
            layer0++;
            layer1++;
            valmax = dcimg[IDimHDRc1].array.F[layer0 * xsize1 * ysize1 + ij1];
        }
        //float valmin = dcimg[IDimHDRc1].array.F[layer1*xsize1*ysize1+ij1];

        //float x = valmax/satvalue;
        //float c1 = pow( 0.5*(1.0+cos(x*M_PI)), alpha1);
        //float c2 = 1.0-c1;

        dcimg[IDlayer].array.F[ij1]    = 1.0f * layer0;
        dcimg[IDlayermin].array.F[ij1] = 1.0 * layer0; // don't go below this layer
    }

    {
        printf("---------------- Convolve layer image ------------\n");
        fflush(stdout);
        int    NBfiter = 500;
        float *pixcol  = (float *) malloc(sizeof(float) * ysize1);
        float *pixline = (float *) malloc(sizeof(float) * xsize1);
        for (int fiter = 0; fiter < NBfiter; fiter++)
        {
            printf(".");
            fflush(stdout);
            for (uint32_t jj1 = 0; jj1 < ysize1; jj1++)
            {
                for (uint32_t ii1 = 1; ii1 < xsize1 - 1; ii1++)
                {
                    pixline[ii1] = 0.3 * dcimg[IDlayer].array.F[jj1 * xsize1 + ii1 - 1] +
                                   0.4f * dcimg[IDlayer].array.F[jj1 * xsize1 + ii1] +
                                   0.3 * dcimg[IDlayer].array.F[jj1 * xsize1 + ii1 + 1];
                }
                for (uint32_t ii1 = 1; ii1 < xsize1 - 1; ii1++)
                {
                    dcimg[IDlayer].array.F[jj1 * xsize1 + ii1] = pixline[ii1];
                }
            }

            for (uint32_t ii1 = 0; ii1 < xsize1; ii1++)
            {
                for (uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                {
                    pixcol[jj1] = 0.3 * dcimg[IDlayer].array.F[(jj1 - 1) * xsize1 + ii1] +
                                  0.4f * dcimg[IDlayer].array.F[jj1 * xsize1 + ii1] +
                                  0.3 * dcimg[IDlayer].array.F[(jj1 + 1) * xsize1 + ii1];
                }
                for (uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                {
                    dcimg[IDlayer].array.F[jj1 * xsize1 + ii1] = pixcol[jj1];
                }
            }

            for (uint32_t ii1 = 0; ii1 < xsize1; ii1++)
            {
                for (uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                {
                    if (dcimg[IDlayer].array.F[jj1 * xsize1 + ii1] <
                        dcimg[IDlayermin].array.F[jj1 * xsize1 + ii1])
                    {
                        dcimg[IDlayer].array.F[jj1 * xsize1 + ii1] =
                            dcimg[IDlayermin].array.F[jj1 * xsize1 + ii1];
                    }
                }
            }
        }
        free(pixcol);
        free(pixline);
        printf("\n");
        printf("---------------- DONE ------------\n");
        fflush(stdout);
    }

    gauss_filter("imlayer", "imlayerg", 50.0, 150);
    imageID IDlayerg = image_ID("imlayerg", dcimg, dcnimg);

    // construct HDR image
    imageID IDout;
    create_2Dimage_ID(outimname, xsize, ysize, &IDout);

    for (uint32_t jj = 0; jj < ysize; jj++)
    {
        float    y   = 1.0 * jj / ysize;
        uint32_t jj1 = (uint32_t) (y * ysize1);
        if (jj1 == ysize1 - 1)
        {
            jj1 = ysize1 - 2;
        }
        float jj1frac = y * ysize1 - jj1;

        for (uint32_t ii = 0; ii < xsize; ii++)
        {
            float    x   = 1.0 * ii / xsize;
            uint32_t ii1 = (uint32_t) (x * xsize1);
            if (ii1 == xsize1 - 1)
            {
                ii1 = xsize1 - 2;
            }
            float ii1frac = x * xsize1 - ii1;

            // get layer
            float layer00 = dcimg[IDlayer].array.F[jj1 * xsize1 + ii1];
            float layer10 = dcimg[IDlayer].array.F[jj1 * xsize1 + ii1 + 1];
            float layer01 = dcimg[IDlayer].array.F[(jj1 + 1) * xsize1 + ii1];
            float layer11 = dcimg[IDlayer].array.F[(jj1 + 1) * xsize1 + ii1 + 1];

            float layer = layer00 * (1.0 - ii1frac) * (1.0 - jj1frac) +
                          layer01 * (1.0 - ii1frac) * jj1frac +
                          layer10 * ii1frac * (1.0 - jj1frac) + layer11 * ii1frac * jj1frac;

            uint32_t layer0 = (uint32_t) layer;
            uint32_t layer1 = layer0 + 1;
            if (layer1 == zsize)
            {
                layer1 = layer0;
            }
            float layercoeff = layer - 1.0 * layer0;

            float pval0 = dcimg[IDimHDRc].array.F[layer0 * xsize * ysize + jj * xsize + ii] /
                          etimearray[layer0];
            float pval1 = dcimg[IDimHDRc].array.F[layer1 * xsize * ysize + jj * xsize + ii] /
                          etimearray[layer1];

            double alpha0 = 10.0;
            //double alpha1 = 2.5;
            /*
            double alpha1 = 6.0;
            double alpha3 = 3.0;
            double alpha4 = 3.0;
            double layermax = 2.0;
            if(layer>layermax)
            {
                layer = layermax;
            }
            double x1 = 1.0 / pow( 1.0 + 1.0/pow(layer/alpha0,alpha1), 1.0/alpha1);
            double layercoeff1 = 1.0 / ( 1.0 + alpha3*pow(6.0, alpha4*x1) );
            */
            double layerg = dcimg[IDlayerg].array.F[jj1 * xsize1 + ii1];
            if (layerg > 3.0)
            {
                layerg = 3.0;
            }
            double layercoeff1 = 1.0 / pow(alpha0, layerg);

            //dcimg[IDout].array.F[jj*xsize+ii] = (pval0 * (1.0f-layercoeff) + pval1 * layercoeff);
            dcimg[IDout].array.F[jj * xsize + ii] =
                layercoeff1 * (pval0 * (1.0 - layercoeff) + pval1 * layercoeff);
        }
    }

    return RETURN_SUCCESS;
}

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    printf("satlevel  = %f\n", *satlevel);
    printf("biaslevel = %f\n", *biaslevel);

    combine_HDR_image(flistname, *satlevel, *biaslevel, outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

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

errno_t CLIADDCMD_image_format__combineHDR()
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
