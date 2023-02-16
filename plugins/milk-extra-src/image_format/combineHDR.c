/** @file combineHDR.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"

#include "image_filter/image_filter.h"

// Local variables pointers
static char   *flistname;
static double *satlevel;
static double *biaslevel;
static char   *outimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_STR,
        ".flistname",
        "file list name",
        "HDRfilelist.txt",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &flistname,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".satlevel",
        "Saturation level",
        "satval",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &satlevel,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".biaslevel",
        "Bias level",
        "biasval",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &biaslevel,
        NULL
    },
    {
        CLIARG_STR_NOT_IMG,
        ".outimname",
        "output image",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "combineHDR", "combine HDR image", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("combine HDR image\n");

    return RETURN_SUCCESS;
}

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
        while(fscanf(fpin, "%s %f %s\n", FITSfname, &etime, timestring) == 3)
        {
            imageID ID;
            printf("Input file [%11.6f] : %s\n", etime, FITSfname);
            etimearray[HDRindex] = etime;
            sprintf(imHDRin, "imHRDin_%03d", HDRindex);
            load_fits(FITSfname, imHDRin, 2, &ID);
            IDarray[HDRindex] = ID;
            HDRindex++;
        }
        fclose(fpin);
        NB_HDRindex = HDRindex;
    }

    printf("PARAMS : %20f %20f\n", biasvalue, 1.0 * satvalue);

    uint32_t xsize = data.image[IDarray[0]].md->size[0];
    uint32_t ysize = data.image[IDarray[0]].md->size[1];
    uint32_t zsize = NB_HDRindex;

    int      binstep = 5;
    uint32_t xsize1  = (uint32_t)(xsize / binstep);
    uint32_t ysize1  = (uint32_t)(ysize / binstep);
    //
    // Assemble cube and subsampled cube
    //
    imageID IDimHDRc;
    create_3Dimage_ID("imHDRc", xsize, ysize, zsize, &IDimHDRc);

    imageID IDimHDRc1;
    create_3Dimage_ID("imHDRc1", xsize1, ysize1, zsize, &IDimHDRc1);

    imageID IDimHDRc1w;
    create_3Dimage_ID("imHDRc1w", xsize1, ysize1, zsize, &IDimHDRc1w);

    for(uint32_t kk = 0; kk < zsize; kk++)
    {
        for(uint32_t jj = 0; jj < ysize; jj++)
        {
            float    y   = 1.0 * jj / ysize;
            uint32_t jj1 = (uint32_t)(y * ysize1);

            for(uint32_t ii = 0; ii < xsize; ii++)
            {
                float    x   = 1.0 * ii / xsize;
                uint32_t ii1 = (uint32_t)(x * xsize1);

                float pval =
                    1.0 * data.image[IDarray[kk]].array.F[jj * xsize + ii] -
                    biasvalue;

                data.image[IDimHDRc]
                .array.F[kk * xsize * ysize + jj * xsize + ii] = pval;

                data.image[IDimHDRc1]
                .array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] += pval;
                data.image[IDimHDRc1w]
                .array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] += 1.0;
            }
        }
    }

    for(uint32_t kk = 0; kk < zsize; kk++)
    {
        for(uint32_t jj1 = 0; jj1 < ysize1; jj1++)
        {
            for(uint32_t ii1 = 0; ii1 < xsize1; ii1++)
            {
                data.image[IDimHDRc1]
                .array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1] /=
                    data.image[IDimHDRc1w]
                    .array.F[kk * xsize1 * ysize1 + jj1 * xsize1 + ii1];
            }
        }
    }

    {
        printf("---------------- Convolve binned image ------------\n");
        fflush(stdout);
        int    NBfiter = 5;
        float *pixcol  = (float *) malloc(sizeof(float) * ysize1);
        float *pixline = (float *) malloc(sizeof(float) * xsize1);
        for(int fiter = 0; fiter < NBfiter; fiter++)
        {
            printf(".");
            fflush(stdout);

            for(uint32_t kk = 0; kk < zsize; kk++)
            {
                for(uint32_t jj1 = 0; jj1 < ysize1; jj1++)
                {
                    for(uint32_t ii1 = 1; ii1 < xsize1 - 1; ii1++)
                    {
                        pixline[ii1] =
                            0.3 * data.image[IDimHDRc1]
                            .array.F[kk * xsize1 * ysize1 +
                                        jj1 * xsize1 + ii1 - 1] +
                            0.4 * data.image[IDimHDRc1]
                            .array.F[kk * xsize1 * ysize1 +
                                        jj1 * xsize1 + ii1] +
                            0.3 * data.image[IDimHDRc1]
                            .array.F[kk * xsize1 * ysize1 +
                                        jj1 * xsize1 + ii1 + 1];
                    }
                    for(uint32_t ii1 = 1; ii1 < xsize1 - 1; ii1++)
                    {
                        data.image[IDimHDRc1].array.F[kk * xsize1 * ysize1 +
                                                      jj1 * xsize1 + ii1] =
                                                          pixline[ii1];
                    }
                }

                for(uint32_t ii1 = 0; ii1 < xsize1; ii1++)
                {
                    for(uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                    {
                        pixcol[jj1] =
                            0.3 * data.image[IDimHDRc1]
                            .array.F[kk * xsize1 * ysize1 +
                                        (jj1 - 1) * xsize1 + ii1] +
                            0.4 * data.image[IDimHDRc1]
                            .array.F[kk * xsize1 * ysize1 +
                                        jj1 * xsize1 + ii1] +
                            0.3 * data.image[IDimHDRc1]
                            .array.F[kk * xsize1 * ysize1 +
                                        (jj1 + 1) * xsize1 + ii1];
                    }
                    for(uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                    {
                        data.image[IDimHDRc1].array.F[kk * xsize1 * ysize1 +
                                                      jj1 * xsize1 + ii1] =
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

    for(uint32_t ij1 = 0; ij1 < xsize1 * ysize1; ij1++)
    {
        //float layer = 0.0;
        uint32_t layer0 = 0;
        uint32_t layer1 = 0;
        uint32_t kk     = 0;
        while((kk < zsize) &&
                (data.image[IDimHDRc1].array.F[kk * xsize1 * ysize1 + ij1] >
                 satvalue))
        {
            layer0 = kk;
            kk++;
        }

        layer1 = layer0 + 1;
        if(layer1 == zsize)
        {
            layer1 = zsize - 1;
        }

        float valmax =
            data.image[IDimHDRc1].array.F[layer0 * xsize1 * ysize1 + ij1];
        if((valmax > satvalue) && (layer1 < zsize - 1))
        {
            // increment layers
            layer0++;
            layer1++;
            valmax =
                data.image[IDimHDRc1].array.F[layer0 * xsize1 * ysize1 + ij1];
        }
        //float valmin = data.image[IDimHDRc1].array.F[layer1*xsize1*ysize1+ij1];

        //float x = valmax/satvalue;
        //float c1 = pow( 0.5*(1.0+cos(x*M_PI)), alpha1);
        //float c2 = 1.0-c1;

        data.image[IDlayer].array.F[ij1] = 1.0 * layer0;
        data.image[IDlayermin].array.F[ij1] =
            1.0 * layer0; // don't go below this layer
    }

    {
        printf("---------------- Convolve layer image ------------\n");
        fflush(stdout);
        int    NBfiter = 500;
        float *pixcol  = (float *) malloc(sizeof(float) * ysize1);
        float *pixline = (float *) malloc(sizeof(float) * xsize1);
        for(int fiter = 0; fiter < NBfiter; fiter++)
        {
            printf(".");
            fflush(stdout);
            for(uint32_t jj1 = 0; jj1 < ysize1; jj1++)
            {
                for(uint32_t ii1 = 1; ii1 < xsize1 - 1; ii1++)
                {
                    pixline[ii1] =
                        0.3 * data.image[IDlayer]
                        .array.F[jj1 * xsize1 + ii1 - 1] +
                        0.4 * data.image[IDlayer].array.F[jj1 * xsize1 + ii1] +
                        0.3 *
                        data.image[IDlayer].array.F[jj1 * xsize1 + ii1 + 1];
                }
                for(uint32_t ii1 = 1; ii1 < xsize1 - 1; ii1++)
                {
                    data.image[IDlayer].array.F[jj1 * xsize1 + ii1] =
                        pixline[ii1];
                }
            }

            for(uint32_t ii1 = 0; ii1 < xsize1; ii1++)
            {
                for(uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                {
                    pixcol[jj1] =
                        0.3 * data.image[IDlayer]
                        .array.F[(jj1 - 1) * xsize1 + ii1] +
                        0.4 * data.image[IDlayer].array.F[jj1 * xsize1 + ii1] +
                        0.3 * data.image[IDlayer]
                        .array.F[(jj1 + 1) * xsize1 + ii1];
                }
                for(uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                {
                    data.image[IDlayer].array.F[jj1 * xsize1 + ii1] =
                        pixcol[jj1];
                }
            }

            for(uint32_t ii1 = 0; ii1 < xsize1; ii1++)
            {
                for(uint32_t jj1 = 1; jj1 < ysize1 - 1; jj1++)
                {
                    if(data.image[IDlayer].array.F[jj1 * xsize1 + ii1] <
                            data.image[IDlayermin].array.F[jj1 * xsize1 + ii1])
                    {
                        data.image[IDlayer].array.F[jj1 * xsize1 + ii1] =
                            data.image[IDlayermin].array.F[jj1 * xsize1 + ii1];
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
    imageID IDlayerg = image_ID("imlayerg");

    // construct HDR image
    imageID IDout;
    create_2Dimage_ID(outimname, xsize, ysize, &IDout);

    for(uint32_t jj = 0; jj < ysize; jj++)
    {
        float    y   = 1.0 * jj / ysize;
        uint32_t jj1 = (uint32_t)(y * ysize1);
        if(jj1 == ysize1 - 1)
        {
            jj1 = ysize1 - 2;
        }
        float jj1frac = y * ysize1 - jj1;

        for(uint32_t ii = 0; ii < xsize; ii++)
        {
            float    x   = 1.0 * ii / xsize;
            uint32_t ii1 = (uint32_t)(x * xsize1);
            if(ii1 == xsize1 - 1)
            {
                ii1 = xsize1 - 2;
            }
            float ii1frac = x * xsize1 - ii1;

            // get layer
            float layer00 = data.image[IDlayer].array.F[jj1 * xsize1 + ii1];
            float layer10 = data.image[IDlayer].array.F[jj1 * xsize1 + ii1 + 1];
            float layer01 =
                data.image[IDlayer].array.F[(jj1 + 1) * xsize1 + ii1];
            float layer11 =
                data.image[IDlayer].array.F[(jj1 + 1) * xsize1 + ii1 + 1];

            float layer = layer00 * (1.0 - ii1frac) * (1.0 - jj1frac) +
                          layer01 * (1.0 - ii1frac) * jj1frac +
                          layer10 * ii1frac * (1.0 - jj1frac) +
                          layer11 * ii1frac * jj1frac;

            uint32_t layer0 = (uint32_t) layer;
            uint32_t layer1 = layer0 + 1;
            if(layer1 == zsize)
            {
                layer1 = layer0;
            }
            float layercoeff = layer - 1.0 * layer0;

            float pval0 =
                data.image[IDimHDRc]
                .array.F[layer0 * xsize * ysize + jj * xsize + ii] /
                etimearray[layer0];
            float pval1 =
                data.image[IDimHDRc]
                .array.F[layer1 * xsize * ysize + jj * xsize + ii] /
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
            double layerg = data.image[IDlayerg].array.F[jj1 * xsize1 + ii1];
            if(layerg > 3.0)
            {
                layerg = 3.0;
            }
            double layercoeff1 = 1.0 / pow(alpha0, layerg);

            //data.image[IDout].array.F[jj*xsize+ii] = (pval0 * (1.0-layercoeff) + pval1 * layercoeff);
            data.image[IDout].array.F[jj * xsize + ii] =
                layercoeff1 * (pval0 * (1.0 - layercoeff) + pval1 * layercoeff);
        }
    }

    return RETURN_SUCCESS;
}

static errno_t compute_function()
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

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_image_format__combineHDR()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
