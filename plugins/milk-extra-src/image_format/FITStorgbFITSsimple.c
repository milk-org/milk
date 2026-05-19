/**
 * @file FITStorgbFITSsimple.c
 * @brief Fitstorgbfitssimple module
 */

/** @file FITStorgbFITSsimple.h
 */

#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

static float FLUXFACTOR = 1.0;

// convers a single raw bayer FITS frame into RGB FITS
// uses "bias", "badpix" and "flat" if they exist
// output is imr, img, imb
// this is a simple interpolation routine
// IMPORTANT: input will be modified
// Sampling factor : 0=full resolution (slow), 1=half resolution (fast), 2=quarter resolution (very fast)
// Fast mode does not reject bad pixels
errno_t convert_rawbayerFITStorgbFITS_simple(const char *__restrict ID_name,
        const char *__restrict ID_name_r,
        const char *__restrict ID_name_g,
        const char *__restrict ID_name_b,
        int SamplFactor)
{
    imageID ID;
    long    Xsize, Ysize;
    imageID IDr, IDg, IDb, IDrc, IDgc, IDbc, IDbp;
    imageID IDbadpix;
    imageID IDflat;
    imageID IDdark;
    imageID IDbias;
    long   ii, jj, ii1, jj1, ii2, jj2, iistart, iiend, jjstart, jjend, dii, djj;
    double v1, v2, v, vc, tmp1;
    long   cnt;
    double coeff;
    imageID ID00, ID01, ID10, ID11;
    imageID ID00c, ID01c, ID10c, ID11c;
    double  eps     = 1.0e-8;
    int     RGBmode = 0;

    int FastMode = 0;

    if(variable_ID("_RGBfast") != -1)
    {
        FastMode = 1;
    }

    ID    = image_ID(ID_name, dcimg, dcnimg);
    Xsize = dcimg[ID].md[0].size[0];
    Ysize = dcimg[ID].md[0].size[1];

    printf("X Y  = %ld %ld\n", Xsize, Ysize);

    if((Xsize == 4290) && (Ysize == 2856))
    {
        RGBmode = 1;
    }
    if((Xsize == 4770) && (Ysize == 3178))
    {
        RGBmode = 1;
    }
    if((Xsize == 5202) && (Ysize == 3465))
    {
        RGBmode = 2;
    }
    if((Xsize == 5208) && (Ysize == 3476))
    {
        RGBmode = 1;
    }

    if(RGBmode == 0)
    {
        printf("image size : %ld %ld\n", Xsize, Ysize);
        RGBmode = 1;
        //PRINT_ERROR("Unknown RGB image mode\n");
        //exit(0);
    }

    printf("FAST MODE = %d\n", FastMode);
    printf("RGBmode   = %d\n", RGBmode);
    //exit(0);

    if(FastMode == 0)
    {
        // bias
        IDbias = image_ID("bias", dcimg, dcnimg);
        if(IDbias == -1)
        {
            create_2Dimage_ID("bias", Xsize, Ysize, &IDbias);
            for(ii = 0; ii < Xsize * Ysize; ii++)
            {
                dcimg[IDbias].array.F[ii] = 0.0f;
            }
        }

        // dark
        IDdark = image_ID("dark", dcimg, dcnimg);
        if(IDdark == -1)
        {
            create_2Dimage_ID("dark", Xsize, Ysize, &IDdark);
            for(ii = 0; ii < Xsize * Ysize; ii++)
            {
                dcimg[IDdark].array.F[ii] = 0.0f;
            }
        }

        // bad pixel map
        IDbadpix = image_ID("badpix", dcimg, dcnimg);
        if(IDbadpix == -1)
        {
            create_2Dimage_ID("badpix", Xsize, Ysize, &IDbadpix);
            for(ii = 0; ii < Xsize * Ysize; ii++)
            {
                dcimg[IDbadpix].array.F[ii] = 0.0f;
            }
        }

        copy_image_ID("badpix", "badpix1", 0);
        IDbp = image_ID("badpix1", dcimg, dcnimg);

        // flat field
        IDflat = image_ID("flat", dcimg, dcnimg);
        if(IDflat == -1)
        {
            create_2Dimage_ID("flat", Xsize, Ysize, &IDflat);
            for(ii = 0; ii < Xsize * Ysize; ii++)
            {
                dcimg[IDflat].array.F[ii] = 1.0f;
            }
            //      arith_image_cstadd_inplace("flat",1.0);
        }

        // remove bias
        if(IDbias != -1)
        {
            for(ii = 0; ii < Xsize; ii++)
                for(jj = 0; jj < Ysize; jj++)
                {
                    dcimg[ID].array.F[jj * Xsize + ii] -=
                        dcimg[IDbias].array.F[jj * Xsize + ii];
                }
        }
        // remove dark
        if(IDdark != -1)
        {
            for(ii = 0; ii < Xsize; ii++)
                for(jj = 0; jj < Ysize; jj++)
                {
                    dcimg[ID].array.F[jj * Xsize + ii] -=
                        dcimg[IDdark].array.F[jj * Xsize + ii];
                }
        }

        // remove obvious isolated hot pixels
        cnt = 0;
        for(ii = 0; ii < Xsize; ii++)
            for(jj = 0; jj < Ysize; jj++)
            {
                v1      = dcimg[ID].array.F[jj * Xsize + ii];
                iistart = ii - 2;
                iiend   = ii + 2;
                if(iistart < 0)
                {
                    iistart = 0;
                }
                if(iiend > Xsize - 1)
                {
                    iiend = Xsize - 1;
                }
                jjstart = jj - 2;
                jjend   = jj + 2;
                if(jjstart < 0)
                {
                    jjstart = 0;
                }
                if(jjend > Ysize - 1)
                {
                    jjend = Ysize - 1;
                }
                v2 = 0.0;
                for(ii1 = iistart; ii1 < iiend; ii1++)
                    for(jj1 = jjstart; jj1 < jjend; jj1++)
                        if((ii1 != ii) || (jj1 != jj))
                        {
                            tmp1 = dcimg[ID].array.F[jj1 * Xsize + ii1];
                            if(tmp1 > v2)
                            {
                                v2 = tmp1;
                            }
                        }
                if(v1 > 4.0 * v2 + 500.0)
                {
                    dcimg[ID].array.F[jj * Xsize + ii] = v2;
                    //		dcimg[IDbp].array.F[jj*Xsize+ii] = 1.0f;
                    cnt++;
                }
            }
        printf("%ld hot pixels removed\n", cnt);

        for(ii = 0; ii < Xsize; ii++)
            for(jj = 0; jj < Ysize; jj++)
            {
                dcimg[ID].array.F[jj * Xsize + ii] *= FLUXFACTOR;
            }
    }

    switch(SamplFactor)
    {

        case 0:

            if(image_ID(ID_name_r, dcimg, dcnimg) != -1)
            {
                delete_image_ID(ID_name_r, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_r, Xsize, Ysize, &IDr);
            create_2Dimage_ID("imrc", Xsize, Ysize, &IDrc);

            if(image_ID(ID_name_g, dcimg, dcnimg) != -1)
            {
                delete_image_ID(ID_name_g, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_g, Xsize, Ysize, &IDg);
            create_2Dimage_ID("imgc", Xsize, Ysize, &IDgc);

            if(image_ID(ID_name_b, dcimg, dcnimg) != -1)
            {
                delete_image_ID(ID_name_b, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_b, Xsize, Ysize, &IDb);
            create_2Dimage_ID("imbc", Xsize, Ysize, &IDbc);

            if(RGBmode == 1)  // GBRG
            {
                ID00  = IDg;
                ID00c = IDgc;

                ID10  = IDb;
                ID10c = IDbc;

                ID01  = IDr;
                ID01c = IDrc;

                ID11  = IDg;
                ID11c = IDgc;
            }

            if(RGBmode == 2)
            {
                ID00  = IDr;
                ID00c = IDrc;

                ID10  = IDg;
                ID10c = IDgc;

                ID01  = IDg;
                ID01c = IDgc;

                ID11  = IDb;
                ID11c = IDbc;
            }

            if(FastMode == 0)
            {
                for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                    for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                    {
                        ii = ii1 * 2;
                        jj = jj1 * 2;

                        ii2 = ii;
                        jj2 = jj + 1;
                        dcimg[ID01].array.F[jj2 * Xsize + ii2] =
                            dcimg[ID].array.F[jj2 * Xsize + ii2] /
                            dcimg[IDflat].array.F[jj2 * Xsize + ii2];
                        dcimg[ID01c].array.F[jj2 * Xsize + ii2] =
                            1.0f - dcimg[IDbp].array.F[jj2 * Xsize + ii2];

                        ii2 = ii + 1;
                        jj2 = jj + 1;
                        dcimg[ID11].array.F[jj2 * Xsize + ii2] =
                            dcimg[ID].array.F[jj2 * Xsize + ii2] /
                            dcimg[IDflat].array.F[jj2 * Xsize + ii2];
                        dcimg[ID11c].array.F[jj2 * Xsize + ii2] =
                            1.0f - dcimg[IDbp].array.F[jj2 * Xsize + ii2];

                        ii2 = ii;
                        jj2 = jj;
                        dcimg[ID00].array.F[jj2 * Xsize + ii2] =
                            dcimg[ID].array.F[jj2 * Xsize + ii2] /
                            dcimg[IDflat].array.F[jj2 * Xsize + ii2];
                        dcimg[ID00c].array.F[jj2 * Xsize + ii2] =
                            1.0f - dcimg[IDbp].array.F[jj2 * Xsize + ii2];

                        ii2 = ii + 1;
                        jj2 = jj;
                        dcimg[ID10].array.F[jj2 * Xsize + ii2] =
                            dcimg[ID].array.F[jj2 * Xsize + ii2] /
                            dcimg[IDflat].array.F[jj2 * Xsize + ii2];
                        dcimg[ID10c].array.F[jj2 * Xsize + ii2] =
                            1.0f - dcimg[IDbp].array.F[jj2 * Xsize + ii2];
                    }

                for(ii = 0; ii < Xsize; ii++)
                    for(jj = 0; jj < Ysize; jj++)
                    {
                        if(dcimg[IDrc].array.F[jj * Xsize + ii] < 0.5f)
                        {
                            v  = 0.0;
                            vc = 0.0;
                            for(dii = -2; dii < 3; dii++)
                                for(djj = -2; djj < 3; djj++)
                                {
                                    ii1 = ii + dii;
                                    jj1 = jj + djj;
                                    if((ii1 > -1) && (jj1 > -1) && (ii1 < Xsize) &&
                                            (jj1 < Ysize))
                                        if((dii != 0) || (djj != 0))
                                        {
                                            if(dcimg[IDrc]
                                                    .array.F[jj1 * Xsize + ii1] >
                                                    0.5)
                                            {
                                                coeff = exp(
                                                            -5.0 * (dii * dii + djj * djj));
                                                vc += coeff;
                                                v += dcimg[IDr]
                                                     .array
                                                     .F[jj1 * Xsize + ii1] *
                                                     coeff;
                                            }
                                        }
                                }
                            dcimg[IDr].array.F[jj * Xsize + ii] = v / vc;
                        }

                        if(dcimg[IDgc].array.F[jj * Xsize + ii] < 0.5f)
                        {
                            v  = 0.0;
                            vc = 0.0;
                            for(dii = -2; dii < 3; dii++)
                                for(djj = -2; djj < 3; djj++)
                                {
                                    ii1 = ii + dii;
                                    jj1 = jj + djj;
                                    if((ii1 > -1) && (jj1 > -1) && (ii1 < Xsize) &&
                                            (jj1 < Ysize))
                                        if((dii != 0) || (djj != 0))
                                        {
                                            if(dcimg[IDgc]
                                                    .array.F[jj1 * Xsize + ii1] >
                                                    0.5)
                                            {
                                                coeff = exp(
                                                            -5.0 * (dii * dii + djj * djj));
                                                vc += coeff;
                                                v += dcimg[IDg]
                                                     .array
                                                     .F[jj1 * Xsize + ii1] *
                                                     coeff;
                                            }
                                        }
                                }
                            dcimg[IDg].array.F[jj * Xsize + ii] = v / vc;
                        }

                        if(dcimg[IDbc].array.F[jj * Xsize + ii] < 0.5f)
                        {
                            v  = 0.0;
                            vc = 0.0;
                            for(dii = -2; dii < 3; dii++)
                                for(djj = -2; djj < 3; djj++)
                                {
                                    ii1 = ii + dii;
                                    jj1 = jj + djj;
                                    if((ii1 > -1) && (jj1 > -1) && (ii1 < Xsize) &&
                                            (jj1 < Ysize))
                                        if((dii != 0) || (djj != 0))
                                        {
                                            if(dcimg[IDbc]
                                                    .array.F[jj1 * Xsize + ii1] >
                                                    0.5)
                                            {
                                                coeff = exp(
                                                            -5.0 * (dii * dii + djj * djj));
                                                vc += coeff;
                                                v += dcimg[IDb]
                                                     .array
                                                     .F[jj1 * Xsize + ii1] *
                                                     coeff;
                                            }
                                        }
                                }
                            dcimg[IDb].array.F[jj * Xsize + ii] = v / vc;
                        }
                    }
            }
            else
            {
                if(RGBmode == 1)  // GBRG
                {
                    // G
                    for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj;
                            dcimg[IDg].array.F[jj2 * Xsize + ii2] =
                                dcimg[ID].array.F[jj2 * Xsize + ii2];
                            ii2 = ii + 1;
                            jj2 = jj + 1;
                            dcimg[IDg].array.F[jj2 * Xsize + ii2] =
                                dcimg[ID].array.F[jj2 * Xsize + ii2];
                        }
                    // replace blue pixels
                    for(ii1 = 0; ii1 < Xsize / 2 - 1; ii1++)
                        for(jj1 = 1; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii + 1;
                            jj2 = jj;
                            dcimg[IDg].array.F[jj2 * Xsize + ii2] =
                                0.25 *
                                (dcimg[ID].array.F[jj2 * Xsize + (ii2 - 1)] +
                                 dcimg[ID].array.F[jj2 * Xsize + (ii2 + 1)] +
                                 dcimg[ID].array.F[(jj2 + 1) * Xsize + ii2] +
                                 dcimg[ID].array.F[(jj2 - 1) * Xsize + ii2]);
                        }
                    // replace red pixels
                    for(ii1 = 1; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2 - 1; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj + 1;
                            dcimg[IDg].array.F[jj2 * Xsize + ii2] =
                                0.25 *
                                (dcimg[ID].array.F[jj2 * Xsize + (ii2 - 1)] +
                                 dcimg[ID].array.F[jj2 * Xsize + (ii2 + 1)] +
                                 dcimg[ID].array.F[(jj2 + 1) * Xsize + ii2] +
                                 dcimg[ID].array.F[(jj2 - 1) * Xsize + ii2]);
                        }

                    // R
                    for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                        {
                            ii  = ii1 * 2;
                            jj  = jj1 * 2;
                            ii2 = ii;
                            jj2 = jj + 1;
                            dcimg[IDr].array.F[jj2 * Xsize + ii2] =
                                dcimg[ID].array.F[jj2 * Xsize + ii2];
                        }
                    // replace g1 pixels
                    for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 1; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj;
                            dcimg[IDr].array.F[jj2 * Xsize + ii2] =
                                0.5 *
                                (dcimg[ID].array.F[(jj2 - 1) * Xsize + ii2] +
                                 dcimg[ID].array.F[(jj2 + 1) * Xsize + ii2]);
                        }
                    // replace g2 pixels
                    for(ii1 = 0; ii1 < Xsize / 2 - 1; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii + 1;
                            jj2 = jj + 1;
                            dcimg[IDr].array.F[jj2 * Xsize + ii2] =
                                0.5 *
                                (dcimg[ID].array.F[jj2 * Xsize + (ii2 - 1)] +
                                 dcimg[ID].array.F[jj2 * Xsize + (ii2 + 1)]);
                        }
                    // replace b pixels
                    for(ii1 = 0; ii1 < Xsize / 2 - 1; ii1++)
                        for(jj1 = 1; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii + 1;
                            jj2 = jj;
                            dcimg[IDr].array.F[jj2 * Xsize + ii2] =
                                0.25 *
                                (dcimg[ID]
                                 .array.F[(jj2 - 1) * Xsize + (ii2 - 1)] +
                                 dcimg[ID]
                                 .array.F[(jj2 - 1) * Xsize + (ii2 + 1)] +
                                 dcimg[ID]
                                 .array.F[(jj2 + 1) * Xsize + (ii2 - 1)] +
                                 dcimg[ID]
                                 .array.F[(jj2 + 1) * Xsize + (ii2 + 1)]);
                        }

                    // B
                    for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                        {
                            ii  = ii1 * 2;
                            jj  = jj1 * 2;
                            ii2 = ii + 1;
                            jj2 = jj;
                            dcimg[IDb].array.F[jj2 * Xsize + ii2] =
                                dcimg[ID].array.F[jj2 * Xsize + ii2];
                        }

                    // replace g2 pixels
                    for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2 - 1; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii + 1;
                            jj2 = jj + 1;
                            dcimg[IDb].array.F[jj2 * Xsize + ii2] =
                                0.5 *
                                (dcimg[ID].array.F[(jj2 - 1) * Xsize + ii2] +
                                 dcimg[ID].array.F[(jj2 + 1) * Xsize + ii2]);
                        }
                    // replace g1 pixels
                    for(ii1 = 1; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj;
                            dcimg[IDb].array.F[jj2 * Xsize + ii2] =
                                0.5 *
                                (dcimg[ID].array.F[jj2 * Xsize + (ii2 - 1)] +
                                 dcimg[ID].array.F[jj2 * Xsize + (ii2 + 1)]);
                        }
                    // replace r pixels
                    for(ii1 = 1; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2 - 1; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj + 1;
                            dcimg[IDb].array.F[jj2 * Xsize + ii2] =
                                0.25 *
                                (dcimg[ID]
                                 .array.F[(jj2 - 1) * Xsize + (ii2 - 1)] +
                                 dcimg[ID]
                                 .array.F[(jj2 - 1) * Xsize + (ii2 + 1)] +
                                 dcimg[ID]
                                 .array.F[(jj2 + 1) * Xsize + (ii2 - 1)] +
                                 dcimg[ID]
                                 .array.F[(jj2 + 1) * Xsize + (ii2 + 1)]);
                        }
                }
            }

            //  delete_image_ID("badpix1");

            delete_image_ID("imrc", DELETE_IMAGE_ERRMODE_WARNING);
            delete_image_ID("imgc", DELETE_IMAGE_ERRMODE_WARNING);
            delete_image_ID("imbc", DELETE_IMAGE_ERRMODE_WARNING);
            //  delete_image_ID("imraw");
            break;

        case 1:
            if(image_ID(ID_name_r, dcimg, dcnimg) != -1)
            {
                delete_image_ID(ID_name_r, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_r, Xsize / 2, Ysize / 2, &IDr);
            create_2Dimage_ID("imrc", Xsize / 2, Ysize / 2, &IDrc);

            if(image_ID(ID_name_g, dcimg, dcnimg) != -1)
            {
                delete_image_ID(ID_name_g, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_g, Xsize / 2, Ysize / 2, &IDg);
            create_2Dimage_ID("imgc", Xsize / 2, Ysize / 2, &IDgc);

            if(image_ID(ID_name_b, dcimg, dcnimg) != -1)
            {
                delete_image_ID(ID_name_b, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_b, Xsize / 2, Ysize / 2, &IDb);
            create_2Dimage_ID("imbc", Xsize / 2, Ysize / 2, &IDbc);

            if(RGBmode == 1)  // GBRG
            {
                ID00  = IDg;
                ID00c = IDgc;

                ID10  = IDb;
                ID10c = IDbc;

                ID01  = IDr;
                ID01c = IDrc;

                ID11  = IDg;
                ID11c = IDgc;
            }

            if(RGBmode == 2)
            {
                ID00  = IDr;
                ID00c = IDrc;

                ID10  = IDg;
                ID10c = IDgc;

                ID01  = IDg;
                ID01c = IDgc;

                ID11  = IDb;
                ID11c = IDbc;
            }

            for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                {
                    ii = ii1 * 2;
                    jj = jj1 * 2;

                    ii2 = ii;
                    jj2 = jj + 1;
                    dcimg[ID01].array.F[jj1 * Xsize / 2 + ii1] +=
                        dcimg[ID].array.F[jj2 * Xsize + ii2] /
                        dcimg[IDflat].array.F[jj2 * Xsize + ii2];
                    dcimg[ID01c].array.F[jj1 * Xsize / 2 + ii1] +=
                        1.0f - dcimg[IDbp].array.F[jj2 * Xsize + ii2];

                    ii2 = ii + 1;
                    jj2 = jj + 1;
                    dcimg[ID11].array.F[jj1 * Xsize / 2 + ii1] +=
                        dcimg[ID].array.F[jj2 * Xsize + ii2] /
                        dcimg[IDflat].array.F[jj2 * Xsize + ii2];
                    dcimg[ID11c].array.F[jj1 * Xsize / 2 + ii1] +=
                        1.0f - dcimg[IDbp].array.F[jj2 * Xsize + ii2];

                    ii2 = ii;
                    jj2 = jj;
                    dcimg[ID00].array.F[jj1 * Xsize / 2 + ii1] +=
                        dcimg[ID].array.F[jj2 * Xsize + ii2] /
                        dcimg[IDflat].array.F[jj2 * Xsize + ii2];
                    dcimg[ID00c].array.F[jj1 * Xsize / 2 + ii1] +=
                        1.0f - dcimg[IDbp].array.F[jj2 * Xsize + ii2];

                    ii2 = ii + 1;
                    jj2 = jj;
                    dcimg[ID10].array.F[jj1 * Xsize / 2 + ii1] +=
                        dcimg[ID].array.F[jj2 * Xsize + ii2] /
                        dcimg[IDflat].array.F[jj2 * Xsize + ii2];
                    dcimg[ID10c].array.F[jj1 * Xsize / 2 + ii1] +=
                        1.0f - dcimg[IDbp].array.F[jj2 * Xsize + ii2];

                    dcimg[IDr].array.F[jj1 * Xsize / 2 + ii1] /=
                        dcimg[IDrc].array.F[jj1 * Xsize / 2 + ii1] + eps;
                    dcimg[IDg].array.F[jj1 * Xsize / 2 + ii1] /=
                        dcimg[IDgc].array.F[jj1 * Xsize / 2 + ii1] + eps;
                    dcimg[IDb].array.F[jj1 * Xsize / 2 + ii1] /=
                        dcimg[IDbc].array.F[jj1 * Xsize / 2 + ii1] + eps;
                }

            delete_image_ID("imrc", DELETE_IMAGE_ERRMODE_WARNING);
            delete_image_ID("imgc", DELETE_IMAGE_ERRMODE_WARNING);
            delete_image_ID("imbc", DELETE_IMAGE_ERRMODE_WARNING);

            break;
    }

    return RETURN_SUCCESS;
}
