/** @file FITStorgbFITSsimple.h
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

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

    ID    = image_ID(ID_name);
    Xsize = data.image[ID].md[0].size[0];
    Ysize = data.image[ID].md[0].size[1];

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
        IDbias = image_ID("bias");
        if(IDbias == -1)
        {
            create_2Dimage_ID("bias", Xsize, Ysize, &IDbias);
            for(ii = 0; ii < Xsize * Ysize; ii++)
            {
                data.image[IDbias].array.F[ii] = 0.0;
            }
        }

        // dark
        IDdark = image_ID("dark");
        if(IDdark == -1)
        {
            create_2Dimage_ID("dark", Xsize, Ysize, &IDdark);
            for(ii = 0; ii < Xsize * Ysize; ii++)
            {
                data.image[IDdark].array.F[ii] = 0.0;
            }
        }

        // bad pixel map
        IDbadpix = image_ID("badpix");
        if(IDbadpix == -1)
        {
            create_2Dimage_ID("badpix", Xsize, Ysize, &IDbadpix);
            for(ii = 0; ii < Xsize * Ysize; ii++)
            {
                data.image[IDbadpix].array.F[ii] = 0.0;
            }
        }

        copy_image_ID("badpix", "badpix1", 0);
        IDbp = image_ID("badpix1");

        // flat field
        IDflat = image_ID("flat");
        if(IDflat == -1)
        {
            create_2Dimage_ID("flat", Xsize, Ysize, &IDflat);
            for(ii = 0; ii < Xsize * Ysize; ii++)
            {
                data.image[IDflat].array.F[ii] = 1.0;
            }
            //      arith_image_cstadd_inplace("flat",1.0);
        }

        // remove bias
        if(IDbias != -1)
        {
            for(ii = 0; ii < Xsize; ii++)
                for(jj = 0; jj < Ysize; jj++)
                {
                    data.image[ID].array.F[jj * Xsize + ii] -=
                        data.image[IDbias].array.F[jj * Xsize + ii];
                }
        }
        // remove dark
        if(IDdark != -1)
        {
            for(ii = 0; ii < Xsize; ii++)
                for(jj = 0; jj < Ysize; jj++)
                {
                    data.image[ID].array.F[jj * Xsize + ii] -=
                        data.image[IDdark].array.F[jj * Xsize + ii];
                }
        }

        // remove obvious isolated hot pixels
        cnt = 0;
        for(ii = 0; ii < Xsize; ii++)
            for(jj = 0; jj < Ysize; jj++)
            {
                v1      = data.image[ID].array.F[jj * Xsize + ii];
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
                            tmp1 = data.image[ID].array.F[jj1 * Xsize + ii1];
                            if(tmp1 > v2)
                            {
                                v2 = tmp1;
                            }
                        }
                if(v1 > 4.0 * v2 + 500.0)
                {
                    data.image[ID].array.F[jj * Xsize + ii] = v2;
                    //		data.image[IDbp].array.F[jj*Xsize+ii] = 1.0;
                    cnt++;
                }
            }
        printf("%ld hot pixels removed\n", cnt);

        for(ii = 0; ii < Xsize; ii++)
            for(jj = 0; jj < Ysize; jj++)
            {
                data.image[ID].array.F[jj * Xsize + ii] *= FLUXFACTOR;
            }
    }

    switch(SamplFactor)
    {

        case 0:

            if(image_ID(ID_name_r) != -1)
            {
                delete_image_ID(ID_name_r, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_r, Xsize, Ysize, &IDr);
            create_2Dimage_ID("imrc", Xsize, Ysize, &IDrc);

            if(image_ID(ID_name_g) != -1)
            {
                delete_image_ID(ID_name_g, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_g, Xsize, Ysize, &IDg);
            create_2Dimage_ID("imgc", Xsize, Ysize, &IDgc);

            if(image_ID(ID_name_b) != -1)
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
                        data.image[ID01].array.F[jj2 * Xsize + ii2] =
                            data.image[ID].array.F[jj2 * Xsize + ii2] /
                            data.image[IDflat].array.F[jj2 * Xsize + ii2];
                        data.image[ID01c].array.F[jj2 * Xsize + ii2] =
                            1.0 - data.image[IDbp].array.F[jj2 * Xsize + ii2];

                        ii2 = ii + 1;
                        jj2 = jj + 1;
                        data.image[ID11].array.F[jj2 * Xsize + ii2] =
                            data.image[ID].array.F[jj2 * Xsize + ii2] /
                            data.image[IDflat].array.F[jj2 * Xsize + ii2];
                        data.image[ID11c].array.F[jj2 * Xsize + ii2] =
                            1.0 - data.image[IDbp].array.F[jj2 * Xsize + ii2];

                        ii2 = ii;
                        jj2 = jj;
                        data.image[ID00].array.F[jj2 * Xsize + ii2] =
                            data.image[ID].array.F[jj2 * Xsize + ii2] /
                            data.image[IDflat].array.F[jj2 * Xsize + ii2];
                        data.image[ID00c].array.F[jj2 * Xsize + ii2] =
                            1.0 - data.image[IDbp].array.F[jj2 * Xsize + ii2];

                        ii2 = ii + 1;
                        jj2 = jj;
                        data.image[ID10].array.F[jj2 * Xsize + ii2] =
                            data.image[ID].array.F[jj2 * Xsize + ii2] /
                            data.image[IDflat].array.F[jj2 * Xsize + ii2];
                        data.image[ID10c].array.F[jj2 * Xsize + ii2] =
                            1.0 - data.image[IDbp].array.F[jj2 * Xsize + ii2];
                    }

                for(ii = 0; ii < Xsize; ii++)
                    for(jj = 0; jj < Ysize; jj++)
                    {
                        if(data.image[IDrc].array.F[jj * Xsize + ii] < 0.5)
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
                                            if(data.image[IDrc]
                                                    .array.F[jj1 * Xsize + ii1] >
                                                    0.5)
                                            {
                                                coeff = exp(
                                                            -5.0 * (dii * dii + djj * djj));
                                                vc += coeff;
                                                v += data.image[IDr]
                                                     .array
                                                     .F[jj1 * Xsize + ii1] *
                                                     coeff;
                                            }
                                        }
                                }
                            data.image[IDr].array.F[jj * Xsize + ii] = v / vc;
                        }

                        if(data.image[IDgc].array.F[jj * Xsize + ii] < 0.5)
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
                                            if(data.image[IDgc]
                                                    .array.F[jj1 * Xsize + ii1] >
                                                    0.5)
                                            {
                                                coeff = exp(
                                                            -5.0 * (dii * dii + djj * djj));
                                                vc += coeff;
                                                v += data.image[IDg]
                                                     .array
                                                     .F[jj1 * Xsize + ii1] *
                                                     coeff;
                                            }
                                        }
                                }
                            data.image[IDg].array.F[jj * Xsize + ii] = v / vc;
                        }

                        if(data.image[IDbc].array.F[jj * Xsize + ii] < 0.5)
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
                                            if(data.image[IDbc]
                                                    .array.F[jj1 * Xsize + ii1] >
                                                    0.5)
                                            {
                                                coeff = exp(
                                                            -5.0 * (dii * dii + djj * djj));
                                                vc += coeff;
                                                v += data.image[IDb]
                                                     .array
                                                     .F[jj1 * Xsize + ii1] *
                                                     coeff;
                                            }
                                        }
                                }
                            data.image[IDb].array.F[jj * Xsize + ii] = v / vc;
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
                            data.image[IDg].array.F[jj2 * Xsize + ii2] =
                                data.image[ID].array.F[jj2 * Xsize + ii2];
                            ii2 = ii + 1;
                            jj2 = jj + 1;
                            data.image[IDg].array.F[jj2 * Xsize + ii2] =
                                data.image[ID].array.F[jj2 * Xsize + ii2];
                        }
                    // replace blue pixels
                    for(ii1 = 0; ii1 < Xsize / 2 - 1; ii1++)
                        for(jj1 = 1; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii + 1;
                            jj2 = jj;
                            data.image[IDg].array.F[jj2 * Xsize + ii2] =
                                0.25 *
                                (data.image[ID].array.F[jj2 * Xsize + (ii2 - 1)] +
                                 data.image[ID].array.F[jj2 * Xsize + (ii2 + 1)] +
                                 data.image[ID].array.F[(jj2 + 1) * Xsize + ii2] +
                                 data.image[ID].array.F[(jj2 - 1) * Xsize + ii2]);
                        }
                    // replace red pixels
                    for(ii1 = 1; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2 - 1; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj + 1;
                            data.image[IDg].array.F[jj2 * Xsize + ii2] =
                                0.25 *
                                (data.image[ID].array.F[jj2 * Xsize + (ii2 - 1)] +
                                 data.image[ID].array.F[jj2 * Xsize + (ii2 + 1)] +
                                 data.image[ID].array.F[(jj2 + 1) * Xsize + ii2] +
                                 data.image[ID].array.F[(jj2 - 1) * Xsize + ii2]);
                        }

                    // R
                    for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                        {
                            ii  = ii1 * 2;
                            jj  = jj1 * 2;
                            ii2 = ii;
                            jj2 = jj + 1;
                            data.image[IDr].array.F[jj2 * Xsize + ii2] =
                                data.image[ID].array.F[jj2 * Xsize + ii2];
                        }
                    // replace g1 pixels
                    for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 1; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj;
                            data.image[IDr].array.F[jj2 * Xsize + ii2] =
                                0.5 *
                                (data.image[ID].array.F[(jj2 - 1) * Xsize + ii2] +
                                 data.image[ID].array.F[(jj2 + 1) * Xsize + ii2]);
                        }
                    // replace g2 pixels
                    for(ii1 = 0; ii1 < Xsize / 2 - 1; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii + 1;
                            jj2 = jj + 1;
                            data.image[IDr].array.F[jj2 * Xsize + ii2] =
                                0.5 *
                                (data.image[ID].array.F[jj2 * Xsize + (ii2 - 1)] +
                                 data.image[ID].array.F[jj2 * Xsize + (ii2 + 1)]);
                        }
                    // replace b pixels
                    for(ii1 = 0; ii1 < Xsize / 2 - 1; ii1++)
                        for(jj1 = 1; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii + 1;
                            jj2 = jj;
                            data.image[IDr].array.F[jj2 * Xsize + ii2] =
                                0.25 *
                                (data.image[ID]
                                 .array.F[(jj2 - 1) * Xsize + (ii2 - 1)] +
                                 data.image[ID]
                                 .array.F[(jj2 - 1) * Xsize + (ii2 + 1)] +
                                 data.image[ID]
                                 .array.F[(jj2 + 1) * Xsize + (ii2 - 1)] +
                                 data.image[ID]
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
                            data.image[IDb].array.F[jj2 * Xsize + ii2] =
                                data.image[ID].array.F[jj2 * Xsize + ii2];
                        }

                    // replace g2 pixels
                    for(ii1 = 0; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2 - 1; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii + 1;
                            jj2 = jj + 1;
                            data.image[IDb].array.F[jj2 * Xsize + ii2] =
                                0.5 *
                                (data.image[ID].array.F[(jj2 - 1) * Xsize + ii2] +
                                 data.image[ID].array.F[(jj2 + 1) * Xsize + ii2]);
                        }
                    // replace g1 pixels
                    for(ii1 = 1; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj;
                            data.image[IDb].array.F[jj2 * Xsize + ii2] =
                                0.5 *
                                (data.image[ID].array.F[jj2 * Xsize + (ii2 - 1)] +
                                 data.image[ID].array.F[jj2 * Xsize + (ii2 + 1)]);
                        }
                    // replace r pixels
                    for(ii1 = 1; ii1 < Xsize / 2; ii1++)
                        for(jj1 = 0; jj1 < Ysize / 2 - 1; jj1++)
                        {
                            ii = ii1 * 2;
                            jj = jj1 * 2;

                            ii2 = ii;
                            jj2 = jj + 1;
                            data.image[IDb].array.F[jj2 * Xsize + ii2] =
                                0.25 *
                                (data.image[ID]
                                 .array.F[(jj2 - 1) * Xsize + (ii2 - 1)] +
                                 data.image[ID]
                                 .array.F[(jj2 - 1) * Xsize + (ii2 + 1)] +
                                 data.image[ID]
                                 .array.F[(jj2 + 1) * Xsize + (ii2 - 1)] +
                                 data.image[ID]
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
            if(image_ID(ID_name_r) != -1)
            {
                delete_image_ID(ID_name_r, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_r, Xsize / 2, Ysize / 2, &IDr);
            create_2Dimage_ID("imrc", Xsize / 2, Ysize / 2, &IDrc);

            if(image_ID(ID_name_g) != -1)
            {
                delete_image_ID(ID_name_g, DELETE_IMAGE_ERRMODE_WARNING);
            }
            create_2Dimage_ID(ID_name_g, Xsize / 2, Ysize / 2, &IDg);
            create_2Dimage_ID("imgc", Xsize / 2, Ysize / 2, &IDgc);

            if(image_ID(ID_name_b) != -1)
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
                    data.image[ID01].array.F[jj1 * Xsize / 2 + ii1] +=
                        data.image[ID].array.F[jj2 * Xsize + ii2] /
                        data.image[IDflat].array.F[jj2 * Xsize + ii2];
                    data.image[ID01c].array.F[jj1 * Xsize / 2 + ii1] +=
                        1.0 - data.image[IDbp].array.F[jj2 * Xsize + ii2];

                    ii2 = ii + 1;
                    jj2 = jj + 1;
                    data.image[ID11].array.F[jj1 * Xsize / 2 + ii1] +=
                        data.image[ID].array.F[jj2 * Xsize + ii2] /
                        data.image[IDflat].array.F[jj2 * Xsize + ii2];
                    data.image[ID11c].array.F[jj1 * Xsize / 2 + ii1] +=
                        1.0 - data.image[IDbp].array.F[jj2 * Xsize + ii2];

                    ii2 = ii;
                    jj2 = jj;
                    data.image[ID00].array.F[jj1 * Xsize / 2 + ii1] +=
                        data.image[ID].array.F[jj2 * Xsize + ii2] /
                        data.image[IDflat].array.F[jj2 * Xsize + ii2];
                    data.image[ID00c].array.F[jj1 * Xsize / 2 + ii1] +=
                        1.0 - data.image[IDbp].array.F[jj2 * Xsize + ii2];

                    ii2 = ii + 1;
                    jj2 = jj;
                    data.image[ID10].array.F[jj1 * Xsize / 2 + ii1] +=
                        data.image[ID].array.F[jj2 * Xsize + ii2] /
                        data.image[IDflat].array.F[jj2 * Xsize + ii2];
                    data.image[ID10c].array.F[jj1 * Xsize / 2 + ii1] +=
                        1.0 - data.image[IDbp].array.F[jj2 * Xsize + ii2];

                    data.image[IDr].array.F[jj1 * Xsize / 2 + ii1] /=
                        data.image[IDrc].array.F[jj1 * Xsize / 2 + ii1] + eps;
                    data.image[IDg].array.F[jj1 * Xsize / 2 + ii1] /=
                        data.image[IDgc].array.F[jj1 * Xsize / 2 + ii1] + eps;
                    data.image[IDb].array.F[jj1 * Xsize / 2 + ii1] /=
                        data.image[IDbc].array.F[jj1 * Xsize / 2 + ii1] + eps;
                }

            delete_image_ID("imrc", DELETE_IMAGE_ERRMODE_WARNING);
            delete_image_ID("imgc", DELETE_IMAGE_ERRMODE_WARNING);
            delete_image_ID("imbc", DELETE_IMAGE_ERRMODE_WARNING);

            break;
    }

    return RETURN_SUCCESS;
}
