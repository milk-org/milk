/** @file measure_transl.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_filter/image_filter.h"
#include "info/info.h"

#include "imcontract.h"

// measure offset between 2 images

double basic_measure_transl(const char *__restrict ID_name1,
                            const char *__restrict ID_name2,
                            long tmax)
{
    imageID ID1, ID2, ID;
    imageID IDout, IDcnt;
    long    dx, dy, ii1, jj1, ii2, jj2, iio, jjo;
    long    sx_out, sy_out;
    long    size1x, size1y;
    long    size2x, size2y;
    double  val;
    double  tmp, v1, v2;
    int     SCALE = 64; // must be power of 2
    long    step1 = 1;
    long    step2 = 1;
    double  vmin;
    double  vdx, vdy;
    long    ii2min, ii2max, jj2min, jj2max;
    long    dxmin, dymin;
    int     SCALEindex;
    long    dsize;
    double  vmincnt;
    long    dx1, dy1;
    int     QUICKMODE = 0;
    long    ii1min, ii1max, jj1min, jj1max;
    long    iiomin, iiomax, jjomin, jjomax;
    imageID ID1mask;
    long    xsizemask, ysizemask;
    double  vlim;
    long    contractfactor;
    long    ii;
    long    ii1m, jj1m;
    double  Mlim;

    double fitval = 0.0;

    step1 = SCALE;
    step2 = SCALE;

    ID1    = image_ID(ID_name1);
    size1x = data.image[ID1].md[0].size[0];
    size1y = data.image[ID1].md[0].size[1];

    ID2    = image_ID(ID_name2);
    size2x = data.image[ID2].md[0].size[0];
    size2y = data.image[ID2].md[0].size[1];

    sx_out = 2 * tmax;
    sy_out = 2 * tmax;
    create_2Dimage_ID("TranslMap", sx_out, sy_out, &IDout);
    create_2Dimage_ID("TranslMapcnt", sx_out, sy_out, &IDcnt);
    for(iio = 0; iio < sx_out; iio++)
        for(jjo = 0; jjo < sy_out; jjo++)
        {
            data.image[IDout].array.F[jjo * sx_out + iio] = 0.0;
            data.image[IDcnt].array.F[jjo * sx_out + iio] = 0.0;
        }

    dxmin      = 0;
    dymin      = 0;
    SCALEindex = 1;

    // STEP 1 : quickly identify regions of image 1 where flux gradient is large
    // select 30% of image pixels
    contractfactor = 2;
    basic_contract(ID_name1, "_im1C", contractfactor, contractfactor);
    gauss_filter("_im1C", "_im1Cg", 5.0, 10);
    execute_arith("_im1HF=_im1C-_im1Cg");
    execute_arith("_im1HF2=_im1HF*_im1HF");
    gauss_filter("_im1HF2", "_im1mask", 5.0, 10);
    delete_image_ID("_im1C", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("_im1HF", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("_im1Cg", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("_im1HF2", DELETE_IMAGE_ERRMODE_WARNING);

    vlim = (double) img_percentile("_im1mask", 0.8);
    printf("vlim = %g\n", vlim);
    save_fl_fits("_im1mask", "_im1mask.0.fits");
    ID1mask   = image_ID("_im1mask");
    xsizemask = data.image[ID1mask].md[0].size[0];
    ysizemask = data.image[ID1mask].md[0].size[1];

    for(ii = 0; ii < xsizemask * ysizemask; ii++)
    {
        if(data.image[ID1mask].array.F[ii] > vlim)
        {
            data.image[ID1mask].array.F[ii] = 1.0;
        }
        else
        {
            data.image[ID1mask].array.F[ii] = 0.0;
        }
    }

    save_fl_fits("_im1mask", "_im1mask.fits");
    //exit(0);

    dsize = tmax * 2;
    while(SCALE != 0)
    {
        step1 = SCALE;
        step2 = 1; //SCALE;

        dsize /= 2; //(long) (1.0*tmax/pow(SCALEindex,2.0));
        if(dsize < 1.2 * SCALE)
        {
            dsize = (long)(1.2 * SCALE);
        }

        //      if(SCALE>1)
        //Mlim = -1;
        // else
        Mlim = 0.5;

        ii1min = 0;
        ii1max = size1x;
        jj1min = 0;
        jj1max = size1y;

        if(QUICKMODE == 1)
        {
            step1 *= 5;
            step2 *= 3;
        }
        if(SCALE == 1)
        {
            step1 = 1;
            step2 = 1;
        }

        for(ii1 = ii1min; ii1 < ii1max; ii1 += step1)
            for(jj1 = jj1min; jj1 < jj1max; jj1 += step1)
            {
                ii1m = (long)(ii1 / contractfactor);
                jj1m = (long)(jj1 / contractfactor);
                if(data.image[ID1mask].array.F[jj1m * xsizemask + ii1m] > Mlim)
                {
                    v1 = data.image[ID1].array.F[jj1 * size1x + ii1];

                    ii2min = ii1 + dxmin - dsize;
                    ii2max = ii1 + dxmin + dsize;
                    while(ii2min < 0)
                    {
                        ii2min += step2;
                    }
                    while(ii2min > size2x - 1)
                    {
                        ii2min -= step2;
                    }
                    while(ii2max < 0)
                    {
                        ii2max += step2;
                    }
                    while(ii2max > size2x - 1)
                    {
                        ii2max -= step2;
                    }

                    jj2min = jj1 + dymin - dsize;
                    jj2max = jj1 + dymin + dsize;
                    while(jj2min < 0)
                    {
                        jj2min += step2;
                    }
                    while(jj2min > size2y - 1)
                    {
                        jj2min -= step2;
                    }
                    while(jj2max < 0)
                    {
                        jj2max += step2;
                    }
                    while(jj2max > size2y - 1)
                    {
                        jj2max -= step2;
                    }

                    for(ii2 = ii2min; ii2 < ii2max; ii2 += step2)
                        for(jj2 = jj2min; jj2 < jj2max; jj2 += step2)
                        {
                            dx  = ii2 - ii1;
                            dy  = jj2 - jj1;
                            dx1 = dx - dxmin;
                            dy1 = dy - dymin;
                            if(dx1 * dx1 + dy1 * dy1 < 1.0 * dsize * dsize)
                            {
                                iio = dx + tmax;
                                jjo = dy + tmax;
                                if((iio > -1) && (iio < sx_out) &&
                                        (jjo > -1) && (jjo < sy_out))
                                {
                                    v2 = data.image[ID2]
                                         .array.F[jj2 * size2x + ii2];
                                    tmp = (v1 - v2);
                                    data.image[IDout]
                                    .array.F[jjo * sx_out + iio] +=
                                        tmp * tmp;
                                    data.image[IDcnt]
                                    .array.F[jjo * sx_out + iio] += 1.0;
                                    //   if((iio == 87)&&(jjo == 100))
                                    //printf("%g (%ld %ld %g) (%ld %ld %g)\n",data.image[IDcnt].array.F[jjo*sx_out+iio], ii1, jj1, v1, ii2, jj2, v2);
                                }
                            }
                        }
                }
            }

        vmin = 1.0e100;
        for(iio = 0; iio < sx_out; iio++)
            for(jjo = 0; jjo < sy_out; jjo++)
            {
                if(data.image[IDcnt].array.F[jjo * sx_out + iio] > 0.1)
                {
                    val = data.image[IDout].array.F[jjo * sx_out + iio] /
                          data.image[IDcnt].array.F[jjo * sx_out + iio];
                    if(val < vmin)
                    {
                        vmin    = val;
                        vmincnt = data.image[IDcnt].array.F[jjo * sx_out + iio];
                        vdx     = 1.0 * iio - tmax;
                        vdy     = 1.0 * jjo - tmax;
                    }
                }
            }
        printf("------- SCALE = %d [%ld] --------\n", SCALE, dsize);
        printf("vdx = %g  (%ld)\n", vdx, dxmin);
        printf("vdy = %g  (%ld)\n", vdy, dymin);
        printf("vmin = %g [%g]\n", vmin, vmincnt);

        dxmin = (long)(vdx + 0.5 + 10000) - 10000;
        dymin = (long)(vdy + 0.5 + 10000) - 10000;

        printf("-------- %ld %ld --------\n", dxmin, dymin);

        if(SCALE == 1)
        {
            SCALE = 0;
        }
        else
        {
            SCALEindex++;
            SCALE /= 2;
        }
    }

    for(iio = 0; iio < sx_out; iio++)
        for(jjo = 0; jjo < sy_out; jjo++)
        {
            if(data.image[IDcnt].array.F[jjo * sx_out + iio] > 0.1)
            {
                data.image[IDout].array.F[jjo * sx_out + iio] /=
                    data.image[IDcnt].array.F[jjo * sx_out + iio];
            }
        }

    ID = gauss_filter("TranslMap", "TranslMapg", 5.0, 10);

    vmin = 1.0e100;

    iiomin = sx_out / 2 + dxmin - 20;
    if(iiomin < 0)
    {
        iiomin = 0;
    }
    iiomax = sx_out / 2 + dxmin + 20;
    if(iiomax > sx_out - 1)
    {
        iiomax = sx_out - 1;
    }

    jjomin = sy_out / 2 + dymin - 20;
    if(jjomin < 0)
    {
        jjomin = 0;
    }
    jjomax = sy_out / 2 + dymin + 20;
    if(jjomax > sy_out - 1)
    {
        jjomax = sy_out - 1;
    }

    for(iio = iiomin; iio < iiomax; iio++)
        for(jjo = jjomin; jjo < jjomax; jjo++)
        {
            if(data.image[IDcnt].array.F[jjo * sx_out + iio] > 0.1)
            {
                val = data.image[ID].array.F[jjo * sx_out + iio];
                if(val < vmin)
                {
                    vmin = val;
                    vdx  = 1.0 * iio - tmax;
                    vdy  = 1.0 * jjo - tmax;
                }
            }
        }

    create_variable_ID("vdx", vdx);
    create_variable_ID("vdy", vdy);
    printf("-------- %f %f --------\n", vdx, vdy);

    save_fl_fits("TranslMapg", "_TranslMap.fits");
    save_fl_fits("TranslMapcnt", "_TranslMapcnt.fits");

    delete_image_ID("TranslMap", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("TranslMapg", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("TranslMapcnt", DELETE_IMAGE_ERRMODE_WARNING);
    //  exit(0);

    return (fitval);
}
