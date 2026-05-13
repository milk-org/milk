/**
 * @file measure_transl.c
 * @brief Measure transl module
 */

/** @file measure_transl.c
 */

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

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_filter/image_filter.h"
#include "info/info.h"

#include "imcontract.h"

/**
 * Measure translation offset between
 * two images using multi-scale search.
 */
double basic_measure_transl(
    const char *__restrict ID_name1,
    const char *__restrict ID_name2,
    long tmax)
{
    int  SCALE = 64; /* must be power of 2 */
    long step1 = 1;
    long step2 = 1;
    int  QUICKMODE = 0;
    long contractfactor;

    double fitval = 0.0;

    step1 = SCALE;
    step2 = SCALE;

    IMGID img1 =
        imgid_make_from_name(ID_name1);
    resolveIMGID(&img1, ERRMODE_WARN,
                 dcimg, dcnimg);
    if (img1.ID == -1) {
        return RETURN_FAILURE;
    }
    long size1x = img1.md->size[0];
    long size1y = img1.md->size[1];

    IMGID img2 =
        imgid_make_from_name(ID_name2);
    resolveIMGID(&img2, ERRMODE_WARN,
                 dcimg, dcnimg);
    if (img2.ID == -1) {
        return RETURN_FAILURE;
    }
    long size2x = img2.md->size[0];
    long size2y = img2.md->size[1];

    long sx_out = 2 * tmax;
    long sy_out = 2 * tmax;

    IMGID imgout =
        imgid_make_from_name_2D(
            "TranslMap", sx_out, sy_out);
    imgout.mdt->shared = 0;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    IMGID imgcnt =
        imgid_make_from_name_2D(
            "TranslMapcnt",
            sx_out, sy_out);
    imgcnt.mdt->shared = 0;
    imgcnt.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgcnt);

    for(long iio = 0;
        iio < sx_out; iio++)
    {
        for(long jjo = 0;
            jjo < sy_out; jjo++)
        {
            imgout.im->array.F[
                jjo * sx_out + iio] =
                0.0;
            imgcnt.im->array.F[
                jjo * sx_out + iio] =
                0.0;
        }
    }

    long dxmin = 0;
    long dymin = 0;
    int  SCALEindex = 1;

    /* STEP 1: identify high-gradient
     * regions in image 1 */
    contractfactor = 2;
    basic_contract(
        ID_name1, "_im1C",
        contractfactor, contractfactor);
    gauss_filter(
        "_im1C", "_im1Cg", 5.0, 10);
    execute_arith("_im1HF=_im1C-_im1Cg");
    execute_arith(
        "_im1HF2=_im1HF*_im1HF");
    gauss_filter(
        "_im1HF2", "_im1mask", 5.0, 10);
    delete_image_ID(
        "_im1C",
        DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(
        "_im1HF",
        DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(
        "_im1Cg",
        DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(
        "_im1HF2",
        DELETE_IMAGE_ERRMODE_WARNING);

    double vlim =
        (double) img_percentile(
            "_im1mask", 0.8);
    printf("vlim = %g\n", vlim);
    save_fl_fits(
        "_im1mask", "_im1mask.0.fits");

    IMGID imgmask =
        imgid_make_from_name(
            "_im1mask");
    resolveIMGID(&imgmask,
                 ERRMODE_ABORT,
                 dcimg, dcnimg);
    long xsizemask =
        imgmask.md->size[0];
    long ysizemask =
        imgmask.md->size[1];

    for(long ii = 0;
        ii < xsizemask * ysizemask;
        ii++)
    {
        if(imgmask.im->array.F[ii]
           > vlim)
        {
            imgmask.im->array.F[ii] =
                1.0;
        }
        else
        {
            imgmask.im->array.F[ii] =
                0.0;
        }
    }

    save_fl_fits(
        "_im1mask", "_im1mask.fits");

    long dsize = tmax * 2;
    double vmin, vdx = 0, vdy = 0;
    double vmincnt = 0;

    while(SCALE != 0)
    {
        step1 = SCALE;
        step2 = 1;

        dsize /= 2;
        if(dsize < 1.2 * SCALE)
        {
            dsize =
                (long)(1.2 * SCALE);
        }

        double Mlim = 0.5;

        long ii1min = 0;
        long ii1max = size1x;
        long jj1min = 0;
        long jj1max = size1y;

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

        for(long ii1 = ii1min;
            ii1 < ii1max;
            ii1 += step1)
        {
            for(long jj1 = jj1min;
                jj1 < jj1max;
                jj1 += step1)
            {
                long ii1m =
                    (long)(ii1
                           / contractfactor);
                long jj1m =
                    (long)(jj1
                           / contractfactor);
                if(imgmask.im->array.F[
                       jj1m * xsizemask
                       + ii1m]
                   > Mlim)
                {
                    double v1 =
                        img1.im
                            ->array.F[
                                jj1
                                * size1x
                                + ii1];

                    long ii2min =
                        ii1 + dxmin
                        - dsize;
                    long ii2max =
                        ii1 + dxmin
                        + dsize;
                    while(ii2min < 0)
                    {
                        ii2min += step2;
                    }
                    while(ii2min
                          > size2x - 1)
                    {
                        ii2min -= step2;
                    }
                    while(ii2max < 0)
                    {
                        ii2max += step2;
                    }
                    while(ii2max
                          > size2x - 1)
                    {
                        ii2max -= step2;
                    }

                    long jj2min =
                        jj1 + dymin
                        - dsize;
                    long jj2max =
                        jj1 + dymin
                        + dsize;
                    while(jj2min < 0)
                    {
                        jj2min += step2;
                    }
                    while(jj2min
                          > size2y - 1)
                    {
                        jj2min -= step2;
                    }
                    while(jj2max < 0)
                    {
                        jj2max += step2;
                    }
                    while(jj2max
                          > size2y - 1)
                    {
                        jj2max -= step2;
                    }

                    for(long ii2 =
                            ii2min;
                        ii2 < ii2max;
                        ii2 += step2)
                    {
                        for(long jj2 =
                                jj2min;
                            jj2 < jj2max;
                            jj2 += step2)
                        {
                            long dx =
                                ii2 - ii1;
                            long dy =
                                jj2 - jj1;
                            long dx1 =
                                dx - dxmin;
                            long dy1 =
                                dy - dymin;
                            if(dx1 * dx1
                               + dy1 * dy1
                               < 1.0
                                 * dsize
                                 * dsize)
                            {
                                long iio =
                                    dx
                                    + tmax;
                                long jjo =
                                    dy
                                    + tmax;
                                if((iio
                                    > -1)
                                   && (iio
                                       < sx_out)
                                   && (jjo
                                       > -1)
                                   && (jjo
                                       < sy_out))
                                {
                                    double
                                        v2 =
                                        img2
                                            .im
                                            ->array
                                            .F[jj2
                                               * size2x
                                               + ii2];
                                    double
                                        tmp =
                                        (v1
                                         - v2);
                                    imgout
                                        .im
                                        ->array
                                        .F[jjo
                                           * sx_out
                                           + iio]
                                        += tmp
                                           * tmp;
                                    imgcnt
                                        .im
                                        ->array
                                        .F[jjo
                                           * sx_out
                                           + iio]
                                        += 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        vmin = 1.0e100;
        for(long iio = 0;
            iio < sx_out; iio++)
        {
            for(long jjo = 0;
                jjo < sy_out; jjo++)
            {
                if(imgcnt.im->array.F[
                       jjo * sx_out
                       + iio]
                   > 0.1)
                {
                    double val =
                        imgout.im
                            ->array.F[
                                jjo
                                * sx_out
                                + iio]
                        / imgcnt.im
                              ->array.F[
                                  jjo
                                  * sx_out
                                  + iio];
                    if(val < vmin)
                    {
                        vmin = val;
                        vmincnt =
                            imgcnt.im
                                ->array
                                .F[jjo
                                   * sx_out
                                   + iio];
                        vdx = 1.0 * iio
                              - tmax;
                        vdy = 1.0 * jjo
                              - tmax;
                    }
                }
            }
        }
        printf(
            "------- SCALE = %d"
            " [%ld] --------\n",
            SCALE, dsize);
        printf(
            "vdx = %g  (%ld)\n",
            vdx, dxmin);
        printf(
            "vdy = %g  (%ld)\n",
            vdy, dymin);
        printf(
            "vmin = %g [%g]\n",
            vmin, vmincnt);

        dxmin =
            (long)(vdx + 0.5 + 10000)
            - 10000;
        dymin =
            (long)(vdy + 0.5 + 10000)
            - 10000;

        printf(
            "-------- %ld %ld"
            " --------\n",
            dxmin, dymin);

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

    for(long iio = 0;
        iio < sx_out; iio++)
    {
        for(long jjo = 0;
            jjo < sy_out; jjo++)
        {
            if(imgcnt.im->array.F[
                   jjo * sx_out + iio]
               > 0.1)
            {
                imgout.im->array.F[
                    jjo * sx_out + iio]
                    /= imgcnt.im
                           ->array.F[
                               jjo
                               * sx_out
                               + iio];
            }
        }
    }

    imageID ID = gauss_filter(
        "TranslMap", "TranslMapg",
        5.0, 10);

    vmin = 1.0e100;

    long iiomin =
        sx_out / 2 + dxmin - 20;
    if(iiomin < 0)
    {
        iiomin = 0;
    }
    long iiomax =
        sx_out / 2 + dxmin + 20;
    if(iiomax > sx_out - 1)
    {
        iiomax = sx_out - 1;
    }

    long jjomin =
        sy_out / 2 + dymin - 20;
    if(jjomin < 0)
    {
        jjomin = 0;
    }
    long jjomax =
        sy_out / 2 + dymin + 20;
    if(jjomax > sy_out - 1)
    {
        jjomax = sy_out - 1;
    }

    for(long iio = iiomin;
        iio < iiomax; iio++)
    {
        for(long jjo = jjomin;
            jjo < jjomax; jjo++)
        {
            if(imgcnt.im->array.F[
                   jjo * sx_out + iio]
               > 0.1)
            {
                double val =
                    dcimg[ID].array.F[
                        jjo * sx_out
                        + iio];
                if(val < vmin)
                {
                    vmin = val;
                    vdx = 1.0 * iio
                          - tmax;
                    vdy = 1.0 * jjo
                          - tmax;
                }
            }
        }
    }

    create_variable_ID("vdx", vdx);
    create_variable_ID("vdy", vdy);
    printf(
        "-------- %f %f --------\n",
        vdx, vdy);

    save_fl_fits(
        "TranslMapg",
        "_TranslMap.fits");
    save_fl_fits(
        "TranslMapcnt",
        "_TranslMapcnt.fits");

    delete_image_ID(
        "TranslMap",
        DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(
        "TranslMapg",
        DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(
        "TranslMapcnt",
        DELETE_IMAGE_ERRMODE_WARNING);

    return (fitval);
}
