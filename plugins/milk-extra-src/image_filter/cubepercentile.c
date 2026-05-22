/**
 * @file cubepercentile.c
 * @brief Cubepercentile module
 */

/** @file cubepercentile.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

/**
 * Compute per-pixel percentile of a
 * 3D image cube.
 */
imageID filter_CubePercentile(const char *__restrict IDcin_name,
                              float perc,
                              const char *__restrict IDout_name)
{
    IMGID imgin = imgid_make_from_name(IDcin_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long xsize = imgin.md->size[0];
    long ysize = imgin.md->size[1];
    long zsize = imgin.md->size[2];

    float *array = (float *) malloc(sizeof(float) * xsize * ysize);
    if (array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IMGID imgout       = imgid_make_from_name_2D(IDout_name, xsize, ysize);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (long ii = 0; ii < xsize * ysize; ii++)
    {
        for (long kk = 0; kk < zsize; kk++)
        {
            array[kk] = imgin.im->array.F[kk * xsize * ysize + ii];
        }

        quick_sort_float(array, zsize);
        imgout.im->array.F[ii] = array[(long) (perc * zsize)];
    }

    free(array);

    return imgout.ID;
}

/**
 * Compute per-pixel percentile of a
 * 3D image cube, excluding values above
 * a threshold.
 */
imageID filter_CubePercentileLimit(const char *__restrict IDcin_name,
                                   float perc,
                                   float limit,
                                   const char *__restrict IDout_name)
{
    IMGID imgin = imgid_make_from_name(IDcin_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long xsize = imgin.md->size[0];
    long ysize = imgin.md->size[1];
    long zsize = imgin.md->size[2];

    float *array = (float *) malloc(sizeof(float) * xsize * ysize);
    if (array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IMGID imgout       = imgid_make_from_name_2D(IDout_name, xsize, ysize);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (long ii = 0; ii < xsize * ysize; ii++)
    {
        long cnt = 0;
        for (long kk = 0; kk < zsize; kk++)
        {
            float v1 = imgin.im->array.F[kk * xsize * ysize + ii];
            if (v1 < limit)
            {
                array[cnt] = v1;
                cnt++;
            }

            if (cnt > 0)
            {
                quick_sort_float(array, zsize);
                imgout.im->array.F[ii] = array[(long) (perc * cnt)];
            }
            else
            {
                imgout.im->array.F[ii] = limit;
            }
        }
    }

    free(array);

    return imgout.ID;
}
