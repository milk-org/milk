// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file medianfilter.c
 * @brief 2D median filter using COREMOD_arith SIMD Young-tableau kernels (3x3, 5x5, 7x7) and OpenMP
 */

#include <stdint.h>
#include <stdlib.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "ImageStreamIO/ImageStreamIO.h"
#    include "computeunit.h"
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_arith/imfunctions.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "libmilkcommon/milk_compiler.h"

imageID median_filter(const char *__restrict ID_name,
                      const char *__restrict out_name,
                      int filter_size)
{
    imageID ID = image_ID(ID_name, dcimg, dcnimg);
    if (ID == -1)
    {
        return -1;
    }

    copy_image_ID(ID_name, out_name, 0);
    imageID IDout = image_ID(out_name, dcimg, dcnimg);
    if (IDout == -1)
    {
        return -1;
    }

    if (filter_size >= 0 && filter_size <= 3)
    {
        arith_image_medfilt2d_im(&dcimg[ID], &dcimg[IDout], filter_size);
        return IDout;
    }

    const long   nx  = (long) dcimg[ID].md[0].size[0];
    const long   ny  = (long) dcimg[ID].md[0].size[1];
    const int    K   = 2 * filter_size + 1;
    const int    K2  = K * K;
    const float *in  = dcimg[ID].array.F;
    float       *out = dcimg[IDout].array.F;
    int          nth = milk_omp_team_size((uint64_t) nx * (uint64_t) ny, 16384);

#ifdef _OPENMP
#    pragma omp parallel num_threads(nth) if (nth > 1)
#endif
    {
        float *array = (float *) malloc((size_t) K2 * sizeof(float));
        if (array != NULL)
        {
#ifdef _OPENMP
#    pragma omp for schedule(static)
#endif
            for (long jj = filter_size; jj < ny - filter_size; jj++)
            {
                for (long ii = filter_size; ii < nx - filter_size; ii++)
                {
                    for (int j = 0; j < K; j++)
                    {
                        const float *row = in + (jj - filter_size + j) * nx + (ii - filter_size);
                        for (int i = 0; i < K; i++)
                        {
                            array[j * K + i] = row[i];
                        }
                    }
                    quick_sort_float(array, K2);
                    out[jj * nx + ii] = array[(K2 - 1) / 2];
                }
            }
            free(array);
        }
    }

    return IDout;
}
