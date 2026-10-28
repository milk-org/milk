// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file naninf2zero.c
 * @brief Replace NaN and Inf pixel values with zero using IEEE-754 bit inspection and OpenMP SIMD
 */

#include <stdint.h>
#include <stdio.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "ImageStreamIO/ImageStreamIO.h"
#    include "computeunit.h"
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/milk_compiler.h"

/* set all nan and inf pixel values to zero */
int basic_naninf2zero(const char *ID_name)
{
    imageID ID = image_ID(ID_name, dcimg, dcnimg);
    if (ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint64_t nelement = dcimg[ID].md[0].nelement;
    uint8_t  dt       = dcimg[ID].md[0].datatype;
    int      nth      = milk_omp_team_size(nelement, 262144);
    long     cnt      = 0;

    if (dt == _DATATYPE_DOUBLE)
    {
        double *MILK_RESTRICT p = MILK_ASSUME_ALIGNED(dcimg[ID].array.D);
#ifdef _OPENMP
#    pragma omp parallel for simd reduction(+ : cnt) num_threads(nth) if (parallel : nth > 1)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            int bad = !arith_isfinite_d(p[ii]);
            p[ii]   = bad ? 0.0 : p[ii];
            cnt += bad;
        }
    }
    else
    {
        float *MILK_RESTRICT p = MILK_ASSUME_ALIGNED(dcimg[ID].array.F);
#ifdef _OPENMP
#    pragma omp parallel for simd reduction(+ : cnt) num_threads(nth) if (parallel : nth > 1)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            int bad = !arith_isfinite_f(p[ii]);
            p[ii]   = bad ? 0.0f : p[ii];
            cnt += bad;
        }
    }

    if (dcdebug > 0)
    {
        printf("%ld values replaced\n", cnt);
    }

    return 0;
}
