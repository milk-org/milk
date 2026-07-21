// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file linregress.c
 * @brief Linregress module
 */

/** @file linregress.c
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#endif

errno_t lin_regress(double      *a,
                    double      *b,
                    double      *Xi2,
                    double      *x,
                    double      *y,
                    double      *sig,
                    unsigned int nb_points)
{
    double S, Sx, Sy, Sxx, Sxy, Syy;
    double delta;

    S   = 0;
    Sx  = 0;
    Sy  = 0;
    Sxx = 0;
    Syy = 0;
    Sxy = 0;

    for (unsigned i = 0; i < nb_points; i++)
    {
        S += 1.0 / sig[i] / sig[i];
        Sx += x[i] / sig[i] / sig[i];
        Sy += y[i] / sig[i] / sig[i];
        Sxx += x[i] * x[i] / sig[i] / sig[i];
        Syy += y[i] * y[i] / sig[i] / sig[i];
        Sxy += x[i] * y[i] / sig[i] / sig[i];
    }

    delta = S * Sxx - Sx * Sx;
    *a    = (Sxx * Sy - Sx * Sxy) / delta;
    *b    = (S * Sxy - Sx * Sy) / delta;
    *Xi2  = Syy - 2 * (*a) * Sy - 2 * (*a) * (*b) * Sx + (*a) * (*a) * S + 2 * (*a) * (*b) * Sx -
           (*b) * (*b) * Sxx;

    return RETURN_SUCCESS;
}
