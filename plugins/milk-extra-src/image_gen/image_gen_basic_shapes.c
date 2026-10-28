// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
#    include "computeunit.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "computeunit.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/create_image.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "statistic/statistic.h"
#include "ImageStreamIO/ImageStreamIO.h"

#include "image_gen.h"
#include "libmilkcommon/milk_compiler.h"


/** @brief creates a double star */
imageID make_double_star(const char *ID_name,
                         uint32_t    l1,
                         uint32_t    l2,
                         double      intensity_1,
                         double      intensity_2,
                         double      separation,
                         double      position_angle)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    dcimg[ID].array.F[((int) (naxes[1] / 2)) * naxes[0] + ((int) (naxes[0] / 2))] = intensity_1;
    dcimg[ID].array.F[((int) (naxes[1] / 2 + separation * cos(position_angle))) * naxes[0] +
                      ((int) (naxes[0] / 2 + separation * sin(position_angle)))]  = intensity_2;

    return (ID);
}

/** @brief creates a disk */
imageID make_disk(const char *ID_name,
                  uint32_t    l1,
                  uint32_t    l2,
                  double      x_center,
                  double      y_center,
                  double      radius)
{
    imageID  ID;
    uint32_t naxes[2];
    long     x1, x2, y1, y2;
    double   r2;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1 = (long) (x_center - radius - 2);
    x2 = (long) (x_center + radius + 2);
    y1 = (long) (y_center - radius - 2);
    y2 = (long) (y_center + radius + 2);

    if (x1 < 0)
    {
        x1 = 0;
    }
    if (x1 > (long) naxes[0])
    {
        x1 = (long) naxes[0];
    }

    if (x2 < 0)
    {
        x2 = 0;
    }
    if (x2 > (long) naxes[0])
    {
        x2 = (long) naxes[0];
    }

    if (y1 < 0)
    {
        y1 = 0;
    }
    if (y1 > (long) naxes[1])
    {
        y1 = (long) naxes[1];
    }

    if (y2 < 0)
    {
        y2 = 0;
    }
    if (y2 > (long) naxes[1])
    {
        y2 = (long) naxes[1];
    }

    r2 = radius * radius;

    float *MILK_RESTRICT out = MILK_ASSUME_ALIGNED(dcimg[ID].array.F);
    int nth = milk_omp_team_size((uint64_t) (y2 - y1) * (uint64_t) (x2 - x1), 262144);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (long jj = y1; jj < y2; jj++)
    {
        float *row = out + (uint64_t) jj * naxes[0];
        double dy  = (double) jj - y_center;
        double dy2 = dy * dy;
#pragma omp simd
        for (long ii = x1; ii < x2; ii++)
        {
            double dx = (double) ii - x_center;
            if (dx * dx + dy2 < r2)
            {
                row[ii] = 1.0f;
            }
        }
    }

    return (ID);
}

imageID make_subpixdisk(const char *ID_name,
                        uint32_t    l1,
                        uint32_t    l2,
                        double      x_center,
                        double      y_center,
                        double      radius)
{
    imageID  ID;
    uint32_t naxes[2];
    int      subgrid = 55;
    double   grid[55];
    long     x1, x2, y1, y2;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1 = (long) (x_center - radius - 2);
    x2 = (long) (x_center + radius + 2);
    y1 = (long) (y_center - radius - 2);
    y2 = (long) (y_center + radius + 2);

    if (x1 < 0)
    {
        x1 = 0;
    }
    if (x1 > (long) naxes[0])
    {
        x1 = (long) naxes[0];
    }
    if (x2 < 0)
    {
        x2 = 0;
    }
    if (x2 > (long) naxes[0])
    {
        x2 = (long) naxes[0];
    }

    if (y1 < 0)
    {
        y1 = 0;
    }
    if (y1 > (long) naxes[1])
    {
        y1 = (long) naxes[1];
    }
    if (y2 < 0)
    {
        y2 = 0;
    }
    if (y2 > (long) naxes[1])
    {
        y2 = (long) naxes[1];
    }

    for (int i = 0; i < subgrid; i++)
    {
        grid[i] = 0.5 - 0.5 / subgrid - (double) i / subgrid;
    }

    const double         r2ref        = radius * radius;
    const double         r_lo         = (radius > 1.5) ? (radius - 1.5) : 0.0;
    const double         r_lo2        = r_lo * r_lo;
    const double         r_hi2        = (radius + 1.5) * (radius + 1.5);
    const double         inv_subgrid2 = 1.0 / ((double) subgrid * (double) subgrid);
    float *MILK_RESTRICT out          = MILK_ASSUME_ALIGNED(dcimg[ID].array.F);
    int nth = milk_omp_team_size((uint64_t) (y2 - y1) * (uint64_t) (x2 - x1), 16384);

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic, 4) num_threads(nth) if (nth > 1)
#endif
    for (long jj = y1; jj < y2; jj++)
    {
        float       *row   = out + (uint64_t) jj * naxes[0];
        const double ydiff = y_center - (double) jj;
        const double y2_d  = ydiff * ydiff;
        for (long ii = x1; ii < x2; ii++)
        {
            const double xdiff = x_center - (double) ii;
            const double r2    = xdiff * xdiff + y2_d;
            if (r2 <= r_lo2 && radius > 1.5)
            {
                row[ii] = 1.0f;
            }
            else if (r2 < r_hi2)
            {
                int tot = 0;
                for (int j = 0; j < subgrid; j++)
                {
                    const double y   = ydiff + grid[j];
                    const double yy2 = y * y;
#pragma omp simd reduction(+ : tot)
                    for (int i = 0; i < subgrid; i++)
                    {
                        const double x = xdiff + grid[i];
                        tot += (x * x + yy2 < r2ref) ? 1 : 0;
                    }
                }
                row[ii] = (float) ((double) tot * inv_subgrid2);
            }
        }
    }

    return (ID);
}

// creates a shape with contour described by sum of sine waves
//
// r = radius + SUM[ ra[i] * cos( ka[i]*PA/2.0/PI + pa[i]) ]

imageID make_subpixdisk_perturb(const char *ID_name,
                                uint32_t    l1,
                                uint32_t    l2,
                                double      x_center,
                                double      y_center,
                                double      radius,
                                long        n,
                                double     *ra,
                                double     *ka,
                                double     *pa)
{
    imageID  ID;
    uint32_t ii, jj;
    uint32_t naxes[2];
    int      i, j;
    double   r;
    double   tot;
    int      subgrid = 55;
    double   grid[55]; // same number of points as subgrid
    double   x, y;
    long     x1, x2, y1, y2;
    long     x1i, x2i, y1i, y2i;
    double   r2, r2ref;
    double   xdiff, ydiff;
    double   subgrid2;
    double   PA;
    double   v0;
    long     k;

    double radius1, radius2;

    radius1 = radius;
    radius2 = radius;
    for (k = 0; k < n; k++)
    {
        radius1 += radius * fabs(ra[k]);
    }
    for (k = 0; k < n; k++)
    {
        radius2 -= radius * fabs(ra[k]);
    }
    if (radius2 < 0.0)
    {
        radius2 = 0.0;
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1  = (long) (x_center - radius1 - 2);
    x2  = (long) (x_center + radius1 + 2);
    y1  = (long) (y_center - radius1 - 2);
    y2  = (long) (y_center + radius1 + 2);
    x1i = (long) (x_center - 0.707106781 * radius2 + 2);
    x2i = (long) (x_center + 0.707106781 * radius2 - 2);
    y1i = (long) (y_center - 0.707106781 * radius2 + 2);
    y2i = (long) (y_center + 0.707106781 * radius2 - 2);

    if (x1 < 0)
    {
        x1 = 0;
    }
    if (x1 > naxes[0])
    {
        x1 = naxes[0];
    }
    if (x2 < 0)
    {
        x2 = 0;
    }
    if (x2 > naxes[0])
    {
        x2 = naxes[0];
    }

    if (y1 < 0)
    {
        y1 = 0;
    }
    if (y1 > naxes[1])
    {
        y1 = naxes[1];
    }
    if (y2 < 0)
    {
        y2 = 0;
    }
    if (y2 > naxes[1])
    {
        y2 = naxes[1];
    }

    if (x1i < 0)
    {
        x1i = 0;
    }
    if (x1i > naxes[0] - 1)
    {
        x1i = naxes[0] - 1;
    }
    if (x2i < 0)
    {
        x2i = 0;
    }
    if (x2i > naxes[0] - 1)
    {
        x2i = naxes[0] - 1;
    }

    if (y1i < 0)
    {
        y1i = 0;
    }
    if (y1i > naxes[1] - 1)
    {
        y1i = naxes[1] - 1;
    }
    if (y2i < 0)
    {
        y2i = 0;
    }
    if (y2i > naxes[1] - 1)
    {
        y2i = naxes[1] - 1;
    }

    r2ref    = radius * radius;
    subgrid2 = subgrid * subgrid;

    for (ii = x1i; ii < x2i; ii++)
    {
        for (jj = y1i; jj < y2i; jj++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
        }
    }

    for (i = 0; i < subgrid; i++)
    {
        grid[i] = (0.5 - 0.5 / subgrid - 1.0 * i / subgrid);
    }

    for (ii = x1; ii < x1i; ii++)
    {
        for (jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;

            v0 = radius;
            for (k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if (r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if (fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for (j = 0; j < subgrid; j++)
                {
                    for (i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;

                        v0 = radius;
                        for (k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;

                        if (r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                }
                tot                                   = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }
    }

    for (ii = x2i; ii < x2; ii++)
    {
        for (jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;

            v0 = radius;
            for (k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if (r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if (fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for (j = 0; j < subgrid; j++)
                {
                    for (i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        r  = x * x + y * y;
                        PA = atan2(y, x);
                        v0 = radius;
                        for (k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;

                        if (r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                }
                tot                                   = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }
    }

    for (ii = x1i; ii < x2i; ii++)
    {
        for (jj = y1; jj < y1i; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;
            v0    = radius;
            for (k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if (r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if (fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for (j = 0; j < subgrid; j++)
                {
                    for (i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;
                        v0 = radius;
                        for (k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;
                        if (r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                }
                tot                                   = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }
    }

    for (ii = x1i; ii < x2i; ii++)
    {
        for (jj = y2i; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;
            v0    = radius;
            for (k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if (r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if (fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for (j = 0; j < subgrid; j++)
                {
                    for (i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;
                        v0 = radius;
                        for (k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;
                        if (r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                }
                tot                                   = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }
    }

    return (ID);
}

/* creates a square */
imageID make_square(const char *ID_name,
                    uint32_t    l1,
                    uint32_t    l2,
                    double      x_center,
                    double      y_center,
                    double      radius)
{
    return make_rectangle(ID_name, l1, l2, x_center, y_center, radius, radius);
}

imageID make_rectangle(const char *ID_name,
                       uint32_t    l1,
                       uint32_t    l2,
                       double      x_center,
                       double      y_center,
                       double      radius1,
                       double      radius2)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    const double         r1sq = radius1 * radius1;
    const double         r2sq = radius2 * radius2;
    float *MILK_RESTRICT out  = MILK_ASSUME_ALIGNED(dcimg[ID].array.F);
    int                  nth  = milk_omp_team_size((uint64_t) naxes[0] * naxes[1], 262144);

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        double dy = (double) jj - y_center;
        if (dy * dy < r2sq)
        {
            float *row = out + (uint64_t) jj * naxes[0];
#pragma omp simd
            for (uint32_t ii = 0; ii < naxes[0]; ii++)
            {
                double dx = (double) ii - x_center;
                if (dx * dx < r1sq)
                {
                    row[ii] = 1.0f;
                }
            }
        }
    }

    return (ID);
}

// line of thickness t from (x1,y1) to (x2,y2)
imageID make_line(const char *IDname,
                  uint32_t    l1,
                  uint32_t    l2,
                  double      x1,
                  double      y1,
                  double      x2,
                  double      y2,
                  double      t)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(IDname, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    const double         r0     = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    const double         PA0    = atan2((y2 - y1), (x2 - x1));
    const double         inv_r0 = (r0 != 0.0) ? (1.0 / r0) : 0.0;
    const double         c_r0   = cos(PA0) * inv_r0;
    const double         s_r0   = sin(PA0) * inv_r0;
    const double         half_t = 0.5 * t * inv_r0;
    float *MILK_RESTRICT out    = MILK_ASSUME_ALIGNED(dcimg[ID].array.F);
    int                  nth    = milk_omp_team_size((uint64_t) naxes[0] * naxes[1], 262144);

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        float *row = out + (uint64_t) jj * naxes[0];
        double y   = (double) jj - y1;
#pragma omp simd
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            double x  = (double) ii - x1;
            double xr = x * c_r0 + y * s_r0;
            double yr = -x * s_r0 + y * c_r0;
            row[ii]   = (xr > 0.0 && xr < 1.0 && yr < half_t && yr > -half_t) ? 1.0f : 0.0f;
        }
    }

    return (ID);
}
