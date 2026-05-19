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
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/create_image.h"
#include "COREMOD_tools/COREMOD_tools.h" 
#include "statistic/statistic.h"
#include "ImageStreamIO/ImageStreamIO.h"

#include "image_gen.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>


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

    dcimg[ID]
    .array.F[((int)(naxes[1] / 2)) * naxes[0] + ((int)(naxes[0] / 2))] =
        intensity_1;
    dcimg[ID]
    .array.F[((int)(naxes[1] / 2 + separation * cos(position_angle))) *
                             naxes[0] +
                             ((int)(naxes[0] / 2 + separation * sin(position_angle)))] =
                 intensity_2;

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
    uint32_t ii, jj;
    uint32_t naxes[2];
    long     x1, x2, y1, y2;
    long     x1i, x2i, y1i, y2i;
    double   r2;
    /*
      int i,j;
      double r;
      double tot;
      int subgrid=100;
      double x,y;
    */

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1  = (long)(x_center - radius - 2);
    x2  = (long)(x_center + radius + 2);
    y1  = (long)(y_center - radius - 2);
    y2  = (long)(y_center + radius + 2);
    x1i = (long)(x_center - 0.707106781 * radius + 2);
    x2i = (long)(x_center + 0.707106781 * radius - 2);
    y1i = (long)(y_center - 0.707106781 * radius + 2);
    y2i = (long)(y_center + 0.707106781 * radius - 2);

    if(x1 < 0)
    {
        x1 = 0;
    }
    if(x1 > naxes[0])
    {
        x1 = naxes[0];
    }

    if(x2 < 0)
    {
        x2 = 0;
    }
    if(x2 > naxes[0])
    {
        x2 = naxes[0];
    }

    if(y1 < 0)
    {
        y1 = 0;
    }
    if(y1 > naxes[1])
    {
        y1 = naxes[1];
    }

    if(y2 > naxes[1])
    {
        y2 = naxes[1];
    }

    if(x1i < 0)
    {
        x1i = 0;
    }
    if(x1i > naxes[0])
    {
        x1i = naxes[0];
    }

    if(x2i < 0)
    {
        x2i = 0;
    }
    if(x2i > naxes[0])
    {
        x2i = naxes[0];
    }

    if(y1i < 0)
    {
        y1i = 0;
    }
    if(y1i > naxes[1])
    {
        y1i = naxes[1];
    }

    if(y2i < 0)
    {
        y2i = 0;
    }
    if(y2i > naxes[1])
    {
        y2i = naxes[1];
    }

    r2 = radius * radius;

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1i; jj < y2i; jj++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
        }

    for(ii = x1; ii < x1i; ii++)
        for(jj = y1; jj < y2; jj++)
            if(((ii - x_center) * (ii - x_center) +
                    (jj - y_center) * (jj - y_center)) < r2)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }

    for(ii = x2i; ii < x2; ii++)
        for(jj = y1; jj < y2; jj++)
            if(((ii - x_center) * (ii - x_center) +
                    (jj - y_center) * (jj - y_center)) < r2)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1; jj < y1i; jj++)
            if(((ii - x_center) * (ii - x_center) +
                    (jj - y_center) * (jj - y_center)) < r2)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y2i; jj < y2; jj++)
            if(((ii - x_center) * (ii - x_center) +
                    (jj - y_center) * (jj - y_center)) < r2)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }

    /*
    for (jj = x1; jj < x2; jj++)
      for (ii = y1; ii < y2; ii++)
        {
    if (((ii-x_center)*(ii-x_center)+(jj-y_center)*(jj-y_center))<r2)
      dcimg[ID].array.F[jj*naxes[0]+ii] = 1;
        }
    */
    /*
      for (jj = 0; jj < naxes[1]; jj++)
      for (ii = 0; ii < naxes[0]; ii++)
      {
      r = sqrt(((ii-x_center)*(ii-x_center)+(jj-y_center)*(jj-y_center)));

      if (r<radius)
      dcimg[ID].array.F[jj*naxes[0]+ii] = 1.0f;
      else
      dcimg[ID].array.F[jj*naxes[0]+ii] = 0.0f;

      if(((radius-r)*(radius-r))<1.5)
      {
      tot = 0;
      for (j = 0; j < subgrid; j++)
      for (i = 0; i < subgrid; i++)
      {
      x = 1.0*ii-0.5+0.5/subgrid+1.0*i/subgrid;
      y = 1.0*jj-0.5+0.5/subgrid+1.0*j/subgrid;
      r = sqrt((x-1.0*x_center)*(x-1.0*x_center)+(y-1.0*y_center)*(y-1.0*y_center));
      if (r < radius)
      tot = tot + 1.0;
      else
      tot = tot + 0.0;
      }
      tot = tot/subgrid/subgrid;
      dcimg[ID].array.F[jj*naxes[0]+ii] = tot;
      }
      }
    */
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

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1  = (long)(x_center - radius - 2);
    x2  = (long)(x_center + radius + 2);
    y1  = (long)(y_center - radius - 2);
    y2  = (long)(y_center + radius + 2);
    x1i = (long)(x_center - 0.707106781 * radius + 2);
    x2i = (long)(x_center + 0.707106781 * radius - 2);
    y1i = (long)(y_center - 0.707106781 * radius + 2);
    y2i = (long)(y_center + 0.707106781 * radius - 2);

    if(x1 < 0)
    {
        x1 = 0;
    }
    if(x1 > naxes[0])
    {
        x1 = naxes[0];
    }
    if(x2 < 0)
    {
        x2 = 0;
    }
    if(x2 > naxes[0])
    {
        x2 = naxes[0];
    }

    if(y1 < 0)
    {
        y1 = 0;
    }
    if(y1 > naxes[1])
    {
        y1 = naxes[1];
    }
    if(y2 < 0)
    {
        y2 = 0;
    }
    if(y2 > naxes[1])
    {
        y2 = naxes[1];
    }

    if(x1i < 0)
    {
        x1i = 0;
    }
    if(x1i > naxes[0] - 1)
    {
        x1i = naxes[0] - 1;
    }
    if(x2i < 0)
    {
        x2i = 0;
    }
    if(x2i > naxes[0] - 1)
    {
        x2i = naxes[0] - 1;
    }

    if(y1i < 0)
    {
        y1i = 0;
    }
    if(y1i > naxes[1] - 1)
    {
        y1i = naxes[1] - 1;
    }
    if(y2i < 0)
    {
        y2i = 0;
    }
    if(y2i > naxes[1] - 1)
    {
        y2i = naxes[1] - 1;
    }

    r2ref    = radius * radius;
    subgrid2 = subgrid * subgrid;

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1i; jj < y2i; jj++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
        }

    for(i = 0; i < subgrid; i++)
    {
        grid[i] = (0.5 - 0.5 / subgrid - 1.0 * i / subgrid);
    }

    for(ii = x1; ii < x1i; ii++)
        for(jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            r2    = xdiff * xdiff + ydiff * ydiff;
            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x = xdiff + grid[i];
                        y = ydiff + grid[j];
                        r = x * x + y * y;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x2i; ii < x2; ii++)
        for(jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            r2    = xdiff * xdiff + ydiff * ydiff;
            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x = xdiff + grid[i];
                        y = ydiff + grid[j];
                        r = x * x + y * y;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1; jj < y1i; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            r2    = xdiff * xdiff + ydiff * ydiff;
            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x = xdiff + grid[i];
                        y = ydiff + grid[j];
                        r = x * x + y * y;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y2i; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            r2    = xdiff * xdiff + ydiff * ydiff;
            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x = xdiff + grid[i];
                        y = ydiff + grid[j];
                        r = x * x + y * y;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
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
    for(k = 0; k < n; k++)
    {
        radius1 += radius * fabs(ra[k]);
    }
    for(k = 0; k < n; k++)
    {
        radius2 -= radius * fabs(ra[k]);
    }
    if(radius2 < 0.0)
    {
        radius2 = 0.0;
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1  = (long)(x_center - radius1 - 2);
    x2  = (long)(x_center + radius1 + 2);
    y1  = (long)(y_center - radius1 - 2);
    y2  = (long)(y_center + radius1 + 2);
    x1i = (long)(x_center - 0.707106781 * radius2 + 2);
    x2i = (long)(x_center + 0.707106781 * radius2 - 2);
    y1i = (long)(y_center - 0.707106781 * radius2 + 2);
    y2i = (long)(y_center + 0.707106781 * radius2 - 2);

    if(x1 < 0)
    {
        x1 = 0;
    }
    if(x1 > naxes[0])
    {
        x1 = naxes[0];
    }
    if(x2 < 0)
    {
        x2 = 0;
    }
    if(x2 > naxes[0])
    {
        x2 = naxes[0];
    }

    if(y1 < 0)
    {
        y1 = 0;
    }
    if(y1 > naxes[1])
    {
        y1 = naxes[1];
    }
    if(y2 < 0)
    {
        y2 = 0;
    }
    if(y2 > naxes[1])
    {
        y2 = naxes[1];
    }

    if(x1i < 0)
    {
        x1i = 0;
    }
    if(x1i > naxes[0] - 1)
    {
        x1i = naxes[0] - 1;
    }
    if(x2i < 0)
    {
        x2i = 0;
    }
    if(x2i > naxes[0] - 1)
    {
        x2i = naxes[0] - 1;
    }

    if(y1i < 0)
    {
        y1i = 0;
    }
    if(y1i > naxes[1] - 1)
    {
        y1i = naxes[1] - 1;
    }
    if(y2i < 0)
    {
        y2i = 0;
    }
    if(y2i > naxes[1] - 1)
    {
        y2i = naxes[1] - 1;
    }

    r2ref    = radius * radius;
    subgrid2 = subgrid * subgrid;

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1i; jj < y2i; jj++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
        }

    for(i = 0; i < subgrid; i++)
    {
        grid[i] = (0.5 - 0.5 / subgrid - 1.0 * i / subgrid);
    }

    for(ii = x1; ii < x1i; ii++)
        for(jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;

            v0 = radius;
            for(k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;

                        v0 = radius;
                        for(k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;

                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x2i; ii < x2; ii++)
        for(jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;

            v0 = radius;
            for(k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        r  = x * x + y * y;
                        PA = atan2(y, x);
                        v0 = radius;
                        for(k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;

                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1; jj < y1i; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;
            v0    = radius;
            for(k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;
                        v0 = radius;
                        for(k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y2i; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;
            v0    = radius;
            for(k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;
                        v0 = radius;
                        for(k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
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
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            if((((ii - x_center) * (ii - x_center)) < (radius * radius)) &&
                    (((jj - y_center) * (jj - y_center)) < (radius * radius)))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }
        }

    return (ID);
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

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            if((((ii - x_center) * (ii - x_center)) < (radius1 * radius1)) &&
                    (((jj - y_center) * (jj - y_center)) < (radius2 * radius2)))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
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
    double   x, y, xr, yr, r0;
    double   PA0;
    uint32_t naxes[2];

    create_2Dimage_ID(IDname, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    r0  = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    PA0 = atan2((y2 - y1), (x2 - x1));
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            x = 1.0 * ii;
            y = 1.0 * jj;
            x -= x1;
            y -= y1;
            xr = x * cos(PA0) + y * sin(PA0);
            yr = -x * sin(PA0) + y * cos(PA0);
            //r=sqrt(xr*xr+yr*yr);
            xr /= r0;
            yr /= r0;
            if((xr > 0) && (xr < 1.0) && (yr < 0.5 * t / r0) &&
                    (yr > -0.5 * t / r0))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            else
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 0.0f;
            }
        }

    return (ID);
}
