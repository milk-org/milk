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
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
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


// draw line crossing point xc, yc with angle, pixel value is coordinate axis perp to line
imageID make_lincoordinate(const char *IDname,
                           uint32_t    l1,
                           uint32_t    l2,
                           double      x_center,
                           double      y_center,
                           double      angle)
{
    imageID  ID;
    uint32_t naxes[2];
    double   x, y, x1;

    create_2Dimage_ID(IDname, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            x  = 1.0 * ii - x_center;
            y  = 1.0 * jj - y_center;
            x1 = x * cos(angle) + y * sin(angle);
            //y1 = -x*sin(angle) + y*cos(angle);
            dcimg[ID].array.F[jj * naxes[0] + ii] = x1;
        }
    }

    return (ID);
}

imageID make_hexagon(const char *IDname,
                     uint32_t    l1,
                     uint32_t    l2,
                     double      x_center,
                     double      y_center,
                     double      radius)
{
    imageID  ID;
    uint32_t ii, jj;
    uint32_t naxes[2];
    float    x, y, r;
    float    value;

    long  iimin, iimax, jjmin, jjmax;
    float radius1, radius0sq;

    radius1   = radius * 2.0 / sqrt(3.0);
    radius0sq = radius * radius;

    printf("Making hexagon at %f x %f\n", x_center, y_center);

    create_2Dimage_ID(IDname, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    iimin = (long) (x_center - radius1 - 1.0);
    if (iimin < 0)
    {
        iimin = 0;
    }
    if (iimin > l1 - 1)
    {
        iimin = l1 - 1;
    }

    iimax = (long) (x_center + radius1 + 1.0);
    if (iimax < 0)
    {
        iimax = 0;
    }
    if (iimax > l1 - 1)
    {
        iimax = l1 - 1;
    }

    jjmin = (long) (y_center - radius1 - 1.0);
    if (jjmin < 0)
    {
        jjmin = 0;
    }
    if (jjmin > l2 - 1)
    {
        jjmin = l2 - 1;
    }

    jjmax = (long) (y_center + radius1 + 1.0);
    if (jjmax < 0)
    {
        jjmax = 0;
    }
    if (jjmax > l2 - 1)
    {
        jjmax = l2 - 1;
    }

#ifdef HAVE_LIBGOMP
#    pragma omp parallel default(shared) private(ii, jj, value, x, y, r)
    {
#    pragma omp for simd
#endif

        for (jj = jjmin; jj < jjmax; jj++)
        {
            for (ii = iimin; ii < iimax; ii++)
            {
                value = 1.0;
                x     = 1.0 * ii - x_center;
                y     = 1.0 * jj - y_center;

                if (x * x + y * y > radius0sq)
                {
                    r = y;
                    if (fabs(r) > radius)
                    {
                        value = 0.0;
                    }
                    else
                    {
                        r = cos(PI / 6.0) * x + sin(PI / 6.0) * y;
                        if (fabs(r) > radius)
                        {
                            value = 0.0;
                        }
                        else
                        {
                            r = cos(-PI / 6.0) * x + sin(-PI / 6.0) * y;
                            if (fabs(r) > radius)
                            {
                                value = 0.0;
                            }
                        }
                    }
                }
                dcimg[ID].array.F[jj * naxes[0] + ii] = value;
            }
        }
#ifdef HAVE_LIBGOMP
    }
#endif

    return (ID);
}

/**
 * @brief Create a regular polygon mask image.
 *
 * Generates a binary mask for a regular N-sided polygon
 * inscribed in a circle of the given radius.  The
 * geometry uses half-plane intersection: a pixel is
 * inside the polygon iff it lies on the interior side
 * of every edge.
 *
 * @param ID_name       Output image name
 * @param l1            Image width (pixels)
 * @param l2            Image height (pixels)
 * @param x_center      Polygon center X coordinate
 * @param y_center      Polygon center Y coordinate
 * @param radius        Circumscribed circle radius
 * @param nsides        Number of sides (>= 3)
 * @param rotation_angle Rotation angle in radians
 *
 * @return Image ID of the created image
 */
imageID make_polygon(const char *ID_name,
                     uint32_t    l1,
                     uint32_t    l2,
                     double      x_center,
                     double      y_center,
                     double      radius,
                     int32_t     nsides,
                     double      rotation_angle)
{
    imageID  ID;
    uint32_t naxes[2];

    if (nsides < 3)
    {
        nsides = 3;
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    /* Pre-compute edge normals and apothem.
     * For a regular N-gon inscribed in a circle of
     * radius R, each edge k has an outward normal at
     * angle = rotation + 2*pi*k/N.  The perpendicular
     * distance from center to each edge (apothem) is
     * R * cos(pi/N). */
    double apothem = radius * cos(M_PI / nsides);

    double *nx = (double *) malloc((size_t) nsides * sizeof(double));
    double *ny = (double *) malloc((size_t) nsides * sizeof(double));

    {
        double dangle = 2.0 * M_PI / nsides;
        for (int32_t k = 0; k < nsides; k++)
        {
            double a = rotation_angle + dangle * k;
            nx[k]    = cos(a);
            ny[k]    = sin(a);
        }
    }

    /* Bounding box: circumscribed circle + 1 pixel */
    long iimin = (long) (x_center - radius - 1.0);
    long iimax = (long) (x_center + radius + 1.0);
    long jjmin = (long) (y_center - radius - 1.0);
    long jjmax = (long) (y_center + radius + 1.0);

    if (iimin < 0)
    {
        iimin = 0;
    }
    if (iimax > (long) naxes[0] - 1)
    {
        iimax = (long) naxes[0] - 1;
    }
    if (jjmin < 0)
    {
        jjmin = 0;
    }
    if (jjmax > (long) naxes[1] - 1)
    {
        jjmax = (long) naxes[1] - 1;
    }

    for (long jj = jjmin; jj <= jjmax; jj++)
    {
        double dy = (double) jj - y_center;
        for (long ii = iimin; ii <= iimax; ii++)
        {
            double dx     = (double) ii - x_center;
            int    inside = 1;
            for (int32_t k = 0; k < nsides; k++)
            {
                double dot = nx[k] * dx + ny[k] * dy;
                if (dot > apothem)
                {
                    inside = 0;
                    break;
                }
            }
            if (inside)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
        }
    }

    free(nx);
    free(ny);

    return (ID);
}

imageID IMAGE_gen_segments2WFmodes(const char *prefix, long ndigit, const char *IDout_name)
{
    imageID IDout = -1;
    long    NBseg;
    long    seg;
    int     OK;
    char    imname[200];
    imageID IDarray[10000];
    long    ii, jj, kk, xsize, ysize, xysize;
    double  x, y;
    imageID IDmask;
    double *segxc;
    double *segyc;
    double *segsum;

    seg = 0;
    OK  = 1;
    while (OK == 1)
    {
        switch (ndigit)
        {
        case 1:
            snprintf(imname, sizeof(imname), "%s%01ld", prefix, seg);
            break;
        case 2:
            snprintf(imname, sizeof(imname), "%s%02ld", prefix, seg);
            break;
        case 3:
            snprintf(imname, sizeof(imname), "%s%03ld", prefix, seg);
            break;
        case 4:
            snprintf(imname, sizeof(imname), "%s%04ld", prefix, seg);
            break;
        case 5:
            snprintf(imname, sizeof(imname), "%s%05ld", prefix, seg);
            break;
        case 6:
            snprintf(imname, sizeof(imname), "%s%06ld", prefix, seg);
            break;

        default:
            printf("ERROR: Invalid number of didits\n");
            exit(0);
        }
        IDarray[seg] = image_ID(imname, dcimg, dcnimg);
        if (IDarray[seg] != -1)
        {
            seg++;
        }
        else
        {
            OK = 0;
        }
    }
    NBseg = seg;
    printf("Processing %ld segments\n", NBseg);
    if (NBseg > 0)
    {
        xsize  = dcimg[IDarray[0]].md[0].size[0];
        ysize  = dcimg[IDarray[0]].md[0].size[1];
        xysize = xsize * ysize;

        segxc = (double *) malloc(sizeof(double) * NBseg);
        if (segxc == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        segyc = (double *) malloc(sizeof(double) * NBseg);
        if (segyc == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        segsum = (double *) malloc(sizeof(double) * NBseg);
        if (segsum == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        create_2Dimage_ID("_pupmask", xsize, ysize, &IDmask);

        for (seg = 0; seg < NBseg; seg++)
        {
            segxc[seg]  = 0.0;
            segyc[seg]  = 0.0;
            segsum[seg] = 0.0;

            for (ii = 0; ii < xsize; ii++)
            {
                for (jj = 0; jj < ysize; jj++)
                {
                    x = 1.0 * ii;
                    y = 1.0 * jj;
                    segxc[seg] += x * dcimg[IDarray[seg]].array.F[jj * xsize + ii];
                    segyc[seg] += y * dcimg[IDarray[seg]].array.F[jj * xsize + ii];
                    segsum[seg] += dcimg[IDarray[seg]].array.F[jj * xsize + ii];

                    dcimg[IDmask].array.F[jj * xsize + ii] +=
                        (1.0 + seg) * dcimg[IDarray[seg]].array.F[jj * xsize + ii];
                }
            }
            segxc[seg] /= segsum[seg];
            segyc[seg] /= segsum[seg];
        }

#ifdef USE_CFITSIO
        save_fits("_pupmask", "_pupmask.fits");
#endif

        //IDtmp = create_2Dimage_ID("_seg2wfm_tmp", xsize, ysize);
        create_3Dimage_ID(IDout_name, xsize, ysize, 3 * NBseg, &IDout);
        kk = 0;
        for (seg = 0; seg < NBseg; seg++) // create modes one at a time
        {
            // piston seg
            for (ii = 0; ii < xsize; ii++)
            {
                for (jj = 0; jj < xsize; jj++)
                {
                    dcimg[IDout].array.F[kk * xysize + jj * xsize + ii] =
                        dcimg[IDarray[seg]].array.F[jj * xsize + ii];
                }
            }
            kk++;

            // Tip
            for (ii = 0; ii < xsize; ii++)
            {
                for (jj = 0; jj < xsize; jj++)
                {
                    dcimg[IDout].array.F[kk * xysize + jj * xsize + ii] =
                        dcimg[IDarray[seg]].array.F[jj * xsize + ii] * (1.0 * ii - segxc[seg]);
                }
            }
            kk++;

            // Tilt
            for (ii = 0; ii < xsize; ii++)
            {
                for (jj = 0; jj < xsize; jj++)
                {
                    dcimg[IDout].array.F[kk * xysize + jj * xsize + ii] =
                        dcimg[IDarray[seg]].array.F[jj * xsize + ii] * (1.0 * jj - segyc[seg]);
                }
            }
            kk++;
        }

        //delete_image_ID("_seg2wfm_tmp", DELETE_IMAGE_ERRMODE_WARNING);

        free(segxc);
        free(segyc);
        free(segsum);
    }

    return (IDout);
}

imageID make_hexsegpupil(const char *IDname, uint32_t size, double radius, double gap, double step)
{
    imageID  ID, ID1, IDp;
    long     x1, y1;
    double   x2, y2;
    imageID  IDdisk;
    uint32_t ii;
    double   tot = 0.0;
    long     size2;

    int    PISTONerr   = 0;
    int    errSEGindex = -1;
    double pampl;
    double piston;
    long   SEGcnt = 0;

    int   mkInfluenceFunctions = 1;
    long  IDif;
    int   seg;
    long  kk, jj;
    float xc, yc, tc;

    int    WriteCIF = 0;
    FILE  *fpmlevel;
    FILE  *fp       = NULL;
    FILE  *fp1      = NULL;
    double pixscale = 1.0;
    long   vID;
    double x, y;
    int    pt;

    long   IDmap1;
    long   index;
    double mapscalefactor = 1.037;
    long   size1;

    long *seglevel;
    long  i;
    long  tmpl1, tmpl2;
    int   segi;
    float segf;
    int   k;

    int *bitval;       // 0 or 1
    int  bitindex = 4; // 0 = MSB

    double vx, vy, rmsx, rmsy;

    if (WriteCIF == 1)
    {
        fp  = fopen("hexcoord.txt", "w");
        fp1 = fopen("hexcoord_pt.txt", "w");

        fprintf(fp, "DS 1 1 1;\n");
    }

    if ((vID = variable_ID("pixscale")) != -1)
    {
        pixscale = dcvar[vID].value.f;
        printf("pixscale = %f\n", pixscale);
    }

    SEGcnt = 100;
    if ((vID = variable_ID("SEGcnt")) != -1)
    {
        SEGcnt = (long) (0.1 + dcvar[vID].value.f);
        printf("SEGcnt = %ld\n", SEGcnt);
    }

    seglevel = (long *) malloc(sizeof(long) * SEGcnt);
    if (seglevel == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    bitval = (int *) malloc(sizeof(int) * SEGcnt);
    if (bitval == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    fpmlevel = fopen("fpm_level.txt", "r");
    if (fpmlevel != NULL)
    {
        for (i = 0; i < SEGcnt; i++)
        {
            int fscanfcnt = fscanf(fpmlevel, "%ld %ld\n", &tmpl1, &tmpl2);
            if (fscanfcnt == EOF)
            {
                if (ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr, "Error: fscanf reached end of file, no matching "
                                    "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if (fscanfcnt != 2)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i "
                        "input items, 2 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }

            seglevel[tmpl1 - 1] = tmpl2 + 15;
        }
        fclose(fpmlevel);
    }

    // SINGLE BIT
    for (i = 0; i < SEGcnt; i++)
    {
        printf("%5ld %5ld   ", i + 1, seglevel[i]);
        segf = 1.0 * seglevel[i] / 16.0;
        for (k = 0; k < 5; k++)
        {
            segi = (int) segf;
            printf(" %d", segi);
            segf -= segi;
            segf *= 2;

            if (k == bitindex)
            {
                bitval[i] = segi;
            }
        }
        printf("\n");
    }

    IDmap1 = image_ID("indexmap", dcimg, dcnimg);
    size1  = dcimg[IDmap1].md[0].size[0];

    size2 = size * size;

    ID = variable_ID("hexpupnoif");
    if (ID != -1)
    {
        mkInfluenceFunctions = 0;
    }

    ID = variable_ID("HEXPISTONerr");
    if (ID != -1)
    {
        PISTONerr = 1;
        pampl     = dcvar[ID].value.f;
        printf("Piston error = %f\n", pampl);
    }
    else
    {
        pampl = 0.0;
    }

    ID = variable_ID("HEXPISTONindex");
    if (ID != -1)
    {
        errSEGindex = (long) (dcvar[ID].value.f + 0.01);
        printf("SEGMENT INDEX = %ld\n", (long) errSEGindex);
    }

    create_2Dimage_ID(IDname, size, size, &ID);
    if (PISTONerr == 1)
    {
        create_2Dimage_ID("hexpupPha", size, size, &IDp);
    }

    IDdisk = make_disk("_TMPdisk", size, size, size / 2, size / 2, radius);
    for (ii = 0; ii < size2; ii++)
    {
        dcimg[IDdisk].array.F[ii] = 1.0f - dcimg[IDdisk].array.F[ii];
    }

    SEGcnt = 0;
    for (x1 = -(long) (2 * size / step); x1 < (long) (2 * size / step); x1++)
    {
        for (y1 = -(long) (2 * size / step); y1 < (long) (2 * size / step); y1++)
        {
            x2 = step * x1 * 3;
            y2 = step * sqrt(3.0) * y1;

            if (sqrt(x2 * x2 + y2 * y2) < radius)
            {
                if (errSEGindex == -1)
                {
                    piston = pampl * (1.0 - 2.0 * ran1());
                }
                else
                {
                    if (errSEGindex == SEGcnt)
                    {
                        piston = pampl;
                    }
                    else
                    {
                        piston = 0.0;
                    }
                }
                printf("Hexagon %ld: ", SEGcnt);
                ID1 = make_hexagon("_TMPhex", size, size, 0.5 * size + x2, 0.5 * size + y2,
                                   (step - gap) * (sqrt(3.0) / 2.0));

                tot = 0.0;
                for (ii = 0; ii < size2; ii++)
                {
                    tot += dcimg[ID1].array.F[ii] * dcimg[IDdisk].array.F[ii];
                }
                if (tot < 0.1)
                {
                    SEGcnt++;
                    if (WriteCIF == 1)
                    {
                        ii    = (long) (0.5 * size1 + x2 * (0.5 * size1 / radius) * mapscalefactor);
                        jj    = (long) (0.5 * size1 + y2 * (0.5 * size1 / radius) * mapscalefactor);
                        index = 0;
                        if (IDmap1 != -1)
                        {
                            index = dcimg[IDmap1].array.UI16[jj * size1 + ii];
                        }

                        //  fprintf(fp, "# hex%03ld     index%03ld   [ %f %f ] -> [ %f %f ]     [%4ld %4ld] %f\n", SEGcnt, index, x2, y2, 0.5*size+x2, 0.5*size+y2, ii, jj, radius);
                        if (bitval[index - 1] == 1)
                        {
                            fprintf(fp, "L %ld;\n", seglevel[index - 1]);
                            fprintf(fp, "P");
                            for (pt = 0; pt < 6; pt++)
                            {
                                x = pixscale * (x2 + 1.0 * cos(2.0 * M_PI * pt / 6) * (step - gap));
                                y = pixscale * (y2 + 1.0 * sin(2.0 * M_PI * pt / 6) * (step - gap));
                                fprintf(fp, " %ld,%ld", (long) (100.0 * x), (long) (100.0 * y));
                                fprintf(fp1, "%ld %ld\n", (long) (100.0 * x), (long) (100.0 * y));
                            }
                            fprintf(fp, ";\n");
                        }
                    }

                    if (PISTONerr == 1)
                    {
                        for (ii = 0; ii < size2; ii++)
                        {
                            dcimg[ID].array.F[ii] += dcimg[ID1].array.F[ii];
                        }
                    }
                    else
                    {
                        for (ii = 0; ii < size2; ii++)
                        {
                            dcimg[ID].array.F[ii] += 1.0f * SEGcnt * dcimg[ID1].array.F[ii];
                        }
                    }

                    if (PISTONerr == 1)
                    {
                        for (ii = 0; ii < size2; ii++)
                        {
                            dcimg[IDp].array.F[ii] += dcimg[ID1].array.F[ii] * piston;
                        }
                    }
                }
                delete_image_ID("_TMPhex", DELETE_IMAGE_ERRMODE_WARNING);
            }

            x2 += step * 1.5;
            y2 += step * sqrt(3.0) / 2.0;
            if (sqrt(x2 * x2 + y2 * y2) < radius)
            {
                // piston = pampl*(1.0-2.0*ran1());
                if (errSEGindex == -1)
                {
                    piston = pampl * (1.0 - 2.0 * ran1());
                }
                else
                {
                    if (errSEGindex == SEGcnt)
                    {
                        piston = pampl;
                    }
                    else
                    {
                        piston = 0.0;
                    }
                }
                printf("Hexagon %ld: ", SEGcnt);
                ID1 = make_hexagon("_TMPhex", size, size, 0.5 * size + x2, 0.5 * size + y2,
                                   (step - gap) * (sqrt(3.0) / 2.0));
                tot = 0.0;
                for (ii = 0; ii < size2; ii++)
                {
                    tot += dcimg[ID1].array.F[ii] * dcimg[IDdisk].array.F[ii];
                }
                if (tot < 0.1)
                {
                    SEGcnt++;

                    if (WriteCIF == 1)
                    {
                        ii    = (long) (0.5 * size1 + x2 * (0.5 * size1 / radius) * mapscalefactor);
                        jj    = (long) (0.5 * size1 + y2 * (0.5 * size1 / radius) * mapscalefactor);
                        index = 0;
                        if (IDmap1 != -1)
                        {
                            index = dcimg[IDmap1].array.UI16[jj * size1 + ii];
                        }

                        // fprintf(fp, "# hex%03ld     index%03ld   [ %f %f ] -> [ %f %f ]   [%4ld %4ld] %f\n", SEGcnt, index, x2, y2, 0.5*size+x2, 0.5*size+y2, ii, jj, radius);

                        if (bitval[index - 1] == 1)
                        {
                            fprintf(fp, "L %ld;\n", seglevel[index - 1]);
                            fprintf(fp, "P");
                            for (pt = 0; pt < 6; pt++)
                            {
                                x = pixscale * (x2 + 1.0 * cos(2.0 * M_PI * pt / 6) * (step - gap));
                                y = pixscale * (y2 + 1.0 * sin(2.0 * M_PI * pt / 6) * (step - gap));
                                fprintf(fp, " %ld,%ld", (long) (100.0 * x), (long) (100.0 * y));
                                fprintf(fp1, "%ld %ld\n", (long) (100.0 * x), (long) (100.0 * y));
                            }
                            fprintf(fp, ";\n");
                        }
                    }

                    if (PISTONerr == 1)
                    {
                        for (ii = 0; ii < size2; ii++)
                        {
                            dcimg[ID].array.F[ii] += dcimg[ID1].array.F[ii];
                        }
                    }
                    else
                    {
                        for (ii = 0; ii < size2; ii++)
                        {
                            dcimg[ID].array.F[ii] += 1.0f * SEGcnt * dcimg[ID1].array.F[ii];
                        }
                    }

                    if (PISTONerr == 1)
                    {
                        for (ii = 0; ii < size2; ii++)
                        {
                            dcimg[IDp].array.F[ii] += dcimg[ID1].array.F[ii] * piston;
                        }
                    }
                }
                delete_image_ID("_TMPhex", DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
    }
    delete_image_ID("_TMPdisk", DELETE_IMAGE_ERRMODE_WARNING);

    printf("%ld segments\n", SEGcnt);

    if (WriteCIF == 1)
    {
        fprintf(fp, "DF;\n");
        fprintf(fp, "E\n");

        fclose(fp);
        fclose(fp1);
    }
    free(seglevel);
    free(bitval);

    if (mkInfluenceFunctions == 1) // TT and focus for each segment
    {
        create_3Dimage_ID("hexpupif", size, size, 3 * SEGcnt, &IDif);
        for (seg = 0; seg < SEGcnt; seg++)
        {
            // piston
            kk = 3 * seg;
            xc = 0.0;
            yc = 0.0;
            tc = 0.0;
            for (ii = 0; ii < size; ii++)
            {
                for (jj = 0; jj < size; jj++)
                {
                    if (fabsf(dcimg[ID].array.F[jj * size + ii] - (seg + 1.0f)) < 0.01f)
                    {
                        dcimg[IDif].array.F[kk * size2 + jj * size + ii] = 1.0;
                        xc += 1.0 * ii;
                        yc += 1.0 * jj;
                        tc += 1.0;
                    }
                }
            }
            xc /= tc;
            yc /= tc;

            // tip and tilt
            rmsx = 0.0;
            rmsy = 0.0;
            for (ii = 0; ii < size; ii++)
            {
                for (jj = 0; jj < size; jj++)
                {
                    if (fabsf(dcimg[ID].array.F[jj * size + ii] - (seg + 1.0f)) < 0.01f)
                    {
                        vx                                                     = 1.0 * ii - xc;
                        dcimg[IDif].array.F[(kk + 1) * size2 + jj * size + ii] = vx;
                        rmsx += vx * vx;

                        vy                                                     = 1.0 * jj - yc;
                        dcimg[IDif].array.F[(kk + 2) * size2 + jj * size + ii] = vy;
                        rmsy += vy * vy;
                    }
                }
            }
            for (ii = 0; ii < size2; ii++)
            {
                dcimg[IDif].array.F[(kk + 1) * size2 + ii] *= sqrt(tc / rmsx);
                dcimg[IDif].array.F[(kk + 2) * size2 + ii] *= sqrt(tc / rmsy);
            }
        }
    }

    return (ID);
}
