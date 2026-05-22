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


imageID make_psf_from_profile(const char *profile_name,
                              const char *ID_name,
                              uint32_t    l1,
                              uint32_t    l2)
{
    imageID  ID;
    uint32_t naxes[2];
    FILE    *fp;
    long     nb_lines;
    char     lstring[1000];
    char     line[200];
    double  *distarr;
    double  *valarr;
    long     i;
    double   dist;
    float    fl1, fl2;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    /* compute number of lines */
    snprintf(lstring, sizeof(lstring), "wc -l %s > tmpcnt.txt", profile_name);
    if (system(lstring) == -1)
    {
        printf("ERROR: system(\"%s\"), %s line %d\n", lstring, __FILE__, __LINE__);
        exit(0);
    }
    if ((fp = fopen("tmpcnt.txt", "r")) == NULL)
    {
        printf("error : can't open file \"tmpcnt.txt\"\n");
    }
    if (fgets(line, 200, fp) == NULL)
    {
        printf("ERROR: fgets, %s line %d\n", __FILE__, __LINE__);
        exit(0);
    }
    fclose(fp);
    printf("%s\n", line);
    fflush(stdout);
    sscanf(line, "%ld %s", &nb_lines, lstring);

    printf("%ld lines\n", nb_lines);

    distarr = (double *) malloc(sizeof(double) * nb_lines);
    if (distarr == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    valarr = (double *) malloc(sizeof(double) * nb_lines);
    if (valarr == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if ((fp = fopen(profile_name, "r")) == NULL)
    {
        printf("error : can't open file \"%s\"\n", profile_name);
        abort();
    }

    for (i = 0; i < nb_lines; i++)
    {
        if (fgets(line, 200, fp) == NULL)
        {
            printf("ERROR: fgets, %s line %d\n", __FILE__, __LINE__);
            exit(0);
        }
        sscanf(line, "%f %f", &fl1, &fl2);
        distarr[i] = fl1;
        valarr[i]  = fl2;
    }
    fclose(fp);

    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dist = sqrt((ii - naxes[0] / 2) * (ii - naxes[0] / 2) +
                        (jj - naxes[1] / 2) * (jj - naxes[1] / 2));
            i    = 0;
            while ((distarr[i] < dist) && (i != nb_lines - 1))
            {
                i++;
            }
            if (i != 0)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] =
                    valarr[i - 1] + (valarr[i] - valarr[i - 1]) * (dist - distarr[i - 1]) /
                                        (distarr[i] - distarr[i - 1]);
            }
            else
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = valarr[0];
            }
        }
    }

    free(distarr);
    free(valarr);

    return (ID);
}

imageID make_offsetHyperGaussian(uint32_t size, double a, double b, long n, const char *IDname)
{
    imageID ID;

    create_2Dimage_ID(IDname, size, size, &ID);
    for (uint32_t ii = 0; ii < size; ii++)
    {
        for (uint32_t jj = 0; jj < size; jj++)
        {
            double x, y, dist;

            x    = 1.0 * ii - size / 2;
            y    = 1.0 * jj - size / 2;
            dist = sqrt(x * x + y * y);
            if (dist < a)
            {
                dcimg[ID].array.F[jj * size + ii] = 0.0f;
            }
            else
            {
                dcimg[ID].array.F[jj * size + ii] = 1.0f - expf(-powf((dist - a) / b, n));
            }
        }
    }

    return (ID);
}

imageID make_cosapoedgePupil(uint32_t size, double a, double b, const char *IDname)
{
    imageID ID;

    create_2Dimage_ID(IDname, size, size, &ID);
    for (uint32_t ii = 0; ii < size; ii++)
    {
        for (uint32_t jj = 0; jj < size; jj++)
        {
            double x, y, dist;

            x    = 1.0 * ii - size / 2;
            y    = 1.0 * jj - size / 2;
            dist = sqrt(x * x + y * y);
            if (dist < a)
            {
                dcimg[ID].array.F[jj * size + ii] = 1.0f;
            }
            else if (dist > b)
            {
                dcimg[ID].array.F[jj * size + ii] = 0.0f;
            }
            else
            {
                dcimg[ID].array.F[jj * size + ii] = 0.5 * (cos(PI * (dist - a) / (b - a)) + 1.0);
            }
        }
    }

    return ID;
}

// make square grid of pixels
imageID make_2Dgridpix(const char *IDname,
                       uint32_t    xsize,
                       uint32_t    ysize,
                       double      pitchx,
                       double      pitchy,
                       double      offsetx,
                       double      offsety)
{
    imageID ID;
    double  x, y;
    long    i, j;
    double  u, t;

    create_2Dimage_ID(IDname, xsize, ysize, &ID);
    for (x = offsetx; x < xsize - 1; x += pitchx)
    {
        for (y = offsety; y < ysize - 1; y += pitchy)
        {
            i                                          = (long) x;
            j                                          = (long) y;
            u                                          = x - i;
            t                                          = y - j;
            dcimg[ID].array.F[j * xsize + i]           = (1.0f - u) * (1.0f - t);
            dcimg[ID].array.F[(j + 1) * xsize + i]     = (1.0f - u) * t;
            dcimg[ID].array.F[j * xsize + i + 1]       = u * (1.0f - t);
            dcimg[ID].array.F[(j + 1) * xsize + i + 1] = u * t;
        }
    }

    return (ID);
}

// make tile
imageID make_tile(const char *IDin_name, uint32_t size, const char *IDout_name)
{
    uint32_t sizex0, sizey0; // input
    imageID  IDin, IDout;

    create_2Dimage_ID(IDout_name, size, size, &IDout);
    IDin   = image_ID(IDin_name, dcimg, dcnimg);
    sizex0 = dcimg[IDin].md[0].size[0];
    sizey0 = dcimg[IDin].md[0].size[1];

    for (uint32_t ii = 0; ii < size; ii++)
    {
        for (uint32_t jj = 0; jj < size; jj++)
        {
            uint32_t ii0                         = ii % sizex0;
            uint32_t jj0                         = jj % sizey0;
            dcimg[IDout].array.F[jj * size + ii] = dcimg[IDin].array.F[jj0 * sizex0 + ii0];
        }
    }

    return (IDout);
}

// make image that is coordinate of input
// for example, if axis = 0
// value = 1.0 x ii
// if axis value is not one of the coordinates, write pixel index
//
imageID image_gen_im2coord(const char *IDin_name, uint8_t axis, const char *IDout_name)
{
    uint8_t  naxis;
    int      OK = 1;
    imageID  IDin;
    imageID  IDout = -1;
    uint32_t xsize, ysize, zsize;

    IDin  = image_ID(IDin_name, dcimg, dcnimg);
    naxis = dcimg[IDin].md[0].naxis;

    if (axis > naxis - 1)
    {
        printf("Image has only %u axis, cannot access axis %u\n", naxis, axis);
        OK = 0;
    }

    if (naxis > 3)
    {
        printf("naxis should be 3 or less\n");
        OK = 0;
    }

    if (OK == 1)
    {
        if (naxis == 1)
        {
            printf("naxis = 1\n");
            fflush(stdout);
            xsize = dcimg[IDin].md[0].size[0];
            create_1Dimage_ID(IDout_name, xsize, &IDout);
            for (uint32_t ii = 0; ii < xsize; ii++)
            {
                dcimg[IDout].array.F[ii] = 1.0f * ii;
            }
        }

        if (naxis == 2)
        {
            printf("naxis = 2\n");
            fflush(stdout);
            xsize = dcimg[IDin].md[0].size[0];
            ysize = dcimg[IDin].md[0].size[1];
            create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);

            switch (axis)
            {
            case 0:
                for (uint32_t ii = 0; ii < xsize; ii++)
                {
                    for (uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.F[jj * xsize + ii] = 1.0f * ii;
                    }
                }
                break;
            case 1:
                for (uint32_t ii = 0; ii < xsize; ii++)
                {
                    for (uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.F[jj * xsize + ii] = 1.0f * jj;
                    }
                }
                break;
            default:
                for (uint32_t ii = 0; ii < xsize; ii++)
                {
                    for (uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.F[jj * xsize + ii] = 1.0 * jj * xsize + ii;
                    }
                }
            }
        }

        if (naxis == 3)
        {
            printf("naxis = 3\n");
            fflush(stdout);
            xsize = dcimg[IDin].md[0].size[0];
            ysize = dcimg[IDin].md[0].size[1];
            zsize = dcimg[IDin].md[0].size[2];
            create_3Dimage_ID(IDout_name, xsize, ysize, zsize, &IDout);

            switch (axis)
            {
            case 0:
                for (uint32_t ii = 0; ii < xsize; ii++)
                {
                    for (uint32_t jj = 0; jj < ysize; jj++)
                    {
                        for (uint32_t kk = 0; kk < zsize; kk++)
                        {
                            dcimg[IDout].array.F[kk * xsize * ysize + jj * xsize + ii] = 1.0 * ii;
                        }
                    }
                }
                break;
            case 1:
                for (uint32_t ii = 0; ii < xsize; ii++)
                {
                    for (uint32_t jj = 0; jj < ysize; jj++)
                    {
                        for (uint32_t kk = 0; kk < zsize; kk++)
                        {
                            dcimg[IDout].array.F[kk * xsize * ysize + jj * xsize + ii] = 1.0 * jj;
                        }
                    }
                }
                break;
            case 2:
                for (uint32_t ii = 0; ii < xsize; ii++)
                {
                    for (uint32_t jj = 0; jj < xsize; jj++)
                    {
                        for (uint32_t kk = 0; kk < zsize; kk++)
                        {
                            dcimg[IDout].array.F[kk * xsize * ysize + jj * xsize + ii] = 1.0 * kk;
                        }
                    }
                }
                break;
            default:
                for (uint32_t ii = 0; ii < xsize; ii++)
                {
                    for (uint32_t jj = 0; jj < xsize; jj++)
                    {
                        for (uint32_t kk = 0; kk < zsize; kk++)
                        {
                            dcimg[IDout].array.F[kk * xsize * ysize + jj * xsize + ii] =
                                1.0 * kk * xsize * ysize + jj * xsize + ii;
                        }
                    }
                }
            }
        }
    }

    return (IDout);
}
