/**
 * @file cubepercentile.c
 * @brief Cubepercentile module
 */

/** @file cubepercentile.c
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

imageID filter_CubePercentile(const char *__restrict IDcin_name,
                              float perc,
                              const char *__restrict IDout_name)
{
    imageID IDcin;
    imageID IDout;
    long    xsize, ysize, zsize;
    long    ii, kk;
    float  *array;

    IDcin = image_ID(IDcin_name, dcimg, dcnimg);
    xsize = dcimg[IDcin].md[0].size[0];
    ysize = dcimg[IDcin].md[0].size[1];
    zsize = dcimg[IDcin].md[0].size[2];

    array = (float *) malloc(sizeof(float) * xsize * ysize);
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        for(kk = 0; kk < zsize; kk++)
        {
            array[kk] = dcimg[IDcin].array.F[kk * xsize * ysize + ii];
        }

        quick_sort_float(array, zsize);
        dcimg[IDout].array.F[ii] = array[(long)(perc * zsize)];
    }

    free(array);

    return IDout;
}

imageID filter_CubePercentileLimit(const char *__restrict IDcin_name,
                                   float perc,
                                   float limit,
                                   const char *__restrict IDout_name)
{
    imageID IDcin;
    imageID IDout;
    long    xsize, ysize, zsize;
    long    ii, kk;
    float  *array;
    long    cnt;
    float   v1;

    IDcin = image_ID(IDcin_name, dcimg, dcnimg);
    xsize = dcimg[IDcin].md[0].size[0];
    ysize = dcimg[IDcin].md[0].size[1];
    zsize = dcimg[IDcin].md[0].size[2];

    array = (float *) malloc(sizeof(float) * xsize * ysize);
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        cnt = 0;
        for(kk = 0; kk < zsize; kk++)
        {
            v1 = dcimg[IDcin].array.F[kk * xsize * ysize + ii];
            if(v1 < limit)
            {
                array[cnt] = v1;
                cnt++;
            }

            if(cnt > 0)
            {
                quick_sort_float(array, zsize);
                dcimg[IDout].array.F[ii] = array[(long)(perc * cnt)];
            }
            else
            {
                dcimg[IDout].array.F[ii] = limit;
            }
        }
    }

    free(array);

    return IDout;
}
