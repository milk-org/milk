/** @file gaussfilter.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

typedef struct
{
    char     *name;
    uint32_t *xsize;
    uint32_t *ysize;
    int      *shared;
    int      *NBkw;
    int      *CBsize;
} LOCVAR_INIMG2D;



imageID gauss_filter(const char *__restrict ID_name,
                     const char *__restrict out_name,
                     float sigma,
                     int   filter_size);


static errno_t gauss_filter_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 1) +
            CLI_checkarg(4, 2) ==
            0)
    {
        gauss_filter(data.cmdargtoken[1].val.string,
                     data.cmdargtoken[2].val.string,
                     data.cmdargtoken[3].val.numf,
                     data.cmdargtoken[4].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



errno_t gaussfilter_addCLIcmd()
{

    RegisterCLIcommand("gaussfilt",
                       __FILE__,
                       gauss_filter_cli,
                       "gaussian 2D filtering",
                       "<input image> <output image> <sigma> <filter box size>",
                       "gaussfilt imin imout 2.3 5",
                       "long gauss_filter(const char *ID_name, const char "
                       "*out_name, float sigma, int filter_size)");

    return RETURN_SUCCESS;
}





imageID gauss_filter(
    const char *__restrict ID_name,
    const char *__restrict out_name,
    float sigma,
    int   filter_size
)
{
    imageID  IDout;
    imageID  IDtmp;

    long     naxes[3];
    long     naxis;
    uint32_t filtersizec;




    imageID ID    = image_ID(ID_name);

    naxis = data.image[ID].md[0].naxis;
    for(int kk = 0; kk < naxis; kk++)
    {
        naxes[kk] = data.image[ID].md[0].size[kk];
    }

    filtersizec = filter_size;
    if(filtersizec > data.image[ID].md[0].size[0] / 2 - 1)
    {
        filtersizec = data.image[ID].md[0].size[0] / 2 - 1;
    }
    if(filtersizec > data.image[ID].md[0].size[1] / 2 - 1)
    {
        filtersizec = data.image[ID].md[0].size[1] / 2 - 1;
    }



    float   *__restrict array;
    array = (float *) malloc((2 * filtersizec + 1) * sizeof(float));
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if(naxis == 2)
    {
        naxes[2] = 1;
    }
    copy_image_ID(ID_name, out_name, 0);

    {
        IMGID outimg = mkIMGID_from_name(out_name);
        resolveIMGID(&outimg, ERRMODE_ABORT);
        image_setzero(outimg);
    }


    create_2Dimage_ID("gtmp", naxes[0], naxes[1], &IDtmp);
    //  copy_image_ID(ID_name,"gtmp", 0);
    // arith_image_zero("gtmp");
    // save_fl_fits("gtmp","gtmp0");
    // IDtmp = image_ID("gtmp");
    IDout = image_ID(out_name);


    {
        float sum = 0.0;
        for(int i = 0; i < (2 * filtersizec + 1); i++)
        {
            array[i] =
                exp(-((i - filtersizec) * (i - filtersizec)) / sigma / sigma);
            sum += array[i];
        }

        for(int i = 0; i < (2 * filtersizec + 1); i++)
        {
            array[i] /= sum;
        }
    }

    for(uint32_t k = 0; k < naxes[2]; k++)
    {
        for(uint64_t ii = 0; ii < naxes[0] * naxes[1]; ii++)
        {
            data.image[IDtmp].array.F[ii] = 0.0;
        }

        for(uint32_t jj = 0; jj < naxes[1]; jj++)
        {
            for(long ii = 0; ii < naxes[0] - (2 * filtersizec + 1); ii++)
            {
                for(int i = 0; i < (2 * filtersizec + 1); i++)
                {
                    data.image[IDtmp]
                    .array.F[jj * naxes[0] + (ii + filtersizec)] +=
                        array[i] *
                        data.image[ID].array.F[k * naxes[0] * naxes[1] +
                                               jj * naxes[0] + (ii + i)];
                }
            }
            for(uint32_t ii = 0; ii < filtersizec; ii++)
            {
                double tot = 0.0;
                for(int i = filtersizec - ii; i < (2 * filtersizec + 1); i++)
                {
                    data.image[IDtmp].array.F[jj * naxes[0] + ii] +=
                        array[i] *
                        data.image[ID]
                        .array.F[k * naxes[0] * naxes[1] + jj * naxes[0] +
                                   (ii - filtersizec + i)];
                    tot += array[i];
                }
                data.image[IDtmp].array.F[jj * naxes[0] + ii] /= tot;
            }
            for(long ii = naxes[0] - filtersizec - 1; ii < naxes[0]; ii++)
            {
                double tot = 0.0;
                for(int i = 0; i < (2 * filtersizec + 1) -
                        (ii - naxes[0] + filtersizec + 1);
                        i++)
                {
                    data.image[IDtmp].array.F[jj * naxes[0] + ii] +=
                        array[i] *
                        data.image[ID]
                        .array.F[k * naxes[0] * naxes[1] + jj * naxes[0] +
                                   (ii - filtersizec + i)];
                    tot += array[i];
                }
                data.image[IDtmp].array.F[jj * naxes[0] + ii] /= tot;
            }
        }

        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            for(long jj = 0; jj < naxes[1] - (2 * filtersizec + 1); jj++)
            {
                for(int j = 0; j < (2 * filtersizec + 1); j++)
                {
                    data.image[IDout]
                    .array.F[k * naxes[0] * naxes[1] +
                               (jj + filtersizec) * naxes[0] + ii] +=
                                 array[j] *
                                 data.image[IDtmp].array.F[(jj + j) * naxes[0] + ii];
                }
            }

            for(long jj = 0; jj < filtersizec; jj++)
            {
                double tot  = 0.0;
                long jmax = (2 * filtersizec + 1);
                if(jj - filtersizec + jmax > naxes[1])
                {
                    jmax = naxes[1] - jj + filtersizec;
                }
                for(int j = filtersizec - jj; j < jmax; j++)
                {
                    data.image[IDout].array.F[k * naxes[0] * naxes[1] +
                                              jj * naxes[0] + ii] +=
                                                  array[j] *
                                                  data.image[IDtmp]
                                                  .array.F[(jj - filtersizec + j) * naxes[0] + ii];
                    tot += array[j];
                }
                data.image[IDout]
                .array.F[k * naxes[0] * naxes[1] + jj * naxes[0] + ii] /=
                    tot;
            }

            for(long jj = naxes[1] - filtersizec - 1; jj < naxes[1]; jj++)
            {
                double tot = 0.0;
                for(int j = 0; j < (2 * filtersizec + 1) -
                        (jj - naxes[1] + filtersizec + 1);
                        j++)
                {
                    data.image[IDout].array.F[k * naxes[0] * naxes[1] +
                                              jj * naxes[0] + ii] +=
                                                  array[j] *
                                                  data.image[IDtmp]
                                                  .array.F[(jj - filtersizec + j) * naxes[0] + ii];
                    tot += array[j];
                }
                data.image[IDout]
                .array.F[k * naxes[0] * naxes[1] + jj * naxes[0] + ii] /=
                    tot;
            }
        }
    }

    delete_image_ID("gtmp", DELETE_IMAGE_ERRMODE_WARNING);

    free(array);

    return IDout;
}



imageID gauss_3Dfilter(
    const char *__restrict ID_name,
    const char *__restrict out_name,
    float sigma,
    int   filter_size
)
{
    imageID ID;
    imageID IDout;
    imageID IDtmp;
    imageID IDtmp1;
    long    naxes[3];


    float *__restrict array;
    array = (float *) malloc((2 * filter_size + 1) * sizeof(float));
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    naxes[2] = data.image[ID].md[0].size[2];

    copy_image_ID(ID_name, out_name, 0);
    {
        IMGID outimg = mkIMGID_from_name(out_name);
        resolveIMGID(&outimg, ERRMODE_ABORT);
        image_setzero(outimg);
    }

    copy_image_ID(ID_name, "gtmp", 0);

    {
        IMGID imggtmp = mkIMGID_from_name("gtmp");
        resolveIMGID(&imggtmp, ERRMODE_ABORT);
        image_setzero(imggtmp);
    }

    copy_image_ID("gtmp", "gtmp1", 0);
    IDtmp  = image_ID("gtmp");
    IDtmp1 = image_ID("gtmp1");
    IDout  = image_ID(out_name);

    {
        float sum = 0.0;
        for(int i = 0; i < (2 * filter_size + 1); i++)
        {
            array[i] =
                exp(-((i - filter_size) * (i - filter_size)) / sigma / sigma);
            sum += array[i];
        }

        for(int i = 0; i < (2 * filter_size + 1); i++)
        {
            array[i] /= sum;
        }
    }

    for(long kk = 0; kk < naxes[2]; kk++)
        for(long jj = 0; jj < naxes[1]; jj++)
            for(long ii = 0; ii < naxes[0] - (2 * filter_size + 1); ii++)
            {
                for(int i = 0; i < (2 * filter_size + 1); i++)
                {
                    data.image[IDtmp]
                    .array.F[kk * naxes[0] * naxes[1] + jj * naxes[0] +
                                (ii + filter_size)] +=
                                 array[i] *
                                 data.image[ID].array.F[kk * naxes[0] * naxes[1] +
                                                        jj * naxes[0] + (ii + i)];
                }
            }

    for(long kk = 0; kk < naxes[2]; kk++)
        for(long ii = 0; ii < naxes[0]; ii++)
            for(long jj = 0; jj < naxes[1] - (2 * filter_size + 1); jj++)
            {
                for(int j = 0; j < (2 * filter_size + 1); j++)
                {
                    data.image[IDtmp1]
                    .array.F[kk * naxes[0] * naxes[1] +
                                (jj + filter_size) * naxes[0] + ii] +=
                                 array[j] *
                                 data.image[IDtmp].array.F[kk * naxes[0] * naxes[1] +
                                                           (jj + j) * naxes[0] + ii];
                }
            }

    for(long ii = 0; ii < naxes[0]; ii++)
        for(long jj = 0; jj < naxes[1]; jj++)
            for(long kk = 0; kk < naxes[2] - (2 * filter_size + 1); kk++)
            {
                for(int k = 0; k < (2 * filter_size + 1); k++)
                {
                    data.image[IDout]
                    .array.F[(kk + filter_size) * naxes[0] * naxes[1] +
                                                jj * naxes[0] + ii] +=
                                 array[k] * data.image[IDtmp1]
                                 .array.F[(kk + k) * naxes[0] * naxes[1] +
                                                   jj * naxes[0] + ii];
                }
            }

    delete_image_ID("gtmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("gtmp1", DELETE_IMAGE_ERRMODE_WARNING);

    free(array);

    return IDout;
}
