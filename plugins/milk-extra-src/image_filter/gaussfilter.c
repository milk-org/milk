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

// Local variables pointers
//static LOCVAR_INIMG2D inim;
//static LOCVAR_OUTIMG2D outim;
//static float *sigmaval;
//static int *filtsizeval;

// ==========================================
// Forward declaration(s)
// ==========================================

imageID gauss_filter(const char *__restrict ID_name,
                     const char *__restrict out_name,
                     float sigma,
                     int   filter_size);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

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

// ==========================================
// Register CLI command(s)
// ==========================================

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

imageID gauss_filter(const char *__restrict ID_name,
                     const char *__restrict out_name,
                     float sigma,
                     int   filter_size)
{
    imageID  ID;
    imageID  IDout;
    imageID  IDtmp;
    float   *array;
    long     ii, jj, kk;
    long     naxes[3];
    long     naxis;
    long     i, j, k;
    float    sum;
    double   tot;
    long     jmax;
    uint32_t filtersizec;

    // printf("sigma = %f\n",sigma);
    // printf("filter size = %d\n",filtersizec);

    ID    = image_ID(ID_name);
    naxis = data.image[ID].md[0].naxis;
    for(kk = 0; kk < naxis; kk++)
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
    arith_image_zero(out_name);
    create_2Dimage_ID("gtmp", naxes[0], naxes[1], &IDtmp);
    //  copy_image_ID(ID_name,"gtmp", 0);
    // arith_image_zero("gtmp");
    // save_fl_fits("gtmp","gtmp0");
    // IDtmp = image_ID("gtmp");
    IDout = image_ID(out_name);

    sum = 0.0;
    for(i = 0; i < (2 * filtersizec + 1); i++)
    {
        array[i] =
            exp(-((i - filtersizec) * (i - filtersizec)) / sigma / sigma);
        sum += array[i];
    }

    for(i = 0; i < (2 * filtersizec + 1); i++)
    {
        array[i] /= sum;
        //    printf("%ld %f\n",i,array[i]);
    }

    for(k = 0; k < naxes[2]; k++)
    {
        for(ii = 0; ii < naxes[0] * naxes[1]; ii++)
        {
            data.image[IDtmp].array.F[ii] = 0.0;
        }

        for(jj = 0; jj < naxes[1]; jj++)
        {
            for(ii = 0; ii < naxes[0] - (2 * filtersizec + 1); ii++)
            {
                for(i = 0; i < (2 * filtersizec + 1); i++)
                {
                    data.image[IDtmp]
                    .array.F[jj * naxes[0] + (ii + filtersizec)] +=
                        array[i] *
                        data.image[ID].array.F[k * naxes[0] * naxes[1] +
                                               jj * naxes[0] + (ii + i)];
                }
            }
            for(ii = 0; ii < filtersizec; ii++)
            {
                tot = 0.0;
                for(i = filtersizec - ii; i < (2 * filtersizec + 1); i++)
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
            for(ii = naxes[0] - filtersizec - 1; ii < naxes[0]; ii++)
            {
                tot = 0.0;
                for(i = 0; i < (2 * filtersizec + 1) -
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

        for(ii = 0; ii < naxes[0]; ii++)
        {
            //   printf("A jj : 0 -> %ld/%ld\n", naxes[1]-(2*filtersizec+1), naxes[1]);
            //   fflush(stdout);
            for(jj = 0; jj < naxes[1] - (2 * filtersizec + 1); jj++)
            {
                //       printf("00: %ld/%ld\n", k*naxes[0]*naxes[1]+(jj+filtersizec)*naxes[0]+ii, naxes[0]*naxes[1]*naxes[2]);
                //       printf("01: %ld/%ld\n", (jj+j)*naxes[0]+ii, naxes[0]*naxes[1]);
                fflush(stdout);
                for(j = 0; j < (2 * filtersizec + 1); j++)
                {
                    data.image[IDout]
                    .array.F[k * naxes[0] * naxes[1] +
                               (jj + filtersizec) * naxes[0] + ii] +=
                                 array[j] *
                                 data.image[IDtmp].array.F[(jj + j) * naxes[0] + ii];
                }
            }

            //    printf("B jj : 0 -> %d/%ld\n", filtersizec, naxes[1]);
            //    fflush(stdout);
            for(jj = 0; jj < filtersizec; jj++)
            {
                tot  = 0.0;
                jmax = (2 * filtersizec + 1);
                if(jj - filtersizec + jmax > naxes[1])
                {
                    jmax = naxes[1] - jj + filtersizec;
                }
                for(j = filtersizec - jj; j < jmax; j++)
                {
                    //           printf("02: %ld/%ld\n", k*naxes[0]*naxes[1]+jj*naxes[0]+ii, naxes[0]*naxes[1]*naxes[2]);
                    //           printf("03: %ld/%ld\n", (jj-filtersizec+j)*naxes[0]+ii, naxes[0]*naxes[1]);
                    fflush(stdout);
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

            for(jj = naxes[1] - filtersizec - 1; jj < naxes[1]; jj++)
            {
                tot = 0.0;
                for(j = 0; j < (2 * filtersizec + 1) -
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

    //  save_fl_fits("gtmp","gtmp");
    delete_image_ID("gtmp", DELETE_IMAGE_ERRMODE_WARNING);

    free(array);

    return IDout;
}

imageID gauss_3Dfilter(const char *__restrict ID_name,
                       const char *__restrict out_name,
                       float sigma,
                       int   filter_size)
{
    imageID ID;
    imageID IDout;
    imageID IDtmp;
    imageID IDtmp1;
    float  *array;
    long    ii, jj, kk;
    long    naxes[3];
    int     i, j, k;
    float   sum;

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
    arith_image_zero(out_name);
    copy_image_ID(ID_name, "gtmp", 0);
    arith_image_zero("gtmp");
    copy_image_ID("gtmp", "gtmp1", 0);
    IDtmp  = image_ID("gtmp");
    IDtmp1 = image_ID("gtmp1");
    IDout  = image_ID(out_name);

    sum = 0.0;
    for(i = 0; i < (2 * filter_size + 1); i++)
    {
        array[i] =
            exp(-((i - filter_size) * (i - filter_size)) / sigma / sigma);
        sum += array[i];
    }

    for(i = 0; i < (2 * filter_size + 1); i++)
    {
        array[i] /= sum;
    }

    for(kk = 0; kk < naxes[2]; kk++)
        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0] - (2 * filter_size + 1); ii++)
            {
                for(i = 0; i < (2 * filter_size + 1); i++)
                {
                    data.image[IDtmp]
                    .array.F[kk * naxes[0] * naxes[1] + jj * naxes[0] +
                                (ii + filter_size)] +=
                                 array[i] *
                                 data.image[ID].array.F[kk * naxes[0] * naxes[1] +
                                                        jj * naxes[0] + (ii + i)];
                }
            }

    for(kk = 0; kk < naxes[2]; kk++)
        for(ii = 0; ii < naxes[0]; ii++)
            for(jj = 0; jj < naxes[1] - (2 * filter_size + 1); jj++)
            {
                for(j = 0; j < (2 * filter_size + 1); j++)
                {
                    data.image[IDtmp1]
                    .array.F[kk * naxes[0] * naxes[1] +
                                (jj + filter_size) * naxes[0] + ii] +=
                                 array[j] *
                                 data.image[IDtmp].array.F[kk * naxes[0] * naxes[1] +
                                                           (jj + j) * naxes[0] + ii];
                }
            }

    for(ii = 0; ii < naxes[0]; ii++)
        for(jj = 0; jj < naxes[1]; jj++)
            for(kk = 0; kk < naxes[2] - (2 * filter_size + 1); kk++)
            {
                for(k = 0; k < (2 * filter_size + 1); k++)
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
