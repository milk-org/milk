/** @file imgetcircasym.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID IMAGE_BASIC_get_circasym_component(const char *__restrict ID_name,
        const char *__restrict ID_out_name,
        float       xcenter,
        float       ycenter,
        const char *options);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t IMAGE_BASIC_get_circasym_component_cli()
{
    if(0 + CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 1) +
            CLI_checkarg(4, 1) ==
            0)
    {
        IMAGE_BASIC_get_circasym_component(data.cmdargtoken[1].val.string,
                                           data.cmdargtoken[2].val.string,
                                           data.cmdargtoken[3].val.numf,
                                           data.cmdargtoken[4].val.numf,
                                           "");

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

errno_t __attribute__((cold)) imgetcircasym_addCLIcmd()
{

    RegisterCLIcommand("imgetcircasym",
                       __FILE__,
                       IMAGE_BASIC_get_circasym_component_cli,
                       "extract non-circular symmetric part of image",
                       "<inim> <outim> <xcenter> <ycenter>",
                       "imcgetcircassym imin imout 256.0 230.5",
                       "long IMAGE_BASIC_get_asym_component(const char "
                       "*ID_name, const char *ID_out_name, float "
                       "xcenter, float ycenter, const char *options)");

    return RETURN_SUCCESS;
}

imageID
IMAGE_BASIC_get_circasym_component_byID(imageID ID,
                                        const char *__restrict ID_out_name,
                                        float       xcenter,
                                        float       ycenter,
                                        const char *options)
{
    float    step = 1.0;
    uint32_t naxes[2];
    float    distance;
    float   *dist;
    float   *mean;
    float   *rms;
    long    *counts;
    long     i;
    long     nb_step;
    imageID  IDout;
    char     input[50];
    int      str_pos;
    float    perc;
    float    ifloat, x;

    if(strstr(options, "-perc ") != NULL)
    {
        str_pos = strstr(options, "-perc ") - options;
        str_pos = str_pos + strlen("-perc ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i] = '\0';
        perc     = atof(input);
        printf("percentile is %f\n", perc);
    }

    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    nb_step  = naxes[0] / 2;

    dist = (float *) malloc(sizeof(float) * nb_step);
    if(dist == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    mean = (float *) malloc(sizeof(float) * nb_step);
    if(mean == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    rms = (float *) malloc(sizeof(float) * nb_step);
    if(rms == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    counts = (long *) malloc(sizeof(long) * nb_step);
    if(counts == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for(i = 0; i < nb_step; i++)
    {
        dist[i]   = 0;
        mean[i]   = 0;
        rms[i]    = 0;
        counts[i] = 0;
    }

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                            (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            i        = (long)(1.0 * distance / step + 0.5);
            if(i < nb_step)
            {
                dist[i] += distance;
                mean[i] += data.image[ID].array.F[jj * naxes[0] + ii];
                rms[i] += data.image[ID].array.F[jj * naxes[0] + ii] *
                          data.image[ID].array.F[jj * naxes[0] + ii];
                counts[i] += 1;
            }
        }

    for(i = 0; i < nb_step; i++)
    {
        dist[i] /= counts[i];
        mean[i] /= counts[i];
        rms[i] = sqrt(rms[i] - 1.0 * counts[i] * mean[i] * mean[i]) /
                 sqrt(counts[i]);
    }

    printf("%u %u\n", naxes[0], naxes[1]);
    create_2Dimage_ID(ID_out_name, naxes[0], naxes[1], NULL);
    IDout = image_ID(ID_out_name);
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                            (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            i        = (long)(1.0 * distance / step);
            ifloat   = 1.0 * distance / step;
            x        = ifloat - i;

            if((i + 1) < nb_step)
            {
                data.image[IDout].array.F[jj * naxes[0] + ii] =
                    data.image[ID].array.F[jj * naxes[0] + ii] -
                    ((1.0 - x) * mean[i] + x * mean[i + 1]);
            }
        }

    free(counts);
    free(dist);
    free(mean);
    free(rms);

    return (IDout);
}

imageID IMAGE_BASIC_get_circasym_component(const char *__restrict ID_name,
        const char *__restrict ID_out_name,
        float       xcenter,
        float       ycenter,
        const char *options)
{
    imageID IDout;
    imageID ID;

    printf("get non-circular symmetric component from image %s\n", ID_name);
    fflush(stdout);

    ID = image_ID(ID_name);

    IDout = IMAGE_BASIC_get_circasym_component_byID(ID,
            ID_out_name,
            xcenter,
            ycenter,
            options);

    return IDout;
}
