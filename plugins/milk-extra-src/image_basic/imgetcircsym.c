/** @file imgetcircsym.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID IMAGE_BASIC_get_circsym_component(const char *__restrict ID_name,
        const char *__restrict ID_out_name,
        float xcenter,
        float ycenter);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t IMAGE_BASIC_get_circsym_component_cli()
{
    if(0 + CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 1) +
            CLI_checkarg(4, 1) ==
            0)
    {
        IMAGE_BASIC_get_circsym_component(data.cmdargtoken[1].val.string,
                                          data.cmdargtoken[2].val.string,
                                          data.cmdargtoken[3].val.numf,
                                          data.cmdargtoken[4].val.numf);
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

errno_t __attribute__((cold)) imgetcircsym_addCLIcmd()
{

    RegisterCLIcommand("imgetcircsym",
                       __FILE__,
                       IMAGE_BASIC_get_circsym_component_cli,
                       "extract circular symmetric part of image",
                       "<inim> <outim> <xcenter> <ycenter>",
                       "imcgetcircsym imin imout 256.0 230.5",
                       "long IMAGE_BASIC_get_sym_component(const char "
                       "*ID_name, const char *ID_out_name, float "
                       "xcenter, float ycenter)");

    return RETURN_SUCCESS;
}

imageID IMAGE_BASIC_get_circsym_component(const char *__restrict ID_name,
        const char *__restrict ID_out_name,
        float xcenter,
        float ycenter)
{
    float    step = 1.0;
    imageID  ID;
    uint32_t naxes[2];
    float    distance;
    float   *dist;
    float   *mean;
    float   *rms;
    long    *counts;
    long     i;
    long     nb_step;
    imageID  IDout;
    float    ifloat, x;

    ID       = image_ID(ID_name);
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
    create_2Dimage_ID(ID_out_name, naxes[0], naxes[1], &IDout);
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
                    ((1.0 - x) * mean[i] + x * mean[i + 1]);
            }
        }

    free(counts);
    free(dist);
    free(mean);
    free(rms);

    return (IDout);
}
