/**
 * @file    improfile.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"


// ==========================================
// Forward declaration(s)
// ==========================================

errno_t profile(const char *ID_name,
                const char *outfile,
                double      xcenter,
                double      ycenter,
                double      step,
                long        nb_step);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t info_profile_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_FLOAT64) + CLI_checkarg(4, CLIARG_FLOAT64) +
            CLI_checkarg(5, CLIARG_FLOAT64) + CLI_checkarg(6, CLIARG_INT64) ==
            0)
    {
        profile(data.cmdargtoken[1].val.string,
                data.cmdargtoken[2].val.string,
                data.cmdargtoken[3].val.numf,
                data.cmdargtoken[4].val.numf,
                data.cmdargtoken[5].val.numf,
                data.cmdargtoken[6].val.numl);
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

errno_t improfile_addCLIcmd()
{
    RegisterCLIcommand(
        "profile",
        __FILE__,
        info_profile_cli,
        "radial profile",
        "<image> <output file> <xcenter> <ycenter> <step> <Nbstep>",
        "profile psf psf.prof 256 256 1.0 100",
        "int profile(const char *ID_name, const char *outfile, double xcenter, "
        "double ycenter, double "
        "step, long nb_step)");

    return RETURN_SUCCESS;
}

errno_t profile(const char *ID_name,
                const char *outfile,
                double      xcenter,
                double      ycenter,
                double      step,
                long        nb_step)
{
    imageID  ID;
    uint32_t naxes[2];
    uint64_t nelements;
    double   distance;
    double  *dist;
    double  *mean;
    double  *rms;
    long    *counts;
    FILE    *fp;
    long     i;

    int *mask;
    long IDmask; // if profmask exists

    ID        = image_ID(ID_name);
    naxes[0]  = data.image[ID].md[0].size[0];
    naxes[1]  = data.image[ID].md[0].size[1];
    nelements = naxes[0] * naxes[1];

    dist = (double *) malloc(nb_step * sizeof(double));
    if(dist == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    mean = (double *) malloc(nb_step * sizeof(double));
    if(mean == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    rms = (double *) malloc(nb_step * sizeof(double));
    if(rms == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    counts = (long *) malloc(nb_step * sizeof(long));
    if(counts == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    mask = (int *) malloc(sizeof(int) * nelements);
    if(mask == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IDmask = image_ID("profmask");
    if(IDmask != -1)
    {
        for(unsigned long ii = 0; ii < nelements; ii++)
        {
            if(data.image[IDmask].array.F[ii] > 0.5)
            {
                mask[ii] = 1;
            }
            else
            {
                mask[ii] = 0;
            }
        }
    }
    else
        for(unsigned long ii = 0; ii < nelements; ii++)
        {
            mask[ii] = 1;
        }

    //  if( Debug )
    // printf("Function profile. center = %f %f, step = %f, NBstep =
    // %ld\n",xcenter,ycenter,step,nb_step);

    for(i = 0; i < nb_step; i++)
    {
        dist[i]   = 0.0;
        mean[i]   = 0.0;
        rms[i]    = 0.0;
        counts[i] = 0;
    }

    if((fp = fopen(outfile, "w")) == NULL)
    {
        printf("error : can't open file %s\n", outfile);
    }

    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                            (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            i        = (long)(distance / step);
            if((i < nb_step) && (mask[jj * naxes[0] + ii] == 1))
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
        rms[i] = 0.0;
    }

    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                            (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            i        = (long) distance / step;
            if((i < nb_step) && (mask[jj * naxes[0] + ii] == 1))
            {
                rms[i] +=
                    (data.image[ID].array.F[jj * naxes[0] + ii] - mean[i]) *
                    (data.image[ID].array.F[jj * naxes[0] + ii] - mean[i]);
                //	  counts[i] += 1;
            }
        }

    for(i = 0; i < nb_step; i++)
    {
        if(counts[i] > 0)
        {
            //     dist[i] /= counts[i];
            // mean[i] /= counts[i];
            // rms[i] =
            // sqrt(rms[i]-1.0*counts[i]*mean[i]*mean[i])/sqrt(counts[i]);
            rms[i] = sqrt(rms[i] / counts[i]);
            fprintf(fp,
                    "%.18f %.18g %.18g %ld %ld\n",
                    dist[i],
                    mean[i],
                    rms[i],
                    counts[i],
                    i);
        }
    }

    fclose(fp);
    free(mask);

    free(counts);
    free(dist);
    free(mean);
    free(rms);

    return RETURN_SUCCESS;
}

errno_t profile2im(const char   *profile_name,
                   long          nbpoints,
                   unsigned long size,
                   double        xcenter,
                   double        ycenter,
                   double        radius,
                   const char   *out)
{
    DEBUG_TRACE_FSTART();

    FILE   *fp;
    imageID ID;
    double *profile_array;
    long    i;
    long    index;
    double  tmp;
    double  r, x;

    FUNC_CHECK_RETURN(create_2Dimage_ID(out, size, size, &ID));

    profile_array = (double *) malloc(sizeof(double) * nbpoints);
    if(profile_array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if((fp = fopen(profile_name, "r")) == NULL)
    {
        printf("ERROR: cannot open profile file \"%s\"\n", profile_name);
        exit(0);
    }
    for(i = 0; i < nbpoints; i++)
    {
        if(fscanf(fp, "%ld %lf\n", &index, &tmp) != 2)
        {
            printf("ERROR: fscanf, %s line %d\n", __FILE__, __LINE__);
            exit(0);
        }
        profile_array[i] = tmp;
    }
    fclose(fp);

    for(unsigned long ii = 0; ii < size; ii++)
        for(unsigned long jj = 0; jj < size; jj++)
        {
            r = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                     (1.0 * jj - ycenter) * (1.0 * jj - ycenter)) /
                radius;
            i = (long)(r * nbpoints);
            x = r * nbpoints - i; // 0<x<1

            if(i + 1 < nbpoints)
            {
                data.image[ID].array.F[jj * size + ii] =
                    (1.0 - x) * profile_array[i] + x * profile_array[i + 1];
            }
            else if(i < nbpoints)
            {
                data.image[ID].array.F[jj * size + ii] = profile_array[i];
            }
        }

    free(profile_array);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
