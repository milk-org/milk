/**
 * @file improfile.c
 * @brief Radial profile
 */

#include <math.h>

#include "CLIcore.h"
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
errno_t profile(
    const char *ID_name,
    const char *outfile,
    double      xcenter,
    double      ycenter,
    double      step,
    long        nb_step);

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "psf";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "psf.prof";
static double p_xcenter = 256.0;
static double p_ycenter = 256.0;
static double p_step    = 1.0;
static long long p_nbstep = 100;

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "profile",
    .cmdkey      = "profile",
    .description = "radial profile"
};

#define FPS_PARAMS(X) \
    X(".in_name", p_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".out_name", p_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output file") \
    X(".xcenter", &p_xcenter, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x center") \
    X(".ycenter", &p_ycenter, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y center") \
    X(".step", &p_step, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "step size") \
    X(".nbstep", &p_nbstep, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "number of steps")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};
static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cms = {0};

static __attribute__((constructor))
void init_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static errno_t compute_function()
{
    profile(p_in, p_out,
            p_xcenter, p_ycenter,
            p_step, (long) p_nbstep);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_info__improfile()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
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

    ID        = image_ID(ID_name, dcimg, dcnimg);
    naxes[0]  = dcimg[ID].md[0].size[0];
    naxes[1]  = dcimg[ID].md[0].size[1];
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

    IDmask = image_ID("profmask", dcimg, dcnimg);
    if(IDmask != -1)
    {
        for(unsigned long ii = 0; ii < nelements; ii++)
        {
            if(dcimg[IDmask].array.F[ii] > 0.5)
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
                mean[i] += dcimg[ID].array.F[jj * naxes[0] + ii];
                rms[i] += dcimg[ID].array.F[jj * naxes[0] + ii] *
                          dcimg[ID].array.F[jj * naxes[0] + ii];
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
                    (dcimg[ID].array.F[jj * naxes[0] + ii] - mean[i]) *
                    (dcimg[ID].array.F[jj * naxes[0] + ii] - mean[i]);
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
                dcimg[ID].array.F[jj * size + ii] =
                    (1.0 - x) * profile_array[i] + x * profile_array[i + 1];
            }
            else if(i < nbpoints)
            {
                dcimg[ID].array.F[jj * size + ii] = profile_array[i];
            }
        }

    free(profile_array);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
