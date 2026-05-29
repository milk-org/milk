/**
 * @file improfile.c
 * @brief Radial profile
 */

#include <math.h>

#include "CLIcore.h"
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
errno_t profile(const char *ID_name,
                const char *outfile,
                double      xcenter,
                double      ycenter,
                double      step,
                long        nb_step);

static char      p_in[FUNCTION_PARAMETER_STRMAXLEN]  = "psf";
static char      p_out[FUNCTION_PARAMETER_STRMAXLEN] = "psf.prof";
static double    p_xcenter                           = 256.0;
static double    p_ycenter                           = 256.0;
static double    p_step                              = 1.0;
static long long p_nbstep                            = 100;

static FPS_APP_INFO FPS_app_info = { .fps_name    = "profile",
                                     .cmdkey      = "profile",
                                     .description = "radial profile",
                                     .description_long =
                                         "Compute the azimuthally-averaged radial profile of a 2D "
                                         "image centered on a specified point." };

#define FPS_PARAMS(X)                                                              \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output file")   \
    X(".xcenter", &p_xcenter, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &p_ycenter, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y center") \
    X(".step", &p_step, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "step size")      \
    X(".nbstep", &p_nbstep, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "number of steps")

FPS_V2_SECTION5(FPS_PARAMS)

static MILK_HOT errno_t compute_function()
{
    profile(p_in, p_out, p_xcenter, p_ycenter, p_step, (long) p_nbstep);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_info__improfile()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

/**
 * Compute radial profile of an image.
 */
errno_t profile(const char *ID_name,
                const char *outfile,
                double      xcenter,
                double      ycenter,
                double      step,
                long        nb_step)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t naxes[2];
    naxes[0]           = imgin.md->size[0];
    naxes[1]           = imgin.md->size[1];
    uint64_t nelements = naxes[0] * naxes[1];

    double *dist = (double *) malloc(nb_step * sizeof(double));
    if (dist == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    double *mean = (double *) malloc(nb_step * sizeof(double));
    if (mean == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    double *rms = (double *) malloc(nb_step * sizeof(double));
    if (rms == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    long *counts = (long *) malloc(nb_step * sizeof(long));
    if (counts == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    int *mask = (int *) malloc(sizeof(int) * nelements);
    if (mask == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IMGID imgmask = imgid_make_from_name("profmask");
    resolveIMGID(&imgmask, ERRMODE_WARN, dcimg, dcnimg);
    if (imgmask.ID != -1)
    {
        for (unsigned long ii = 0; ii < nelements; ii++)
        {
            if (imgmask.im->array.F[ii] > 0.5)
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
    {
        for (unsigned long ii = 0; ii < nelements; ii++)
        {
            mask[ii] = 1;
        }
    }

    for (long i = 0; i < nb_step; i++)
    {
        dist[i]   = 0.0;
        mean[i]   = 0.0;
        rms[i]    = 0.0;
        counts[i] = 0;
    }

    FILE *fp;
    if ((fp = fopen(outfile, "w")) == NULL)
    {
        printf("error : can't open "
               "file %s\n",
               outfile);
    }

    for (unsigned long jj = 0; jj < naxes[1]; jj++)
    {
        for (unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            double distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                                   (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            long   i        = (long) (distance / step);
            if ((i < nb_step) && (mask[jj * naxes[0] + ii] == 1))
            {
                dist[i] += distance;
                mean[i] += imgin.im->array.F[jj * naxes[0] + ii];
                rms[i] +=
                    imgin.im->array.F[jj * naxes[0] + ii] * imgin.im->array.F[jj * naxes[0] + ii];
                counts[i] += 1;
            }
        }
    }

    for (long i = 0; i < nb_step; i++)
    {
        dist[i] /= counts[i];
        mean[i] /= counts[i];
        rms[i] = 0.0;
    }

    for (unsigned long jj = 0; jj < naxes[1]; jj++)
    {
        for (unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            double distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                                   (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            long   i        = (long) distance / step;
            if ((i < nb_step) && (mask[jj * naxes[0] + ii] == 1))
            {
                rms[i] += (imgin.im->array.F[jj * naxes[0] + ii] - mean[i]) *
                          (imgin.im->array.F[jj * naxes[0] + ii] - mean[i]);
            }
        }
    }

    for (long i = 0; i < nb_step; i++)
    {
        if (counts[i] > 0)
        {
            rms[i] = sqrt(rms[i] / counts[i]);
            fprintf(fp,
                    "%.18f %.18g "
                    "%.18g %ld %ld\n",
                    dist[i], mean[i], rms[i], counts[i], i);
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

/**
 * Construct a 2D image from a radial
 * profile read from a text file.
 */
errno_t profile2im(const char   *profile_name,
                   long          nbpoints,
                   unsigned long size,
                   double        xcenter,
                   double        ycenter,
                   double        radius,
                   const char   *out)
{
    DEBUG_TRACE_FSTART();

    IMGID imgout       = imgid_make_from_name_2D(out, size, size);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    double *profile_array = (double *) malloc(sizeof(double) * nbpoints);
    if (profile_array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    FILE *fp;
    if ((fp = fopen(profile_name, "r")) == NULL)
    {
        printf("ERROR: cannot open "
               "profile file \"%s\"\n",
               profile_name);
        exit(0);
    }
    for (long i = 0; i < nbpoints; i++)
    {
        long   index;
        double tmp;
        if (fscanf(fp, "%ld %lf\n", &index, &tmp) != 2)
        {
            printf("ERROR: fscanf, "
                   "%s line %d\n",
                   __FILE__, __LINE__);
            exit(0);
        }
        profile_array[i] = tmp;
    }
    fclose(fp);

    for (unsigned long ii = 0; ii < size; ii++)
    {
        for (unsigned long jj = 0; jj < size; jj++)
        {
            double r = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                            (1.0 * jj - ycenter) * (1.0 * jj - ycenter)) /
                       radius;
            long   i = (long) (r * nbpoints);
            double x = r * nbpoints - i;

            if (i + 1 < nbpoints)
            {
                imgout.im->array.F[jj * size + ii] =
                    (1.0 - x) * profile_array[i] + x * profile_array[i + 1];
            }
            else if (i < nbpoints)
            {
                imgout.im->array.F[jj * size + ii] = profile_array[i];
            }
        }
    }

    free(profile_array);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
