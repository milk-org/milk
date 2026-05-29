/**
 * @file imgetcircsym.c
 * @brief Extract circular symmetric part
 */

#include <math.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID IMAGE_BASIC_get_circsym_component(const char *__restrict ID_name,
                                          const char *__restrict ID_out_name,
                                          float xcenter,
                                          float ycenter);

static char   p_in[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static char   p_out[FUNCTION_PARAMETER_STRMAXLEN] = "imout";
static double p_xcenter                           = 256.0;
static double p_ycenter                           = 230.5;

static FPS_APP_INFO FPS_app_info = { .fps_name    = "imgetcircsym",
                                     .cmdkey      = "imgetcircsym",
                                     .description = "extract circular symmetric part",
                                     .description_long =
                                         "Extract the circularly-symmetric component of a 2D image "
                                         "by computing the azimuthal average around the center." };

#define FPS_PARAMS(X)                                                              \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".xcenter", &p_xcenter, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &p_ycenter, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y center")

FPS_V2_SECTION5(FPS_PARAMS)

static MILK_HOT errno_t compute_function()
{
    IMAGE_BASIC_get_circsym_component(p_in, p_out, (float) p_xcenter, (float) p_ycenter);
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_basic__imgetcircsym()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

/**
 * Extract the circularly symmetric
 * component of a 2D image.
 */
imageID IMAGE_BASIC_get_circsym_component(const char *__restrict ID_name,
                                          const char *__restrict ID_out_name,
                                          float xcenter,
                                          float ycenter)
{
    float step = 1.0;

    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t naxes[2];
    naxes[0]     = imgin.md->size[0];
    naxes[1]     = imgin.md->size[1];
    long nb_step = naxes[0] / 2;

    float *dist = (float *) malloc(sizeof(float) * nb_step);
    if (dist == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    float *mean = (float *) malloc(sizeof(float) * nb_step);
    if (mean == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    float *rms = (float *) malloc(sizeof(float) * nb_step);
    if (rms == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    long *counts = (long *) malloc(sizeof(long) * nb_step);
    if (counts == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for (long i = 0; i < nb_step; i++)
    {
        dist[i]   = 0;
        mean[i]   = 0;
        rms[i]    = 0;
        counts[i] = 0;
    }

    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            float distance = sqrtf((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                                   (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            long  i        = (long) (1.0 * distance / step + 0.5);
            if (i < nb_step)
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
        rms[i] = sqrt(rms[i] - 1.0 * counts[i] * mean[i] * mean[i]) / sqrt(counts[i]);
    }

    printf("%u %u\n", naxes[0], naxes[1]);

    IMGID imgout       = imgid_make_from_name_2D(ID_out_name, naxes[0], naxes[1]);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            float distance = sqrtf((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                                   (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            long  i        = (long) (1.0 * distance / step);
            float ifloat   = 1.0 * distance / step;
            float x        = ifloat - i;

            if ((i + 1) < nb_step)
            {
                imgout.im->array.F[jj * naxes[0] + ii] = ((1.0 - x) * mean[i] + x * mean[i + 1]);
            }
        }
    }

    free(counts);
    free(dist);
    free(mean);
    free(rms);

    return imgout.ID;
}
