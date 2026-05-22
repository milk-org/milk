#include "ImageStreamIO/ImageStruct.h"
/**
 * @file    stream_temporal_stats.c
 * @brief   Publishes average and standard dev of image stream at regular intervals
 *
 * Type specs: all input integer types + float32 allowed
 *             output posted as float32
 *             headache will come later for float64 and complex
 *
 * Input: raw camera stream name (string)
 * Input: count per stat batch (int), disregarded if <= 0
 * Input: time timeout (float), disregarded if <= 0.0
 *
 * Output: Post UTR reduced stream (float 32)
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"
#include "milk_type_dispatch.h"

/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = { .fps_name    = "stream_av_std",
                                     .cmdkey      = "stream_av_std",
                                     .description = "RT compute of ave/std of image streams",
                                     .description_long =
                                         "Compute running temporal statistics (mean, standard "
                                         "deviation) of a shared memory stream in real-time." };


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    *in_name      = NULL;
static int32_t *ptr_n_frames = NULL;
static double  *ptr_timeout  = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                  \
    X(".in_name", &in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".n_frames", &ptr_n_frames, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT,               \
      "Stats every n frames max")                                                      \
    X(".timeout", &ptr_timeout, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "Stats at timeout (sec)")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

/*
THE IMPORTANT, CUSTOM PART
*/

#define FOREACH_CAST(start, end, in_arr, out_type)                  \
    {                                                               \
        int       i;                                                \
        int       j = end;                                          \
        out_type  val;                                              \
        out_type *ptr_sumx  = (out_type *) sum_x;                   \
        out_type *ptr_sumxx = (out_type *) sum_xx;                  \
        for (i = start; i < j; i++)                                 \
        {                                                           \
            val          = (out_type) (in_img.im->array.in_arr[i]); \
            ptr_sumx[i]  = val;                                     \
            ptr_sumxx[i] = val * val;                               \
        }                                                           \
    }

#define FOREACH_CASTADD(start, end, in_arr, out_type)      \
    {                                                      \
        int       i;                                       \
        int       j = end;                                 \
        out_type  val;                                     \
        out_type *ptr_sumx  = (out_type *) sum_x;          \
        out_type *ptr_sumxx = (out_type *) sum_xx;         \
        for (i = start; i < j; i++)                        \
        {                                                  \
            val = (out_type) (in_img.im->array.in_arr[i]); \
            ptr_sumx[i] += val;                            \
            ptr_sumxx[i] += val * val;                     \
        }                                                  \
    }

static errno_t ave_std_accumulate(IMGID in_img, void *sum_x, void *sum_xx, int reset)
{
    int n_pixels = in_img.md->size[0] * in_img.md->size[1];

    if (reset)
    {
#define ACCUM_RESET_BODY_F(MBR) FOREACH_CAST(0, n_pixels, MBR, float)
#define ACCUM_RESET_BODY_D(MBR) FOREACH_CAST(0, n_pixels, MBR, double)

        uint8_t datatype = in_img.md->datatype;
        MILK_FOR_EACH_DATATYPE(datatype, ACCUM_RESET_BODY_F, ACCUM_RESET_BODY_D(D))
        else
        {
            PRINT_ERROR("COMPLEX TYPES UNSUPPORTED");
            return RETURN_FAILURE;
        }

#undef ACCUM_RESET_BODY_F
#undef ACCUM_RESET_BODY_D
    }
    else
    {
#define ACCUM_ADD_BODY_F(MBR) FOREACH_CASTADD(0, n_pixels, MBR, float)
#define ACCUM_ADD_BODY_D(MBR) FOREACH_CASTADD(0, n_pixels, MBR, double)

        uint8_t datatype = in_img.md->datatype;
        MILK_FOR_EACH_DATATYPE(datatype, ACCUM_ADD_BODY_F, ACCUM_ADD_BODY_D(D))
        else
        {
            PRINT_ERROR("COMPLEX TYPES UNSUPPORTED");
            return RETURN_FAILURE;
        }

#undef ACCUM_ADD_BODY_F
#undef ACCUM_ADD_BODY_D
    }

    return RETURN_SUCCESS;
}

errno_t ave_finalize(IMGID out_ave_img, void *sum_x, int n_frames_acc)
{
    int n_pixels = out_ave_img.md->size[0] * out_ave_img.md->size[1];
    // TODO MACRO this if a third type may occur

    out_ave_img.md->write = TRUE;

    // Two possible datatypes: float or double
    if (out_ave_img.md->datatype == _DATATYPE_FLOAT)
    {
        float *ptr_sumx = (float *) sum_x;
        for (int ii = 0; ii < n_pixels; ++ii)
        {
            out_ave_img.im->array.F[ii] = ptr_sumx[ii] / n_frames_acc;
        }
    }
    else if (out_ave_img.md->datatype == _DATATYPE_DOUBLE)
    {
        double *ptr_sumx = (double *) sum_x;
        for (int ii = 0; ii < n_pixels; ++ii)
        {
            out_ave_img.im->array.D[ii] = ptr_sumx[ii] / n_frames_acc;
        }
    }
    else
    {
        PRINT_ERROR("TYPE UNSUPPORTED");
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

errno_t std_finalize(IMGID out_std_img, void *sum_x, void *sum_xx, int n_frames_acc)
{
    int n_pixels = out_std_img.md->size[0] * out_std_img.md->size[1];

    out_std_img.md->write = TRUE;

    // Two possible datatypes: float or double
    if (out_std_img.md->datatype == _DATATYPE_FLOAT)
    {
        float *ptr_sumx  = (float *) sum_x;
        float *ptr_sumxx = (float *) sum_xx;
        for (int ii = 0; ii < n_pixels; ++ii)
        {
            out_std_img.im->array.F[ii] =
                sqrt(ptr_sumxx[ii] / (n_frames_acc - 1) -
                     ptr_sumx[ii] * (ptr_sumx[ii] / n_frames_acc) / (n_frames_acc - 1));
        }
    }
    else if (out_std_img.md->datatype == _DATATYPE_DOUBLE)
    {
        double *ptr_sumx  = (double *) sum_x;
        double *ptr_sumxx = (double *) sum_xx;
        for (int ii = 0; ii < n_pixels; ++ii)
        {
            out_std_img.im->array.D[ii] =
                sqrt(ptr_sumxx[ii] / (n_frames_acc - 1) -
                     ptr_sumx[ii] * (ptr_sumx[ii] / n_frames_acc) / (n_frames_acc - 1));
        }
    }
    else
    {
        PRINT_ERROR("TYPE UNSUPPORTED");
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

/*
BOILERPLATE
*/

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID in_img = imgid_make_from_name(in_name);
    resolveIMGID(&in_img, ERRMODE_WARN, dcimg, dcnimg);

    // Set in_img to be the trigger
    snprintf(CLIcmddata.cmdsettings->triggerstreamname,
             sizeof(CLIcmddata.cmdsettings->triggerstreamname), "%s", in_name);
    if (in_img.ID == -1)
    {
        return RETURN_FAILURE;
    }
    // for FPS mode:
    if (dcfpsptr != NULL)
    {
        snprintf(dcfpsptr->cmdset.triggerstreamname, sizeof(dcfpsptr->cmdset.triggerstreamname),
                 "%s", in_name);
    }

    // HANDLE DATATYPES
    uint8_t _DATATYPE_INPUT        = in_img.md->datatype;
    uint8_t _DATATYPE_OUTPUT       = ImageStreamIO_floattype(_DATATYPE_INPUT);
    uint8_t SIZEOF_DATATYPE_OUTPUT = ImageStreamIO_typesize(_DATATYPE_OUTPUT);

    char out_ave_name[200];
    snprintf(out_ave_name, sizeof(out_ave_name), "%s_ave", in_name);

    char out_std_name[200];
    snprintf(out_std_name, sizeof(out_std_name), "%s_std", in_name);

    // Resolve or create outputs, per need
    IMGID out_ave_img = imgid_make_from_name(out_ave_name);
    if (resolveIMGID(&out_ave_img, ERRMODE_WARN, dcimg, dcnimg))
    {
        PRINT_WARNING("WARNING - output average image not found and being created");
        in_img.mdt->datatype = _DATATYPE_OUTPUT; // To be passed to out_ave_img
        imcreatelikewiseIMGID(&out_ave_img, &in_img);
        in_img.mdt->datatype = _DATATYPE_INPUT; // Revert !
        resolveIMGID(&out_ave_img, ERRMODE_WARN, dcimg, dcnimg);
    }

    IMGID out_std_img = imgid_make_from_name(out_std_name);
    if (resolveIMGID(&out_std_img, ERRMODE_WARN, dcimg, dcnimg))
    {
        PRINT_WARNING("WARNING - output std image not found and being created");
        in_img.mdt->datatype = _DATATYPE_OUTPUT; // To be passed to out_std_img
        imcreatelikewiseIMGID(&out_std_img, &in_img);
        in_img.mdt->datatype = _DATATYPE_INPUT; // Revert !
        resolveIMGID(&out_std_img, ERRMODE_WARN, dcimg, dcnimg);
    }

    if (out_ave_img.ID == -1)
    {
        return RETURN_FAILURE;
    }
    if (out_std_img.ID == -1)
    {
        return RETURN_FAILURE;
    }

    for (int kw = 0; kw < in_img.md->NBkw; ++kw)
    {
        // AVE
        snprintf(out_ave_img.im->kw[kw].name, sizeof(out_ave_img.im->kw[kw].name), "%s",
                 in_img.im->kw[kw].name);
        out_ave_img.im->kw[kw].type  = in_img.im->kw[kw].type;
        out_ave_img.im->kw[kw].value = in_img.im->kw[kw].value;
        snprintf(out_ave_img.im->kw[kw].comment, sizeof(out_ave_img.im->kw[kw].comment), "%s",
                 in_img.im->kw[kw].comment);
        // STD
        snprintf(out_std_img.im->kw[kw].name, sizeof(out_std_img.im->kw[kw].name), "%s",
                 in_img.im->kw[kw].name);
        out_std_img.im->kw[kw].type  = in_img.im->kw[kw].type;
        out_std_img.im->kw[kw].value = in_img.im->kw[kw].value;
        snprintf(out_std_img.im->kw[kw].comment, sizeof(out_std_img.im->kw[kw].comment), "%s",
                 in_img.im->kw[kw].comment);
    }

    /*
    SETUP
    */

    int n_pixels = in_img.md->size[0] * in_img.md->size[1];

    void *sum_x  = malloc(n_pixels * SIZEOF_DATATYPE_OUTPUT);
    void *sum_xx = malloc(n_pixels * SIZEOF_DATATYPE_OUTPUT);

    // HOUSEKEEPING
    int n_frames_acc   = 0;
    int just_published = FALSE;

    struct timespec time1;
    struct timespec time2;

    clock_gettime(CLOCK_MILK, &time1);

    PRINT_WARNING("Timeout: %f", *ptr_timeout);
    PRINT_WARNING("Frames: %d", *ptr_n_frames);

    /*
    PROCESSINFO INIT
    */
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    // PROCESSINFO* processinfo now available

    /*
    LOOP
    */

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART

    {
        /*
        ACCUMULATE
        */
        ave_std_accumulate(in_img, sum_x, sum_xx, just_published);
        just_published = FALSE;
        ++n_frames_acc;
        /*
        PRE - FINALIZE
        */

        /*
        FINALIZATION AND PUBLISH
        */
        clock_gettime(CLOCK_MILK, &time2);

        if ((n_frames_acc >= *ptr_n_frames || timespec_diff_double(time1, time2) > *ptr_timeout))
        {
            if (n_frames_acc >= 1)
            {
                // Keyword value carry-over
                for (int kw = 0; kw < in_img.md->NBkw; ++kw)
                {
                    out_ave_img.im->kw[kw].value = in_img.im->kw[kw].value;
                    out_std_img.im->kw[kw].value = in_img.im->kw[kw].value;
                }

                ave_finalize(out_ave_img, sum_x, n_frames_acc);
                processinfo_update_output_stream(processinfo, out_ave_img.im, NULL);

                if (n_frames_acc >= 2)
                {
                    std_finalize(out_std_img, sum_x, sum_xx, n_frames_acc);
                    processinfo_update_output_stream(processinfo, out_std_img.im, NULL);
                }

                // TODO update the timeout timespec

                just_published = TRUE;
                clock_gettime(CLOCK_MILK, &time1);
                n_frames_acc = 0;
            }
        }
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    /*
    TEARDOWN
    */

    free(sum_x);
    free(sum_xx);

    imgid_free(&in_img);
    imgid_free(&out_ave_img);
    imgid_free(&out_std_img);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/*
CLI boilerplate
*/

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_format__temporal_stats()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
