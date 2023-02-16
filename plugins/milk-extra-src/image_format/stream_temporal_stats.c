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

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.c"
#include "CommandLineInterface/timeutils.h"


// Local variables pointers
static char    *in_name;
static int32_t *ptr_n_frames;
static double  *ptr_timeout;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".in_name",
        "input image",
        "in_name",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &in_name,
        NULL
    },
    {
        CLIARG_INT32,
        ".n_frames",
        "Stats every n frames max",
        "n_frames",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &ptr_n_frames,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".timeout",
        "Stats at timeout (sec)",
        "timeout",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &ptr_timeout,
        NULL
    }
};

static CLICMDDATA CLIcmddata = {"stream_av_std",
                                "RT compute of ave/std of image streams",
                                CLICMD_FIELDS_DEFAULTS
                               };

static errno_t help_function()
{
    printf("Compute temporal average and st-dev of image stream\n");
    return RETURN_SUCCESS;
}

/*
THE IMPORTANT, CUSTOM PART
*/

#define FOREACH_CAST(start, end, in_arr, out_type)                             \
    {                                                                          \
        int       i;                                                           \
        int       j = end;                                                     \
        out_type  val;                                                         \
        out_type *ptr_sumx  = (out_type *) sum_x;                              \
        out_type *ptr_sumxx = (out_type *) sum_xx;                             \
        for (i = start; i < j; i++)                                            \
        {                                                                      \
            val          = (out_type) (in_img.im->array.in_arr[i]);            \
            ptr_sumx[i]  = val;                                                \
            ptr_sumxx[i] = val * val;                                          \
        }                                                                      \
    }

#define FOREACH_CASTADD(start, end, in_arr, out_type)                          \
    {                                                                          \
        int       i;                                                           \
        int       j = end;                                                     \
        out_type  val;                                                         \
        out_type *ptr_sumx  = (out_type *) sum_x;                              \
        out_type *ptr_sumxx = (out_type *) sum_xx;                             \
        for (i = start; i < j; i++)                                            \
        {                                                                      \
            val = (out_type) (in_img.im->array.in_arr[i]);                     \
            ptr_sumx[i] += val;                                                \
            ptr_sumxx[i] += val * val;                                         \
        }                                                                      \
    }

static errno_t
ave_std_accumulate(IMGID in_img, void *sum_x, void *sum_xx, int reset)
{
    int n_pixels = in_img.md->size[0] * in_img.md->size[1];

    if(reset)
    {
        switch(in_img.datatype)
        {
            case _DATATYPE_UINT8:
                FOREACH_CAST(0, n_pixels, UI8, float);
                break;
            case _DATATYPE_INT8:
                FOREACH_CAST(0, n_pixels, SI8, float);
                break;
            case _DATATYPE_UINT16:
                FOREACH_CAST(0, n_pixels, UI16, float);
                break;
            case _DATATYPE_INT16:
                FOREACH_CAST(0, n_pixels, SI16, float);
                break;
            case _DATATYPE_UINT32:
                FOREACH_CAST(0, n_pixels, UI32, float);
                break;
            case _DATATYPE_INT32:
                FOREACH_CAST(0, n_pixels, SI32, float);
                break;
            case _DATATYPE_UINT64:
                FOREACH_CAST(0, n_pixels, UI64, double);
                break;
            case _DATATYPE_INT64:
                FOREACH_CAST(0, n_pixels, SI64, double);
                break;
            case _DATATYPE_FLOAT:
                FOREACH_CAST(0, n_pixels, F, float);
                break;
            case _DATATYPE_DOUBLE:
                FOREACH_CAST(0, n_pixels, D, double);
                break;
            case _DATATYPE_COMPLEX_FLOAT:
            case _DATATYPE_COMPLEX_DOUBLE:
            default:
                PRINT_ERROR("COMPLEX TYPES UNSUPPORTED");
                return RETURN_FAILURE;
        }
    }
    else
    {
        switch(in_img.datatype)
        {
            case _DATATYPE_UINT8:
                FOREACH_CASTADD(0, n_pixels, UI8, float);
                break;
            case _DATATYPE_INT8:
                FOREACH_CASTADD(0, n_pixels, SI8, float);
                break;
            case _DATATYPE_UINT16:
                FOREACH_CASTADD(0, n_pixels, UI16, float);
                break;
            case _DATATYPE_INT16:
                FOREACH_CASTADD(0, n_pixels, SI16, float);
                break;
            case _DATATYPE_UINT32:
                FOREACH_CASTADD(0, n_pixels, UI32, float);
                break;
            case _DATATYPE_INT32:
                FOREACH_CASTADD(0, n_pixels, SI32, float);
                break;
            case _DATATYPE_UINT64:
                FOREACH_CASTADD(0, n_pixels, UI64, double);
                break;
            case _DATATYPE_INT64:
                FOREACH_CASTADD(0, n_pixels, SI64, double);
                break;
            case _DATATYPE_FLOAT:
                FOREACH_CASTADD(0, n_pixels, F, float);
                break;
            case _DATATYPE_DOUBLE:
                FOREACH_CASTADD(0, n_pixels, D, double);
                break;
            case _DATATYPE_COMPLEX_FLOAT:
            case _DATATYPE_COMPLEX_DOUBLE:
            default:
                PRINT_ERROR("COMPLEX TYPES UNSUPPORTED");
                return RETURN_FAILURE;
        }
    }


    return RETURN_SUCCESS;
}

errno_t ave_finalize(IMGID out_ave_img, void *sum_x, int n_frames_acc)
{
    int n_pixels = out_ave_img.md->size[0] * out_ave_img.md->size[1];
    // TODO MACRO this if a third type may occur

    out_ave_img.md->write = TRUE;

    // Two possible datatypes: float or double
    if(out_ave_img.datatype == _DATATYPE_FLOAT)
    {
        float *ptr_sumx = (float *) sum_x;
        for(int ii = 0; ii < n_pixels; ++ii)
        {
            out_ave_img.im->array.F[ii] = ptr_sumx[ii] / n_frames_acc;
        }
    }
    else if(out_ave_img.datatype == _DATATYPE_DOUBLE)
    {
        double *ptr_sumx = (double *) sum_x;
        for(int ii = 0; ii < n_pixels; ++ii)
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

errno_t
std_finalize(IMGID out_std_img, void *sum_x, void *sum_xx, int n_frames_acc)
{
    int n_pixels = out_std_img.md->size[0] * out_std_img.md->size[1];

    out_std_img.md->write = TRUE;

    // Two possible datatypes: float or double
    if(out_std_img.datatype == _DATATYPE_FLOAT)
    {
        float *ptr_sumx  = (float *) sum_x;
        float *ptr_sumxx = (float *) sum_xx;
        for(int ii = 0; ii < n_pixels; ++ii)
        {
            out_std_img.im->array.F[ii] =
                sqrt(ptr_sumxx[ii] / (n_frames_acc - 1) -
                     ptr_sumx[ii] * (ptr_sumx[ii] / n_frames_acc) /
                     (n_frames_acc - 1));
        }
    }
    else if(out_std_img.datatype == _DATATYPE_DOUBLE)
    {
        double *ptr_sumx  = (double *) sum_x;
        double *ptr_sumxx = (double *) sum_xx;
        for(int ii = 0; ii < n_pixels; ++ii)
        {
            out_std_img.im->array.D[ii] =
                sqrt(ptr_sumxx[ii] / (n_frames_acc - 1) -
                     ptr_sumx[ii] * (ptr_sumx[ii] / n_frames_acc) /
                     (n_frames_acc - 1));
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

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID in_img = mkIMGID_from_name(in_name);
    resolveIMGID(&in_img, ERRMODE_ABORT);

    // Set in_img to be the trigger
    strcpy(CLIcmddata.cmdsettings->triggerstreamname, in_name);
    // for FPS mode:
    if(data.fpsptr != NULL)
    {
        strcpy(data.fpsptr->cmdset.triggerstreamname, in_name);
    }

    // HANDLE DATATYPES
    uint8_t _DATATYPE_INPUT        = in_img.md->datatype;
    uint8_t _DATATYPE_OUTPUT       = ImageStreamIO_floattype(_DATATYPE_INPUT);
    uint8_t SIZEOF_DATATYPE_OUTPUT = ImageStreamIO_typesize(_DATATYPE_OUTPUT);

    char out_ave_name[200];
    strcpy(out_ave_name, in_name);
    strcat(out_ave_name, "_ave");

    char out_std_name[200];
    strcpy(out_std_name, in_name);
    strcat(out_std_name, "_std");

    // Resolve or create outputs, per need
    IMGID out_ave_img = mkIMGID_from_name(out_ave_name);
    if(resolveIMGID(&out_ave_img, ERRMODE_WARN))
    {
        PRINT_WARNING(
            "WARNING - output average image not found and being created");
        in_img.datatype = _DATATYPE_OUTPUT; // To be passed to out_ave_img
        imcreatelikewiseIMGID(&out_ave_img, &in_img);
        in_img.datatype = _DATATYPE_INPUT; // Revert !
        resolveIMGID(&out_ave_img, ERRMODE_ABORT);
    }

    IMGID out_std_img = mkIMGID_from_name(out_std_name);
    if(resolveIMGID(&out_std_img, ERRMODE_WARN))
    {
        PRINT_WARNING("WARNING - output std image not found and being created");
        in_img.datatype = _DATATYPE_OUTPUT; // To be passed to out_std_img
        imcreatelikewiseIMGID(&out_std_img, &in_img);
        in_img.datatype = _DATATYPE_INPUT; // Revert !
        resolveIMGID(&out_std_img, ERRMODE_ABORT);
    }

    /*
     Keyword setup - initialization
    */

    for(int kw = 0; kw < in_img.md->NBkw; ++kw)
    {
        // AVE
        strcpy(out_ave_img.im->kw[kw].name, in_img.im->kw[kw].name);
        out_ave_img.im->kw[kw].type  = in_img.im->kw[kw].type;
        out_ave_img.im->kw[kw].value = in_img.im->kw[kw].value;
        strcpy(out_ave_img.im->kw[kw].comment, in_img.im->kw[kw].comment);
        // STD
        strcpy(out_std_img.im->kw[kw].name, in_img.im->kw[kw].name);
        out_std_img.im->kw[kw].type  = in_img.im->kw[kw].type;
        out_std_img.im->kw[kw].value = in_img.im->kw[kw].value;
        strcpy(out_std_img.im->kw[kw].comment, in_img.im->kw[kw].comment);
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

        if((n_frames_acc >= *ptr_n_frames ||
                timespec_diff_double(time1, time2) > *ptr_timeout))
        {
            if(n_frames_acc >= 1)
            {
                // Keyword value carry-over
                for(int kw = 0; kw < in_img.md->NBkw; ++kw)
                {
                    out_ave_img.im->kw[kw].value = in_img.im->kw[kw].value;
                    out_std_img.im->kw[kw].value = in_img.im->kw[kw].value;
                }

                ave_finalize(out_ave_img, sum_x, n_frames_acc);
                processinfo_update_output_stream(processinfo, out_ave_img.ID);

                if(n_frames_acc >= 2)
                {
                    std_finalize(out_std_img, sum_x, sum_xx, n_frames_acc);
                    processinfo_update_output_stream(processinfo,
                                                     out_std_img.ID);
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

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/*
CLI boilerplate
*/
INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_image_format__temporal_stats()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
