/**
 * @file    stream_merge.c
 * @brief   Merge n independently triggers streams into one
 *          This early version relies on very static naming conventions
 *          And will merge <shmname>_[0-N] into <shmname>
 *          Assuming equal framerates
 *
 *          Designed for parallel MVM computations.
 */

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

//#include "image_ID.h"
//#include "stream_sem.h"

//#include "COREMOD_tools/COREMOD_tools.h"



// variables local to this translation unit
static char *stream_basename; // stream basename, which also is the output name.
static int32_t *ptr_n_input; // How many streams to merge?
// static int32_t *ptr_concat_axis; // FUTURE - actually perform smarter concatenation along any axis.

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, // This being CLIARG_IMG assumes the output stream has been created already...
        ".stream_basename",
        "output stream & input stream basename",
        "stream_basename",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &stream_basename,
        NULL
    },
    {
        CLIARG_INT32,
        ".n_input",
        "number of inputs to concatenate",
        "n_input",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &ptr_n_input,
        NULL
    }
};

static CLICMDDATA CLIcmddata = {"shmimmerge",
                                "Merge N in stream into out stream",
                                CLICMD_FIELDS_DEFAULTS
                               };

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



// Wrapper function, used by all CLI calls
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    int32_t n_input = *ptr_n_input;

    // Open array of input images
    IMGID *img_in_arr = (IMGID *) malloc(n_input * sizeof(IMGID));
    char input_name[200];
    for(int ii = 0; ii < n_input; ++ii)
    {
        sprintf(input_name, "%s_%d", stream_basename, ii);
        img_in_arr[ii] = mkIMGID_from_name(input_name);
        resolveIMGID(&img_in_arr[ii], ERRMODE_ABORT);
    }

    // Open output image
    IMGID img_out = mkIMGID_from_name(stream_basename);
    resolveIMGID(&img_out, ERRMODE_WARN);

    // Perform some data offset computations.
    // So that we know WHERE the memcopies should go and how big they should be.
    int32_t *offset_bytes = (int32_t *) malloc(n_input * sizeof(int32_t));
    if(offset_bytes == NULL) {
        PRINT_ERROR("malloc returns NULL pointer, size %ld", (long) (n_input * sizeof(int32_t)));
        abort();
    }

    int32_t *size_bytes = (int32_t *) malloc(n_input * sizeof(int32_t));
    if(size_bytes == NULL) {
        PRINT_ERROR("malloc returns NULL pointer, size %ld", (long) (n_input * sizeof(int32_t)));
        abort();
    }

    int32_t *sem_idxs = (int32_t *) malloc(n_input * sizeof(int32_t));
    if(sem_idxs == NULL) {
        PRINT_ERROR("malloc returns NULL pointer, size %ld", (long) (n_input * sizeof(int32_t)));
        abort();
    }

    int acc = 0;
    for(int kk = 0;  kk < n_input; ++kk)
    {
        offset_bytes[kk] = acc;
        size_bytes[kk] = img_in_arr[kk].size[0] * ImageStreamIO_typesize(
                             img_in_arr[kk].datatype);
        acc += size_bytes[kk]; // Wildly assuming naxis=1 here.

        sem_idxs[kk] = ImageStreamIO_getsemwaitindex(img_in_arr[kk].im, 0);
    }


    struct timespec t_spec1;
    struct timespec t_spec2;

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // Note - by DESIGN, this functions handles its own multi-stream sync
    // processinfo trigger should be put in freerun mode.
    processinfo->triggermode = PROCESSINFO_TRIGGERMODE_IMMEDIATE;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        // One sec timeout
        milk_clock_gettime(&t_spec1);
        t_spec2.tv_sec = t_spec1.tv_sec + 1;
        t_spec2.tv_nsec = t_spec1.tv_nsec;


        for(int kk = 0; kk < n_input; kk++)
        {
            ImageStreamIO_semtimedwait(img_in_arr[kk].im, sem_idxs[kk], &t_spec2);
            // In case one input got ahead of the others
            ImageStreamIO_semflush(img_in_arr[kk].im, sem_idxs[kk]);
        }
        img_out.md->write = TRUE;
        for(int kk = 0; kk < n_input; kk++)
        {
            memcpy(img_out.im->array.raw + offset_bytes[kk], img_in_arr[kk].im->array.raw,
                   size_bytes[kk]);
        }


        // What about keywords?

        // Finito!
        processinfo_update_output_stream(processinfo, img_out.ID);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    // Mem cleanup
    free(img_in_arr);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD_memory__stream_merge()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
