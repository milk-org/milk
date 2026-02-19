#include "ImageStreamIO/ImageStruct.h"
/**
 * @file    im3D_to_stream2D.c
 * @brief   convert 3D image to 2D stream
 */

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Local variables pointers
static char *inimname;
static char *outname;
static long *slice_index;
static int *loop_mode;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in_name",
        "input 3D image",
        "im3D",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outname",
        "output stream name",
        "out1",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &outname,
        NULL
    },
    {
        CLIARG_INT64,
        ".slice_index",
        "initial slice index",
        "0",
        FPFLAG_DEFAULT_INPUT,
        (void **) &slice_index,
        NULL
    }
    ,
    {
        CLIARG_ONOFF,
        ".loop_mode",
        "loop through slices",
        "1",
        FPFLAG_DEFAULT_INPUT,
        (void **) &loop_mode,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "im3D_to_stream2D",
    "convert 3D image to 2D stream",
    CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

/**
 * @brief Extract a 2D slice from a 3D image and write it to a 2D image.
 *
 * @param inimg Pointer to the input 3D image IMGID.
 * @param outimg Pointer to the output 2D image IMGID.
 * @param slice_idx The index of the slice to extract.
 * @return errno_t RETURN_SUCCESS on success, RETURN_FAILURE otherwise.
 */
static errno_t extract_slice_to_2D(IMGID *inimg, IMGID *outimg, long slice_idx)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(inimg, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    if (inimg->md->naxis != 3)
    {
        FUNC_RETURN_FAILURE("Input image is not 3D");
    }


    uint32_t xsize = inimg->mdt->size[0];
    uint32_t ysize = inimg->mdt->size[1];
    uint32_t zsize = inimg->mdt->size[2];

    if (slice_idx < 0 || slice_idx >= zsize)
    {
        FUNC_RETURN_FAILURE("Slice index out of bounds");
    }

    // Create output image if needed
    imcreateIMGID(outimg);

    outimg->md->write = 1;

    long framesize = xsize * ysize * ImageStreamIO_typesize(inimg->md->datatype);

    // Copy the slice data
    memcpy(outimg->im->array.raw,
           inimg->im->array.raw + slice_idx * framesize,
           framesize);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    // create stream with name outname
    IMGID outimg;
    outimg = imgid_make_from_name_2D(outname, inimg.mdt->size[0], inimg.mdt->size[1]);
    outimg.mdt->shared = 1;
    outimg.mdt->datatype = inimg.md->datatype;


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {   if (*loop_mode == 0)
        {
            extract_slice_to_2D(&inimg, &outimg, *slice_index);
        }
        else
        {
            // Loop through slices
            *slice_index = (*slice_index + 1) % inimg.mdt->size[2];
            extract_slice_to_2D(&inimg, &outimg, *slice_index);
        }

        extract_slice_to_2D(&inimg, &outimg, *slice_index);
        processinfo_update_output_stream(processinfo, outimg.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


INSERT_STD_FPSCLIfunctions


// Register function in CLI
errno_t
CLIADDCMD_COREMOD_memory__im3D_to_stream2D()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}