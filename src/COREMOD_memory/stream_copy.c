/**
 * @file    stream_copy.c
 * @brief   copy image stream
 */

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"

// variables local to this translation unit
static char *inimname;
static char *outimname;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in_sname",
        "input stream",
        "ims1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_IMG,
        ".out_sname",
        "output stream",
        "ims2",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata = {"shmimcopy",
                                "copy in stream to existing out stream",
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

    IMGID imgin = mkIMGID_from_name(inimname);
    resolveIMGID(&imgin, ERRMODE_ABORT);

    IMGID imgout = mkIMGID_from_name(outimname);
    resolveIMGID(&imgout, ERRMODE_ABORT);

    uint64_t im_in_datasize = ImageStreamIO_typesize(imgin.im->md->datatype) *
                              imgin.im->md->nelement;
    uint64_t im_out_datasize = ImageStreamIO_typesize(imgin.im->md->datatype) *
                               imgout.im->md->nelement;
    uint64_t byte_copy_size = im_in_datasize < im_out_datasize ? im_in_datasize :
                              im_out_datasize;

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    memcpy(
        imgout.im->array.F,
        imgin.im->array.F,
        byte_copy_size
    );

    processinfo_update_output_stream(processinfo, imgout.ID);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD_memory__stream_copy()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
