/**
 * @file    stream_copy.c
 * @brief   copy image stream
 *
 * Uses FPS V2 framework.
 */

#include "CLIcore.h"
#include "fps.h"

#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "shmimcopy",
    .cmdkey      = "shmimcopy",
    .description =
        "copy in stream to existing out stream"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char *inimname  = NULL;
static char *outimname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_sname", &inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream") \
    X(".out_sname", &outimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output stream")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgin =
        imgid_make_from_name(inimname);
    resolveIMGID(
        &imgin, ERRMODE_ABORT,
        dcimg, dcnimg);

    IMGID imgout =
        imgid_make_from_name(outimname);
    resolveIMGID(
        &imgout, ERRMODE_ABORT,
        dcimg, dcnimg);

    uint64_t im_in_datasize =
        ImageStreamIO_typesize(
            imgin.im->md->datatype)
        * imgin.im->md->nelement;
    uint64_t im_out_datasize =
        ImageStreamIO_typesize(
            imgin.im->md->datatype)
        * imgout.im->md->nelement;
    uint64_t byte_copy_size =
        im_in_datasize < im_out_datasize
        ? im_in_datasize : im_out_datasize;

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {
        ImageStreamIO_semflush(
            imgin.im,
            processinfo->triggersem);

        memcpy(
            imgout.im->array.F,
            imgin.im->array.F,
            byte_copy_size);

        processinfo_update_output_stream(
            processinfo, imgout.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__stream_copy()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif
