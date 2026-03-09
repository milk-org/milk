#include "ImageStreamIO/ImageStruct.h"
#include "CLIcore.h"
#include "fps.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "cropmask",
    .cmdkey      = "cropmask",
    .description = "crop and mask image"
};

static char *cminsname;
static char *masksname;
static char *outsname;
static uint32_t *cropxstart;
static uint32_t *cropxsize;
static uint32_t *cropystart;
static uint32_t *cropysize;

#define FPS_PARAMS(X) \
    X(".insname", &cminsname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input stream name") \
    X(".masksname", &masksname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "mask stream name") \
    X(".outsname", &outsname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "output stream name") \
    X(".cropxstart", &cropxstart, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "crop x coord start") \
    X(".cropxsize", &cropxsize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "crop x coord size") \
    X(".cropystart", &cropystart, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "crop y coord start") \
    X(".cropysize", &cropysize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "crop y coord size")

static errno_t customCONFsetup()
{
    if(dcfpsptr != NULL)
    {
        long fpi = functionparameter_GetParamIndex(dcfpsptr, ".insname");
        if(fpi >= 0)
        {
            dcfpsptr->parray[fpi].fpflag |=
                FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
        }
    }
    return RETURN_SUCCESS;
}

static errno_t customCONFcheck()
{
    return RETURN_SUCCESS;
}

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

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // CONNECT TO INPUT STREAM
    IMGID imgin = imgid_make_from_name(cminsname);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);
    printf("Input stream size : %u %u\n", imgin.md->size[0], imgin.md->size[1]);
    //long m = imgin.md->size[0] * imgin.md->size[1];

    // CONNNECT TO OR CREATE MASK STREAM
    IMGID imgmask = stream_connect_create_2Df32(masksname, *cropxsize, *cropysize);

    // CONNNECT TO OR CREATE OUTPUT STREAM
    IMGID imgout = stream_connect_create_2Df32(outsname, *cropxsize, *cropysize);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        const uint32_t in_xsize = imgin.md->size[0];
        const uint32_t crop_xsize = *cropxsize;
        const uint32_t crop_ysize = *cropysize;
        const uint32_t crop_ystart = *cropystart;
        const uint32_t crop_xstart = *cropxstart;

        for(uint32_t jj = 0; jj < crop_ysize; jj++)
        {
            const float *__restrict imgin_row =
                &imgin.im->array.F[(jj + crop_ystart) * in_xsize + crop_xstart];
            const float *__restrict imgmask_row = &imgmask.im->array.F[jj * crop_xsize];
            float *__restrict imgout_row        = &imgout.im->array.F[jj * crop_xsize];

            for(uint32_t ii = 0; ii < crop_xsize; ii++)
            {
                imgout_row[ii] = imgmask_row[ii] * imgin_row[ii];
            }
        }
        processinfo_update_output_stream(processinfo, imgout.im, NULL);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

// Register function in CLI
errno_t
CLIADDCMD_COREMODE_arith__cropmask()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(
    FPS_app_info,
    FPS_PARAMS,
    compute_function,
    customCONFcheck)
#endif
