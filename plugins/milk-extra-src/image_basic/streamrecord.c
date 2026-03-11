/**
 * @file streamrecord.c
 * @brief Record stream of images
 */

#include <sched.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID IMAGE_BASIC_streamrecord(
    const char *__restrict streamname,
    long NBframes,
    const char *__restrict IDname);

static char p_stream[FUNCTION_PARAMETER_STRMAXLEN]
    = "imstream";
static long long p_nframes = 100;
static char p_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "imrec";

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imgstreamrec",
    .cmdkey      = "imgstreamrec",
    .description =
        "record stream of images"
};

#define FPS_PARAMS(X) \
    X(".stream", p_stream, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream") \
    X(".nframes", &p_nframes, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "number of frames") \
    X(".out_name", p_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image")

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
    IMAGE_BASIC_streamrecord(
        p_stream, (long) p_nframes, p_out);
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
CLIADDCMD_image_basic__streamrecord()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

// works only for floats
//
imageID IMAGE_BASIC_streamrecord(const char *__restrict streamname,
                                 long NBframes,
                                 const char *__restrict IDname)
{
    imageID       ID;
    imageID       IDstream;
    long          xsize, ysize, zsize, xysize;
    unsigned long cnt;
    long          waitdelayus = 50;
    long          kk;
    char         *ptr;

    IDstream = image_ID(streamname, dcimg, dcnimg);
    xsize    = dcimg[IDstream].md[0].size[0];
    ysize    = dcimg[IDstream].md[0].size[1];
    zsize    = NBframes;
    xysize   = xsize * ysize;

    create_3Dimage_ID(IDname, xsize, ysize, zsize, &ID);
    cnt = dcimg[IDstream].md[0].cnt0;

    kk = 0;

    ptr = (char *) dcimg[ID].array.F;
    while(kk != NBframes)
    {
        while(cnt > dcimg[IDstream].md[0].cnt0)
        {
            usleep(waitdelayus);
        }

        cnt++;

        printf("\r%ld / %ld  [%ld %ld]      ",
               kk,
               NBframes,
               cnt,
               dcimg[ID].md[0].cnt0);
        fflush(stdout);

        memcpy(ptr, dcimg[IDstream].array.F, sizeof(float) * xysize);
        ptr += sizeof(float) * xysize;
        kk++;
    }
    printf("\n\n");

    return ID;
}
