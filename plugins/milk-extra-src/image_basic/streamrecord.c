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
        "record stream of images",
    .description_long =
        "Record frames from a 2D shared memory stream into a 3D cube or sequence of FITS files on disk."
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

static MILK_HOT errno_t compute_function()
{
    IMAGE_BASIC_streamrecord(
        p_stream, (long) p_nframes, p_out);
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) CLIfunction(void)
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

/**
 * Record NBframes from a float stream
 * into a 3D cube.
 */
imageID IMAGE_BASIC_streamrecord(
    const char *__restrict streamname,
    long NBframes,
    const char *__restrict IDname)
{
    IMGID imgin =
        imgid_make_from_name(streamname);
    resolveIMGID(&imgin, ERRMODE_WARN,
                 dcimg, dcnimg);
    if (imgin.ID == -1) {
        return RETURN_FAILURE;
    }

    long xsize  = imgin.md->size[0];
    long ysize  = imgin.md->size[1];
    long xysize = xsize * ysize;

    IMGID imgout =
        imgid_make_from_name_3D(
            IDname,
            xsize, ysize, NBframes);
    imgout.mdt->shared = 0;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    unsigned long cnt = imgin.md->cnt0;
    long kk = 0;
    long waitdelayus = 50;

    char *ptr =
        (char *) imgout.im->array.F;
    while(kk != NBframes)
    {
        while(cnt > imgin.md->cnt0)
        {
            usleep(waitdelayus);
        }

        cnt++;

        printf("\r%ld / %ld  [%lu %lu]"
               "      ",
               kk, NBframes,
               cnt,
               imgout.md->cnt0);
        fflush(stdout);

        memcpy(ptr,
               imgin.im->array.F,
               sizeof(float) * xysize);
        ptr += sizeof(float) * xysize;
        kk++;
    }
    printf("\n\n");

    return imgout.ID;
}
