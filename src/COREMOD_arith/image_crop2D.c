#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "image_crop2D.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *cropinsname = NULL;
char     *outsname    = NULL;
uint32_t *cropxstart  = NULL;
uint32_t *cropxsize   = NULL;
uint32_t *cropystart  = NULL;
uint32_t *cropysize   = NULL;

static FPS_APP_INFO app_info = {
    .fps_name    = "crop2D",
    .cmdkey      = "crop2D",
    .description = "crop 2D image"
};

static uint64_t processinfo_change_cnt_local;

void image_crop2D_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *input_image,
    IMAGE *output_image
)
{
    if (fps && fps->md->processinfo_change_cnt
        != processinfo_change_cnt_local)
    {
        fps_to_processinfo(fps, processinfo);
        processinfo_change_cnt_local =
            fps->md->processinfo_change_cnt;
    }
    if (!cropxstart || !cropxsize
        || !cropystart || !cropysize)
    {
        return;
    }
    uint32_t xs = *cropxstart;
    uint32_t xw = *cropxsize;
    uint32_t ys = *cropystart;
    uint32_t yw = *cropysize;
    uint32_t iw =
        input_image->md[0].size[0];
    uint32_t ih =
        input_image->md[0].size[1];
    size_t ts = ImageStreamIO_typesize(
        input_image->md[0].datatype);
    for (uint32_t j = 0; j < yw; j++) {
        uint32_t oj = j + ys;
        if (oj >= ih) {
            continue;
        }
        memcpy(
            ((char *)
             output_image->array.raw)
            + j * xw * ts,
            ((char *)
             input_image->array.raw)
            + (oj * iw + xs) * ts,
            xw * ts);
    }
}

errno_t image_crop2D_validate()
{
    if (!cropinsname || !cropxstart
        || !cropxsize || !cropystart
        || !cropysize)
    {
        return RETURN_SUCCESS;
    }
    IMAGE im;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(
            cropinsname, &im) == 0)
    {
        uint32_t w = im.md[0].size[0];
        uint32_t h = im.md[0].size[1];
        if (*cropxstart + *cropxsize > w) {
            if (*cropxstart >= w) {
                *cropxstart = 0;
            }
            if (*cropxstart + *cropxsize
                > w)
            {
                *cropxsize =
                    w - *cropxstart;
            }
        }
        if (*cropystart + *cropysize > h) {
            if (*cropystart >= h) {
                *cropystart = 0;
            }
            if (*cropystart + *cropysize
                > h)
            {
                *cropysize =
                    h - *cropystart;
            }
        }
        ImageStreamIO_closeIm(&im);
    }
    return RETURN_SUCCESS;
}


static FPS_CLI_BINDING bindings[] = {
    CROP2D_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    CROP2D_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "crop2D", "crop 2D image",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_crop2D(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "crop2D", "crop 2D image",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    IMGID iin =
        imgid_make_from_name(cropinsname);
    resolveIMGID(
        &iin, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    IMGID iout = stream_connect_create_2D(
        outsname, *cropxsize, *cropysize,
        iin.md->datatype);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_crop2D_compute(
        data.fpsptr, processinfo,
        iin.im, iout.im);
    processinfo_update_output_stream(
        processinfo, iout.im, iin.im);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t CLIADDCMD_COREMODE_arith__crop2D()
{
    CLIcmddata.FPS_customCONFcheck =
        image_crop2D_validate;
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    CROP2D_PARAMS,
    compute_function
)
#endif