#include "ImageStreamIO/ImageStruct.h"
/**
 * @file    imresize.c
 * @brief   Resize 2D image
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "imresize.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#include "COREMOD_memory/COREMOD_memory.h"

char *imresize_inimname  = NULL;
char *imresize_outimname = NULL;
long *imresize_xsize     = NULL;
long *imresize_ysize     = NULL;

static void imresize_step(IMAGE *imgin, IMAGE *imgout)
{
    uint32_t nx_in = imgin->md[0].size[0];
    uint32_t ny_in = imgin->md[0].size[1];
    uint32_t nx_out = imgout->md[0].size[0];
    uint32_t ny_out = imgout->md[0].size[1];
    if(imgin->md[0].datatype == _DATATYPE_FLOAT) {
        for(uint32_t ii = 0; ii < nx_out; ii++) {
            for(uint32_t jj = 0; jj < ny_out; jj++) {
                float xf1 = (float)ii * nx_in / nx_out;
                float yf1 = (float)jj * ny_in / ny_out;
                long ii1 = (long)xf1; long jj1 = (long)yf1;
                float uf = xf1 - (float)ii1; float tf = yf1 - (float)jj1;
                if((ii1 >= 0) && (ii1 + 1 < (long)nx_in) && (jj1 >= 0) && (jj1 + 1 < (long)ny_in)) {
                    float v00 = imgin->array.F[jj1 * nx_in + ii1];
                    float v01 = imgin->array.F[(jj1 + 1) * nx_in + ii1];
                    float v10 = imgin->array.F[jj1 * nx_in + ii1 + 1];
                    float v11 = imgin->array.F[(jj1 + 1) * nx_in + ii1 + 1];
                    imgout->array.F[jj * nx_out + ii] = v00*(1.0-uf)*(1.0-tf) + v10*uf*(1.0-tf) + v01*(1.0-uf)*tf + v11*uf*tf;
                } else { imgout->array.F[jj * nx_out + ii] = 0.0; }
            }
        }
    }
}

#ifndef FPS_STANDALONE
long basic_resizeim(const char *imname_in, const char *imname_out, long xsizeout, long ysizeout)
{
    IMGID in = imgid_make_from_name(imname_in); resolveIMGID(&in, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    IMGID out = stream_connect_create_2Df32(imname_out, xsizeout, ysizeout);
    imresize_step(in.im, out.im);
    ImageStreamIO_UpdateIm(out.im); return 0;
}

static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    IMRESIZE_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};
static CLICMDDATA CLIcmddata = { "resizeim", "resize 2D image", CLICMD_FIELDS_DEFAULTS };
static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }
static errno_t compute_function() { basic_resizeim(imresize_inimname, imresize_outimname, *imresize_xsize, *imresize_ysize); return RETURN_SUCCESS; }
INSERT_STD_FPSCLIfunctions
errno_t imresize_addCLIcmd() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif

#ifdef FPS_STANDALONE
int FPSINIT_resizeim(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps; FPS_INIT_STD_PREAMBLE(fps, fps_name, keywords, description, IMRESIZE_HELPTEXT); FPS_INIT_PROCINFO_DEFAULTS(fps, "im1", 1);
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    IMRESIZE_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps); function_parameter_FPCONFexit(&fps); return 0;
}
int FPSCONF_resizeim(const char *fps_name, int loop) { FPS_CONF_STD_BODY(fps_name, loop, { imresize_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name"); imresize_outimname = functionparameter_GetParamPtr_STRING(&fps, ".out_name"); imresize_xsize = functionparameter_GetParamPtr_INT64(&fps, ".xsize"); imresize_ysize = functionparameter_GetParamPtr_INT64(&fps, ".ysize"); }, { }); return 0; }
FPS_MAKE_STANDALONE_CONFSTOP(resizeim) FPS_MAKE_STANDALONE_RUNSTOP(resizeim)
int FPSRUN_resizeim(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps; FPS_RUN_STD_PREAMBLE(fps_name, fps, { imresize_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name"); imresize_outimname = functionparameter_GetParamPtr_STRING(&fps, ".out_name"); imresize_xsize = functionparameter_GetParamPtr_INT64(&fps, ".xsize"); imresize_ysize = functionparameter_GetParamPtr_INT64(&fps, ".ysize"); });
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(imresize_inimname, &iin) != 0) return 1;
    IMAGE iout; uint32_t size[2] = { (uint32_t)*imresize_xsize, (uint32_t)*imresize_ysize };
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(imresize_outimname, &iout) != 0) { ImageStreamIO_createIm(&iout, imresize_outimname, 2, size, _DATATYPE_FLOAT, 1, 10, 0); }
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) { processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); imresize_step(&iin, &iout); processinfo_exec_end(pinfo); ImageStreamIO_UpdateIm(&iout);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}
FPS_MAIN_STANDALONE("resizeim", resizeim, IMRESIZE_HELPTEXT, IMRESIZE_PARAMS)
#endif
