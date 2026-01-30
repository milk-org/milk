#include "ImageStreamIO/ImageStruct.h"
/**
 * @file    imrotate.c
 * @brief   Rotate 2D image
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "imrotate.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#include "COREMOD_memory/COREMOD_memory.h"

char  *imrotate_inimname  = NULL;
char  *imrotate_outimname = NULL;
float *imrotate_angle     = NULL;

static void imrotate_step(IMAGE *imgin, IMAGE *imgout, float angle)
{
    uint32_t nx = imgin->md[0].size[0];
    uint32_t ny = imgin->md[0].size[1];
    float c = cos(angle);
    float s = sin(angle);
    for(uint32_t jj = 0; jj < ny; jj++) {
        for(uint32_t ii = 0; ii < nx; ii++) {
            long iis = (long)(nx / 2 + (ii - (int)nx / 2) * c + (jj - (int)ny / 2) * s);
            long jjs = (long)(ny / 2 - (ii - (int)nx / 2) * s + (jj - (int)ny / 2) * c);
            if((iis >= 0) && (jjs >= 0) && (iis < (long)nx) && (jjs < (long)ny)) {
                imgout->array.F[jj * nx + ii] = imgin->array.F[jjs * nx + iis];
            } else {
                imgout->array.F[jj * nx + ii] = 0.0;
            }
        }
    }
}

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    IMROTATE_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};
static CLICMDDATA CLIcmddata = { "rotateim", "rotate 2D image", CLICMD_FIELDS_DEFAULTS };
static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

imageID basic_rotate(const char *__restrict ID_name, const char *__restrict IDout_name, float angle)
{
    IMGID in = mkIMGID_from_name(ID_name); resolveIMGID(&in, ERRMODE_ABORT);
    IMGID out = stream_connect_create_2Df32(IDout_name, in.md->size[0], in.md->size[1]);
    imrotate_step(in.im, out.im, angle);
    ImageStreamIO_UpdateIm(out.im); return out.ID;
}

static errno_t compute_function() { basic_rotate(imrotate_inimname, imrotate_outimname, *imrotate_angle); return RETURN_SUCCESS; }
INSERT_STD_FPSCLIfunctions
errno_t imrotate_addCLIcmd() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif

#ifdef FPS_STANDALONE
int FPSINIT_rotateim(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps; FPS_INIT_STD_PREAMBLE(fps, fps_name, keywords, description, IMROTATE_HELPTEXT); FPS_INIT_PROCINFO_DEFAULTS(fps, "im1", 1);
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    IMROTATE_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps); function_parameter_FPCONFexit(&fps); return 0;
}
int FPSCONF_rotateim(const char *fps_name, int loop) { FPS_CONF_STD_BODY(fps_name, loop, { imrotate_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name"); imrotate_outimname = functionparameter_GetParamPtr_STRING(&fps, ".out_name"); imrotate_angle = functionparameter_GetParamPtr_FLOAT32(&fps, ".angle"); }, { }); return 0; }
FPS_MAKE_STANDALONE_CONFSTOP(rotateim) FPS_MAKE_STANDALONE_RUNSTOP(rotateim)
int FPSRUN_rotateim(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps; FPS_RUN_STD_PREAMBLE(fps_name, fps, { imrotate_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name"); imrotate_outimname = functionparameter_GetParamPtr_STRING(&fps, ".out_name"); imrotate_angle = functionparameter_GetParamPtr_FLOAT32(&fps, ".angle"); });
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(imrotate_inimname, &iin) != 0) return 1;
    IMAGE iout; uint32_t size[2] = { iin.md[0].size[0], iin.md[0].size[1] };
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(imrotate_outimname, &iout) != 0) { ImageStreamIO_createIm(&iout, imrotate_outimname, 2, size, _DATATYPE_FLOAT, 1, 10, 0); }
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) { processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); imrotate_step(&iin, &iout, *imrotate_angle); processinfo_exec_end(pinfo); ImageStreamIO_UpdateIm(&iout);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}
FPS_MAIN_STANDALONE("rotateim", rotateim, IMROTATE_HELPTEXT, IMROTATE_PARAMS)
#endif
