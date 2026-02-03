#include "ImageStreamIO/ImageStruct.h"
/**
 * @file    gaussfilter.c
 * @brief   Image filtering
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "gaussfilter.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

char  *gaussfilt_inimname    = NULL;
char  *gaussfilt_outimname   = NULL;
float *gaussfilt_sigma       = NULL;
int   *gaussfilt_filtersize = NULL;

static void gauss_filter_step(IMAGE *imgin, IMAGE *imgout, float sigma, int filter_size)
{
    uint32_t nx = imgin->md[0].size[0];
    uint32_t ny = imgin->md[0].size[1];
    uint32_t nz = (imgin->md[0].naxis == 3) ? imgin->md[0].size[2] : 1;
    int fsize = filter_size;
    if(fsize > (int)nx/2-1) fsize = nx/2-1;
    if(fsize > (int)ny/2-1) fsize = ny/2-1;

    float *array = (float *) malloc((2 * fsize + 1) * sizeof(float));
    float sum = 0.0;
    for(int i = 0; i < (2 * fsize + 1); i++) {
        array[i] = exp(-((i - fsize) * (i - fsize)) / sigma / sigma);
        sum += array[i];
    }
    for(int i = 0; i < (2 * fsize + 1); i++) array[i] /= sum;

    float *tmp = (float *) calloc(nx * ny, sizeof(float));
    for(uint32_t k = 0; k < nz; k++) {
        float *pl_in = imgin->array.F + k * nx * ny;
        float *pl_out = imgout->array.F + k * nx * ny;
        memset(tmp, 0, nx * ny * sizeof(float));
        for(uint32_t j = 0; j < ny; j++) {
            for(uint32_t i = fsize; i < nx - fsize; i++) {
                for(int ii = -fsize; ii <= fsize; ii++) tmp[j * nx + i] += array[ii + fsize] * pl_in[j * nx + i + ii];
            }
        }
        for(uint32_t i = 0; i < nx; i++) {
            for(uint32_t j = fsize; j < ny - fsize; j++) {
                float v = 0;
                for(int jj = -fsize; jj <= fsize; jj++) v += array[jj + fsize] * tmp[(j + jj) * nx + i];
                pl_out[j * nx + i] = v;
            }
        }
    }
    free(tmp); free(array);
}

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    GAUSSFILT_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};
static CLICMDDATA CLIcmddata = { "gaussfilt", "gaussian 2D filtering", CLICMD_FIELDS_DEFAULTS };
static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

imageID gauss_filter(const char *ID_name, const char *out_name, float sigma, int filter_size)
{
    IMGID in = imgid_make_from_name(ID_name); resolveIMGID(&in, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    IMGID out = stream_connect_create_2Df32(out_name, in.md->size[0], in.md->size[1]);
    gauss_filter_step(in.im, out.im, sigma, filter_size);
    ImageStreamIO_UpdateIm(out.im); return out.ID;
}

static errno_t compute_function() { gauss_filter(gaussfilt_inimname, gaussfilt_outimname, *gaussfilt_sigma, *gaussfilt_filtersize); return RETURN_SUCCESS; }
INSERT_STD_FPSCLIfunctions
errno_t gaussfilter_addCLIcmd() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif

#ifdef FPS_STANDALONE
int FPSINIT_gaussfilt(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps; FPS_INIT_STD_PREAMBLE(fps, fps_name, keywords, description, GAUSSFILT_HELPTEXT); FPS_INIT_PROCINFO_DEFAULTS(fps, "im1", 1);
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    GAUSSFILT_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps); function_parameter_FPCONFexit(&fps); return 0;
}
int FPSCONF_gaussfilt(const char *fps_name, int loop) { FPS_CONF_STD_BODY(fps_name, loop, { gaussfilt_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name"); gaussfilt_outimname = functionparameter_GetParamPtr_STRING(&fps, ".out_name"); gaussfilt_sigma = functionparameter_GetParamPtr_FLOAT32(&fps, ".sigma"); gaussfilt_filtersize = functionparameter_GetParamPtr_INT32(&fps, ".filter_size"); }, { }); return 0; }
FPS_MAKE_STANDALONE_CONFSTOP(gaussfilt) FPS_MAKE_STANDALONE_RUNSTOP(gaussfilt)
int FPSRUN_gaussfilt(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps; FPS_RUN_STD_PREAMBLE(fps_name, fps, { gaussfilt_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name"); gaussfilt_outimname = functionparameter_GetParamPtr_STRING(&fps, ".out_name"); gaussfilt_sigma = functionparameter_GetParamPtr_FLOAT32(&fps, ".sigma"); gaussfilt_filtersize = functionparameter_GetParamPtr_INT32(&fps, ".filter_size"); });
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(gaussfilt_inimname, &iin) != 0) return 1;
    IMAGE iout; uint32_t size[2] = { iin.md[0].size[0], iin.md[0].size[1] };
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(gaussfilt_outimname, &iout) != 0) {
        ImageStreamIO_createIm(&iout, gaussfilt_outimname, 2, size, _DATATYPE_FLOAT, 1, 10, 0);
    }
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) { processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); gauss_filter_step(&iin, &iout, *gaussfilt_sigma, *gaussfilt_filtersize); processinfo_exec_end(pinfo); ImageStreamIO_UpdateIm(&iout);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}
FPS_MAIN_STANDALONE("gaussfilt", gaussfilt, GAUSSFILT_HELPTEXT, GAUSSFILT_PARAMS)
#endif
