#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "CLIcore.h"
#include "image_crop2D.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *cropinsname = NULL;
char     *outsname    = NULL;
uint32_t *cropxstart  = NULL;
uint32_t *cropxsize   = NULL;
uint32_t *cropystart  = NULL;
uint32_t *cropysize   = NULL;
static uint64_t processinfo_change_cnt_local = 0;

void image_crop2D_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *input_image, IMAGE *output_image) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    if (!cropxstart || !cropxsize || !cropystart || !cropysize) return;
    uint32_t xs = *cropxstart, xw = *cropxsize, ys = *cropystart, yw = *cropysize;
    uint32_t iw = input_image->md[0].size[0], ih = input_image->md[0].size[1];
    size_t ts = ImageStreamIO_typesize(input_image->md[0].datatype);
    for(uint32_t j=0; j<yw; j++) {
        uint64_t oj = j + ys; if (oj >= ih) continue;
        memcpy(((char*)output_image->array.raw) + j*xw*ts, ((char*)input_image->array.raw) + (oj*iw+xs)*ts, xw*ts);
    }
}

errno_t image_crop2D_validate() {
    if (!cropinsname || !cropxstart || !cropxsize || !cropystart || !cropysize) return RETURN_SUCCESS;
    IMAGE im; if (ImageStreamIO_read_sharedmem_image_toIMAGE(cropinsname, &im) == 0) {
        uint32_t w = im.md[0].size[0], h = im.md[0].size[1];
        if (*cropxstart + *cropxsize > w) { if (*cropxstart >= w) *cropxstart = 0; if (*cropxstart + *cropxsize > w) *cropxsize = w - *cropxstart; }
        if (*cropystart + *cropysize > h) { if (*cropystart >= h) *cropystart = 0; if (*cropystart + *cropysize > h) *cropysize = h - *cropystart; }
        ImageStreamIO_closeIm(&im);
    }
    return RETURN_SUCCESS;
}

/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_crop2D(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, CROP2D_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    CROP2D_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_crop2D(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        cropinsname = functionparameter_GetParamPtr_STRING(&fps, ".insname");
        outsname    = functionparameter_GetParamPtr_STRING(&fps, ".outsname");
        cropxstart  = functionparameter_GetParamPtr_UINT32(&fps, ".cropxstart");
        cropxsize   = functionparameter_GetParamPtr_UINT32(&fps, ".cropxsize");
        cropystart  = functionparameter_GetParamPtr_UINT32(&fps, ".cropystart");
        cropysize   = functionparameter_GetParamPtr_UINT32(&fps, ".cropysize");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { if (function_parameter_FPCONFloopstep(&fps)) image_crop2D_validate(); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(crop2D)
FPS_MAKE_STANDALONE_RUNSTOP(crop2D)

int FPSRUN_crop2D(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    cropinsname = functionparameter_GetParamPtr_STRING(&fps, ".insname");
    outsname    = functionparameter_GetParamPtr_STRING(&fps, ".outsname");
    cropxstart  = functionparameter_GetParamPtr_UINT32(&fps, ".cropxstart");
    cropxsize   = functionparameter_GetParamPtr_UINT32(&fps, ".cropxsize");
    cropystart  = functionparameter_GetParamPtr_UINT32(&fps, ".cropystart");
    cropysize   = functionparameter_GetParamPtr_UINT32(&fps, ".cropysize");
    IMAGE iin, iout; if (ImageStreamIO_read_sharedmem_image_toIMAGE(cropinsname, &iin) != 0) return 1;
    uint32_t d[2] = {*cropxsize, *cropysize};
    if (ImageStreamIO_createIm_gpu(&iout, outsname, 2, d, iin.md[0].datatype, -1, 1, 10, 0, 0, 0) != 0) return 1;
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "Crop2D Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); image_crop2D_compute(&fps, pinfo, &iin, &iout); processinfo_exec_end(pinfo);
        processinfo_update_output_stream(pinfo, &iout, &iin);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("crop2D", crop2D, CROP2D_HELPTEXT)
#endif

static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    CROP2D_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "crop2D", "crop 2D image", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID iin = mkIMGID_from_name(cropinsname); resolveIMGID(&iin, ERRMODE_ABORT);
    IMGID iout = stream_connect_create_2D(outsname, *cropxsize, *cropysize, iin.md->datatype);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_crop2D_compute(data.fpsptr, processinfo, iin.im, iout.im);
    processinfo_update_output_stream(processinfo, iout.im, iin.im);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMODE_arith__crop2D() { CLIcmddata.FPS_customCONFcheck = image_crop2D_validate; INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }