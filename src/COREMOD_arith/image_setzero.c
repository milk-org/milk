#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "CLIcore.h"
#include "image_setzero.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char *imsetzero_inimname = NULL;
static uint64_t processinfo_change_cnt_local = 0;

#ifndef FPS_STANDALONE
errno_t image_setzero_IMGID(IMGID *inimg) {
    resolveIMGID(inimg, ERRMODE_ABORT);
    memset(inimg->im->array.raw, 0, ImageStreamIO_typesize(inimg->md->datatype) * inimg->md->nelement);
    return RETURN_SUCCESS;
}

errno_t image_setzero(IMGID inimg) {
    return image_setzero_IMGID(&inimg);
}
#endif

void image_setzero_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    memset(inimg->array.raw, 0, ImageStreamIO_typesize(inimg->md[0].datatype) * inimg->md[0].nelement);
}

/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_imzero(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, IMSETZERO_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    IMSETZERO_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_imzero(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        imsetzero_inimname = functionparameter_GetParamPtr_STRING(&fps, ".imname");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { function_parameter_FPCONFloopstep(&fps); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(imzero)
FPS_MAKE_STANDALONE_RUNSTOP(imzero)

int FPSRUN_imzero(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    imsetzero_inimname = functionparameter_GetParamPtr_STRING(&fps, ".imname");
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(imsetzero_inimname, &iin) != 0) return 1;
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "imzero Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); image_setzero_compute(&fps, pinfo, &iin); processinfo_exec_end(pinfo);
        processinfo_update_output_stream(pinfo, &iin, NULL);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("imzero", imzero, IMSETZERO_HELPTEXT, IMSETZERO_PARAMS)
#endif

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    IMSETZERO_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "imzero", "set all image pixels to zero value", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID in = mkIMGID_from_name(imsetzero_inimname); resolveIMGID(&in, ERRMODE_ABORT);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_setzero_compute(data.fpsptr, processinfo, in.im);
    processinfo_update_output_stream(processinfo, in.im, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMOD_arith__imsetzero() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif
