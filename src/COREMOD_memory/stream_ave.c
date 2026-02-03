#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "CLIcore.h"
#include "stream_ave.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *streamave_inimname    = NULL;
char     *streamave_outimave    = NULL;
uint32_t *streamave_outimshared = NULL;
char     *streamave_outimrms    = NULL;
uint64_t *streamave_NBcoadd     = NULL;
uint64_t *streamave_cntindex    = NULL;
uint64_t *streamave_compave     = NULL;
uint64_t *streamave_comprms     = NULL;
static uint64_t processinfo_change_cnt_local = 0;

void stream_ave_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *imgin, IMAGE *imgoutave, IMAGE *imgoutrms, double *imdataarray, double *imdataarrayPOW) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    uint64_t xysize = imgin->md[0].size[0] * imgin->md[0].size[1];
    if (*streamave_cntindex == 0) {
        for(uint64_t i=0; i<xysize; i++) {
            double v = 0; switch(imgin->md[0].datatype) { case _DATATYPE_FLOAT: v = imgin->array.F[i]; break; case _DATATYPE_DOUBLE: v = imgin->array.D[i]; break; }
            imdataarray[i] = v; if (*streamave_comprms) imdataarrayPOW[i] = v*v;
        }
    } else {
        for(uint64_t i=0; i<xysize; i++) {
            double v = 0; switch(imgin->md[0].datatype) { case _DATATYPE_FLOAT: v = imgin->array.F[i]; break; case _DATATYPE_DOUBLE: v = imgin->array.D[i]; break; }
            imdataarray[i] += v; if (*streamave_comprms) imdataarrayPOW[i] += v*v;
        }
    }
    (*streamave_cntindex)++;
    if (*streamave_cntindex >= *streamave_NBcoadd) {
        if (*streamave_compave && imgoutave) { for(uint64_t i=0; i<xysize; i++) imgoutave->array.F[i] = imdataarray[i] / (*streamave_cntindex); processinfo_update_output_stream(processinfo, imgoutave, NULL); }
        if (*streamave_comprms && imgoutrms) { for(uint64_t i=0; i<xysize; i++) imgoutrms->array.F[i] = sqrt(imdataarrayPOW[i]) / (*streamave_cntindex); processinfo_update_output_stream(processinfo, imgoutrms, NULL); }
        *streamave_cntindex = 0;
    }
}


/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_stream_ave(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, STREAMAVE_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    STREAMAVE_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_stream_ave(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        streamave_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
        streamave_outimave = functionparameter_GetParamPtr_STRING(&fps, ".outave_name");
        streamave_NBcoadd = functionparameter_GetParamPtr_UINT64(&fps, ".NBcoadd");
        streamave_cntindex = functionparameter_GetParamPtr_UINT64(&fps, ".cntindex");
        streamave_compave = (uint64_t*)functionparameter_GetParamPtr_INT64(&fps, ".comp.ave");
        streamave_comprms = (uint64_t*)functionparameter_GetParamPtr_INT64(&fps, ".comp.rms");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { function_parameter_FPCONFloopstep(&fps); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(stream_ave)
FPS_MAKE_STANDALONE_RUNSTOP(stream_ave)

int FPSRUN_stream_ave(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    streamave_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
    streamave_NBcoadd = functionparameter_GetParamPtr_UINT64(&fps, ".NBcoadd");
    streamave_cntindex = functionparameter_GetParamPtr_UINT64(&fps, ".cntindex");
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(streamave_inimname, &iin) != 0) return 1;
    uint64_t xys = iin.md[0].size[0] * iin.md[0].size[1];
    double *d1 = malloc(sizeof(double)*xys), *d2 = malloc(sizeof(double)*xys);
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); stream_ave_compute(&fps, pinfo, &iin, NULL, NULL, d1, d2); processinfo_exec_end(pinfo);
    }
    free(d1); free(d2); processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("stream_ave", stream_ave, STREAMAVE_HELPTEXT, STREAMAVE_PARAMS)
#endif

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    STREAMAVE_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "streamave", "average stream of images", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID in = imgid_make_from_name(streamave_inimname); resolveIMGID(&in, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    uint64_t xys = in.md[0].size[0] * in.md[0].size[1];
    double *d1 = malloc(sizeof(double)*xys), *d2 = malloc(sizeof(double)*xys);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    stream_ave_compute(data.fpsptr, processinfo, in.im, NULL, NULL, d1, d2);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    free(d1); free(d2); return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_streamaverage() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif