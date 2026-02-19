#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "CLIcore.h"
#include "image_set_col.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *setcol_inimname = NULL;
float    *setcol_pixval   = NULL;
uint32_t *setcol_colindex = NULL;
static uint64_t processinfo_change_cnt_local = 0;

errno_t image_set_col(IMGID inimg, double value, uint32_t colindex) {
    if (colindex >= inimg.md->size[0]) return RETURN_FAILURE;
    uint32_t xsize = inimg.md->size[0], ysize = inimg.md->size[1];
    switch (inimg.md->datatype) {
        case _DATATYPE_FLOAT: for(uint32_t j=0; j<ysize; j++) inimg.im->array.F[j*xsize + colindex] = (float)value; break;
        case _DATATYPE_DOUBLE: for(uint32_t j=0; j<ysize; j++) inimg.im->array.D[j*xsize + colindex] = value; break;
    }
    return RETURN_SUCCESS;
}

void image_set_col_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    if (!setcol_pixval || !setcol_colindex) return;
    IMGID id; id.im = inimg; id.md = &inimg->md[0];
    image_set_col(id, *setcol_pixval, *setcol_colindex);
}

/* ==================================================================
 * STANDALONE IMPLEMENTATION                                          
 */

int FPSINIT_setcol(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, SETCOL_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags) \
{ \
    c_type val = def_val; \
    void *vptr = &val; \
    if (FPTYPE_IS_STRING(fps_type)) { \
        vptr = *(void**)&val; \
    } \
    function_parameter_add_entry(&fps, key, descr, fps_type, cli_flags, vptr, NULL); \
}
    SETCOL_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_setcol(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        setcol_inimname = functionparameter_GetParamPtr_STRING(&fps, ".imname");
        setcol_pixval   = functionparameter_GetParamPtr_FLOAT32(&fps, ".pixval");
        setcol_colindex = functionparameter_GetParamPtr_UINT32(&fps, ".col");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { function_parameter_FPCONFloopstep(&fps); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(setcol)
FPS_MAKE_STANDALONE_RUNSTOP(setcol)

int FPSRUN_setcol(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    setcol_inimname = functionparameter_GetParamPtr_STRING(&fps, ".imname");
    setcol_pixval   = functionparameter_GetParamPtr_FLOAT32(&fps, ".pixval");
    setcol_colindex = functionparameter_GetParamPtr_UINT32(&fps, ".col");
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(setcol_inimname, &iin) != 0) return 1;
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "setcol Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); image_set_col_compute(&fps, pinfo, &iin); processinfo_exec_end(pinfo);
        processinfo_update_output_stream(pinfo, &iin, NULL);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("setcol", setcol, SETCOL_HELPTEXT, SETCOL_PARAMS)
#endif

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags) { fps_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    SETCOL_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "setcol", "set image column pixels values", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID in = imgid_make_from_name(setcol_inimname); resolveIMGID(&in, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_set_col_compute(data.fpsptr, processinfo, in.im);
    processinfo_update_output_stream(processinfo, in.im, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMOD_arith__imset_col() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif
