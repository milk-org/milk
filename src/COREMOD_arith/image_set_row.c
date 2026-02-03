#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "CLIcore.h"
#include "image_set_row.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *setrow_inimname = NULL;
float    *setrow_pixval   = NULL;
uint32_t *setrow_rowindex = NULL;
static uint64_t processinfo_change_cnt_local = 0;

errno_t image_set_row(IMGID inimg, double value, uint32_t rowindex) {
    if (rowindex >= inimg.md->size[1]) return RETURN_FAILURE;
    uint32_t xsize = inimg.md->size[0];
    switch (inimg.md->datatype) {
        case _DATATYPE_FLOAT: for(uint32_t i=0; i<xsize; i++) inimg.im->array.F[rowindex*xsize + i] = (float)value; break;
        case _DATATYPE_DOUBLE: for(uint32_t i=0; i<xsize; i++) inimg.im->array.D[rowindex*xsize + i] = value; break;
    }
    return RETURN_SUCCESS;
}

void image_set_row_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    if (!setrow_pixval || !setrow_rowindex) return;
    IMGID id; id.im = inimg; id.md = &inimg->md[0];
    image_set_row(id, *setrow_pixval, *setrow_rowindex);
}

/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_setrow(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, SETROW_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    SETROW_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_setrow(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        setrow_inimname = functionparameter_GetParamPtr_STRING(&fps, ".imname");
        setrow_pixval   = functionparameter_GetParamPtr_FLOAT32(&fps, ".pixval");
        setrow_rowindex = functionparameter_GetParamPtr_UINT32(&fps, ".row");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { function_parameter_FPCONFloopstep(&fps); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(setrow)
FPS_MAKE_STANDALONE_RUNSTOP(setrow)

int FPSRUN_setrow(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    setrow_inimname = functionparameter_GetParamPtr_STRING(&fps, ".imname");
    setrow_pixval   = functionparameter_GetParamPtr_FLOAT32(&fps, ".pixval");
    setrow_rowindex = functionparameter_GetParamPtr_UINT32(&fps, ".row");
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(setrow_inimname, &iin) != 0) return 1;
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "setrow Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); image_set_row_compute(&fps, pinfo, &iin); processinfo_exec_end(pinfo);
        processinfo_update_output_stream(pinfo, &iin, NULL);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("setrow", setrow, SETROW_HELPTEXT, SETROW_PARAMS)
#endif

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    SETROW_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "setrow", "set image row pixels values", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID in = imgid_make_from_name(setrow_inimname); resolveIMGID(&in, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_set_row_compute(data.fpsptr, processinfo, in.im);
    processinfo_update_output_stream(processinfo, in.im, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMOD_arith__imset_row() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif
