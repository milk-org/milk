#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "CLIcore.h"
#include "image_set_3Daxes.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *set3d_inimname = NULL;
uint32_t *set3d_size0    = NULL;
uint32_t *set3d_size1    = NULL;
uint32_t *set3d_size2    = NULL;
static uint64_t processinfo_change_cnt_local = 0;

errno_t image_set_3Daxes(IMGID inimg, uint32_t imsize0, uint32_t imsize1, uint32_t imsize2) {
    long nelem = inimg.md->nelement;
    uint32_t s0 = (imsize0 == 0) ? inimg.md->size[0] : imsize0;
    uint32_t s1 = (imsize1 == 0) ? ((inimg.md->naxis < 2) ? 1 : inimg.md->size[1]) : imsize1;
    uint32_t s2 = (imsize2 == 0) ? ((inimg.md->naxis < 3) ? 1 : inimg.md->size[2]) : imsize2;
    if((long)s0 * s1 * s2 == nelem) {
        inimg.md->naxis = 3; inimg.md->size[0] = s0; inimg.md->size[1] = s1; inimg.md->size[2] = s2;
    }
    return RETURN_SUCCESS;
}

void image_set_3Daxes_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    if (!set3d_size0 || !set3d_size1 || !set3d_size2) return;
    IMGID id; id.im = inimg; id.md = &inimg->md[0];
    image_set_3Daxes(id, *set3d_size0, *set3d_size1, *set3d_size2);
}

/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_set3Daxes(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, SET3DAXES_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    SET3DAXES_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_set3Daxes(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        set3d_inimname = functionparameter_GetParamPtr_STRING(&fps, ".imname");
        set3d_size0    = functionparameter_GetParamPtr_UINT32(&fps, ".size0");
        set3d_size1    = functionparameter_GetParamPtr_UINT32(&fps, ".size1");
        set3d_size2    = functionparameter_GetParamPtr_UINT32(&fps, ".size2");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { function_parameter_FPCONFloopstep(&fps); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(set3Daxes)
FPS_MAKE_STANDALONE_RUNSTOP(set3Daxes)

int FPSRUN_set3Daxes(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    set3d_inimname = functionparameter_GetParamPtr_STRING(&fps, ".imname");
    set3d_size0    = functionparameter_GetParamPtr_UINT32(&fps, ".size0");
    set3d_size1    = functionparameter_GetParamPtr_UINT32(&fps, ".size1");
    set3d_size2    = functionparameter_GetParamPtr_UINT32(&fps, ".size2");
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(set3d_inimname, &iin) != 0) return 1;
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "set3Daxes Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); image_set_3Daxes_compute(&fps, pinfo, &iin); processinfo_exec_end(pinfo);
        processinfo_update_output_stream(pinfo, &iin, NULL);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("set3Daxes", set3Daxes, SET3DAXES_HELPTEXT)
#endif

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    SET3DAXES_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "set3Daxes", "set 3D image axes size", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID in = mkIMGID_from_name(set3d_inimname); resolveIMGID(&in, ERRMODE_ABORT);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_set_3Daxes_compute(data.fpsptr, processinfo, in.im);
    processinfo_update_output_stream(processinfo, in.im, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMOD_arith__imset_3Daxes() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif
