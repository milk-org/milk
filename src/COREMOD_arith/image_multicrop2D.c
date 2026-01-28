#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "CLIcore.h"
#include "image_multicrop2D.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *multicrop_insname  = NULL;
char     *multicrop_outsname = NULL;
uint32_t *multicrop_outxsize = NULL;
uint32_t *multicrop_outysize = NULL;
int64_t  *multicrop_wactive[MAXNB_CROPWINDOW];
int64_t  *multicrop_waddmode[MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropxstart[MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropxsize[MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropystart[MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropysize[MAXNB_CROPWINDOW];
uint32_t *multicrop_wbinfact[MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropxpos[MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropypos[MAXNB_CROPWINDOW];
static uint64_t processinfo_change_cnt_local = 0;

void image_multicrop2D_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *imgin, IMAGE *imgout) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    uint32_t ox = *multicrop_outxsize, oy = *multicrop_outysize;
    size_t ts = ImageStreamIO_typesize(imgin->md[0].datatype);
    memset(imgout->array.raw, 0, ts * ox * oy);
    for(int w=0; w < MAXNB_CROPWINDOW ; w++) {
        if (multicrop_wactive[w] && *multicrop_wactive[w] == 1) {
            uint32_t xs = *multicrop_wcropxstart[w], ys = *multicrop_wcropystart[w], xw = *multicrop_wcropxsize[w], yw = *multicrop_wcropysize[w], xp = *multicrop_wcropxpos[w], yp = *multicrop_wcropypos[w], bf = *multicrop_wbinfact[w];
            if (bf < 1) bf = 1;
            uint32_t cxw = xw; if (xp + cxw/bf > ox) cxw = (ox - xp) * bf; if (xs + cxw > imgin->md[0].size[0]) cxw = imgin->md[0].size[0] - xs;
            uint32_t cyw = yw; if (yp + cyw/bf > oy) cyw = (oy - yp) * bf; if (ys + cyw > imgin->md[0].size[1]) cyw = imgin->md[0].size[1] - ys;
            for(uint32_t j=0; j<cyw; j++) {
                uint64_t ioff = (uint64_t)(ys + j) * imgin->md[0].size[0] + xs, ooff = (uint64_t)(yp + j/bf) * ox + xp;
                if (*multicrop_waddmode[w] == 0) memcpy(((char*)imgout->array.raw) + ooff*ts, ((char*)imgin->array.raw) + ioff*ts, ts * (cxw/bf));
                else if (imgin->md[0].datatype == _DATATYPE_FLOAT) for(uint32_t i=0; i<cxw; i++) imgout->array.F[ooff + i/bf] += imgin->array.F[ioff + i];
            }
        }
    }
}

errno_t image_multicrop2D_validate() {
    if (multicrop_outxsize && *multicrop_outxsize < 1) *multicrop_outxsize = 1;
    if (multicrop_outysize && *multicrop_outysize < 1) *multicrop_outysize = 1;
    return RETURN_SUCCESS;
}

/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_multicrop(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, MULTICROP2D_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) \
    { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    MULTICROP2D_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_multicrop(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        multicrop_insname = functionparameter_GetParamPtr_STRING(&fps, ".insname");
        multicrop_outsname = functionparameter_GetParamPtr_STRING(&fps, ".outsname");
        multicrop_outxsize = functionparameter_GetParamPtr_UINT32(&fps, ".outxsize");
        multicrop_outysize = functionparameter_GetParamPtr_UINT32(&fps, ".outysize");
        for(int i=0; i<MAXNB_CROPWINDOW; i++) {
            char key[64]; sprintf(key, ".w%d.active", i); multicrop_wactive[i] = functionparameter_GetParamPtr_INT64(&fps, key);
            sprintf(key, ".w%d.addmode", i); multicrop_waddmode[i] = functionparameter_GetParamPtr_INT64(&fps, key);
            sprintf(key, ".w%d.cropxstart", i); multicrop_wcropxstart[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
            sprintf(key, ".w%d.cropxsize", i); multicrop_wcropxsize[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
            sprintf(key, ".w%d.cropystart", i); multicrop_wcropystart[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
            sprintf(key, ".w%d.cropysize", i); multicrop_wcropysize[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
            sprintf(key, ".w%d.cropxpos", i); multicrop_wcropxpos[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
            sprintf(key, ".w%d.cropypos", i); multicrop_wcropypos[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
            sprintf(key, ".w%d.cropbinfact", i); multicrop_wbinfact[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
        }
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { if (function_parameter_FPCONFloopstep(&fps)) image_multicrop2D_validate(); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(multicrop)
FPS_MAKE_STANDALONE_RUNSTOP(multicrop)

int FPSRUN_multicrop(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    multicrop_insname = functionparameter_GetParamPtr_STRING(&fps, ".insname");
    multicrop_outsname = functionparameter_GetParamPtr_STRING(&fps, ".outsname");
    multicrop_outxsize = functionparameter_GetParamPtr_UINT32(&fps, ".outxsize");
    multicrop_outysize = functionparameter_GetParamPtr_UINT32(&fps, ".outysize");
    for(int i=0; i<MAXNB_CROPWINDOW; i++) {
        char key[64]; sprintf(key, ".w%d.active", i); multicrop_wactive[i] = functionparameter_GetParamPtr_INT64(&fps, key);
        sprintf(key, ".w%d.addmode", i); multicrop_waddmode[i] = functionparameter_GetParamPtr_INT64(&fps, key);
        sprintf(key, ".w%d.cropxstart", i); multicrop_wcropxstart[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
        sprintf(key, ".w%d.cropxsize", i); multicrop_wcropxsize[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
        sprintf(key, ".w%d.cropystart", i); multicrop_wcropystart[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
        sprintf(key, ".w%d.cropysize", i); multicrop_wcropysize[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
        sprintf(key, ".w%d.cropxpos", i); multicrop_wcropxpos[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
        sprintf(key, ".w%d.cropypos", i); multicrop_wcropypos[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
        sprintf(key, ".w%d.cropbinfact", i); multicrop_wbinfact[i] = functionparameter_GetParamPtr_UINT32(&fps, key);
    }
    IMAGE iin, iout; if (ImageStreamIO_read_sharedmem_image_toIMAGE(multicrop_insname, &iin) != 0) return 1;
    uint32_t dims[2] = {*multicrop_outxsize, *multicrop_outysize};
    if (ImageStreamIO_createIm_gpu(&iout, multicrop_outsname, 2, dims, iin.md[0].datatype, -1, 1, 10, 0, 0, 0) != 0) return 1;
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "multicrop Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); image_multicrop2D_compute(&fps, pinfo, &iin, &iout); processinfo_exec_end(pinfo);
        processinfo_update_output_stream(pinfo, &iout, &iin);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("multicrop", multicrop, MULTICROP2D_HELPTEXT)
#endif

static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) \
    { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    MULTICROP2D_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "multicrop2D", "crop 2D image, multiple crops", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID in = mkIMGID_from_name(multicrop_insname); resolveIMGID(&in, ERRMODE_ABORT);
    IMGID out = stream_connect_create_2D(multicrop_outsname, *multicrop_outxsize, *multicrop_outysize, in.md->datatype);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_multicrop2D_compute(data.fpsptr, processinfo, in.im, out.im);
    processinfo_update_output_stream(processinfo, out.im, in.im);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMODE_arith__multicrop2D() { CLIcmddata.FPS_customCONFcheck = image_multicrop2D_validate; INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }