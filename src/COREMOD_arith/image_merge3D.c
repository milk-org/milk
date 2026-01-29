#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "CLIcore.h"
#include "image_merge3D.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *immerge_inimname0 = NULL;
char     *immerge_inimname1 = NULL;
char     *immerge_outimname = NULL;
uint32_t *immerge_mergeaxis = NULL;
static uint64_t processinfo_change_cnt_local = 0;

errno_t image_marge(IMGID inimg0, IMGID inimg1, IMGID *outimg, uint8_t mergeaxis) {
#ifndef FPS_STANDALONE
    resolveIMGID(&inimg0, ERRMODE_ABORT);
    resolveIMGID(&inimg1, ERRMODE_ABORT);
    resolveIMGID(outimg, ERRMODE_NULL);
#endif
    if(outimg->ID == -1) copyIMGID(&inimg0, outimg);
    if (mergeaxis < 3) {
        uint32_t s0 = (inimg0.md->size[mergeaxis] == 0) ? 1 : inimg0.md->size[mergeaxis];
        uint32_t s1 = (inimg1.md->size[mergeaxis] == 0) ? 1 : inimg1.md->size[mergeaxis];
        outimg->size[mergeaxis] = s0 + s1;
    } else return RETURN_FAILURE;
    outimg->naxis = (outimg->size[2] > 1) ? 3 : ((outimg->size[1] > 1) ? 2 : 1);
#ifndef FPS_STANDALONE
    createimagefromIMGID(outimg);
#else
    outimg->im = (IMAGE*) malloc(sizeof(IMAGE));
    strncpy(outimg->name, immerge_outimname, 79);
    ImageStreamIO_createIm_gpu(outimg->im, outimg->name, outimg->naxis, outimg->size, outimg->datatype, -1, 1, 10, 0, 0, 0);
    outimg->md = outimg->im->md;
#endif
    size_t ts = ImageStreamIO_typesize(outimg->datatype);
    if (mergeaxis == outimg->naxis-1) {
        size_t sz0 = ts * inimg0.md->nelement;
        memcpy(outimg->im->array.raw, inimg0.im->array.raw, sz0);
        memcpy(((char*)outimg->im->array.raw) + sz0, inimg1.im->array.raw, ts * inimg1.md->nelement);
    } else {
        uint64_t b0 = inimg0.size[0], b1 = inimg1.size[0];
        if (mergeaxis == 1) { b0 *= inimg0.size[1]; b1 *= inimg1.size[1]; }
        uint64_t po = 0, p0 = 0, p1 = 0;
        while (po < outimg->md->nelement) {
            memcpy(((char*)outimg->im->array.raw) + po*ts, ((char*)inimg0.im->array.raw) + p0*ts, ts*b0);
            p0 += b0; po += b0;
            memcpy(((char*)outimg->im->array.raw) + po*ts, ((char*)inimg1.im->array.raw) + p1*ts, ts*b1);
            p1 += b1; po += b1;
        }
    }
    return RETURN_SUCCESS;
}

void image_merge_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg0, IMAGE *inimg1, IMAGE *outimg) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    if (!immerge_mergeaxis) return;
    IMGID id0, id1, idout;
    id0.im = inimg0; id0.md = &inimg0->md[0];
    id1.im = inimg1; id1.md = &inimg1->md[0];
    idout.im = outimg; idout.md = &outimg->md[0];
    image_marge(id0, id1, &idout, *immerge_mergeaxis);
}

/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_immerge(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, IMMERGE_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    IMMERGE_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_immerge(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        immerge_inimname0 = functionparameter_GetParamPtr_STRING(&fps, ".in0name");
        immerge_inimname1 = functionparameter_GetParamPtr_STRING(&fps, ".in1name");
        immerge_outimname = functionparameter_GetParamPtr_STRING(&fps, ".outname");
        immerge_mergeaxis = functionparameter_GetParamPtr_UINT32(&fps, ".axis");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { function_parameter_FPCONFloopstep(&fps); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(immerge)
FPS_MAKE_STANDALONE_RUNSTOP(immerge)

int FPSRUN_immerge(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    immerge_inimname0 = functionparameter_GetParamPtr_STRING(&fps, ".in0name");
    immerge_inimname1 = functionparameter_GetParamPtr_STRING(&fps, ".in1name");
    immerge_outimname = functionparameter_GetParamPtr_STRING(&fps, ".outname");
    immerge_mergeaxis = functionparameter_GetParamPtr_UINT32(&fps, ".axis");
    IMAGE i0, i1;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(immerge_inimname0, &i0) != 0) return 1;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(immerge_inimname1, &i1) != 0) return 1;
    IMGID id0, id1, idout; id0.im = &i0; id0.md = &i0.md[0]; id1.im = &i1; id1.md = &i1.md[0];
#ifndef FPS_STANDALONE
    idout = makeIMGID_blank(); 
#else
    idout.ID = -1; idout.im = NULL; idout.md = NULL;
#endif
    image_marge(id0, id1, &idout, *immerge_mergeaxis);
    PROCESSINFO *processinfo = processinfo_setup((char*)fps_name, "immerge Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(processinfo, &i0, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, processinfo); processinfo_loopstart(processinfo);
    while(processinfo_loopstep(processinfo)) {
        processinfo_waitoninputstream(processinfo); if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(processinfo); image_merge_compute(&fps, processinfo, &i0, &i1, idout.im); processinfo_exec_end(processinfo);
        processinfo_update_output_stream(processinfo, idout.im, &i0);
    }
    processinfo_cleanExit(processinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("immerge", immerge, IMMERGE_HELPTEXT)
#endif

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) \
    { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    IMMERGE_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "immerge", "merge images along axis", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID id0 = mkIMGID_from_name(immerge_inimname0); resolveIMGID(&id0, ERRMODE_ABORT);
    IMGID id1 = mkIMGID_from_name(immerge_inimname1); resolveIMGID(&id1, ERRMODE_ABORT);
    IMGID idout = mkIMGID_from_name(immerge_outimname);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_merge_compute(data.fpsptr, processinfo, id0.im, id1.im, idout.im);
    processinfo_update_output_stream(processinfo, idout.im, id0.im);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMOD_arith__image_merge() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif