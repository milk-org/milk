#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "CLIcore.h"
#include "image_norm.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *norm_inimname  = NULL;
char     *norm_outimname = NULL;
uint32_t *norm_sliceaxis = NULL;
static uint64_t processinfo_change_cnt_local = 0;

errno_t image_slicenorm_IMGID(IMGID *inimg, IMGID *outimg, uint8_t sliceaxis) {
#ifndef FPS_STANDALONE
    resolveIMGID(inimg, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    resolveIMGID(outimg, ERRMODE_NULL, data.image, data.NB_MAX_IMAGE);
#endif
    if(outimg->ID == -1) imgid_copy(inimg, outimg);
    for (uint8_t axis = 0; axis < inimg->md->naxis; axis++)
        if (axis != sliceaxis) outimg->mdt->size[axis] = 1;
    outimg->mdt->datatype = _DATATYPE_FLOAT;
#ifndef FPS_STANDALONE
    createimagefromIMGID(outimg);
#else
    outimg->im = (IMAGE*) malloc(sizeof(IMAGE));
    strncpy(outimg->name, norm_outimname, 79);
    ImageStreamIO_createIm_gpu(outimg->im, outimg->name, outimg->mdt->naxis, outimg->mdt->size, outimg->mdt->datatype, -1, 1, 10, 0, 0, 0);
    outimg->md = outimg->im->md;
#endif
    uint32_t sizes[3] = {inimg->md->size[0], inimg->md->size[1], inimg->md->size[2]};
    if(inimg->md->naxis < 3) sizes[2] = 1;
    if(inimg->md->naxis < 2) sizes[1] = 1;
    double *normarray = (double*)calloc(sizes[sliceaxis], sizeof(double));
    for(uint32_t i=0; i<sizes[0]; i++)
        for(uint32_t j=0; j<sizes[1]; j++)
            for(uint32_t k=0; k<sizes[2]; k++) {
                uint64_t idx = (uint64_t)k*sizes[1]*sizes[0] + (uint64_t)j*sizes[0] + i;
                double v = 0;
                switch(inimg->mdt->datatype) {
                    case _DATATYPE_FLOAT: v = inimg->im->array.F[idx]; break;
                    case _DATATYPE_DOUBLE: v = inimg->im->array.D[idx]; break;
                }
                uint32_t coords[3] = {i,j,k}; normarray[coords[sliceaxis]] += v*v;
            }
    for(uint32_t i=0; i<sizes[sliceaxis]; i++) outimg->im->array.F[i] = sqrt(normarray[i]);
    free(normarray);
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
errno_t image_slicenorm(const char *inname, const char *outname, uint8_t sliceaxis) {
    IMGID idin = imgid_make_from_name(inname), idout = imgid_make_from_name(outname);
    errno_t ret = image_slicenorm_IMGID(&idin, &idout, sliceaxis);
    imgid_free(&idin); imgid_free(&idout);
    return ret;
}
#endif

void image_norm_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg, IMAGE *outimg) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    if (!norm_sliceaxis) return;
    IMGID idin, idout; 
    idin = imgid_make(); idin.im = inimg; idin.md = &inimg->md[0]; imgid_update_creationparams(&idin);
    idout = imgid_make(); idout.im = outimg; idout.md = &outimg->md[0]; imgid_update_creationparams(&idout);
    image_slicenorm_IMGID(&idin, &idout, *norm_sliceaxis);
    imgid_free(&idin); imgid_free(&idout);
}


/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_normslice(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, NORMSLICE_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
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
    NORMSLICE_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_normslice(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        norm_inimname  = functionparameter_GetParamPtr_STRING(&fps, ".in0name");
        norm_outimname = functionparameter_GetParamPtr_STRING(&fps, ".outname");
        norm_sliceaxis = functionparameter_GetParamPtr_UINT32(&fps, ".axis");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { function_parameter_FPCONFloopstep(&fps); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(normslice)
FPS_MAKE_STANDALONE_RUNSTOP(normslice)

int FPSRUN_normslice(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    norm_inimname  = functionparameter_GetParamPtr_STRING(&fps, ".in0name");
    norm_outimname = functionparameter_GetParamPtr_STRING(&fps, ".outname");
    norm_sliceaxis = functionparameter_GetParamPtr_UINT32(&fps, ".axis");
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(norm_inimname, &iin) != 0) return 1;
    IMGID idin, idout; 
    idin = imgid_make(); idin.im = &iin; idin.md = &iin.md[0]; imgid_update_creationparams(&idin);
    idout = imgid_make();
    image_slicenorm_IMGID(&idin, &idout, *norm_sliceaxis);
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "normslice Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); image_norm_compute(&fps, pinfo, &iin, idout.im); processinfo_exec_end(pinfo);
        processinfo_update_output_stream(pinfo, idout.im, &iin);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); 
    imgid_free(&idin); imgid_free(&idout);
    return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("normslice", normslice, NORMSLICE_HELPTEXT, NORMSLICE_PARAMS)
#endif

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags) { fps_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    NORMSLICE_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "normslice", "image norm by slice", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID idin = imgid_make_from_name(norm_inimname); resolveIMGID(&idin, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    IMGID idout = imgid_make_from_name(norm_outimname);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_norm_compute(data.fpsptr, processinfo, idin.im, idout.im);
    processinfo_update_output_stream(processinfo, idout.im, idin.im);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    imgid_free(&idin); imgid_free(&idout);
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMOD_arith__image_normslice() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif