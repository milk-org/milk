#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include "CLIcore.h"
#include "savefits.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"
#include "COREMOD_iofits_common.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "check_fitsio_status.h"
#include "file_exists.h"
#include "is_fits_file.h"

extern COREMOD_IOFITS_DATA COREMOD_iofits_data;
char *savefits_inimname  = NULL;
char *savefits_outfname  = NULL;
int  *savefits_outbitpix = NULL;
char *savefits_inheader  = NULL;
static uint64_t processinfo_change_cnt_local = 0;

errno_t saveFITS_opt_trunc_IMGID(IMGID *imgin, int truncate, const char *outputFITSname, int outputbitpix, const char *importheaderfile, IMAGE_KEYWORD *kwarray, int kwarraysize, const char *FITSIOext) {
    COREMOD_iofits_data.FITSIO_status = 0;
    pthread_t self_id = pthread_self();
    char fnametmp[STRINGMAXLEN_FILENAME], fnametmpext[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fnametmp, "%s.%d.%ld.tmp", outputFITSname, (int) getpid(), (long) self_id);
    WRITE_FILENAME(fnametmpext, "%s%s", fnametmp, FITSIOext);
    resolveIMGID(imgin, ERRMODE_WARN);
    if(imgin->ID == -1) return RETURN_SUCCESS;
    int bitpix = (outputbitpix != 0) ? outputbitpix : ImageStreamIO_FITSIObitpix(imgin->md->datatype);
    if (bitpix == -1) bitpix = FLOAT_IMG;
    fitsfile *fptr;
    fits_create_file(&fptr, fnametmpext, &COREMOD_iofits_data.FITSIO_status);
    if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0) return RETURN_FAILURE;
    int naxis = imgin->md->naxis; long naxesl[3], nelements = 1;
    for(int i = 0; i < naxis; i++) {
        naxesl[i] = (long) imgin->md->size[i];
        if (truncate >= 0 && i == naxis -1) naxesl[naxis - 1] = truncate;
        nelements *= naxesl[i];
    }
    fits_create_img(fptr, bitpix, naxis, naxesl, &COREMOD_iofits_data.FITSIO_status);
    if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0) { remove(fnametmp); return RETURN_FAILURE; }
    fits_write_img(fptr, ImageStreamIO_FITSIOdatatype(imgin->md->datatype), 1, nelements, imgin->im->array.raw, &COREMOD_iofits_data.FITSIO_status);
    fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
    rename(fnametmp, outputFITSname);
    return RETURN_SUCCESS;
}

errno_t saveFITS_opt_trunc(const char *inputimname, int truncate, const char *outputFITSname, int outputbitpix, const char *importheaderfile, IMAGE_KEYWORD *kwarray, int kwarraysize, const char *FITSIOext) {
    IMGID id = mkIMGID_from_name(inputimname);
    return saveFITS_opt_trunc_IMGID(&id, truncate, outputFITSname, outputbitpix, importheaderfile, kwarray, kwarraysize, FITSIOext);
}

errno_t save_fl_fits(const char *inputimname, const char *outputFITSname) {
    return saveFITS_opt_trunc(inputimname, -1, outputFITSname, -32, NULL, NULL, 0, "");
}

errno_t saveFITS(const char *inputimname, const char *outputFITSname, int outputbitpix, const char *importheaderfile, IMAGE_KEYWORD *kwarray, int kwarraysize) {
    return saveFITS_opt_trunc(inputimname, -1, outputFITSname, outputbitpix, importheaderfile, kwarray, kwarraysize, "");
}

errno_t saveall_fits(const char *savedirname) {
    for (int i = 0; i < data.NB_MAX_IMAGE; i++) {
        if (data.image[i].used == 1) {
            char fname[STRINGMAXLEN_FILENAME];
            WRITE_FILENAME(fname, "%s/%s.fits", savedirname, data.image[i].name);
            saveFITS(data.image[i].name, fname, 0, NULL, NULL, 0);
        }
    }
    return RETURN_SUCCESS;
}

errno_t save_fits(const char *inputimname, const char *outputFITSname) {
    return saveFITS(inputimname, outputFITSname, 0, NULL, NULL, 0);
}

void savefits_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *imgin) {
    if (fps && fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
        fps_to_processinfo(fps, processinfo); processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
    }
    if (!savefits_outfname || !savefits_outbitpix) return;
    IMGID id; id.im = imgin; id.md = &imgin->md[0];
    saveFITS_opt_trunc_IMGID(&id, -1, savefits_outfname, *savefits_outbitpix, savefits_inheader, NULL, 0, "");
}

/* ================================================================== */
/* STANDALONE IMPLEMENTATION                                          */

int FPSINIT_savefits(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    if (keywords) strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN-1);
    if (description) strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN-1);
    strncpy(fps.md->helptext, SAVEFITS_HELPTEXT, FPS_HELPTEXT_STRMAXLEN-1);
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    SAVEFITS_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps); return 0;
}

int FPSCONF_savefits(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (loop) {
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);
        savefits_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
        savefits_outfname = functionparameter_GetParamPtr_STRING(&fps, ".out_fname");
        savefits_outbitpix = (int*)functionparameter_GetParamPtr_INT32(&fps, ".bitpix");
        savefits_inheader = functionparameter_GetParamPtr_STRING(&fps, ".in_header");
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { function_parameter_FPCONFloopstep(&fps); usleep(10000); }
    } else { fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT); function_parameter_FPCONFloopstep(&fps); }
    function_parameter_FPCONFexit(&fps); return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(savefits)
FPS_MAKE_STANDALONE_RUNSTOP(savefits)

int FPSRUN_savefits(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) return 1;
    savefits_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
    savefits_outfname = functionparameter_GetParamPtr_STRING(&fps, ".out_fname");
    savefits_outbitpix = (int*)functionparameter_GetParamPtr_INT32(&fps, ".bitpix");
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(savefits_inimname, &iin) != 0) return 1;
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) {
        processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); savefits_compute(&fps, pinfo, &iin); processinfo_exec_end(pinfo);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE("savefits", savefits, SAVEFITS_HELPTEXT)
#endif

static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    SAVEFITS_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

static CLICMDDATA CLIcmddata = { "saveFITS", "save image as FITS", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

static errno_t compute_function() {
    IMGID in = mkIMGID_from_name(savefits_inimname); resolveIMGID(&in, ERRMODE_ABORT);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    savefits_compute(data.fpsptr, processinfo, in.im);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions
errno_t CLIADDCMD_COREMOD_iofits__saveFITS() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }