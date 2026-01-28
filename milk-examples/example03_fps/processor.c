/**
 * @file milk-example-03-processor.c
 * @brief Integration of ImageStreamIO, libprocessinfo, and libfps.
 *
 * This source file contains:
 * 1. Global parameter pointers (shared with module).
 * 2. Shared processing and validation logic.
 * 3. Standalone implementation (FPS loop, main).
 *
 * It is compiled as a standalone executable (processor.c)
 * AND linked into the shared object module (processor03.so) via CMake.
 * When compiled for the module (-DMILK_MODULE), main() is excluded.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <sys/mman.h>

// Main FPS headers
#include "fps.h"
#include "fps_add_entry.h"
#include "fps_paramvalue.h"
#include "fps_FPCONFsetup.h"
#include "fps_FPCONFloopstep.h"
#include "fps_FPCONFexit.h"
#include "fps_CONFstop.h"
#include "fps_RUNstop.h"
#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_RUNexit.h"
#include "fps_tmux.h"
#include "fps_processinfo.h"

// ProcessInfo headers
#include "processinfo.h"
#include "processinfo_shm_link.h"
#include "processinfo_procdirname.h"
#include "processtools_trigger.h"
#include "processinfo_update_output_stream.h"
#include "processinfo_setup.h"
#include "processinfo_loopstep.h"
#include "processinfo_exec_start.h"
#include "processinfo_exec_end.h"
#include "processinfo_signals.h"
#include "fps_processinfo_entries.h"

#include "ImageStreamIO.h"
#include "processor.h"

// ===============================================================================================
// GLOBAL PARAMETERS (SHARED)
// ===============================================================================================
char *in_name_ptr = NULL;
char *proc_out_name_ptr = NULL;
uint32_t *roi_size_ptr = NULL;
uint32_t *off_x_ptr = NULL;

static uint64_t processinfo_change_cnt_local = 0;


/* =============================================================================================== */
/* =============================================================================================== */
/* SHARED LOGIC                                                                                    */
/* =============================================================================================== */

/**
 * @brief Shared processing logic for one loop iteration.
 */
void processor03_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *input_image,
    IMAGE *output_image)
{
    if (fps) {
        if(fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
            fps_to_processinfo(fps, processinfo);
            processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
        }
    }

    if (!off_x_ptr || !roi_size_ptr) return;

    uint32_t off_x = *off_x_ptr;
    uint32_t roi_size = *roi_size_ptr;

    uint32_t in_w = input_image->md[0].size[0];
    float *in_data = (float*)input_image->array.raw;
    float *out_data = (float*)output_image->array.raw;

    for(uint32_t y=0; y<roi_size; y++) {
        for(uint32_t x=0; x<roi_size; x++) {
            if (x + off_x < in_w)
                out_data[y*roi_size + x] = in_data[y*in_w + (x + off_x)];
            else
                out_data[y*roi_size + x] = 0;
        }
    }
    printf(".");
    fflush(stdout);
}

/**
 * @brief Shared validation logic for configuration loop.
 */
void processor03_validate() {
    if (!in_name_ptr || !roi_size_ptr || !off_x_ptr) return;

    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name_ptr, &input_image) == 0) {
        uint32_t width = input_image.md[0].size[0];

        if (*off_x_ptr + *roi_size_ptr > width) {
            if (*off_x_ptr > width) {
                *off_x_ptr = 0;
            }
            if (*off_x_ptr + *roi_size_ptr > width) {
                if (*roi_size_ptr > width) {
                    *roi_size_ptr = width;
                    *off_x_ptr = 0;
                } else {
                    *off_x_ptr = width - *roi_size_ptr;
                }
            }
        }
        ImageStreamIO_closeIm(&input_image);
    }
}


/* =============================================================================================== */
/* =============================================================================================== */
/* STANDALONE IMPLEMENTATION                                                                       */
/* =============================================================================================== */

int FPSINIT_processor(
    const char *fps_name,
    const char *keywords,
    const char *description)
{
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Initializing FPS '%s'...\n", fps_name);

    fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    strncpy(fps.md->sourcefname, __FILE__, FPS_SRCDIR_STRLENMAX - 1);
    fps.md->sourceline = __LINE__;

    if (keywords != NULL) {
        strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN - 1);
    }
    if (description != NULL) {
        strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN - 1);
    }

    // Set detailed help text
    strncpy(fps.md->helptext, PROCESSOR_HELPTEXT, FPS_HELPTEXT_STRMAXLEN - 1);

    strncpy(fps.cmdset.triggerstreamname, "stream03", STRINGMAXLEN_IMAGE_NAME - 1);
    fps.cmdset.procinfo_loopcntMax = -1;
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
    fps.cmdset.triggertimeout.tv_sec = 10;
    fps.cmdset.triggertimeout.tv_nsec = 0;

#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) \
    { \
        c_type val = def_val; \
        function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); \
    }
    PROCESSOR_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT

    fps_add_processinfo_entries(&fps);
    functionparameter_SetParamValue_ONOFF(&fps, ".procinfo.MeasureTiming", 1);

    function_parameter_FPCONFexit(&fps);
    return 0;
}

int FPSCONF_processor(
    const char *fps_name,
    int loop)
{
    FUNCTION_PARAMETER_STRUCT fps;

    if (loop) {
        printf("Starting configuration process loop for '%s'\n", fps_name);
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);

        in_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
        roi_size_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".roi_size");
        off_x_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".off_x");

        if (!in_name_ptr || !roi_size_ptr || !off_x_ptr) {
            fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
            function_parameter_FPCONFexit(&fps);
            return 1;
        }

        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) {
            if (function_parameter_FPCONFloopstep(&fps)) {
                processor03_validate();
            }
            usleep(10000);
        }
    } else {
        printf("Running single configuration step for '%s'\n", fps_name);
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
        function_parameter_FPCONFloopstep(&fps);
    }

    function_parameter_FPCONFexit(&fps);
    return 0;
}

FPS_MAKE_STANDALONE_CONFSTOP(processor)
FPS_MAKE_STANDALONE_RUNSTOP(processor)

int FPSRUN_processor(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;

    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found. Run 'fpsinit' first.\n", fps_name);
        return 1;
    }

    in_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
    proc_out_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".out_name");
    roi_size_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".roi_size");
    off_x_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".off_x");

    if (!in_name_ptr || !proc_out_name_ptr || !roi_size_ptr || !off_x_ptr) {
        fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
        function_parameter_struct_disconnect(&fps);
        return 1;
    }

    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name_ptr, &input_image) != 0) {
        fprintf(stderr, "Error connecting to input %s\n", in_name_ptr);
        return 1;
    }

    IMAGE output_image;
    uint32_t dims[2] = {*roi_size_ptr, *roi_size_ptr};
    if (ImageStreamIO_createIm_gpu(&output_image, proc_out_name_ptr, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0) != 0) {
        return 1;
    }

    PROCESSINFO *processinfo = processinfo_setup((char*)fps_name, "Ex03 Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    if (!processinfo) return 1;

    processinfo_CatchSignals();
    processinfo_waitoninputstream_init(processinfo, &input_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, processinfo);
    processinfo_loopstart(processinfo);

    int loopOK = 1;
    while(loopOK) {
        loopOK = processinfo_loopstep(processinfo);
        if(!loopOK) break;

        processinfo_waitoninputstream(processinfo);
        if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;

        processinfo_exec_start(processinfo);

        processor03_compute(&fps, processinfo, &input_image, &output_image);

        processinfo_exec_end(processinfo);
        processinfo_update_output_stream(processinfo, &output_image, &input_image);
    }

    processinfo_cleanExit(processinfo);
    function_parameter_struct_disconnect(&fps);
    return 0;
}

#ifndef MILK_MODULE
FPS_MAIN_STANDALONE("processor03", processor, PROCESSOR_HELPTEXT)
#endif
