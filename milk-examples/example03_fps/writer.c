/**
 * @file writer.c
 * @brief Dynamic pattern generator with full FPS and ProcessInfo support.
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
#include "writer.h"

/* =============================================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                                                      */
/* =============================================================================================== */
char *out_name_ptr = NULL;
uint32_t *width_ptr = NULL;
uint32_t *height_ptr = NULL;
float *freq_x_ptr = NULL;
float *freq_y_ptr = NULL;

static uint64_t processinfo_change_cnt_local = 0;

/* =============================================================================================== */
/* SHARED LOGIC                                                                                    */
/* =============================================================================================== */

/**
 * @brief Computation logic for pattern generation.
 * 
 * Generates a scrolling 2D pattern based on the current loop counter.
 */
void writer03_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *output_image)
{
    // Sync external FPS changes to local ProcessInfo
    if (fps) {
        if(fps->md->processinfo_change_cnt != processinfo_change_cnt_local) {
            fps_to_processinfo(fps, processinfo);
            processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
        }
    }

    if (!width_ptr || !height_ptr || !freq_x_ptr || !freq_y_ptr) return;

    uint32_t width = *width_ptr;
    uint32_t height = *height_ptr;
    float freq_x = *freq_x_ptr;
    float freq_y = *freq_y_ptr;

    float *out_data = (float*)output_image->array.raw;
    uint64_t counter = processinfo->loopcnt;

    // Pattern generation: sum of sine waves
    for(uint32_t y=0; y<height; y++) {
        for(uint32_t x=0; x<width; x++) {
            out_data[y*width + x] = 0.5 * sin((x + counter)*freq_x) + 0.5 * cos((y + counter)*freq_y);
        }
    }
}

/**
 * @brief Basic parameter validation.
 */
void writer03_validate() {
    if (width_ptr && *width_ptr < 1) *width_ptr = 1;
    if (height_ptr && *height_ptr < 1) *height_ptr = 1;
}

/* =============================================================================================== */
/* STANDALONE IMPLEMENTATION                                                                       */
/* =============================================================================================== */

/**
 * @brief Initialize FPS metadata for the writer.
 */
int FPSINIT_writer(
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

    strncpy(fps.md->helptext, WRITER_HELPTEXT, FPS_HELPTEXT_STRMAXLEN - 1);

    fps.cmdset.procinfo_loopcntMax = -1;
    // Free-running mode
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_IMMEDIATE; 

#define X_FPS_INIT(fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags) \
{ \
    c_type val = def_val; \
    void *vptr = &val; \
    if (FPTYPE_IS_STRING(fps_type)) { \
        vptr = *(void**)&val; \
    } \
    function_parameter_add_entry(&fps, key, descr, fps_type, cli_flags, vptr, NULL); \
}
    WRITER_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT

    fps_add_processinfo_entries(&fps);
    functionparameter_SetParamValue_ONOFF(&fps, ".procinfo.MeasureTiming", 1);

    function_parameter_FPCONFexit(&fps);
    return 0;
}

/**
 * @brief Config loop: handles real-time parameter validation.
 */
int FPSCONF_writer(
    const char *fps_name,
    int loop)
{
    FUNCTION_PARAMETER_STRUCT fps;

    if (loop) {
        printf("Starting configuration process loop for '%s'\n", fps_name);
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);

        out_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".out_name");
        width_ptr    = functionparameter_GetParamPtr_UINT32(&fps, ".width");
        height_ptr   = functionparameter_GetParamPtr_UINT32(&fps, ".height");
        freq_x_ptr   = functionparameter_GetParamPtr_FLOAT32(&fps, ".freq_x");
        freq_y_ptr   = functionparameter_GetParamPtr_FLOAT32(&fps, ".freq_y");

        if (!out_name_ptr || !width_ptr || !height_ptr || !freq_x_ptr || !freq_y_ptr) {
            fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
            function_parameter_FPCONFexit(&fps);
            return 1;
        }

        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) {
            if (function_parameter_FPCONFloopstep(&fps)) {
                writer03_validate();
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

FPS_MAKE_STANDALONE_CONFSTOP(writer)
FPS_MAKE_STANDALONE_RUNSTOP(writer)

/**
 * @brief Main execution loop for the writer.
 */
int FPSRUN_writer(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;

    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found. Run 'fpsinit' first.\n", fps_name);
        return 1;
    }

    out_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".out_name");
    width_ptr    = functionparameter_GetParamPtr_UINT32(&fps, ".width");
    height_ptr   = functionparameter_GetParamPtr_UINT32(&fps, ".height");
    freq_x_ptr   = functionparameter_GetParamPtr_FLOAT32(&fps, ".freq_x");
    freq_y_ptr   = functionparameter_GetParamPtr_FLOAT32(&fps, ".freq_y");

    if (!out_name_ptr || !width_ptr || !height_ptr || !freq_x_ptr || !freq_y_ptr) {
        fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
        function_parameter_struct_disconnect(&fps);
        return 1;
    }

    IMAGE output_image;
    uint32_t dims[2] = {*width_ptr, *height_ptr};
    if (ImageStreamIO_createIm_gpu(&output_image, out_name_ptr, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0) != 0) {
        return 1;
    }

    PROCESSINFO *processinfo = processinfo_setup((char*)fps_name, "Ex03 Writer", "Looping", __FUNCTION__, __FILE__, __LINE__);
    if (!processinfo) return 1;

    processinfo_CatchSignals();
    fps_to_processinfo(&fps, processinfo);
    processinfo_loopstart(processinfo);

    int loopOK = 1;
    while(loopOK) {
        loopOK = processinfo_loopstep(processinfo);
        if(!loopOK) break;

        processinfo_exec_start(processinfo);

        writer03_compute(&fps, processinfo, &output_image);

        processinfo_exec_end(processinfo);
        // Post frame to shared memory (increment cnt0, post semaphores)
        processinfo_update_output_stream(processinfo, &output_image, NULL);
        
        usleep(10000); // Target 100 Hz
    }

    processinfo_cleanExit(processinfo);
    function_parameter_struct_disconnect(&fps);
    return 0;
}

#ifndef MILK_MODULE
FPS_MAIN_STANDALONE("writer03", writer, WRITER_HELPTEXT, WRITER_PARAMS)
#endif