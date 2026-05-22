/**
 * @file processor.c
 * @brief Logic for ImageStreamIO + ProcessInfo + FPS integration.
 *
 * This file contains:
 * 1. Global parameter pointers (shared between standalone and module).
 * 2. core compute logic used in both run modes.
 * 3. Validation logic used in both configuration modes.
 * 4. Standalone implementation of FPS/ProcessInfo hooks (main, start/stop).
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

// ProcessInfo headers
#include "processinfo.h"
#include "processtools.h"

#include "ImageStreamIO.h"
#include "processor.h"

/* =============================================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                                                      */
/* =============================================================================================== */
char     *in_name_ptr       = NULL;
char     *proc_out_name_ptr = NULL;
uint32_t *roi_size_ptr      = NULL;
uint32_t *off_x_ptr         = NULL;

/** @brief Tracks whether processinfo settings need to be re-synced from FPS. */
static uint64_t processinfo_change_cnt_local = 0;


/* =============================================================================================== */
/* SHARED LOGIC                                                                                    */
/* =============================================================================================== */

/**
 * @brief Core ROI extraction computation.
 *
 * This function is called by both the standalone RUN loop and the CLI module compute function.
 */
void processor03_compute(FUNCTION_PARAMETER_STRUCT *fps,
                         PROCESSINFO               *processinfo,
                         IMAGE                     *input_image,
                         IMAGE                     *output_image)
{
    // Sync processinfo settings (priority, CPU mask, etc.) if they were changed via FPS TUI/CLI
    if (fps)
    {
        if (fps->md->processinfo_change_cnt != processinfo_change_cnt_local)
        {
            fps_to_processinfo(fps, processinfo);
            processinfo_change_cnt_local = fps->md->processinfo_change_cnt;
        }
    }

    if (!off_x_ptr || !roi_size_ptr)
    {
        return;
    }

    uint32_t off_x    = *off_x_ptr;
    uint32_t roi_size = *roi_size_ptr;

    uint32_t in_w     = input_image->md[0].size[0];
    float   *in_data  = (float *) input_image->array.raw;
    float   *out_data = (float *) output_image->array.raw;

    // Simple 2D copy with horizontal offset
    for (uint32_t y = 0; y < roi_size; y++)
    {
        for (uint32_t x = 0; x < roi_size; x++)
        {
            if (x + off_x < in_w)
            {
                out_data[y * roi_size + x] = in_data[y * in_w + (x + off_x)];
            }
            else
            {
                out_data[y * roi_size + x] = 0; // Padding if offset is out of bounds
            }
        }
    }

    // Visual progress indicator
    printf(".");
    fflush(stdout);
}

/**
 * @brief Validates parameter constraints.
 *
 * Ensures the requested ROI window is physically possible within the current input stream.
 * Automatically adjusts offset or size to maintain validity.
 */
void processor03_validate()
{
    if (!in_name_ptr || !roi_size_ptr || !off_x_ptr)
    {
        return;
    }

    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name_ptr, &input_image) == 0)
    {
        uint32_t width = input_image.md[0].size[0];

        if (*off_x_ptr + *roi_size_ptr > width)
        {
            if (*off_x_ptr > width)
            {
                *off_x_ptr = 0;
            }
            if (*off_x_ptr + *roi_size_ptr > width)
            {
                if (*roi_size_ptr > width)
                {
                    *roi_size_ptr = width;
                    *off_x_ptr    = 0;
                }
                else
                {
                    *off_x_ptr = width - *roi_size_ptr;
                }
            }
        }
        ImageStreamIO_closeIm(&input_image);
    }
}


/* =============================================================================================== */
/* STANDALONE IMPLEMENTATION                                                                       */
/* =============================================================================================== */

/**
 * @brief Initialize the FPS shared memory segment.
 *
 * Uses the PROCESSOR_PARAMS X-Macro to populate the structure.
 */
int FPSINIT_processor(const char *fps_name, const char *keywords, const char *description)
{
    FUNCTION_PARAMETER_STRUCT fps;
    FPS_INIT_STD_PREAMBLE(fps, fps_name, keywords, description, PROCESSOR_HELPTEXT);
    FPS_INIT_PROCINFO_DEFAULTS(fps, "stream03", 10);

    // Use X-Macro to add all parameters to the FPS
#define X_FPS_INIT(fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags)  \
    {                                                                                    \
        c_type val  = def_val;                                                           \
        void  *vptr = &val;                                                              \
        if (FPTYPE_IS_STRING(fps_type))                                                  \
        {                                                                                \
            vptr = *(void **) &val;                                                      \
        }                                                                                \
        function_parameter_add_entry(&fps, key, descr, fps_type, cli_flags, vptr, NULL); \
    }
    PROCESSOR_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT

    // Add standard ProcessInfo parameters (CPU mask, RT priority, etc.)
    fps_add_processinfo_entries(&fps);
    functionparameter_SetParamValue_ONOFF(&fps, ".procinfo.MeasureTiming", 1);

    function_parameter_FPCONFexit(&fps);
    return 0;
}

/**
 * @brief Run the configuration monitoring loop.
 *
 * Links the local pointers to the shared memory entries and runs the validation logic.
 */
int FPSCONF_processor(const char *fps_name, int loop)
{
    FPS_CONF_STD_BODY(
        fps_name, loop,
        {
            // Map local pointers to FPS shared memory entries
            in_name_ptr  = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
            roi_size_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".roi_size");
            off_x_ptr    = functionparameter_GetParamPtr_UINT32(&fps, ".off_x");

            if (!in_name_ptr || !roi_size_ptr || !off_x_ptr)
            {
                fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
                function_parameter_FPCONFexit(&fps);
                return 1;
            }
        },
        { processor03_validate(); });
    return 0;
}

// Generate standard stop commands
FPS_MAKE_STANDALONE_CONFSTOP(processor)
FPS_MAKE_STANDALONE_RUNSTOP(processor)

/**
 * @brief Main processing loop (RUN mode).
 *
 * Synchronizes with the input stream and performs the ROI extraction.
 */
int FPSRUN_processor(const char *fps_name)
{
    FUNCTION_PARAMETER_STRUCT fps;
    FPS_RUN_STD_PREAMBLE(fps_name, fps, {
        // Map local pointers
        in_name_ptr       = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
        proc_out_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".out_name");
        roi_size_ptr      = functionparameter_GetParamPtr_UINT32(&fps, ".roi_size");
        off_x_ptr         = functionparameter_GetParamPtr_UINT32(&fps, ".off_x");

        if (!in_name_ptr || !proc_out_name_ptr || !roi_size_ptr || !off_x_ptr)
        {
            fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
            function_parameter_struct_disconnect(&fps);
            return 1;
        }
    });

    // Connect to source stream
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name_ptr, &input_image) != 0)
    {
        fprintf(stderr, "Error connecting to input %s\n", in_name_ptr);
        return 1;
    }

    // Create/Connect to output stream
    IMAGE    output_image;
    uint32_t dims[2] = { *roi_size_ptr, *roi_size_ptr };
    if (ImageStreamIO_createIm_gpu(&output_image, proc_out_name_ptr, 2, dims, _DATATYPE_FLOAT, -1,
                                   1, 10, 0, 0, 0) != 0)
    {
        return 1;
    }

    PROCESSINFO *processinfo;
    FPS_RUN_PROCESSINFO_SETUP(processinfo, fps_name, "Ex03 Run", "Looping", &input_image, fps);

    FPS_RUN_PROCESSINFO_LOOP(processinfo, fps, &input_image, &output_image, {
        // Perform computation
        processor03_compute(&fps, processinfo, &input_image, &output_image);
    });

    return 0;
}

// Generate the standard main function for standalone build
#ifndef MILK_MODULE
FPS_MAIN_STANDALONE("processor03", processor, PROCESSOR_HELPTEXT, PROCESSOR_PARAMS)
#endif
