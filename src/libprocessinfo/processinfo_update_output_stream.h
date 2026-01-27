#ifndef _PROCESSINFO_UPDATE_OUTPUT_STREAM_H
#define _PROCESSINFO_UPDATE_OUTPUT_STREAM_H

#include "processinfo.h"
#include "ImageStreamIO/ImageStruct.h"

/**
 * @brief Update output stream metadata and telemetry.
 *
 * This function updates the output image's shared memory metadata, including
 * write PID, timestamp, and propagates processing trace from input to output.
 *
 * @param[in,out] processinfo  Pointer to the PROCESSINFO structure.
 * @param[in,out] output_image Pointer to the output stream's IMAGE.
 * @param[in]     input_image  Pointer to the input stream's IMAGE.
 * @return RETURN_SUCCESS on success.
 */
errno_t processinfo_update_output_stream(PROCESSINFO *processinfo,
        IMAGE        *output_image,
        IMAGE        *input_image);

#endif