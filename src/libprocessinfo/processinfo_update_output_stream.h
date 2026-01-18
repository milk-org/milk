#ifndef _PROCESSINFO_UPDATE_OUTPUT_STREAM_H
#define _PROCESSINFO_UPDATE_OUTPUT_STREAM_H

#include "processinfo.h"
#include "ImageStreamIO/ImageStruct.h"

errno_t processinfo_update_output_stream(PROCESSINFO *processinfo,
        IMAGE        *output_image,
        IMAGE        *input_image);

#endif
