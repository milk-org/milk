#ifndef FPS_INTERNAL_H
#define FPS_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#define RETURN_SUCCESS 0
#define RETURN_FAILURE 1

#define DEBUG_TRACE_FSTART(...)
#define DEBUG_TRACE_FEXIT(...)
#define DEBUG_TRACEPOINT(...)

#define PRINT_ERROR(...) fprintf(stderr, __VA_ARGS__)

#define SNPRINTF_CHECK(str, size, format, ...) \
    snprintf(str, size, format, ##__VA_ARGS__)

#include "ImageStreamIO/ImageStruct.h" // For STRINGMAXLEN_IMAGE_NAME if needed

#include "milkDebugTools.h" // for STRINGMAXLEN_FPSPROCESSTYPE

#endif
