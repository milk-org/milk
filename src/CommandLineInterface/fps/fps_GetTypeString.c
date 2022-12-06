/**
 * @file    fps_GetTypeString.c
 */

#include "CommandLineInterface/CLIcore.h"

errno_t functionparameter_GetTypeString(
    uint32_t type,
    char *typestring
)
{

    snprintf(typestring, STRINGMAXLEN_FPSTYPE, " ");

    // using if statements (not switch) to allow for multiple types
    if(type & FPTYPE_UNDEF)
    {
        strncat(typestring, "UNDEF ", STRINGMAXLEN_FPSTYPE - 1);
    }

    if(type & FPTYPE_INT32)
    {
        strncat(typestring, "INT32 ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_UINT32)
    {
        strncat(typestring, "UINT32 ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_INT64)
    {
        strncat(typestring, "INT64 ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_UINT64)
    {
        strncat(typestring, "UINT64 ", STRINGMAXLEN_FPSTYPE - 1);
    }

    if(type & FPTYPE_FLOAT64)
    {
        strncat(typestring, "FLOAT64 ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_FLOAT32)
    {
        strncat(typestring, "FLOAT32 ", STRINGMAXLEN_FPSTYPE - 1);
    }

    if(type & FPTYPE_PID)
    {
        strncat(typestring, "PID ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_TIMESPEC)
    {
        strncat(typestring, "TIMESPEC ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_FILENAME)
    {
        strncat(typestring, "FILENAME ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_FITSFILENAME)
    {
        strncat(typestring, "FITSFILENAME ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_EXECFILENAME)
    {
        strncat(typestring, "EXECFILENAME", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_DIRNAME)
    {
        strncat(typestring, "DIRNAME", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_STREAMNAME)
    {
        strncat(typestring, "STREAMNAME", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_STRING)
    {
        strncat(typestring, "STRING ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_ONOFF)
    {
        strncat(typestring, "ONOFF ", STRINGMAXLEN_FPSTYPE - 1);
    }
    if(type & FPTYPE_FPSNAME)
    {
        strncat(typestring, "FPSNAME ", STRINGMAXLEN_FPSTYPE - 1);
    }

    return RETURN_SUCCESS;
}
