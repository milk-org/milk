/**
 * @file    fps_GetTypeString.c
 */


#include "CommandLineInterface/CLIcore.h"



errno_t functionparameter_GetTypeString(
    uint32_t type,
    char *typestring
)
{

    sprintf(typestring, " ");

    // using if statements (not switch) to allow for multiple types
    if(type & FPTYPE_UNDEF)
    {
        strcat(typestring, "UNDEF ");
    }

    if(type & FPTYPE_INT32)
    {
        strcat(typestring, "INT32 ");
    }
    if(type & FPTYPE_UINT32)
    {
        strcat(typestring, "UINT32 ");
    }
    if(type & FPTYPE_INT64)
    {
        strcat(typestring, "INT64 ");
    }
    if(type & FPTYPE_UINT64)
    {
        strcat(typestring, "UINT64 ");
    }


    if(type & FPTYPE_FLOAT64)
    {
        strcat(typestring, "FLOAT64 ");
    }
    if(type & FPTYPE_FLOAT32)
    {
        strcat(typestring, "FLOAT32 ");
    }


    if(type & FPTYPE_PID)
    {
        strcat(typestring, "PID ");
    }
    if(type & FPTYPE_TIMESPEC)
    {
        strcat(typestring, "TIMESPEC ");
    }
    if(type & FPTYPE_FILENAME)
    {
        strcat(typestring, "FILENAME ");
    }
    if(type & FPTYPE_FITSFILENAME)
    {
        strcat(typestring, "FITSFILENAME ");
    }
    if(type & FPTYPE_EXECFILENAME)
    {
        strcat(typestring, "EXECFILENAME");
    }
    if(type & FPTYPE_DIRNAME)
    {
        strcat(typestring, "DIRNAME");
    }
    if(type & FPTYPE_STREAMNAME)
    {
        strcat(typestring, "STREAMNAME");
    }
    if(type & FPTYPE_STRING)
    {
        strcat(typestring, "STRING ");
    }
    if(type & FPTYPE_ONOFF)
    {
        strcat(typestring, "ONOFF ");
    }
    if(type & FPTYPE_FPSNAME)
    {
        strcat(typestring, "FPSNAME ");
    }

    return RETURN_SUCCESS;
}

