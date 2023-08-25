/**
 * @file    fps_printparameter_valuestring.h
 * @brief   print parameter value string
 */

#include "CommandLineInterface/CLIcore.h"

errno_t functionparameter_PrintParameter_ValueString(
    FUNCTION_PARAMETER *fpsentry,
    char *outstring,
    int stringmaxlen
)
{
    int cmdOK = 0;

    switch(fpsentry->type)
    {

        case FPTYPE_UINT32:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s UINT32 %u",
                           fpsentry->keywordfull,
                           fpsentry->val.ui32[0]);
            cmdOK = 1;
            break;

        case FPTYPE_INT32:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s INT32  %d",
                           fpsentry->keywordfull,
                           fpsentry->val.i32[0]);
            cmdOK = 1;
            break;

        case FPTYPE_UINT64:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s UINT64 %lu",
                           fpsentry->keywordfull,
                           fpsentry->val.ui64[0]);
            cmdOK = 1;
            break;

        case FPTYPE_INT64:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s INT64  %ld",
                           fpsentry->keywordfull,
                           fpsentry->val.i64[0]);
            cmdOK = 1;
            break;

        case FPTYPE_FLOAT64:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s FLOAT64 %f",
                           fpsentry->keywordfull,
                           fpsentry->val.f64[0]);
            cmdOK = 1;
            break;

        case FPTYPE_FLOAT32:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s FLOAT32 %f",
                           fpsentry->keywordfull,
                           fpsentry->val.f32[0]);
            cmdOK = 1;
            break;

        case FPTYPE_PID:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s PID %ld",
                           fpsentry->keywordfull,
                           fpsentry->val.i64[0]);
            cmdOK = 1;
            break;

        case FPTYPE_TIMESPEC:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s TIMESPEC %ld.%09ld",
                           fpsentry->keywordfull,
                           fpsentry->val.ts->tv_sec,
                           fpsentry->val.ts->tv_nsec);
            break;

        case FPTYPE_FILENAME:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s FILENAME %s",
                           fpsentry->keywordfull,
                           fpsentry->val.string[0]);
            cmdOK = 1;
            break;

        case FPTYPE_FITSFILENAME:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s FITSFILENAME %s",
                           fpsentry->keywordfull,
                           fpsentry->val.string[0]);
            cmdOK = 1;
            break;

        case FPTYPE_EXECFILENAME:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s EXECFILENAME %s",
                           fpsentry->keywordfull,
                           fpsentry->val.string[0]);
            cmdOK = 1;
            break;

        case FPTYPE_DIRNAME:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s DIRNAME %s",
                           fpsentry->keywordfull,
                           fpsentry->val.string[0]);
            cmdOK = 1;
            break;

        case FPTYPE_STREAMNAME:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s STREAMNAME %s",
                           fpsentry->keywordfull,
                           fpsentry->val.string[0]);
            cmdOK = 1;
            break;

        case FPTYPE_STRING:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s STRING %s",
                           fpsentry->keywordfull,
                           fpsentry->val.string[0]);
            cmdOK = 1;
            break;

        case FPTYPE_ONOFF:
            if(fpsentry->fpflag & FPFLAG_ONOFF)
            {
                SNPRINTF_CHECK(outstring,
                               stringmaxlen,
                               "%s ONOFF ON",
                               fpsentry->keywordfull);
            }
            else
            {
                SNPRINTF_CHECK(outstring,
                               stringmaxlen,
                               "%s ONOFF OFF",
                               fpsentry->keywordfull);
            }
            cmdOK = 1;
            break;

        case FPTYPE_FPSNAME:
            SNPRINTF_CHECK(outstring,
                           stringmaxlen,
                           "%s FPSNAME %s",
                           fpsentry->keywordfull,
                           fpsentry->val.string[0]);
            cmdOK = 1;
            break;
    }

    if(cmdOK == 1)
    {
        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
    }
}
