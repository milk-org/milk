/**
 * @file    fps_printparameter_valuestring.c
 * @brief   print parameter value string
 */

#include "fps.h"
#include "fps_internal.h"
#include "ImageStreamIO/ImageStreamIO.h"

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
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s UINT32 %u", fpsentry->keywordfull, fpsentry->val.ui32[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_INT32:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s INT32  %d", fpsentry->keywordfull, fpsentry->val.i32[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_UINT64:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s UINT64 %lu", fpsentry->keywordfull, fpsentry->val.ui64[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_INT64:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s INT64  %ld", fpsentry->keywordfull, fpsentry->val.i64[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_FLOAT64:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s FLOAT64 %f", fpsentry->keywordfull, fpsentry->val.f64[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_FLOAT32:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s FLOAT32 %f", fpsentry->keywordfull, fpsentry->val.f32[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_PID:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s PID %ld", fpsentry->keywordfull, (long)fpsentry->val.pid[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_TIMESPEC:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s TIMESPEC %ld.%09ld", fpsentry->keywordfull, fpsentry->val.ts[0].tv_sec, fpsentry->val.ts[0].tv_nsec);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_FILENAME:
        case FPTYPE_FITSFILENAME:
        case FPTYPE_EXECFILENAME:
        case FPTYPE_DIRNAME:
        case FPTYPE_STREAMNAME:
        case FPTYPE_STRING:
        case FPTYPE_FPSNAME:
        case FPTYPE_PROCESS:
        case FPTYPE_STRING_NOT_STREAM:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s %s %s", fpsentry->keywordfull, 
                                     fpsentry->type == FPTYPE_FILENAME ? "FILENAME" :
                                     fpsentry->type == FPTYPE_FITSFILENAME ? "FITSFILENAME" :
                                     fpsentry->type == FPTYPE_EXECFILENAME ? "EXECFILENAME" :
                                     fpsentry->type == FPTYPE_DIRNAME ? "DIRNAME" :
                                     fpsentry->type == FPTYPE_STREAMNAME ? "STREAMNAME" :
                                     fpsentry->type == FPTYPE_FPSNAME ? "FPSNAME" :
                                     fpsentry->type == FPTYPE_PROCESS ? "PROCESS" :
                                     fpsentry->type == FPTYPE_STRING_NOT_STREAM ? "STRING_NOT_STREAM" : "STRING",
                                     fpsentry->val.string[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_ONOFF:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s ONOFF %s", fpsentry->keywordfull, 
                                     (fpsentry->val.i32[0]) ? "ON" : "OFF");
                if (_slen >= 0) cmdOK = 1;
            }
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

errno_t functionparameter_GetParamValueString(
    FUNCTION_PARAMETER *fpsentry,
    char *outstring,
    int stringmaxlen
)
{
    int cmdOK = 0;

    switch(fpsentry->type)
    {

        case FPTYPE_UINT32:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%u", fpsentry->val.ui32[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_INT32:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%d", fpsentry->val.i32[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_UINT64:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%lu", fpsentry->val.ui64[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_INT64:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%ld", fpsentry->val.i64[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_FLOAT64:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%f", fpsentry->val.f64[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_FLOAT32:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%f", fpsentry->val.f32[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_PID:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%ld", (long)fpsentry->val.pid[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_TIMESPEC:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%ld.%09ld", fpsentry->val.ts[0].tv_sec, fpsentry->val.ts[0].tv_nsec);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_STREAMNAME:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s", fpsentry->val.string[0]);
                if (_slen >= 0) {
                    IMAGE tmpimg;
                    if (ImageStreamIO_openIm(&tmpimg, fpsentry->val.string[0]) == IMAGESTREAMIO_SUCCESS) {
                        const char* type_str = ImageStreamIO_typename(tmpimg.md->datatype);
                        
                        char size_str[64];
                        if (tmpimg.md->naxis == 1) {
                            snprintf(size_str, 64, "%u", tmpimg.md->size[0]);
                        } else if (tmpimg.md->naxis == 2) {
                            snprintf(size_str, 64, "%ux%u", tmpimg.md->size[0], tmpimg.md->size[1]);
                        } else {
                            snprintf(size_str, 64, "%ux%ux%u", tmpimg.md->size[0], tmpimg.md->size[1], tmpimg.md->size[2]);
                        }

                        int _slen2 = snprintf(outstring + _slen, stringmaxlen - _slen, 
                                              " [%s %s cnt=%lu]", type_str, size_str, tmpimg.md->cnt0);
                        if (_slen2 >= 0) cmdOK = 1;
                        ImageStreamIO_closeIm(&tmpimg);
                    } else {
                        // Stream not found
                        int _slen2 = snprintf(outstring + _slen, stringmaxlen - _slen, " [NOTFOUND]");
                        if (_slen2 >= 0) cmdOK = 1;
                    }
                }
            }
            break;

        case FPTYPE_FILENAME:
        case FPTYPE_FITSFILENAME:
        case FPTYPE_EXECFILENAME:
        case FPTYPE_DIRNAME:
        case FPTYPE_STRING:
        case FPTYPE_FPSNAME:
        case FPTYPE_PROCESS:
        case FPTYPE_STRING_NOT_STREAM:
            {
                int _slen = snprintf(outstring, stringmaxlen, "%s", fpsentry->val.string[0]);
                if (_slen >= 0) cmdOK = 1;
            }
            break;

        case FPTYPE_ONOFF:
            if(fpsentry->val.i32[0])
            {
                int _slen = snprintf(outstring, stringmaxlen, "ON");
                if (_slen >= 0) cmdOK = 1;
            }
            else
            {
                int _slen = snprintf(outstring, stringmaxlen, "OFF");
                if (_slen >= 0) cmdOK = 1;
            }
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