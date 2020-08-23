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
    case FPTYPE_INT64:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s INT64      %ld %ld %ld %ld",
            fpsentry->keywordfull,
            fpsentry->val.l[0],
            fpsentry->val.l[1],
            fpsentry->val.l[2],
            fpsentry->val.l[3]);
        cmdOK = 1;
        break;

    case FPTYPE_FLOAT64:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s FLOAT64    %f %f %f %f",
            fpsentry->keywordfull,
            fpsentry->val.f[0],
            fpsentry->val.f[1],
            fpsentry->val.f[2],
            fpsentry->val.f[3]);
        cmdOK = 1;
        break;

    case FPTYPE_FLOAT32:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s FLOAT32    %f %f %f %f",
            fpsentry->keywordfull,
            fpsentry->val.s[0],
            fpsentry->val.s[1],
            fpsentry->val.s[2],
            fpsentry->val.s[3]);
        cmdOK = 1;
        break;

    case FPTYPE_PID:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s PID        %ld",
            fpsentry->keywordfull,
            fpsentry->val.l[0]);
        cmdOK = 1;
        break;

    case FPTYPE_TIMESPEC:
        //
        break;

    case FPTYPE_FILENAME:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s FILENAME   %s",
            fpsentry->keywordfull,
            fpsentry->val.string[0]);
        cmdOK = 1;
        break;

    case FPTYPE_FITSFILENAME:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s FITSFILENAME   %s",
            fpsentry->keywordfull,
            fpsentry->val.string[0]);
        cmdOK = 1;
        break;

    case FPTYPE_EXECFILENAME:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s EXECFILENAME   %s",
            fpsentry->keywordfull,
            fpsentry->val.string[0]);
        cmdOK = 1;
        break;

    case FPTYPE_DIRNAME:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s DIRNAME    %s",
            fpsentry->keywordfull,
            fpsentry->val.string[0]);
        cmdOK = 1;
        break;

    case FPTYPE_STREAMNAME:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s STREAMNAME %s",
            fpsentry->keywordfull,
            fpsentry->val.string[0]);
        cmdOK = 1;
        break;

    case FPTYPE_STRING:
        SNPRINTF_CHECK(
            outstring,
            stringmaxlen,
            "%-40s STRING     %s",
            fpsentry->keywordfull,
            fpsentry->val.string[0]);
        cmdOK = 1;
        break;

    case FPTYPE_ONOFF:
        if(fpsentry->fpflag & FPFLAG_ONOFF)
        {
            SNPRINTF_CHECK(outstring, stringmaxlen, "%-40s ONOFF      ON",
                           fpsentry->keywordfull);
        }
        else
        {
            SNPRINTF_CHECK(outstring, stringmaxlen, "%-40s ONOFF      OFF",
                           fpsentry->keywordfull);
        }
        cmdOK = 1;
        break;


    case FPTYPE_FPSNAME:
        SNPRINTF_CHECK(outstring, stringmaxlen, "%-40s FPSNAME   %s",
                       fpsentry->keywordfull, fpsentry->val.string[0]);
        cmdOK = 1;
        break;

    }


	if(cmdOK==1)
	{
		return RETURN_SUCCESS;
	}
	else
	{
		return RETURN_FAILURE;
	}

}
