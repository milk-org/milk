/**
 * @file    fps_paramvalue.c
 * @brief   set and get parameter values
 */



#include "CommandLineInterface/CLIcore.h"


long functionparameter_GetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    long value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.l[0];
    fps->parray[fpsi].val.l[3] = value;

    return value;
}


int functionparameter_SetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    long value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.l[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}


long *functionparameter_GetParamPtr_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    long *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.l[0];

    return ptr;
}





double functionparameter_GetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    double value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.f[0];
    fps->parray[fpsi].val.f[3] = value;

    return value;
}

int functionparameter_SetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    double value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.f[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}

double *functionparameter_GetParamPtr_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    double *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.f[0];

    return ptr;
}


float functionparameter_GetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    float value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.s[0];
    fps->parray[fpsi].val.s[3] = value;

    return value;
}

int functionparameter_SetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    float value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.s[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}

float *functionparameter_GetParamPtr_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    float *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.s[0];

    return ptr;
}



char *functionparameter_GetParamPtr_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    return fps->parray[fpsi].val.string[0];
}

int functionparameter_SetParamValue_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    const char *stringvalue
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    strncpy(fps->parray[fpsi].val.string[0], stringvalue,
            FUNCTION_PARAMETER_STRMAXLEN);
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}




int functionparameter_GetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    if(fps->parray[fpsi].fpflag & FPFLAG_ONOFF)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}



int functionparameter_SetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int ONOFFvalue
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    if(ONOFFvalue == 1)
    {
        fps->parray[fpsi].fpflag |= FPFLAG_ONOFF;
        fps->parray[fpsi].val.l[0] = 1;
    }
    else
    {
        fps->parray[fpsi].fpflag &= ~FPFLAG_ONOFF;
        fps->parray[fpsi].val.l[0] = 0;
    }

    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}



uint64_t *functionparameter_GetParamPtr_fpflag(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    uint64_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].fpflag;

    return ptr;
}




