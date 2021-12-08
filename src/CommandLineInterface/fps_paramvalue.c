/**
 * @file    fps_paramvalue.c
 * @brief   set and get parameter values
 */



#include "CommandLineInterface/CLIcore.h"


#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_paramvalue.h"
#include "fps_GetParamIndex.h"





/**
 * @brief  Get pointer to value and FPS index
 *
 */
int64_t *functionparameter_GetParamPtr_generic(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    long *paramindex
)
{
    int64_t *ptr;

    long fpsi = functionparameter_GetParamIndex(fps, paramname);

    // type is arbitrary
    ptr = &fps->parray[fpsi].val.i64[0];

    if(paramindex != NULL)
    {
        *paramindex = fpsi;
    }

    return ptr;
}











// INT64


int64_t functionparameter_GetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    int64_t value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.i64[0];
    fps->parray[fpsi].val.i64[3] = value;

    return value;
}


errno_t functionparameter_SetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int64_t value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.i64[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}


//
// stand-alone function to set parameter value
//
errno_t function_parameter_SetValue_int64(
    char *keywordfull,
    int64_t val
)
{
    FUNCTION_PARAMETER_STRUCT fps;
    char tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                        FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int keywordlevel = 0;
    char *pch;


    // break full keyword into keywords
    strncpy(tmpstring, keywordfull,
            FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL - 1);
    keywordlevel = 0;
    pch = strtok(tmpstring, ".");
    while(pch != NULL)
    {
        strncpy(keyword[keywordlevel], pch, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN - 1);
        keywordlevel++;
        pch = strtok(NULL, ".");
    }

    function_parameter_struct_connect(keyword[9], &fps, FPSCONNECT_SIMPLE);

    int pindex = functionparameter_GetParamIndex(&fps, keywordfull);


    fps.parray[pindex].val.i64[0] = val;

    function_parameter_struct_disconnect(&fps);

    return RETURN_SUCCESS;
}


int64_t *functionparameter_GetParamPtr_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    int64_t *ptr;

    long fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.i64[0];

    return ptr;
}










uint64_t functionparameter_GetParamValue_UINT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    uint64_t value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.ui64[0];
    fps->parray[fpsi].val.ui64[3] = value;

    return value;
}


errno_t functionparameter_SetParamValue_UINT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    uint64_t value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.ui64[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}


uint64_t *functionparameter_GetParamPtr_UINT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    uint64_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.ui64[0];

    return ptr;
}












int32_t functionparameter_GetParamValue_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    int32_t value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.i32[0];
    fps->parray[fpsi].val.i32[3] = value;

    return value;
}


errno_t functionparameter_SetParamValue_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int32_t value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.i32[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}


int32_t *functionparameter_GetParamPtr_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    int32_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.i32[0];

    return ptr;
}
















uint32_t functionparameter_GetParamValue_UINT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    long value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.ui32[0];
    fps->parray[fpsi].val.ui32[3] = value;

    return value;
}


errno_t functionparameter_SetParamValue_UINT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    uint32_t value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.ui32[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}


uint32_t *functionparameter_GetParamPtr_UINT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    uint32_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.ui32[0];

    return ptr;
}













double functionparameter_GetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    double value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.f64[0];
    fps->parray[fpsi].val.f64[3] = value;

    return value;
}

errno_t functionparameter_SetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    double value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.f64[0] = value;
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
    ptr = &fps->parray[fpsi].val.f64[0];

    return ptr;
}









float functionparameter_GetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    float value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.f32[0];
    fps->parray[fpsi].val.f32[3] = value;

    return value;
}

int functionparameter_SetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    float value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.f32[0] = value;
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
    ptr = &fps->parray[fpsi].val.f32[0];

    return ptr;
}









float functionparameter_GetParamValue_TIMESPEC(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    long value_sec;
    long value_nsec;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value_sec = fps->parray[fpsi].val.ts[0].tv_sec;
    value_nsec = fps->parray[fpsi].val.ts[0].tv_nsec;
    fps->parray[fpsi].val.ts[3].tv_sec = value_sec;
    fps->parray[fpsi].val.ts[3].tv_nsec = value_nsec;

    float value = 1.0 * value_sec + 1.0e-9 * value_nsec;
    return value;
}

int functionparameter_SetParamValue_TIMESPEC(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    float value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    long valuesec = (long) value;
    long valuensec = (long) (1.0e9 * (value-valuesec));
    fps->parray[fpsi].val.ts[0].tv_sec = valuesec;
    fps->parray[fpsi].val.ts[0].tv_nsec = valuensec;

    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}

struct timespec *functionparameter_GetParamPtr_TIMESPEC(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    struct timespec *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.ts[0];

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
            FUNCTION_PARAMETER_STRMAXLEN - 1);
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
        fps->parray[fpsi].val.i64[0] = 1;
    }
    else
    {
        fps->parray[fpsi].fpflag &= ~FPFLAG_ONOFF;
        fps->parray[fpsi].val.i64[0] = 0;
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
