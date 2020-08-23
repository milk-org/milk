/**
 * @file    fps_paramvalue.h
 * @brief   set and get parameter values
 */

#ifndef FPS_PARAMVALUE_H
#define FPS_PARAMVALUE_H

long functionparameter_GetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    long value
);


int function_parameter_SetValue_int64(
    char *keywordfull,
    long val
);


long *functionparameter_GetParamPtr_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);



double functionparameter_GetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    double value
);


double *functionparameter_GetParamPtr_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


float functionparameter_GetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    float value
);


float *functionparameter_GetParamPtr_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


char *functionparameter_GetParamPtr_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    const char *stringvalue
);


int functionparameter_GetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int ONOFFvalue
);


uint64_t *functionparameter_GetParamPtr_fpflag(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);

#endif
