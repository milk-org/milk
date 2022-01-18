/**
 * @file    fps_paramvalue.h
 * @brief   set and get parameter values
 */

#ifndef FPS_PARAMVALUE_H
#define FPS_PARAMVALUE_H

int64_t *functionparameter_GetParamPtr_generic(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, long *paramindex);

// =====================================================================
// INT32
// =====================================================================

errno_t functionparameter_SetParamValue_INT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, int32_t value);

int32_t functionparameter_GetParamValue_INT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

int32_t *functionparameter_GetParamPtr_INT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

// Creates variable "_varname"
//
#define FPS_GETPARAM_INT32(varname, pname)                                                                             \
    int32_t _##varname = 0;                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        _##varname = functionparameter_GetParamValue_INT32(&fps, pname);                                               \
        (void)_##varname;                                                                                              \
    } while (0)

// =====================================================================
// UINT32
// =====================================================================

errno_t functionparameter_SetParamValue_UINT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, uint32_t value);

uint32_t functionparameter_GetParamValue_UINT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

uint32_t *functionparameter_GetParamPtr_UINT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

// Creates variable "_varname"
//
#define FPS_GETPARAM_UINT32(varname, pname)                                                                            \
    int32_t _##varname = 0;                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        _##varname = functionparameter_GetParamValue_UINT32(&fps, pname);                                              \
        (void)_##varname;                                                                                              \
    } while (0)

// =====================================================================
// INT64
// =====================================================================

errno_t functionparameter_SetParamValue_INT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, int64_t value);

int64_t functionparameter_GetParamValue_INT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

int64_t *functionparameter_GetParamPtr_INT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

// Creates variable "_varname"
//
#define FPS_GETPARAM_INT64(varname, pname)                                                                             \
    int64_t _##varname = 0;                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        _##varname = functionparameter_GetParamValue_INT64(&fps, pname);                                               \
        (void)_##varname;                                                                                              \
    } while (0)

// =====================================================================
// UINT64
// =====================================================================

errno_t functionparameter_SetParamValue_UINT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, uint64_t value);

uint64_t functionparameter_GetParamValue_UINT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

uint64_t *functionparameter_GetParamPtr_UINT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

// Creates variable "_varname"
//
#define FPS_GETPARAM_UINT64(varname, pname)                                                                            \
    int64_t _##varname = 0;                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        _##varname = functionparameter_GetParamValue_UINT64(&fps, pname);                                              \
        (void)_##varname;                                                                                              \
    } while (0)

// =====================================================================
// FLOAT32
// =====================================================================

float functionparameter_GetParamValue_FLOAT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

errno_t functionparameter_SetParamValue_FLOAT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, float value);

float *functionparameter_GetParamPtr_FLOAT32(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

// Creates variable "_varname"
//
#define FPS_GETPARAM_FLOAT32(varname, pname)                                                                           \
    float _##varname = 0;                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        _##varname = functionparameter_GetParamValue_FLOAT32(&fps, pname);                                             \
        (void)_##varname;                                                                                              \
    } while (0)

// =====================================================================
// FLOAT64
// =====================================================================

double functionparameter_GetParamValue_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

errno_t functionparameter_SetParamValue_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, double value);

double *functionparameter_GetParamPtr_FLOAT64(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

// =====================================================================
// TIMESPEC
// =====================================================================

float functionparameter_GetParamValue_TIMESPEC(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

int functionparameter_SetParamValue_TIMESPEC(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, float value);

struct timespec *functionparameter_GetParamPtr_TIMESPEC(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

// =====================================================================
// STRING
// =====================================================================

char *functionparameter_GetParamPtr_STRING(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

errno_t functionparameter_SetParamValue_STRING(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname,
                                               const char *stringvalue);

// =====================================================================
// ON/OFF
// =====================================================================

int functionparameter_GetParamValue_ONOFF(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

errno_t functionparameter_SetParamValue_ONOFF(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname, int ONOFFvalue);

uint64_t *functionparameter_GetParamPtr_fpflag(FUNCTION_PARAMETER_STRUCT *fps, const char *paramname);

#endif
