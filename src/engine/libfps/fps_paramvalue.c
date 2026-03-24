/**
 * @file    fps_paramvalue.c
 * @brief   Typed accessors for FPS parameter values
 *
 * Provides Get, Set, and GetPtr functions for every
 * FPS parameter type (INT64, UINT64, INT32, UINT32,
 * FLOAT64, FLOAT32, TIMESPEC, STRING, ONOFF, fpflag).
 *
 * Accessor pattern (repeated per type):
 *  - GetParamValue_TYPE()  — read current value and
 *    snapshot it into val[3] for change-detection.
 *  - SetParamValue_TYPE()  — write value and bump
 *    cnt0 + value_cnt so watchers detect the update.
 *  - GetParamPtr_TYPE()    — return a direct pointer
 *    to val[0] for zero-copy hot-path reads.
 *
 * The val[] array stores:
 *   [0] = current, [1] = min, [2] = max, [3] = last.
 * GetParamValue copies [0]→[3] so callers can later
 * compare [0] vs [3] to detect changes.
 */

#include "fps.h"
#include "fps_internal.h"

#include "fps_GetParamIndex.h"
#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_paramvalue.h"

/**
 * @brief Get generic pointer to a parameter value
 *
 * Returns a raw int64_t pointer to val[0] regardless
 * of actual type. Optionally outputs the parameter
 * index for further operations.
 *
 * @param fps        Connected FPS
 * @param paramname  Dot-separated parameter keyword
 * @param paramindex If non-NULL, receives the index
 * @return Pointer to val.i64[0] (cast as needed)
 */
int64_t *functionparameter_GetParamPtr_generic(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname,
        long       *paramindex)
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

/* ============================================================
 * INT64 accessors
 * ========================================================== */

/**
 * @brief Read INT64 parameter and snapshot for change
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Current int64_t value
 */

int64_t functionparameter_GetParamValue_INT64(FUNCTION_PARAMETER_STRUCT *fps,
        const char *paramname)
{
    int64_t value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value    = fps->parray[fpsi].val.i64[0];
    fps->parray[fpsi].val.i64[3] = value;

    return value;
}

/**
 * @brief Write INT64 parameter value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @param value      New value
 * @return EXIT_SUCCESS
 */
errno_t functionparameter_SetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int64_t     value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.i64[0] = value;
    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Standalone INT64 setter via full keyword
 *
 * Connects to the FPS, sets the value, and
 * disconnects. Used by external tools that need
 * to poke a single parameter without holding a
 * persistent connection.
 *
 * @param keywordfull  Full dotted keyword path
 * @param val          New int64_t value
 * @return RETURN_SUCCESS
 */
errno_t function_parameter_SetValue_int64(char *keywordfull, int64_t val)
{
    FUNCTION_PARAMETER_STRUCT fps;
    char                      tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                                             FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char                      keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL]
    [FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int   keywordlevel = 0;
    char *pch;

    // break full keyword into keywords
    strncpy(tmpstring,
            keywordfull,
            FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
            FUNCTION_PARAMETER_KEYWORD_MAXLEVEL -
            1);
    keywordlevel = 0;
    pch          = strtok(tmpstring, ".");
    while(pch != NULL)
    {
        strncpy(keyword[keywordlevel],
                pch,
                FUNCTION_PARAMETER_KEYWORD_STRMAXLEN - 1);
        keywordlevel++;
        pch = strtok(NULL, ".");
    }

    function_parameter_struct_connect(keyword[9], &fps, FPSCONNECT_SIMPLE);

    int pindex = functionparameter_GetParamIndex(&fps, keywordfull);

    fps.parray[pindex].val.i64[0] = val;

    function_parameter_struct_disconnect(&fps);

    return RETURN_SUCCESS;
}

/**
 * @brief Get direct pointer to INT64 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.i64[0]
 */
int64_t *functionparameter_GetParamPtr_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    int64_t *ptr;

    long fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr       = &fps->parray[fpsi].val.i64[0];

    return ptr;
}

/* ============================================================
 * UINT64 accessors
 * ========================================================== */

/**
 * @brief Read UINT64 parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Current uint64_t value
 */
uint64_t functionparameter_GetParamValue_UINT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    uint64_t value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value    = fps->parray[fpsi].val.ui64[0];
    fps->parray[fpsi].val.ui64[3] = value;

    return value;
}

/**
 * @brief Write UINT64 parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @param value      New value
 * @return EXIT_SUCCESS
 */
errno_t functionparameter_SetParamValue_UINT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    uint64_t    value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.ui64[0] = value;
    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to UINT64 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.ui64[0]
 */
uint64_t *functionparameter_GetParamPtr_UINT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    uint64_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr      = &fps->parray[fpsi].val.ui64[0];

    return ptr;
}

/* ============================================================
 * INT32 accessors
 * ========================================================== */

/**
 * @brief Read INT32 parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Current int32_t value
 */
int32_t functionparameter_GetParamValue_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    int32_t value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value    = fps->parray[fpsi].val.i32[0];
    fps->parray[fpsi].val.i32[3] = value;

    return value;
}

/**
 * @brief Write INT32 parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @param value      New value
 * @return EXIT_SUCCESS
 */
errno_t functionparameter_SetParamValue_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int32_t     value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.i32[0] = value;
    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to INT32 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.i32[0]
 */
int32_t *functionparameter_GetParamPtr_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    int32_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr      = &fps->parray[fpsi].val.i32[0];

    return ptr;
}

/* ============================================================
 * UINT32 accessors
 * ========================================================== */

/**
 * @brief Read UINT32 parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Current uint32_t value
 */
uint32_t functionparameter_GetParamValue_UINT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    long value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value    = fps->parray[fpsi].val.ui32[0];
    fps->parray[fpsi].val.ui32[3] = value;

    return value;
}

/**
 * @brief Write UINT32 parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @param value      New value
 * @return EXIT_SUCCESS
 */
errno_t functionparameter_SetParamValue_UINT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    uint32_t    value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.ui32[0] = value;
    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to UINT32 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.ui32[0]
 */
uint32_t *functionparameter_GetParamPtr_UINT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    uint32_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr      = &fps->parray[fpsi].val.ui32[0];

    return ptr;
}

/* ============================================================
 * FLOAT64 (double) accessors
 * ========================================================== */

/**
 * @brief Read FLOAT64 (double) parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Current double value
 */
double functionparameter_GetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    double value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value    = fps->parray[fpsi].val.f64[0];
    fps->parray[fpsi].val.f64[3] = value;

    return value;
}

/**
 * @brief Write FLOAT64 (double) parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @param value      New value
 * @return EXIT_SUCCESS
 */
errno_t functionparameter_SetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    double      value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.f64[0] = value;
    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to FLOAT64 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.f64[0]
 */
double *functionparameter_GetParamPtr_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    double *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr      = &fps->parray[fpsi].val.f64[0];

    return ptr;
}

/* ============================================================
 * FLOAT32 (float) accessors
 * ========================================================== */

/**
 * @brief Read FLOAT32 (float) parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Current float value
 */
float functionparameter_GetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    float value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value    = fps->parray[fpsi].val.f32[0];
    fps->parray[fpsi].val.f32[3] = value;

    return value;
}

/**
 * @brief Write FLOAT32 (float) parameter
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @param value      New value
 * @return EXIT_SUCCESS
 */
int functionparameter_SetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    float       value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.f32[0] = value;
    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to FLOAT32 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.f32[0]
 */
float *functionparameter_GetParamPtr_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    float *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr      = &fps->parray[fpsi].val.f32[0];

    return ptr;
}

/* ============================================================
 * TIMESPEC accessors
 * ========================================================== */

/**
 * @brief Read TIMESPEC as seconds (float)
 *
 * Converts tv_sec + tv_nsec to a single float.
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Time in seconds
 */
float functionparameter_GetParamValue_TIMESPEC(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    long value_sec;
    long value_nsec;

    int fpsi   = functionparameter_GetParamIndex(fps, paramname);
    value_sec  = fps->parray[fpsi].val.ts[0].tv_sec;
    value_nsec = fps->parray[fpsi].val.ts[0].tv_nsec;
    fps->parray[fpsi].val.ts[3].tv_sec  = value_sec;
    fps->parray[fpsi].val.ts[3].tv_nsec = value_nsec;

    float value = 1.0 * value_sec + 1.0e-9 * value_nsec;
    return value;
}

/**
 * @brief Write TIMESPEC from seconds (float)
 *
 * Decomposes float seconds into tv_sec + tv_nsec.
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @param value      Time in seconds
 * @return EXIT_SUCCESS
 */
int functionparameter_SetParamValue_TIMESPEC(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    float       value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    long valuesec                       = (long) value;
    long valuensec                      = (long)(1.0e9 * (value - valuesec));
    fps->parray[fpsi].val.ts[0].tv_sec  = valuesec;
    fps->parray[fpsi].val.ts[0].tv_nsec = valuensec;

    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to TIMESPEC value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.ts[0]
 */
struct timespec *
functionparameter_GetParamPtr_TIMESPEC(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *paramname)
{
    struct timespec *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr      = &fps->parray[fpsi].val.ts[0];

    return ptr;
}

/* ============================================================
 * STRING accessors
 * ========================================================== */

/**
 * @brief Get pointer to STRING parameter buffer
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.string[0] (mutable)
 */
char *functionparameter_GetParamPtr_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    return fps->parray[fpsi].val.string[0];
}

/**
 * @brief Write STRING parameter
 *
 * @param fps          Connected FPS
 * @param paramname    Parameter keyword
 * @param stringvalue  New string value
 * @return EXIT_SUCCESS
 */
int functionparameter_SetParamValue_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *paramname,
    const char *stringvalue)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    strncpy(fps->parray[fpsi].val.string[0],
            stringvalue,
            FUNCTION_PARAMETER_STRMAXLEN - 1);
    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/* ============================================================
 * ONOFF (boolean toggle) accessors
 * ========================================================== */

/**
 * @brief Read ONOFF parameter
 *
 * Returns 1 if FPFLAG_ONOFF is set, 0 otherwise.
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return 1 = ON, 0 = OFF
 */
int functionparameter_GetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *paramname)
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

/**
 * @brief Write ONOFF parameter
 *
 * Sets or clears FPFLAG_ONOFF and mirrors the
 * value into val.i64[0] for consistent reads.
 *
 * @param fps         Connected FPS
 * @param paramname   Parameter keyword
 * @param ONOFFvalue  1 = ON, 0 = OFF
 * @return EXIT_SUCCESS
 */
int functionparameter_SetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *paramname,
    int                        ONOFFvalue)
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

    fps->parray[fpsi].cnt0++; fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/* ============================================================
 * Flag pointer accessor
 * ========================================================== */

/**
 * @brief Get pointer to parameter's fpflag word
 *
 * Allows direct bit manipulation of the parameter
 * flags without going through Set/Get wrappers.
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to fpflag (uint64_t)
 */
uint64_t *functionparameter_GetParamPtr_fpflag(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname)
{
    uint64_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr      = &fps->parray[fpsi].fpflag;

    return ptr;
}
