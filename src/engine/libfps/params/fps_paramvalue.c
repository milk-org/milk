// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_paramvalue.c
 * @brief   Typed accessors for FPS parameter values
 *
 * Provides Get, Set, and GetPtr functions for every
 * FPS parameter type (INT64, UINT64, INT32, UINT32,
 * FLOAT64, FLOAT32, TIMESPEC, STRING, ONOFF, fpflag).
 *
 * Accessor pattern (repeated per type):
 *  - GetParamValue_TYPE()  -- read current value and
 *    snapshot it into val[3] for change-detection.
 *  - SetParamValue_TYPE()  -- write value and bump
 *    cnt0 + value_cnt so watchers detect the update.
 *  - GetParamPtr_TYPE()    -- return a direct pointer
 *    to val[0] for zero-copy hot-path reads.
 *
 * The val[] array stores:
 *   [0] = current, [1] = min, [2] = max, [3] = last.
 * GetParamValue copies [0]->[3] so callers can later
 * compare [0] vs [3] to detect changes.
 */

#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "fps.h"
#include "fps_internal.h"

#include "fps_GetParamIndex.h"
#include "fps_GetTypeString.h"
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
int64_t *functionparameter_GetParamPtr_generic(FPS *fps, const char *paramname, long *paramindex)
{
    int64_t *ptr;

    long fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        if (paramindex != NULL)
        {
            *paramindex = -1;
        }
        return NULL;
    }

    // type is arbitrary
    ptr = &fps->parray[fpsi].val.i64[0];

    if (paramindex != NULL)
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

int64_t functionparameter_GetParamValue_INT64(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return 0;
    }
    int64_t value                = fps->parray[fpsi].val.i64[0];
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
errno_t functionparameter_SetParamValue_INT64(FPS *fps, const char *paramname, int64_t value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }
    fps->parray[fpsi].val.i64[0] = value;
    fps->parray[fpsi].cnt0++;
    fps->parray[fpsi].value_cnt++;

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
    FPS   fps;
    char  tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char  keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int   keywordlevel = 0;
    char *pch;

    // break full keyword into keywords
    strncpy(tmpstring, keywordfull,
            FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL - 1);
    keywordlevel = 0;
    pch          = strtok(tmpstring, ".");
    while (pch != NULL)
    {
        strncpy(keyword[keywordlevel], pch, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN - 1);
        keywordlevel++;
        pch = strtok(NULL, ".");
    }

    fps_connect(keyword[9], &fps, FPSCONNECT_SIMPLE);

    int pindex = functionparameter_GetParamIndex(&fps, keywordfull);
    if (pindex < 0)
    {
        fps_disconnect(&fps);
        return RETURN_FAILURE;
    }

    fps.parray[pindex].val.i64[0] = val;

    fps_disconnect(&fps);

    return RETURN_SUCCESS;
}

/**
 * @brief Get direct pointer to INT64 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.i64[0]
 */
int64_t *functionparameter_GetParamPtr_INT64(FPS *fps, const char *paramname)
{
    long fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return NULL;
    }
    return &fps->parray[fpsi].val.i64[0];
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
uint64_t functionparameter_GetParamValue_UINT64(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return 0;
    }
    uint64_t value                = fps->parray[fpsi].val.ui64[0];
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
errno_t functionparameter_SetParamValue_UINT64(FPS *fps, const char *paramname, uint64_t value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }
    fps->parray[fpsi].val.ui64[0] = value;
    fps->parray[fpsi].cnt0++;
    fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to UINT64 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.ui64[0]
 */
uint64_t *functionparameter_GetParamPtr_UINT64(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return NULL;
    }
    return &fps->parray[fpsi].val.ui64[0];
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
int32_t functionparameter_GetParamValue_INT32(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return 0;
    }
    int32_t value                = fps->parray[fpsi].val.i32[0];
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
errno_t functionparameter_SetParamValue_INT32(FPS *fps, const char *paramname, int32_t value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }
    fps->parray[fpsi].val.i32[0] = value;
    fps->parray[fpsi].cnt0++;
    fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to INT32 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.i32[0]
 */
int32_t *functionparameter_GetParamPtr_INT32(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return NULL;
    }
    return &fps->parray[fpsi].val.i32[0];
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
uint32_t functionparameter_GetParamValue_UINT32(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return 0;
    }
    uint32_t value                = fps->parray[fpsi].val.ui32[0];
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
errno_t functionparameter_SetParamValue_UINT32(FPS *fps, const char *paramname, uint32_t value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }
    fps->parray[fpsi].val.ui32[0] = value;
    fps->parray[fpsi].cnt0++;
    fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to UINT32 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.ui32[0]
 */
uint32_t *functionparameter_GetParamPtr_UINT32(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return NULL;
    }
    return &fps->parray[fpsi].val.ui32[0];
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
double functionparameter_GetParamValue_FLOAT64(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return 0.0;
    }
    double value                 = fps->parray[fpsi].val.f64[0];
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
errno_t functionparameter_SetParamValue_FLOAT64(FPS *fps, const char *paramname, double value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }
    fps->parray[fpsi].val.f64[0] = value;
    fps->parray[fpsi].cnt0++;
    fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to FLOAT64 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.f64[0]
 */
double *functionparameter_GetParamPtr_FLOAT64(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return NULL;
    }
    return &fps->parray[fpsi].val.f64[0];
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
float functionparameter_GetParamValue_FLOAT32(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return 0.0f;
    }
    float value                  = fps->parray[fpsi].val.f32[0];
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
int functionparameter_SetParamValue_FLOAT32(FPS *fps, const char *paramname, float value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }
    fps->parray[fpsi].val.f32[0] = value;
    fps->parray[fpsi].cnt0++;
    fps->parray[fpsi].value_cnt++;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to FLOAT32 value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.f32[0]
 */
float *functionparameter_GetParamPtr_FLOAT32(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return NULL;
    }
    return &fps->parray[fpsi].val.f32[0];
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
struct timespec functionparameter_GetParamValue_TIMESPEC(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        struct timespec t = { .tv_sec = 0, .tv_nsec = 0 };
        return t;
    }
    fps->parray[fpsi].val.ts[3] = fps->parray[fpsi].val.ts[0];
    return fps->parray[fpsi].val.ts[0];
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
int functionparameter_SetParamValue_TIMESPEC(FPS *fps, const char *paramname, struct timespec value)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }
    fps->parray[fpsi].cnt0++;
    fps->parray[fpsi].value_cnt++;
    fps->parray[fpsi].val.ts[0] = value;

    return EXIT_SUCCESS;
}

/**
 * @brief Get direct pointer to TIMESPEC value
 *
 * @param fps        Connected FPS
 * @param paramname  Parameter keyword
 * @return Pointer to val.ts[0]
 */
struct timespec *functionparameter_GetParamPtr_TIMESPEC(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return NULL;
    }
    return &fps->parray[fpsi].val.ts[0];
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
char *functionparameter_GetParamPtr_STRING(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return NULL;
    }
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
int functionparameter_SetParamValue_STRING(FPS *fps, const char *paramname, const char *stringvalue)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }

    strncpy(fps->parray[fpsi].val.string[0], stringvalue, FUNCTION_PARAMETER_STRMAXLEN - 1);
    fps->parray[fpsi].cnt0++;
    fps->parray[fpsi].value_cnt++;

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
int functionparameter_GetParamValue_ONOFF(FPS *fps, const char *paramname)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return 0;
    }

    if (fps->parray[fpsi].fpflag & FPFLAG_ONOFF)
    {
        return 1;
    }
    return 0;
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
int functionparameter_SetParamValue_ONOFF(FPS *fps, const char *paramname, int ONOFFvalue)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    if (fpsi < 0)
    {
        return EXIT_FAILURE;
    }

    if (ONOFFvalue == 1)
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
    fps->parray[fpsi].value_cnt++;

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
uint64_t *functionparameter_GetParamPtr_fpflag(FPS *fps, const char *paramname)
{
    uint64_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr      = &fps->parray[fpsi].fpflag;

    return ptr;
}


/* ============================================================
 * Generic string-to-typed setter
 * ========================================================== */

/**
 * functionparameter_SetParamValue_arg - set an FPS parameter to a value read by milk_argval_parse()
 * @fps:    connected FPS structure
 * @pindex: index into fps->parray[]
 * @ad:     the value, in the field of the parameter's type
 *
 * Writes through the typed setters (an ONOFF value also sets FPFLAG_ONOFF). A string must fit the FPS
 * (FUNCTION_PARAMETER_STRMAXLEN - 1 characters): it is never cut.
 *
 * Returns 0 on success, -1 if pindex is out of range or the value does not fit.
 */
int functionparameter_SetParamValue_arg(FPS *fps, int pindex, const CLICMDARGDATA *ad)
{
    if (pindex < 0 || pindex >= fps->md->NBparamMAX)
    {
        return -1;
    }
    const char *kw = fps->parray[pindex].keywordfull;
    int         rc = EXIT_FAILURE;

    switch (fps->parray[pindex].type)
    {
    case FPTYPE_INT32:
        rc = functionparameter_SetParamValue_INT32(fps, kw, ad->val.i32);
        break;
    case FPTYPE_UINT32:
        rc = functionparameter_SetParamValue_UINT32(fps, kw, ad->val.ui32);
        break;
    case FPTYPE_INT64:
    case FPTYPE_PID:
        rc = functionparameter_SetParamValue_INT64(fps, kw, ad->val.i64);
        break;
    case FPTYPE_UINT64:
        rc = functionparameter_SetParamValue_UINT64(fps, kw, ad->val.ui64);
        break;
    case FPTYPE_FLOAT32:
        rc = functionparameter_SetParamValue_FLOAT32(fps, kw, ad->val.f32);
        break;
    case FPTYPE_FLOAT64:
        rc = functionparameter_SetParamValue_FLOAT64(fps, kw, ad->val.f64);
        break;
    case FPTYPE_TIMESPEC:
    {
        struct timespec ts = { .tv_sec = (time_t) ad->val.f64, .tv_nsec = 0 };
        ts.tv_nsec         = (long) ((ad->val.f64 - (double) ts.tv_sec) * 1e9);
        rc                 = functionparameter_SetParamValue_TIMESPEC(fps, kw, ts);
        break;
    }
    case FPTYPE_ONOFF:
        rc = functionparameter_SetParamValue_ONOFF(fps, kw, ad->val.i64 != 0);
        break;
    default:
        rc = (strlen(ad->val.s) < FUNCTION_PARAMETER_STRMAXLEN)
                 ? functionparameter_SetParamValue_STRING(fps, kw, ad->val.s)
                 : EXIT_FAILURE;
        break;
    }
    return (rc == EXIT_SUCCESS) ? 0 : -1;
}

/**
 * functionparameter_SetParamValue_fromString -
 *      set an FPS parameter from a string value.
 * @fps:      connected FPS structure
 * @pindex:   index into fps->parray[]
 * @strval:   NUL-terminated string representation (one trailing newline is ignored)
 *
 * Reads @strval as every argument is read (milk_argval_parse(): calculator literals, on/off/true/false/
 * 1/0), sets the parameter (functionparameter_SetParamValue_arg()) and logs it.
 *
 * Returns 0 on success, -1 if pindex is out of range or @strval is not a value of the parameter.
 */
int functionparameter_SetParamValue_fromString(FPS *fps, int pindex, const char *strval)
{
    CLICMDARGDATA ad;
    char          txt[STRINGMAXLEN_CLICMDARG];
    char          shown[STRINGMAXLEN_CLICMDARG];
    size_t        len = strlen(strval);

    if (pindex < 0 || pindex >= fps->md->NBparamMAX || len >= sizeof(txt))
    {
        return -1;
    }
    memcpy(txt, strval, len + 1);
    if (len > 0 && txt[len - 1] == '\n')
    {
        txt[len - 1] = '\0';
    }
    memset(&ad, 0, sizeof(ad));
    ad.type = fps->parray[pindex].type;
    if (milk_argval_parse(ad.type, txt, &ad) != 0 ||
        functionparameter_SetParamValue_arg(fps, pindex, &ad) != 0)
    {
        return -1;
    }
    char tname[STRINGMAXLEN_FPSTYPE];
    functionparameter_GetTypeString((uint32_t) ad.type, tname); /* " INT32 " */
    milk_argval_format(&ad, shown, sizeof(shown));
    functionparameter_outlog("SETVAL", "%s%s%s", fps->parray[pindex].keywordfull, tname, shown);
    return 0;
}
