// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

#undef STRINGMAXLEN_DIRNAME
#undef STRINGMAXLEN_FULLFILENAME
#undef STRINGMAXLEN_COMMAND
#undef PRINT_ERROR
#include "fps_types.h"
#include "fps_paramvalue.h"
#include "fps_printparameter_valuestring.h"
#include "fps_WriteParameterToDisk.h"
#include "fps_save2disk.h"

static OV_FPS_PARAMS s_active_fps_params;

/* =========================================================
 * FPS cache public accessors
 * ========================================================= */

/**
 * fcache_param - cache entry and parameter index behind display row @disp_idx (s_fcache_mutex held).
 * Return: the entry, or NULL when the row no longer maps to a parameter.
 */
static ov_fps_cache_t *fcache_param(const char *fps_name, int disp_idx, int *pindex)
{
    int ci = fcache_find(fps_name);
    if (ci < 0 || disp_idx < 0 || disp_idx >= s_fcache[ci].dparam_nb)
    {
        return NULL;
    }
    ov_fps_cache_t *ce = &s_fcache[ci];
    if (ce->fps.md == NULL || ce->fps.parray == NULL)
    {
        return NULL;
    }
    long nb = (ce->fps.md->NBparamMAX < ce->nbparam_fit) ? ce->fps.md->NBparamMAX : ce->nbparam_fit;
    *pindex = ce->dparam_idx[disp_idx];
    return (*pindex >= 0 && *pindex < nb) ? ce : NULL;
}

/**
 * fcache_connect_rw - connect read-write, for one edit, to the FPS the cache entry maps, and check
 * that parameter @pindex is still @kw. Return: 0 (caller fps_disconnect()s), OV_PARAM_RO_FILE, or
 * OV_PARAM_MOVED when the FPS was re-created or the parameter moved since the user saw it.
 */
static int fcache_connect_rw(const ov_fps_cache_t *ce, int pindex, const char *kw, FPS *fps)
{
    struct stat st;
    char        path[PATH_MAX];
    memset(fps, 0, sizeof(*fps));
    fps->SMfd = -1;
    snprintf(path, sizeof(path), "%s/%s.fps.shm", ov_get_fpsdir(), ce->fname);
    if (access(path, W_OK) != 0 && errno != ENOENT)
    {
        return OV_PARAM_RO_FILE;
    }
    if (fps_connect(ce->fname, fps, FPSCONNECT_SIMPLE) < 0)
    {
        return OV_PARAM_MOVED;
    }
    if (fstat(fps->SMfd, &st) != 0 || !ov_shm_same_file(&ce->map, &st) ||
        pindex >= fps->md->NBparamMAX || strcmp(fps->parray[pindex].keywordfull, kw) != 0)
    {
        fps_disconnect(fps);
        return OV_PARAM_MOVED;
    }
    return 0;
}

/**
 * @brief Fetch parameter metadata safely under cache lock.
 *
 * @param[in]  fps_name Name of the FPS
 * @param[in]  disp_idx Display parameter index
 * @param[out] info     Output struct populated with parameter info
 * @return 0 on success, -1 if not found or invalid index
 */
int ov_fcache_get_param_info(const char *fps_name, int disp_idx, ov_fps_param_info_t *info)
{
    if (fps_name == NULL || info == NULL)
    {
        return -1;
    }
    memset(info, 0, sizeof(*info));

    pthread_mutex_lock(&s_fcache_mutex);
    int             pindex = -1;
    ov_fps_cache_t *ce     = fcache_param(fps_name, disp_idx, &pindex);
    if (ce == NULL)
    {
        pthread_mutex_unlock(&s_fcache_mutex);
        return -1;
    }
    FPS *fps = &ce->fps;

    FPS_PARAM *fp     = &fps->parray[pindex];
    info->type        = fp->type;
    info->fpflag      = fp->fpflag;
    info->is_writable = (fp->fpflag & FPFLAG_WRITESTATUS) ? 1 : 0;
    info->has_min     = (fp->fpflag & FPFLAG_MINLIMIT) != 0;
    info->has_max     = (fp->fpflag & FPFLAG_MAXLIMIT) != 0;
    switch (fp->type)
    {
    case FPTYPE_INT32:
        info->vmin = fp->val.i32[1];
        info->vmax = fp->val.i32[2];
        break;
    case FPTYPE_UINT32:
        info->vmin = fp->val.ui32[1];
        info->vmax = fp->val.ui32[2];
        break;
    case FPTYPE_INT64:
        info->vmin = (double) fp->val.i64[1];
        info->vmax = (double) fp->val.i64[2];
        break;
    case FPTYPE_UINT64:
        info->vmin = (double) fp->val.ui64[1];
        info->vmax = (double) fp->val.ui64[2];
        break;
    case FPTYPE_FLOAT32:
        info->vmin = fp->val.f32[1];
        info->vmax = fp->val.f32[2];
        break;
    case FPTYPE_FLOAT64:
        info->vmin = fp->val.f64[1];
        info->vmax = fp->val.f64[2];
        break;
    default:
        info->has_min = info->has_max = 0; /* limits apply to numbers only */
        break;
    }

    strncpy(info->keyword, fp->keywordfull, sizeof(info->keyword) - 1);
    info->keyword[sizeof(info->keyword) - 1] = '\0';

    functionparameter_GetParamValueString(fp, info->valstr, (int) sizeof(info->valstr));

    /* Strip FPS name prefix from keyword if present */
    const char *dkw        = fp->keywordfull;
    int         prefix_len = (int) strlen(fps->md->name);
    if (strncmp(dkw, fps->md->name, (size_t) prefix_len) == 0 && dkw[prefix_len] == '.')
    {
        dkw += prefix_len + 1;
    }
    strncpy(info->display_kw, dkw, sizeof(info->display_kw) - 1);
    info->display_kw[sizeof(info->display_kw) - 1] = '\0';

    pthread_mutex_unlock(&s_fcache_mutex);
    return 0;
}

/**
 * @brief Toggle an ONOFF parameter through an on-demand read-write connection.
 *
 * @param[in]  fps_name    Name of the FPS
 * @param[in]  disp_idx    Display parameter index
 * @param[out] out_keyword Optional buffer to receive parameter keyword (can be NULL)
 * @param[in]  kw_size     Size of out_keyword buffer
 * @param[out] out_newval  Optional pointer to receive new value (0 or 1, can be NULL)
 * @return 0 on success, -1 on error, or OV_PARAM_MOVED / OV_PARAM_RO_FILE / OV_PARAM_LOCKED
 */
int ov_fcache_toggle_param(const char *fps_name,
                           int         disp_idx,
                           char       *out_keyword,
                           size_t      kw_size,
                           int        *out_newval)
{
    int pindex = -1;
    FPS fps;
    if (fps_name == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&s_fcache_mutex);
    ov_fps_cache_t *ce = fcache_param(fps_name, disp_idx, &pindex);
    if (ce == NULL || ce->fps.parray[pindex].type != FPTYPE_ONOFF)
    {
        pthread_mutex_unlock(&s_fcache_mutex);
        return -1;
    }
    if (!(ce->fps.parray[pindex].fpflag & FPFLAG_WRITESTATUS))
    {
        pthread_mutex_unlock(&s_fcache_mutex);
        return OV_PARAM_LOCKED;
    }
    char kw[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    snprintf(kw, sizeof(kw), "%s", ce->fps.parray[pindex].keywordfull);
    int rc = fcache_connect_rw(ce, pindex, kw, &fps);
    pthread_mutex_unlock(&s_fcache_mutex);
    if (rc != 0)
    {
        return rc;
    }

    FPS_PARAM *fp     = &fps.parray[pindex];
    int        newval = (fp->fpflag & FPFLAG_ONOFF) ? 0 : 1;
    functionparameter_SetParamValue_ONOFF(&fps, fp->keywordfull, newval);
    fps.md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
    if (fp->fpflag & FPFLAG_SAVEONCHANGE)
    {
        functionparameter_WriteParameterToDisk(&fps, pindex, "setval", "milk-CTRL_toggle");
        functionparameter_SaveFPS2disk(&fps);
    }
    fps_disconnect(&fps);

    if (out_keyword != NULL && kw_size > 0)
    {
        snprintf(out_keyword, kw_size, "%s", kw);
    }
    if (out_newval != NULL)
    {
        *out_newval = newval;
    }
    return 0;
}

/**
 * @brief Set an FPS parameter from a value string through an on-demand read-write connection.
 *
 * @param[in] fps_name Name of the FPS
 * @param[in] disp_idx Display parameter index
 * @param[in] kw       keywordfull the user edited (checked against the live FPS)
 * @param[in] valstr   New value string to parse and apply
 * @return 0 on success, -1 on error or invalid value, or OV_PARAM_MOVED / OV_PARAM_RO_FILE /
 *         OV_PARAM_LOCKED
 */
int ov_fcache_set_param_value(const char *fps_name,
                              int         disp_idx,
                              const char *kw,
                              const char *valstr)
{
    int pindex = -1;
    FPS fps;
    if (fps_name == NULL || kw == NULL || valstr == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&s_fcache_mutex);
    ov_fps_cache_t *ce = fcache_param(fps_name, disp_idx, &pindex);
    if (ce == NULL)
    {
        pthread_mutex_unlock(&s_fcache_mutex);
        return -1;
    }
    if (!(ce->fps.parray[pindex].fpflag & FPFLAG_WRITESTATUS))
    {
        pthread_mutex_unlock(&s_fcache_mutex);
        return OV_PARAM_LOCKED;
    }
    int rc = fcache_connect_rw(ce, pindex, kw, &fps);
    pthread_mutex_unlock(&s_fcache_mutex);
    if (rc != 0)
    {
        return rc;
    }

    FPS_PARAM *fp = &fps.parray[pindex];
    rc = (functionparameter_SetParamValue_fromString(&fps, pindex, valstr) == 0) ? 0 : -1;
    if (rc == 0)
    {
        fps.md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
        if (strncmp(fp->keywordfull, ".procinfo.", 10) == 0)
        {
            fps.md->processinfo_change_cnt++;
        }
        if (fp->fpflag & FPFLAG_SAVEONCHANGE)
        {
            functionparameter_WriteParameterToDisk(&fps, pindex, "setval",
                                                   "milk-CTRL_SetParamValue");
            functionparameter_SaveFPS2disk(&fps);
        }
    }
    fps_disconnect(&fps);
    return rc;
}

/**
 * ov_fps_get_params - fetch display parameters for an FPS on-demand.
 * @fps_name: name of the FPS
 *
 * Return: pointer to thread-safe static parameters struct, or NULL.
 */
const OV_FPS_PARAMS *ov_fps_get_params(const char *fps_name)
{
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return NULL;
    }

    pthread_mutex_lock(&s_fcache_mutex);
    int ci = fcache_find(fps_name);
    if (ci < 0)
    {
        pthread_mutex_unlock(&s_fcache_mutex);
        return NULL;
    }

    ov_fps_cache_t *ce = &s_fcache[ci];
    if (!ce->sparam_cached)
    {
        fcache_build_params(ce);
    }

    FPS *fpsp = &ce->fps;
    if (fpsp->md == NULL || fpsp->parray == NULL)
    {
        pthread_mutex_unlock(&s_fcache_mutex);
        return NULL;
    }

    s_active_fps_params.nb_disp_params = ce->dparam_nb;
    for (int dp = 0; dp < ce->dparam_nb; dp++)
    {
        FPS_PARAM *fp = &fpsp->parray[ce->dparam_idx[dp]];
        strncpy(s_active_fps_params.disp_param_name[dp], ce->dparam_key[dp],
                FUNCTION_PARAMETER_STRMAXLEN - 1);
        s_active_fps_params.disp_param_name[dp][FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';

        char valstr[FUNCTION_PARAMETER_STRMAXLEN] = { 0 };
        switch (fp->type)
        {
        case FPTYPE_UNDEF:
            snprintf(valstr, sizeof(valstr), "[UNDEF]");
            break;
        case FPTYPE_INT32:
            snprintf(valstr, sizeof(valstr), "%" PRIi32, fp->val.i32[0]);
            break;
        case FPTYPE_UINT32:
            snprintf(valstr, sizeof(valstr), "%" PRIu32, fp->val.ui32[0]);
            break;
        case FPTYPE_INT64:
            snprintf(valstr, sizeof(valstr), "%" PRIi64, fp->val.i64[0]);
            break;
        case FPTYPE_UINT64:
            snprintf(valstr, sizeof(valstr), "%" PRIu64, fp->val.ui64[0]);
            break;
        case FPTYPE_FLOAT64:
            snprintf(valstr, sizeof(valstr), "%g", fp->val.f64[0]);
            break;
        case FPTYPE_FLOAT32:
            snprintf(valstr, sizeof(valstr), "%g", fp->val.f32[0]);
            break;
        case FPTYPE_PID:
            snprintf(valstr, sizeof(valstr), "%d", (int) fp->val.pid[0]);
            break;
        case FPTYPE_TIMESPEC:
        {
            double secs = fp->val.ts[0].tv_sec + (fp->val.ts[0].tv_nsec / 1e9);
            snprintf(valstr, sizeof(valstr), "%g s", secs);
            break;
        }
        case FPTYPE_ONOFF:
            snprintf(valstr, sizeof(valstr), "%s", fp->val.i64[0] ? "ON" : "OFF");
            break;
        case FPTYPE_FPSNAME:
        case FPTYPE_STREAMNAME:
        case FPTYPE_STRING:
        case FPTYPE_DIRNAME:
        case FPTYPE_FILENAME:
        case FPTYPE_EXECFILENAME:
        case FPTYPE_FITSFILENAME:
        case FPTYPE_PROCESS:
        case FPTYPE_STRING_NOT_STREAM:
            strncpy(valstr, fp->val.string[0], sizeof(valstr) - 1);
            break;
        default:
            snprintf(valstr, sizeof(valstr), "[Type %d]", fp->type);
            break;
        }
        strncpy(s_active_fps_params.disp_param_value[dp], valstr, FUNCTION_PARAMETER_STRMAXLEN - 1);
        s_active_fps_params.disp_param_value[dp][FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';
        s_active_fps_params.disp_param_type[dp]                                    = fp->type;
        s_active_fps_params.disp_param_flags[dp]                                   = fp->fpflag;

        strncpy(s_active_fps_params.disp_param_descr[dp], fp->description,
                FUNCTION_PARAMETER_DESCR_STRMAXLEN - 1);
        s_active_fps_params.disp_param_descr[dp][FUNCTION_PARAMETER_DESCR_STRMAXLEN - 1] = '\0';

        s_active_fps_params.disp_param_has_min[dp] = (fp->fpflag & FPFLAG_MINLIMIT) != 0;
        s_active_fps_params.disp_param_has_max[dp] = (fp->fpflag & FPFLAG_MAXLIMIT) != 0;

        if (s_active_fps_params.disp_param_has_min[dp])
        {
            char minstr[FUNCTION_PARAMETER_STRMAXLEN] = { 0 };
            switch (fp->type)
            {
            case FPTYPE_INT64:
                snprintf(minstr, sizeof(minstr), "%" PRIi64, fp->val.i64[1]);
                break;
            case FPTYPE_INT32:
                snprintf(minstr, sizeof(minstr), "%" PRIi32, fp->val.i32[1]);
                break;
            case FPTYPE_UINT64:
                snprintf(minstr, sizeof(minstr), "%" PRIu64, fp->val.ui64[1]);
                break;
            case FPTYPE_UINT32:
                snprintf(minstr, sizeof(minstr), "%" PRIu32, fp->val.ui32[1]);
                break;
            case FPTYPE_FLOAT64:
                snprintf(minstr, sizeof(minstr), "%g", fp->val.f64[1]);
                break;
            case FPTYPE_FLOAT32:
                snprintf(minstr, sizeof(minstr), "%g", (double) fp->val.f32[1]);
                break;
            case FPTYPE_TIMESPEC:
            {
                double secs = fp->val.ts[1].tv_sec + (fp->val.ts[1].tv_nsec / 1e9);
                snprintf(minstr, sizeof(minstr), "%g s", secs);
                break;
            }
            default:
                s_active_fps_params.disp_param_has_min[dp] = 0;
                break;
            }
            strncpy(s_active_fps_params.disp_param_min[dp], minstr,
                    FUNCTION_PARAMETER_STRMAXLEN - 1);
            s_active_fps_params.disp_param_min[dp][FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';
        }
        else
        {
            s_active_fps_params.disp_param_min[dp][0] = '\0';
        }

        if (s_active_fps_params.disp_param_has_max[dp])
        {
            char maxstr[FUNCTION_PARAMETER_STRMAXLEN] = { 0 };
            switch (fp->type)
            {
            case FPTYPE_INT64:
                snprintf(maxstr, sizeof(maxstr), "%" PRIi64, fp->val.i64[2]);
                break;
            case FPTYPE_INT32:
                snprintf(maxstr, sizeof(maxstr), "%" PRIi32, fp->val.i32[2]);
                break;
            case FPTYPE_UINT64:
                snprintf(maxstr, sizeof(maxstr), "%" PRIu64, fp->val.ui64[2]);
                break;
            case FPTYPE_UINT32:
                snprintf(maxstr, sizeof(maxstr), "%" PRIu32, fp->val.ui32[2]);
                break;
            case FPTYPE_FLOAT64:
                snprintf(maxstr, sizeof(maxstr), "%g", fp->val.f64[2]);
                break;
            case FPTYPE_FLOAT32:
                snprintf(maxstr, sizeof(maxstr), "%g", (double) fp->val.f32[2]);
                break;
            case FPTYPE_TIMESPEC:
            {
                double secs = fp->val.ts[2].tv_sec + (fp->val.ts[2].tv_nsec / 1e9);
                snprintf(maxstr, sizeof(maxstr), "%g s", secs);
                break;
            }
            default:
                s_active_fps_params.disp_param_has_max[dp] = 0;
                break;
            }
            strncpy(s_active_fps_params.disp_param_max[dp], maxstr,
                    FUNCTION_PARAMETER_STRMAXLEN - 1);
            s_active_fps_params.disp_param_max[dp][FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';
        }
        else
        {
            s_active_fps_params.disp_param_max[dp][0] = '\0';
        }
    }

    pthread_mutex_unlock(&s_fcache_mutex);
    return &s_active_fps_params;
}
