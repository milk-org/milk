#include <pthread.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "overview_data_internal.h"

/* =========================================================
 * Persistent SHM mapping caches
 *
 * These caches keep SHM file descriptors and mappings
 * alive across scan ticks to avoid the overhead of
 * open/mmap/munmap/close on every cycle.
 *
 * Staleness: each tick checks stat() inode; a mismatch
 * or missing file triggers remap/eviction.
 * ========================================================= */

/* --- Stream cache --- */


ov_stream_cache_t s_scache[OV_MAX_STREAMS];
int               s_scache_nb = 0;

/**
 * scache_find - find stream in cache by name.
 *
 * Return: cache index, or -1 if not found.
 */
int scache_find(const char *name)
{
    for (int i = 0; i < s_scache_nb; i++)
    {
        if (strcmp(s_scache[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * scache_evict - close mapping and compact array.
 */
void scache_evict(int ci)
{
    ImageStreamIO_closeIm(&s_scache[ci].img);
    s_scache_nb--;
    if (ci < s_scache_nb)
    {
        s_scache[ci] = s_scache[s_scache_nb];
    }
}

/* --- FPS cache --- */

pthread_mutex_t      s_fcache_mutex = PTHREAD_MUTEX_INITIALIZER;
static OV_FPS_PARAMS s_active_fps_params;

ov_fps_cache_t s_fcache[OV_MAX_FPS];
int            s_fcache_nb = 0;

/**
 * @brief Look up an FPS entry in the connection cache.
 */
int fcache_find(const char *name)
{
    for (int i = 0; i < s_fcache_nb; i++)
    {
        if (strcmp(s_fcache[i].fname, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Evict and disconnect an FPS cache entry.
 */
void fcache_evict(int ci)
{
    pthread_mutex_lock(&s_fcache_mutex);
    fps_disconnect(&s_fcache[ci].fps);
    s_fcache_nb--;
    if (ci < s_fcache_nb)
    {
        s_fcache[ci] = s_fcache[s_fcache_nb];
    }
    pthread_mutex_unlock(&s_fcache_mutex);
}

/* --- Proc cache --- */


ov_proc_cache_t s_pcache[OV_MAX_PROCS];
int             s_pcache_nb = 0;


/**
 * @brief Look up a process by PID in the cache.
 */
int pcache_find_pid(pid_t pid)
{
    for (int i = 0; i < s_pcache_nb; i++)
    {
        if (s_pcache[i].pid == pid)
        {
            return i;
        }
    }
    return -1;
}

void pcache_evict(int ci)
{
    munmap(s_pcache[ci].pinfo, sizeof(PROCESSINFO));
    close(s_pcache[ci].fd);
    s_pcache_nb--;
    if (ci < s_pcache_nb)
    {
        s_pcache[ci] = s_pcache[s_pcache_nb];
    }
}

/* =========================================================
 * FPS cache public accessors
 * ========================================================= */

/**
 * ov_fcache_get_fps - return raw FPS pointer by name.
 *
 * Returns the memory-mapped FPS struct from the cache,
 * or NULL if the FPS is not currently cached.
 */
FPS *ov_fcache_get_fps(const char *name)
{
    pthread_mutex_lock(&s_fcache_mutex);
    int  ci  = fcache_find(name);
    FPS *res = (ci >= 0) ? &s_fcache[ci].fps : NULL;
    pthread_mutex_unlock(&s_fcache_mutex);
    return res;
}

/**
 * ov_fcache_get_param_index - map display index to
 *     raw FPS parameter array index.
 *
 * @fps_name: FPS name to look up in cache
 * @disp_idx: display parameter index (0..nb_disp_params-1)
 *
 * Return: raw parray index, or -1 on error.
 */
int ov_fcache_get_param_index(const char *fps_name, int disp_idx)
{
    pthread_mutex_lock(&s_fcache_mutex);
    int ci = fcache_find(fps_name);
    if (ci < 0 || disp_idx < 0 || disp_idx >= s_fcache[ci].dparam_nb)
    {
        pthread_mutex_unlock(&s_fcache_mutex);
        return -1;
    }
    int res = s_fcache[ci].dparam_idx[disp_idx];
    pthread_mutex_unlock(&s_fcache_mutex);
    return res;
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
