// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_modulevars_resync.c
 * @brief   Copy FPS parameter values into module-local C
 *          variables via FPS_CLI_BINDING
 */

#include "fps_modulevars_resync.h"
#include "fps_GetParamIndex.h"


/**
 * @brief Copy a single FPS parameter value back into
 *        the module-local C variable via the binding.
 */
void sync_fps_to_local(FPS *fps, long pindex, FPS_CLI_BINDING *b)
{
    if (b->type == FPTYPE_FLOAT64)
    {
        *((double *) b->ptr) = fps->parray[pindex].val.f64[0];
    }
    else if (b->type == FPTYPE_FLOAT32)
    {
        *((float *) b->ptr) = fps->parray[pindex].val.f32[0];
    }
    else if (b->type == FPTYPE_INT64)
    {
        *((int64_t *) b->ptr) = fps->parray[pindex].val.i64[0];
    }
    else if (b->type == FPTYPE_UINT64)
    {
        *((uint64_t *) b->ptr) = fps->parray[pindex].val.ui64[0];
    }
    else if (b->type == FPTYPE_INT32)
    {
        *((int32_t *) b->ptr) = fps->parray[pindex].val.i32[0];
    }
    else if (b->type == FPTYPE_ONOFF)
    {
        *((int32_t *) b->ptr) = (fps->parray[pindex].fpflag & FPFLAG_ONOFF) ? 1 : 0;
    }
    else if (b->type == FPTYPE_UINT32)
    {
        *((uint32_t *) b->ptr) = fps->parray[pindex].val.ui32[0];
    }
    else if (b->type == FPTYPE_PID)
    {
        *((pid_t *) b->ptr) = fps->parray[pindex].val.pid[0];
    }
    else if (b->type == FPTYPE_TIMESPEC)
    {
        *((struct timespec *) b->ptr) = fps->parray[pindex].val.ts[0];
    }
    else if (FPTYPE_IS_STRING(b->type))
    {
        int _l = FUNCTION_PARAMETER_STRMAXLEN;
        strncpy((char *) b->ptr, fps->parray[pindex].val.string[0], _l - 1);
        ((char *) b->ptr)[_l - 1] = '\0';
    }
}


/**
 * @brief Copy the module-local C variable value into
 *        the FPS parameter slot via the binding, if it
 *        actually differs from the current FPS value.
 *
 * This makes for pretty heavy code, but is necessary to test for
 * because we NEED to return whether the parameter was updated
 * (since some params are used back and forth / not only input or output)
 *
 * @return 1 if the local value differed and was written
 *         back, 0 if it already matched (nothing done).
 */
int sync_local_to_fps(FPS *fps, long pindex, FPS_CLI_BINDING *b)
{
    int changed = 0;

    if (b->type == FPTYPE_FLOAT64)
    {
        double newval = *((double *) b->ptr);
        if (fps->parray[pindex].val.f64[0] != newval)
        {
            fps->parray[pindex].val.f64[0] = newval;
            changed                        = 1;
        }
    }
    else if (b->type == FPTYPE_FLOAT32)
    {
        float newval = *((float *) b->ptr);
        if (fps->parray[pindex].val.f32[0] != newval)
        {
            fps->parray[pindex].val.f32[0] = newval;
            changed                        = 1;
        }
    }
    else if (b->type == FPTYPE_INT64)
    {
        int64_t newval = *((int64_t *) b->ptr);
        if (fps->parray[pindex].val.i64[0] != newval)
        {
            fps->parray[pindex].val.i64[0] = newval;
            changed                        = 1;
        }
    }
    else if (b->type == FPTYPE_UINT64)
    {
        uint64_t newval = *((uint64_t *) b->ptr);
        if (fps->parray[pindex].val.ui64[0] != newval)
        {
            fps->parray[pindex].val.ui64[0] = newval;
            changed                         = 1;
        }
    }
    else if (b->type == FPTYPE_INT32)
    {
        int32_t newval = *((int32_t *) b->ptr);
        if (fps->parray[pindex].val.i32[0] != newval)
        {
            fps->parray[pindex].val.i32[0] = newval;
            changed                        = 1;
        }
    }
    else if (b->type == FPTYPE_ONOFF)
    {
        int32_t newval = *((int32_t *) b->ptr) ? 1 : 0;
        int32_t curval = (fps->parray[pindex].fpflag & FPFLAG_ONOFF) ? 1 : 0;
        if (curval != newval)
        {
            if (newval)
            {
                fps->parray[pindex].fpflag |= FPFLAG_ONOFF;
            }
            else
            {
                fps->parray[pindex].fpflag &= ~FPFLAG_ONOFF;
            }
            fps->parray[pindex].val.i64[0] = newval;
            changed                        = 1;
        }
    }
    else if (b->type == FPTYPE_UINT32)
    {
        uint32_t newval = *((uint32_t *) b->ptr);
        if (fps->parray[pindex].val.ui32[0] != newval)
        {
            fps->parray[pindex].val.ui32[0] = newval;
            changed                         = 1;
        }
    }
    else if (b->type == FPTYPE_PID)
    {
        pid_t newval = *((pid_t *) b->ptr);
        if (fps->parray[pindex].val.pid[0] != newval)
        {
            fps->parray[pindex].val.pid[0] = newval;
            changed                        = 1;
        }
    }
    else if (b->type == FPTYPE_TIMESPEC)
    {
        struct timespec newval = *((struct timespec *) b->ptr);
        if (fps->parray[pindex].val.ts[0].tv_sec != newval.tv_sec ||
            fps->parray[pindex].val.ts[0].tv_nsec != newval.tv_nsec)
        {
            fps->parray[pindex].val.ts[0] = newval;
            changed                       = 1;
        }
    }
    else if (FPTYPE_IS_STRING(b->type))
    {
        int _l = FUNCTION_PARAMETER_STRMAXLEN;
        if (strncmp(fps->parray[pindex].val.string[0], (char *) b->ptr, _l - 1) != 0)
        {
            strncpy(fps->parray[pindex].val.string[0], (char *) b->ptr, _l - 1);
            fps->parray[pindex].val.string[0][_l - 1] = '\0';
            changed                                   = 1;
        }
    }

    if (changed)
    {
        fps->parray[pindex].cnt0++;
        fps->parray[pindex].value_cnt++;
    }

    return changed;
}

/**
 * @brief Refresh module-local C variables from FPS, resolving
 *        and caching each binding's parameter index on first use
 */
errno_t fpsresync_fps_to_modvar(FPS             *fps,
                                FPS_CLI_BINDING *bindings,
                                int              nb_b,
                                int              force_cache_build)
{
    for (int ii = 0; ii < nb_b; ii++)
    {
        FPS_CLI_BINDING *b = &bindings[ii];

        if (b->_fps_pindex == -2 || force_cache_build) /* -2: unresolved (see fps_cli_binding.h) */
        {
            b->_fps_pindex = functionparameter_GetParamIndex(fps, b->fpskeyword);
            if (b->_fps_pindex < 0)
            {
                b->_fps_pindex = -1; /* not found: never read parray[-1], never try again */
                continue;
            }
            b->_fps_last_cnt = (int64_t) fps->parray[b->_fps_pindex].value_cnt;
        }
        else if (b->_fps_pindex < 0)
        {
            continue; // Not resolved once, won't try to resolve again.
        }
        else if ((int64_t) fps->parray[b->_fps_pindex].value_cnt == b->_fps_last_cnt)
        {
            continue; /* unchanged since the last read: nothing to copy (strings are 64 B each) */
        }

        sync_fps_to_local(fps, b->_fps_pindex, b);
        b->_fps_last_cnt = (int64_t) fps->parray[b->_fps_pindex].value_cnt;
    }

    return RETURN_SUCCESS;
}


/**
 * @brief Refresh module-local C variables from FPS shared memory
 */
errno_t fps_to_modulevars_resync_bindings(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b)
{
    for (int ii = 0; ii < nb_b; ii++)
    {
        long pindex = functionparameter_GetParamIndex(fps, bindings[ii].fpskeyword);

        if (pindex != -1)
        {
            sync_fps_to_local(fps, pindex, &bindings[ii]);
        }
    }

    return RETURN_SUCCESS;
}


/**
 * @brief Push module-local C variable values into FPS
 *        parameter slots, for every binding.
 */
errno_t fpsresync_modvar_to_fps(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b)
{
    for (int ii = 0; ii < nb_b; ii++)
    {
        FPS_CLI_BINDING *b = &bindings[ii];
        if (b->_fps_pindex < 0)
        {
            continue;
        }
        if ((int64_t) fps->parray[b->_fps_pindex].value_cnt != b->_fps_last_cnt)
        {
            /* another process wrote it since this iteration read it (@set, a TUI): that value wins.
             * _fps_last_cnt stays behind on purpose, so the next read sees the change and copies it */
            continue;
        }
        sync_local_to_fps(fps, b->_fps_pindex, b);
        // Must update the counter; if there was a concurrent upgrade it would otherwise supersede
        // at the next iteration
        b->_fps_last_cnt = (int64_t) fps->parray[b->_fps_pindex].value_cnt;
    }

    return RETURN_SUCCESS;
}
