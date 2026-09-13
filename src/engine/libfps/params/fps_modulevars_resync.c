// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_modulevars_resync.c
 * @brief   Copy FPS parameter values into module-local C
 *          variables via FPS_CLI_BINDING
 */

#include "fps_cli_sync.h"
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
    else if (b->type == FPTYPE_INT32 || b->type == FPTYPE_ONOFF)
    {
        *((int32_t *) b->ptr) = fps->parray[pindex].val.i32[0];
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
        strncpy((char *) b->ptr, fps->parray[pindex].val.string[0],
                FUNCTION_PARAMETER_STRMAXLEN - 1);
        ((char *) b->ptr)[FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';
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
    else if (b->type == FPTYPE_INT32 || b->type == FPTYPE_ONOFF)
    {
        int32_t newval = *((int32_t *) b->ptr);
        if (fps->parray[pindex].val.i32[0] != newval)
        {
            fps->parray[pindex].val.i32[0] = newval;
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
        if (strncmp(fps->parray[pindex].val.string[0], (char *) b->ptr,
                    FUNCTION_PARAMETER_STRMAXLEN - 1) != 0)
        {
            strncpy(fps->parray[pindex].val.string[0], (char *) b->ptr,
                    FUNCTION_PARAMETER_STRMAXLEN - 1);
            fps->parray[pindex].val.string[0][FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';
            changed                                                             = 1;
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
 * @brief Refresh module-local C variables from FPS shared
 *        memory, cheaply, for every binding.
 *
 * Resolves+caches each binding's
 * parameter index on first use, then copies the current
 * FPS value into the local variable via sync_fps_to_local()
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter binding array
 * @param nb_b      Number of bindings
 * @return          RETURN_SUCCESS on success
 */
errno_t fps_modulevars_bilateral_bindings_resync(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b)
{
    for (int ii = 0; ii < nb_b; ii++)
    {
        FPS_CLI_BINDING *b = &bindings[ii];

        if (b->_fps_pindex == -2)
        {
            b->_fps_pindex = functionparameter_GetParamIndex(fps, b->fpskeyword);
        }
        else if (b->_fps_pindex == -1)
        {
            continue; // Not resolved once, won't try to resolve again.
        }

        int64_t cnt = (int64_t) fps->parray[b->_fps_pindex].value_cnt;

        int local_changed = sync_local_to_fps(fps, b->_fps_pindex, b);

        if (!local_changed && (cnt != b->_fps_last_cnt))
        {
            sync_fps_to_local(fps, b->_fps_pindex, b);
        }
        b->_fps_last_cnt = (int64_t) fps->parray[b->_fps_pindex].value_cnt;
    }

    return RETURN_SUCCESS;
}
