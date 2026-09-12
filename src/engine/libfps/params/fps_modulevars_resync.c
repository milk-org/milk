// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_modulevars_resync.c
 * @brief   Copy FPS parameter values into module-local C
 *          variables via FPS_CLI_BINDING
 *
 * Engine-tier (no CLI dependency): INSERT_STD_PROCINFO_-
 * COMPUTEFUNC_INIT/LOOPSTART (fps_procinfo_macros.h,
 * CLIcore_utils.h) call fps_to_modulevars_resync_bindings()
 * from every compute unit's compute loop. Unlike
 * fps_process_cli_and_sync() (fps_cli_sync.c), these two
 * functions have no CLI/standalone-specific dependency, so
 * they live in the base milkfps library rather than in the
 * milkfpsStandalone/milkfpsCLI variants -- every compute
 * unit links the base library regardless of build mode.
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
 * @brief Refresh module-local C variables from FPS shared
 *        memory, cheaply, for every binding.
 *
 * No CLI parsing -- only resolves+caches each binding's
 * parameter index on first use, then copies the current
 * FPS value into the local variable via sync_fps_to_local()
 * when fps->parray[pindex].value_cnt has changed since the
 * last call (b->_fps_last_cnt). A param whose value hasn't
 * changed costs one int64_t compare; safe to call every
 * compute-loop iteration.
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter binding array
 * @param nb_b      Number of bindings
 * @return          RETURN_SUCCESS on success
 */
errno_t fps_to_modulevars_resync_bindings(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b)
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
        if (cnt != b->_fps_last_cnt)
        {
            sync_fps_to_local(fps, b->_fps_pindex, b);
            b->_fps_last_cnt = cnt;
        }
    }

    return RETURN_SUCCESS;
}
