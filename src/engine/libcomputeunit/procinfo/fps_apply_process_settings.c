// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_apply_process_settings.c
 * @brief   Apply FPS process settings to the current process
 *
 * Reads the placement settings stored in the FPS and applies them to the running process:
 *
 *   1. taskset (CPU affinity) — milk-makecsetandrt
 *   2. cset (cgroup)          — milk-makecsetandrt
 *   3. RT priority            — milk-makecsetandrt
 *
 * OMP_NUM_THREADS (.procinfo.NBthread) is not set here: libgomp reads it when it loads, so a
 * unit's starter puts it in the worker's environment instead (fps_unit_start()). Under systemd the
 * cpuset move is skipped: on cgroup v2 it takes the process out of its service (U17).
 */

#include <stdio.h>
#include <string.h>

#include "milk_rt.h"
#include "fps_apply_process_settings.h"
#include "fps_unit.h"


/**
 * apply_taskset - Set CPU affinity from FPS
 * @fps: Connected FPS
 */
static void apply_taskset(FPS *fps)
{
    long pindex = functionparameter_GetParamIndex(fps, ".procinfo.taskset");
    if (pindex < 0)
    {
        return;
    }

    const char *spec = fps->parray[pindex].val.string[0];
    if (spec[0] == '\0')
    {
        return;
    }

    milkrt_Tset(spec);
}


/**
 * apply_cset - Migrate to cpuset cgroup from FPS
 * @fps: Connected FPS
 */
static void apply_cset(FPS *fps)
{
    long pindex = functionparameter_GetParamIndex(fps, ".procinfo.cset");
    if (pindex < 0)
    {
        return;
    }

    const char *cset_name = fps->parray[pindex].val.string[0];
    if (cset_name[0] == '\0' || strcmp(cset_name, "system") == 0)
    {
        return;
    }
    char unit[128];
    int  user;
    if (fps_unit_systemd_service(unit, sizeof(unit), &user))
    {
        fprintf(
            stderr,
            "warning: .procinfo.cset %s ignored: %s runs this process, and the move would take it "
            "out of the service (place it with AllowedCPUs= in the unit)\n",
            cset_name, unit);
        return;
    }

    milkrt_CPUset(cset_name);
}


/**
 * apply_rt_priority - Set RT scheduling from FPS
 * @fps: Connected FPS
 */
static void apply_rt_priority(FPS *fps)
{
    long pindex = functionparameter_GetParamIndex(fps, ".procinfo.RTprio");
    if (pindex < 0)
    {
        return;
    }

    long rtprio = functionparameter_GetParamValue_INT64(fps, ".procinfo.RTprio");
    if (rtprio <= 0)
    {
        return;
    }

    milkrt_RTPrio((int) rtprio);
}


/**
 * fps_apply_process_settings - Apply process-level
 *     settings from FPS parameters
 * @fps: Connected FPS to read settings from
 * Return: RETURN_SUCCESS always (failures are warned)
 */
errno_t fps_apply_process_settings(FPS *fps)
{
    if (fps == NULL)
    {
        return RETURN_FAILURE;
    }

    apply_cset(fps);
    apply_taskset(fps);
    apply_rt_priority(fps);

    return RETURN_SUCCESS;
}
