/**
 * @file    fps_apply_process_settings.c
 * @brief   Apply FPS process settings to the current process
 *
 * When a V2 standalone executable runs directly (bypassing
 * tmux dispatch), the cset/taskset/OMP_NUM_THREADS settings
 * stored in FPS shared memory are not applied.  This module
 * reads those parameters and applies them programmatically:
 *
 *   1. OMP_NUM_THREADS  — setenv() before compute_fn()
 *   2. taskset (CPU affinity) — COREMOD_TOOLS_mvProcTset()
 *   3. cset (cgroup)          — COREMOD_TOOLS_mvProcCPUset()
 *   4. RT priority            — COREMOD_TOOLS_mvProcRTPrio()
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_rt.h"
#include "fps_apply_process_settings.h"


/**
 * apply_omp_num_threads - Set OMP_NUM_THREADS from FPS
 * @fps: Connected FPS
 */
static void apply_omp_num_threads(FPS *fps)
{
    long pindex = functionparameter_GetParamIndex(fps, ".procinfo.NBthread");
    if (pindex < 0)
    {
        return;
    }

    long nbthread = functionparameter_GetParamValue_INT64(fps, ".procinfo.NBthread");
    if (nbthread <= 0)
    {
        return;
    }

    char buf[32];
    snprintf(buf, sizeof(buf), "%ld", nbthread);

    if (setenv("OMP_NUM_THREADS", buf, 1) == 0)
    {
        printf("  OMP_NUM_THREADS = %s\n", buf);
    }
    else
    {
        fprintf(stderr,
                "WARNING: setenv OMP_NUM_THREADS"
                " failed: %s\n",
                strerror(errno));
    }
}


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

    apply_omp_num_threads(fps);
    apply_cset(fps);
    apply_taskset(fps);
    apply_rt_priority(fps);

    return RETURN_SUCCESS;
}
