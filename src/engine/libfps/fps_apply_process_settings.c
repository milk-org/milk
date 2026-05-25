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
 *   2. taskset (CPU affinity) — sched_setaffinity()
 *   3. cset (cgroup v2)   — fork+exec milk-makecsetandrt
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "fps_apply_process_settings.h"


/**
 * parse_cpu_list - Parse a CPU list string into a cpu_set_t
 * @spec:   CPU list string (e.g., "0-3,8,10-12")
 * @cpuset: Output cpu_set_t to populate
 *
 * Supports single CPUs ("3"), ranges ("1-5"), and
 * comma-separated combinations ("1-5,8,10-12").
 *
 * Returns 0 on success, -1 on parse error.
 */
static int parse_cpu_list(const char *spec, cpu_set_t *cpuset)
{
    CPU_ZERO(cpuset);

    if (spec == NULL || spec[0] == '\0')
    {
        return -1;
    }

    const char *p = spec;

    while (*p != '\0')
    {
        /* skip leading whitespace/commas */
        while (*p == ',' || *p == ' ')
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }

        /* parse first number */
        char *endp;
        long  lo = strtol(p, &endp, 10);
        if (endp == p || lo < 0)
        {
            return -1;
        }
        p = endp;

        long hi = lo;

        /* check for range */
        if (*p == '-')
        {
            p++;
            hi = strtol(p, &endp, 10);
            if (endp == p || hi < lo)
            {
                return -1;
            }
            p = endp;
        }

        /* set bits in cpu_set_t */
        for (long cpu = lo; cpu <= hi; cpu++)
        {
            if (cpu < CPU_SETSIZE)
            {
                CPU_SET((int) cpu, cpuset);
            }
        }
    }

    return 0;
}


/**
 * apply_omp_num_threads - Set OMP_NUM_THREADS from FPS
 * @fps: Connected FPS
 *
 * Reads .procinfo.NBthread. If the value is > 0, calls
 * setenv("OMP_NUM_THREADS", ...) so the OpenMP runtime
 * picks it up before the first parallel region.
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
 *
 * Reads .procinfo.taskset. If the value is not
 * "system" or empty, parses the CPU list and calls
 * sched_setaffinity() on the current process.
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

    cpu_set_t cpuset;
    if (parse_cpu_list(spec, &cpuset) != 0)
    {
        fprintf(stderr,
                "WARNING: cannot parse taskset"
                " spec \"%s\"\n",
                spec);
        return;
    }

    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0)
    {
        printf("  taskset = %s\n", spec);
    }
    else
    {
        fprintf(stderr,
                "WARNING: sched_setaffinity"
                " failed: %s\n",
                strerror(errno));
    }
}


/**
 * apply_cset - Migrate to cgroup v2 cpuset from FPS
 * @fps: Connected FPS
 *
 * Reads .procinfo.cset. If the value is not "system"
 * or empty, calls milk-makecsetandrt to migrate the
 * current process to the named cgroup.
 *
 * RT priority is handled separately by processinfo,
 * so we pass 0 for the priority argument.
 */
static void apply_cset(FPS *fps)
{
    long pindex = functionparameter_GetParamIndex(fps, ".procinfo.cset");
    if (pindex < 0)
    {
        return;
    }

    const char *cset_name = fps->parray[pindex].val.string[0];

    /* Skip if empty or default "system" */
    if (cset_name[0] == '\0' || strcmp(cset_name, "system") == 0)
    {
        return;
    }

    /* Build: milk-makecsetandrt <pid> <cgroupname> 0 */
    char pid_str[32];
    snprintf(pid_str, sizeof(pid_str), "%d", (int) getpid());

    printf("  cset = %s (via milk-makecsetandrt)\n", cset_name);

    pid_t child = fork();
    if (child == 0)
    {
        /* child process */
        execlp("milk-makecsetandrt", "milk-makecsetandrt", pid_str, cset_name, "0", (char *) NULL);
        /* execlp failed — tool not found */
        _exit(127);
    }
    else if (child > 0)
    {
        /* parent: wait for child */
        int status;
        waitpid(child, &status, 0);

        if (WIFEXITED(status) && WEXITSTATUS(status) == 127)
        {
            fprintf(stderr, "WARNING: milk-makecsetandrt"
                            " not found in PATH."
                            " Cgroup migration skipped.\n");
        }
        else if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        {
            fprintf(stderr,
                    "WARNING: milk-makecsetandrt"
                    " exited with status %d."
                    " Cgroup may not be"
                    " configured.\n",
                    WEXITSTATUS(status));
        }
    }
    else
    {
        fprintf(stderr,
                "WARNING: fork() failed for"
                " milk-makecsetandrt: %s\n",
                strerror(errno));
    }
}


/**
 * fps_apply_process_settings - Apply process-level
 *     settings from FPS parameters
 * @fps: Connected FPS to read settings from
 *
 * Reads .procinfo.NBthread, .procinfo.taskset, and
 * .procinfo.cset from the FPS and applies them to the
 * current process. Order matters:
 *   1. OMP_NUM_THREADS (must be set before OpenMP init)
 *   2. taskset (CPU affinity)
 *   3. cset (cgroup migration, may override taskset)
 *
 * All operations are best-effort: failures print
 * warnings but do not abort the process.
 *
 * Return: RETURN_SUCCESS always (failures are warned)
 */
errno_t fps_apply_process_settings(FPS *fps)
{
    if (fps == NULL)
    {
        return RETURN_FAILURE;
    }

    apply_omp_num_threads(fps);
    apply_taskset(fps);
    apply_cset(fps);

    return RETURN_SUCCESS;
}
