// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_apply_process_settings.c
 * @brief   Apply FPS process settings to the current process
 *
 * Reads placement settings stored in the FPS and applies them before the compute loop:
 *   1. cset     (cgroup)              — milk-makecsetandrt
 *   2. numanode (NUMA physical cores) — in-process sched_setaffinity(2)
 *   3. taskset  (explicit CPU mask)   — milk-makecsetandrt (overrides numanode)
 *   4. RTprio   (SCHED_FIFO priority) — milk-makecsetandrt
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "milk_rt.h"
#include "fps_apply_process_settings.h"
#include "fps_unit_startstop.h"

/** Read first line of @path into @buf (NUL-terminated); return 0 on success. */
static int numanode_read_line(const char *path, char *buf, size_t sz)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL || sz == 0)
    {
        return -1;
    }
    int ok = (fgets(buf, (int) sz, fp) != NULL) ? 0 : -1;
    fclose(fp);
    if (ok == 0)
    {
        size_t len = strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
        {
            buf[--len] = '\0';
        }
    }
    return ok;
}

/** Parse Linux sysfs cpulist ("0-127,256-383") into @set; return 0 if >= 1 CPU set. */
static int numanode_parse_cpulist(const char *spec, cpu_set_t *set)
{
    CPU_ZERO(set);
    if (spec == NULL || *spec == '\0')
    {
        return -1;
    }
    const char *p = spec;
    while (*p != '\0')
    {
        while (*p == ',' || isspace((unsigned char) *p))
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }
        char *end = NULL;
        long  lo  = strtol(p, &end, 10);
        if (end == p || lo < 0 || lo >= CPU_SETSIZE)
        {
            return -1;
        }
        long hi = lo;
        if (*end == '-')
        {
            const char *q = end + 1;
            hi            = strtol(q, &end, 10);
            if (end == q || hi < lo || hi >= CPU_SETSIZE)
            {
                return -1;
            }
        }
        for (long c = lo; c <= hi; c++)
        {
            CPU_SET((int) c, set);
        }
        p = end;
    }
    return (CPU_COUNT(set) > 0) ? 0 : -1;
}

/** Detect PCI GPU NUMA node via sysfs (prefer NVIDIA 0x10de, fallback AMD 0x1002 display). */
static long numanode_detect_gpu(void)
{
    DIR *d = opendir("/sys/bus/pci/devices");
    if (d == NULL)
    {
        return -1;
    }
    long           found = -1;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL)
    {
        if (ent->d_name[0] == '.')
        {
            continue;
        }
        char path[PATH_MAX];
        char val[64];
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", ent->d_name);
        if (numanode_read_line(path, val, sizeof(val)) != 0)
        {
            continue;
        }
        int is_nv  = (strcasecmp(val, "0x10de") == 0);
        int is_amd = (strcasecmp(val, "0x1002") == 0);
        if (!is_nv && !is_amd)
        {
            continue;
        }
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/class", ent->d_name);
        if (numanode_read_line(path, val, sizeof(val)) != 0)
        {
            continue;
        }
        if (strncmp(val, "0x0300", 6) != 0 && strncmp(val, "0x0302", 6) != 0 &&
            strncmp(val, "0x0380", 6) != 0)
        {
            continue;
        }
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", ent->d_name);
        if (numanode_read_line(path, val, sizeof(val)) != 0)
        {
            continue;
        }
        long n = strtol(val, NULL, 10);
        if (n >= 0)
        {
            found = n;
            if (is_nv)
            {
                break;
            }
        }
    }
    closedir(d);
    if (found < 0 && access("/sys/devices/system/node/node1", F_OK) == 0)
    {
        found = 0;
    }
    return found;
}

/** Pin all process threads to physical cores of @fps .procinfo.numanode / MILK_NUMANODE. */
static void apply_numanode(FPS *fps)
{
    long node   = -1;
    long pindex = functionparameter_GetParamIndex(fps, ".procinfo.numanode");
    if (pindex >= 0)
    {
        node = functionparameter_GetParamValue_INT64(fps, ".procinfo.numanode");
    }
    if (node == -1)
    {
        const char *env = getenv("MILK_NUMANODE");
        if (env == NULL || *env == '\0' || strcmp(env, "-1") == 0 || strcasecmp(env, "off") == 0 ||
            strcasecmp(env, "none") == 0)
        {
            return;
        }
        if (strcasecmp(env, "auto") == 0 || strcasecmp(env, "gpu") == 0 || strcmp(env, "-2") == 0)
        {
            node = -2;
        }
        else
        {
            char *endp = NULL;
            long  req  = strtol(env, &endp, 10);
            if (*endp != '\0' || req < 0 || req > 16)
            {
                return;
            }
            node = req;
        }
    }
    if (node < -1)
    {
        node = numanode_detect_gpu();
        if (node < 0)
        {
            return;
        }
    }

    char path[PATH_MAX];
    char cpulist[256];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%ld/cpulist", node);
    if (numanode_read_line(path, cpulist, sizeof(cpulist)) != 0)
    {
        return;
    }

    cpu_set_t raw_mask;
    if (numanode_parse_cpulist(cpulist, &raw_mask) != 0)
    {
        return;
    }

    /* Keep one thread per physical core (first sibling in thread_siblings_list) */
    cpu_set_t phys_mask;
    CPU_ZERO(&phys_mask);
    for (int c = 0; c < CPU_SETSIZE; c++)
    {
        if (!CPU_ISSET(c, &raw_mask))
        {
            continue;
        }
        char sib[64];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
                 c);
        if (numanode_read_line(path, sib, sizeof(sib)) == 0)
        {
            long first_sib = strtol(sib, NULL, 10);
            if (first_sib == (long) c)
            {
                CPU_SET(c, &phys_mask);
            }
        }
        else
        {
            CPU_SET(c, &phys_mask);
        }
    }

    cpu_set_t use_mask = (CPU_COUNT(&phys_mask) > 0) ? phys_mask : raw_mask;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0)
    {
        cpu_set_t intersected;
        CPU_AND(&intersected, &use_mask, &allowed);
        if (CPU_COUNT(&intersected) > 0)
        {
            use_mask = intersected;
        }
    }

    sched_setaffinity(0, sizeof(use_mask), &use_mask);
    DIR *td = opendir("/proc/self/task");
    if (td != NULL)
    {
        struct dirent *ent;
        while ((ent = readdir(td)) != NULL)
        {
            if (ent->d_name[0] == '.')
            {
                continue;
            }
            char *endp  = NULL;
            long  tid_l = strtol(ent->d_name, &endp, 10);
            if (*endp == '\0' && tid_l > 0)
            {
                sched_setaffinity((pid_t) tid_l, sizeof(use_mask), &use_mask);
            }
        }
        closedir(td);
    }
}

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

errno_t fps_apply_process_settings(FPS *fps)
{
    if (fps == NULL)
    {
        return RETURN_FAILURE;
    }

    apply_cset(fps);
    apply_numanode(fps);
    apply_taskset(fps);
    apply_rt_priority(fps);

    return RETURN_SUCCESS;
}
