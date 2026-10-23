// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <signal.h>
#include <pthread.h>

/* Liveness and /proc/<pid>/stat values: overview_data_pid.c */

/**
 * pid_get_core_utilization - get cores actively utilized by a process's threads.
 *
 * Reads the 39th field of /proc/<pid>/task/<tid>/stat for each thread.
 */
int pid_get_core_utilization(pid_t pid, int *cores, int max_cores)
{
    if (pid <= 0 || cores == NULL || max_cores <= 0)
    {
        return 0;
    }

    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/task", (int) pid);

    DIR *d = opendir(path);
    if (!d)
    {
        return 0;
    }

    struct dirent *dir;
    int            count = 0;

    while ((dir = readdir(d)) != NULL && count < max_cores)
    {
        if (dir->d_name[0] == '.')
        {
            continue;
        }

        int tid = atoi(dir->d_name);
        if (tid > 0)
        {
            char stat_path[256];
            snprintf(stat_path, sizeof(stat_path), "%s/%d/stat", path, tid);
            FILE *f = fopen(stat_path, "r");
            if (f)
            {
                int  processor = -1;
                char buf[1024];
                if (fgets(buf, sizeof(buf), f))
                {
                    /* Skip past the comm field which might contain spaces */
                    char *p = strrchr(buf, ')');
                    if (p)
                    {
                        p += 2; /* skip ") " */
                        /* Now p points to field 3 (state). We want field 39.
                           So we skip 36 fields. */
                        for (int i = 0; i < 36 && p; i++)
                        {
                            p = strchr(p, ' ');
                            if (p)
                            {
                                p++;
                            }
                        }
                        if (p)
                        {
                            processor = atoi(p);
                        }
                    }
                }
                fclose(f);
                if (processor >= 0)
                {
                    /* Avoid duplicates if possible, though mostly each thread is on one core.
                       We will just add it, the UI can deduplicate or just show it.
                       Let's deduplicate for a cleaner UI. */
                    int duplicate = 0;
                    for (int i = 0; i < count; i++)
                    {
                        if (cores[i] == processor)
                        {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        cores[count++] = processor;
                    }
                }
            }
        }
    }
    closedir(d);

    /* Sort the cores for a nicer display */
    for (int i = 0; i < count - 1; i++)
    {
        for (int j = i + 1; j < count; j++)
        {
            if (cores[j] < cores[i])
            {
                int tmp  = cores[i];
                cores[i] = cores[j];
                cores[j] = tmp;
            }
        }
    }

    return count;
}


#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

/**
 * pid_get_advanced_stats - get scheduling, memory faults, and thread counts
 */
int pid_get_advanced_stats(pid_t pid, ov_advanced_stats_t *out)
{
    if (pid <= 0 || out == NULL)
    {
        return -1;
    }
    memset(out, 0, sizeof(*out));

    char path[256];

    // 1. Read /proc/<pid>/stat for faults
    snprintf(path, sizeof(path), "/proc/%d/stat", (int) pid);
    FILE *f = fopen(path, "r");
    if (f)
    {
        char buf[1024];
        if (fgets(buf, sizeof(buf), f))
        {
            char *p = strrchr(buf, ')');
            if (p)
            {
                p += 2; // skip ") "
                // p points to field 3 (state)
                // minflt is field 10, so it's 7 fields after state
                for (int i = 0; i < 7 && p; i++)
                {
                    p = strchr(p, ' ');
                    if (p)
                    {
                        p++;
                    }
                }
                if (p)
                {
                    out->minflt = strtoul(p, &p, 10);
                    if (p)
                    {
                        p++;                // skip space to field 11
                        p = strchr(p, ' '); // skip field 11
                        if (p)
                        {
                            p++; // points to field 12
                            out->majflt = strtoul(p, NULL, 10);
                        }
                    }
                }
            }
        }
        fclose(f);
    }

    // 2. Read /proc/<pid>/status for threads, vol_ctxt, nonvol_ctxt
    snprintf(path, sizeof(path), "/proc/%d/status", (int) pid);
    f = fopen(path, "r");
    if (f)
    {
        char buf[256];
        while (fgets(buf, sizeof(buf), f))
        {
            if (strncmp(buf, "Threads:", 8) == 0)
            {
                out->threads = strtoul(buf + 8, NULL, 10);
            }
            else if (strncmp(buf, "voluntary_ctxt_switches:", 24) == 0)
            {
                out->vol_ctxt = strtoul(buf + 24, NULL, 10);
            }
            else if (strncmp(buf, "nonvoluntary_ctxt_switches:", 27) == 0)
            {
                out->nonvol_ctxt = strtoul(buf + 27, NULL, 10);
            }
        }
        fclose(f);
    }

    // 3. Read /proc/<pid>/sched for migrations
    snprintf(path, sizeof(path), "/proc/%d/sched", (int) pid);
    f = fopen(path, "r");
    if (f)
    {
        char buf[256];
        while (fgets(buf, sizeof(buf), f))
        {
            if (strncmp(buf, "se.nr_migrations", 16) == 0)
            {
                char *p = strchr(buf, ':');
                if (p)
                {
                    out->migrations = strtoul(p + 1, NULL, 10);
                }
                break;
            }
        }
        fclose(f);
    }

    return 0;
}
