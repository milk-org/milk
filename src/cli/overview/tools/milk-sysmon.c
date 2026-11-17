// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-sysmon.c
 * @brief Headless CPU, GPU, NUMA, PROCESSINFO, and SHM compute telemetry profiler.
 *
 * Samples system CPU load (/proc/stat), CPU frequencies, NUMA residency,
 * NVIDIA GPU telemetry (nvidia-smi), ImageStreamIO SHM bandwidth, and PROCESSINFO
 * nanosecond timing rings across simulation phases, writing JSON + CSV reports.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include "milk_help.h"
#include "overview_data.h"
#include "overview_defs.h"
#include "processinfo.h"
#include "processinfo_shm_close.h"
#include "processinfo_shm_link.h"

#define SYSMON_MAX_CPUS 1024
#define SYSMON_MAX_NUMA 16
#define SYSMON_MAX_PROCS 512
#define SYSMON_MAX_STREAMS 512
#define SYSMON_MAX_PHASES 128

static volatile sig_atomic_t g_stop = 0;

static void handle_sig(int sig)
{
    (void) sig;
    g_stop = 1;
}

/* =========================================================
 * Utility helpers
 * ======================================================= */

static double now_epoch_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

static double now_mono_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

static double safe_d(double v)
{
    return (isfinite(v)) ? v : 0.0;
}

static void json_escape(FILE *fp, const char *s)
{
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *) s; s && *p; p++)
    {
        unsigned char c = *p;
        if (c == '"' || c == '\\')
        {
            fprintf(fp, "\\%c", c);
        }
        else if (c == '\n')
        {
            fputs("\\n", fp);
        }
        else if (c == '\r')
        {
            fputs("\\r", fp);
        }
        else if (c == '\t')
        {
            fputs("\\t", fp);
        }
        else if (c < 0x20)
        {
            fprintf(fp, "\\u%04x", (unsigned) c);
        }
        else
        {
            fputc(c, fp);
        }
    }
    fputc('"', fp);
}

static uint64_t dtype_bytes(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
    case _DATATYPE_INT8:
        return 1;
    case _DATATYPE_UINT16:
    case _DATATYPE_INT16:
        return 2;
    case _DATATYPE_UINT32:
    case _DATATYPE_INT32:
    case _DATATYPE_FLOAT:
        return 4;
    case _DATATYPE_UINT64:
    case _DATATYPE_INT64:
    case _DATATYPE_DOUBLE:
    case _DATATYPE_COMPLEX_FLOAT:
        return 8;
    case _DATATYPE_COMPLEX_DOUBLE:
        return 16;
    default:
        return 4;
    }
}

static const char *dtype_str(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "UI8";
    case _DATATYPE_INT8:
        return "SI8";
    case _DATATYPE_UINT16:
        return "U16";
    case _DATATYPE_INT16:
        return "S16";
    case _DATATYPE_UINT32:
        return "U32";
    case _DATATYPE_INT32:
        return "S32";
    case _DATATYPE_UINT64:
        return "U64";
    case _DATATYPE_INT64:
        return "S64";
    case _DATATYPE_FLOAT:
        return "F32";
    case _DATATYPE_DOUBLE:
        return "F64";
    case _DATATYPE_COMPLEX_FLOAT:
        return "CF";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "CD";
    default:
        return "UNK";
    }
}

static int read_first_line(const char *path, char *buf, size_t sz)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        return -1;
    }
    int ok = (fgets(buf, (int) sz, fp) != NULL) ? 0 : -1;
    fclose(fp);
    if (ok == 0)
    {
        size_t n = strlen(buf);
        while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
        {
            buf[--n] = '\0';
        }
    }
    return ok;
}

static void trim_str(char *s)
{
    if (!s)
    {
        return;
    }
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char) s[len - 1]))
    {
        s[--len] = '\0';
    }
    char *p = s;
    while (*p && isspace((unsigned char) *p))
    {
        p++;
    }
    if (p != s)
    {
        memmove(s, p, strlen(p) + 1);
    }
}

static int cmp_int64(const void *a, const void *b)
{
    int64_t la = *(const int64_t *) a;
    int64_t lb = *(const int64_t *) b;
    return (la > lb) - (la < lb);
}

/* =========================================================
 * Host, NUMA & GPU static metadata
 * ======================================================= */

typedef struct
{
    char   hostname[128];
    char   kernel[128];
    char   cpu_model[128];
    int    nb_cpus;
    int    nb_numa;
    char   numa_cpulist[SYSMON_MAX_NUMA][128];
    int8_t cpu_to_numa[SYSMON_MAX_CPUS];
    char   governor[64];
    double cpu_max_mhz;
    double ram_total_mb;
    double ram_avail_mb;
    double shm_total_mb;
    double shm_free_mb;
    int    perf_paranoid;
    char   bash_version[128];
    char   shmdir[PATH_MAX];
    char   procdir[PATH_MAX];

    int    gpu_present;
    char   gpu_name[128];
    char   gpu_driver[64];
    char   gpu_pci_bus[32];
    int    gpu_pcie_gen;
    int    gpu_pcie_gen_max;
    int    gpu_pcie_width;
    int    gpu_pcie_width_max;
    double gpu_vram_total_mib;
    double gpu_power_limit_w;
    int    gpu_numa_node;
} host_info_t;

static void parse_cpulist_for_numa(const char *s, int node, host_info_t *h)
{
    const char *p = s;
    while (p && *p)
    {
        char *end = NULL;
        long  a   = strtol(p, &end, 10);
        if (end == p)
        {
            break;
        }
        long b = a;
        if (*end == '-')
        {
            b = strtol(end + 1, &end, 10);
        }
        for (long c = a; c <= b; c++)
        {
            if (c >= 0 && c < SYSMON_MAX_CPUS)
            {
                h->cpu_to_numa[c] = (int8_t) node;
            }
        }
        if (*end == ',')
        {
            p = end + 1;
        }
        else
        {
            break;
        }
    }
}

static void collect_host_info(host_info_t *h)
{
    memset(h, 0, sizeof(*h));
    for (int i = 0; i < SYSMON_MAX_CPUS; i++)
    {
        h->cpu_to_numa[i] = 0;
    }

    snprintf(h->shmdir, sizeof(h->shmdir), "%s", ov_get_shmdir());
    snprintf(h->procdir, sizeof(h->procdir), "%s", ov_get_procdir());

    struct utsname un;
    if (uname(&un) == 0)
    {
        snprintf(h->hostname, sizeof(h->hostname), "%s", un.nodename);
        snprintf(h->kernel, sizeof(h->kernel), "%s %s", un.sysname, un.release);
    }

    h->nb_cpus = (int) sysconf(_SC_NPROCESSORS_ONLN);
    if (h->nb_cpus < 1)
    {
        h->nb_cpus = 1;
    }
    if (h->nb_cpus > SYSMON_MAX_CPUS)
    {
        h->nb_cpus = SYSMON_MAX_CPUS;
    }

    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp)
    {
        char line[256];
        while (fgets(line, sizeof(line), fp))
        {
            if (strncmp(line, "model name", 10) == 0)
            {
                char *col = strchr(line, ':');
                if (col)
                {
                    snprintf(h->cpu_model, sizeof(h->cpu_model), "%s", col + 1);
                    trim_str(h->cpu_model);
                    break;
                }
            }
        }
        fclose(fp);
    }

    for (int n = 0; n < SYSMON_MAX_NUMA; n++)
    {
        char path[128], buf[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
        if (read_first_line(path, buf, sizeof(buf)) != 0)
        {
            break;
        }
        snprintf(h->numa_cpulist[n], sizeof(h->numa_cpulist[n]), "%s", buf);
        parse_cpulist_for_numa(buf, n, h);
        h->nb_numa = n + 1;
    }
    if (h->nb_numa == 0)
    {
        h->nb_numa = 1;
    }

    if (read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", h->governor,
                        sizeof(h->governor)) != 0)
    {
        snprintf(h->governor, sizeof(h->governor), "unknown");
    }

    char mhzbuf[64];
    if (read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", mhzbuf,
                        sizeof(mhzbuf)) == 0)
    {
        h->cpu_max_mhz = atof(mhzbuf) / 1000.0;
    }

    fp = fopen("/proc/meminfo", "r");
    if (fp)
    {
        char line[256];
        while (fgets(line, sizeof(line), fp))
        {
            double val = 0.0;
            if (sscanf(line, "MemTotal: %lf kB", &val) == 1)
            {
                h->ram_total_mb = val / 1024.0;
            }
            else if (sscanf(line, "MemAvailable: %lf kB", &val) == 1)
            {
                h->ram_avail_mb = val / 1024.0;
            }
        }
        fclose(fp);
    }

    struct statvfs sv;
    if (statvfs("/dev/shm", &sv) == 0 || statvfs(h->shmdir, &sv) == 0)
    {
        h->shm_total_mb = ((double) sv.f_blocks * sv.f_frsize) / (1024.0 * 1024.0);
        h->shm_free_mb  = ((double) sv.f_bavail * sv.f_frsize) / (1024.0 * 1024.0);
    }

    char pbuf[32];
    h->perf_paranoid = 99;
    if (read_first_line("/proc/sys/kernel/perf_event_paranoid", pbuf, sizeof(pbuf)) == 0)
    {
        h->perf_paranoid = atoi(pbuf);
    }

    fp = popen("/bin/bash --version 2>/dev/null | head -n1", "r");
    if (fp)
    {
        if (fgets(h->bash_version, sizeof(h->bash_version), fp) != NULL)
        {
            trim_str(h->bash_version);
        }
        pclose(fp);
    }

    h->gpu_numa_node = -1;
    fp               = popen("nvidia-smi --query-gpu=name,driver_version,pci.bus_id,"
                             "pcie.link.gen.current,pcie.link.gen.max,"
                             "pcie.link.width.current,pcie.link.width.max,"
                             "memory.total,power.limit --format=csv,noheader,nounits 2>/dev/null",
                             "r");
    if (fp)
    {
        char line[512];
        if (fgets(line, sizeof(line), fp) != NULL)
        {
            char *cols[9] = { NULL };
            int   nc      = 0;
            char *p       = line;
            while (nc < 9 && p)
            {
                cols[nc++] = p;
                char *cm   = strchr(p, ',');
                if (!cm)
                {
                    break;
                }
                *cm = '\0';
                p   = cm + 1;
            }
            if (nc >= 9)
            {
                for (int k = 0; k < 9; k++)
                {
                    trim_str(cols[k]);
                }
                h->gpu_present = 1;
                snprintf(h->gpu_name, sizeof(h->gpu_name), "%s", cols[0]);
                snprintf(h->gpu_driver, sizeof(h->gpu_driver), "%s", cols[1]);
                snprintf(h->gpu_pci_bus, sizeof(h->gpu_pci_bus), "%s", cols[2]);
                h->gpu_pcie_gen       = atoi(cols[3]);
                h->gpu_pcie_gen_max   = atoi(cols[4]);
                h->gpu_pcie_width     = atoi(cols[5]);
                h->gpu_pcie_width_max = atoi(cols[6]);
                h->gpu_vram_total_mib = atof(cols[7]);
                h->gpu_power_limit_w  = atof(cols[8]);

                char        sysbus[64];
                const char *bp = h->gpu_pci_bus;
                if (strlen(bp) >= 12 && strncmp(bp, "0000", 4) == 0 && bp[8] == ':')
                {
                    bp += 4;
                }
                snprintf(sysbus, sizeof(sysbus), "%s", bp);
                for (char *q = sysbus; *q; q++)
                {
                    *q = (char) tolower((unsigned char) *q);
                }
                char npath[160], nbuf[32];
                snprintf(npath, sizeof(npath), "/sys/bus/pci/devices/%s/numa_node", sysbus);
                if (read_first_line(npath, nbuf, sizeof(nbuf)) == 0)
                {
                    h->gpu_numa_node = atoi(nbuf);
                }
            }
        }
        pclose(fp);
    }
}

/* =========================================================
 * Background GPU Monitor (pipe + compute-apps thread)
 * ======================================================= */

typedef struct
{
    int    valid;
    double gpu_util;
    double mem_util;
    double vram_mib;
    double power_w;
    double sm_mhz;
    double mem_mhz;
    double temp_c;
    char   pstate[16];
} gpu_sample_t;

typedef struct
{
    pid_t pid;
    char  comm[64];
    int   mem_mib;
} gpu_app_t;

#define MAX_GPU_APPS 64

typedef struct
{
    FILE        *stream_fp;
    int          stream_fd;
    char         rdbuf[4096];
    size_t       rdlen;
    gpu_sample_t cur;

    pthread_t       app_thread;
    pthread_mutex_t app_mtx;
    int             app_thread_started;
    volatile int    stop_flag;
    gpu_app_t       apps[MAX_GPU_APPS];
    int             nb_apps;
} gpu_mon_t;

static void parse_gpu_csv_line(const char *line, gpu_sample_t *out)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", line);
    char *cols[8] = { NULL };
    int   nc      = 0;
    char *p       = buf;
    while (nc < 8 && p)
    {
        cols[nc++] = p;
        char *cm   = strchr(p, ',');
        if (!cm)
        {
            break;
        }
        *cm = '\0';
        p   = cm + 1;
    }
    if (nc >= 8)
    {
        for (int i = 0; i < 8; i++)
        {
            trim_str(cols[i]);
        }
        out->valid    = 1;
        out->gpu_util = atof(cols[0]);
        out->mem_util = atof(cols[1]);
        out->vram_mib = atof(cols[2]);
        out->power_w  = atof(cols[3]);
        out->sm_mhz   = atof(cols[4]);
        out->mem_mhz  = atof(cols[5]);
        out->temp_c   = atof(cols[6]);
        snprintf(out->pstate, sizeof(out->pstate), "%s", cols[7]);
    }
}

static void *gpu_app_worker(void *arg)
{
    gpu_mon_t *gm = (gpu_mon_t *) arg;
    while (!gm->stop_flag && !g_stop)
    {
        FILE *fp = popen("nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory "
                         "--format=csv,noheader,nounits 2>/dev/null",
                         "r");
        if (fp)
        {
            gpu_app_t tmp[MAX_GPU_APPS];
            int       n = 0;
            char      line[256];
            while (n < MAX_GPU_APPS && fgets(line, sizeof(line), fp) != NULL)
            {
                char *c1 = strchr(line, ',');
                if (!c1)
                {
                    continue;
                }
                char *c2 = strchr(c1 + 1, ',');
                if (!c2)
                {
                    continue;
                }
                *c1 = '\0';
                *c2 = '\0';
                trim_str(line);
                trim_str(c1 + 1);
                trim_str(c2 + 1);
                pid_t pid = (pid_t) atoi(line);
                if (pid > 0)
                {
                    tmp[n].pid     = pid;
                    const char *bn = strrchr(c1 + 1, '/');
                    snprintf(tmp[n].comm, sizeof(tmp[n].comm), "%s", bn ? (bn + 1) : (c1 + 1));
                    tmp[n].mem_mib = atoi(c2 + 1);
                    n++;
                }
            }
            pclose(fp);
            pthread_mutex_lock(&gm->app_mtx);
            gm->nb_apps = n;
            memcpy(gm->apps, tmp, (size_t) n * sizeof(gpu_app_t));
            pthread_mutex_unlock(&gm->app_mtx);
        }
        for (int i = 0; i < 10 && !gm->stop_flag && !g_stop; i++)
        {
            usleep(100000);
        }
    }
    return NULL;
}

static void gpu_mon_start(gpu_mon_t *gm, int gpu_present)
{
    memset(gm, 0, sizeof(*gm));
    pthread_mutex_init(&gm->app_mtx, NULL);
    gm->stream_fd = -1;
    if (!gpu_present)
    {
        return;
    }

    gm->stream_fp =
        popen("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,"
              "power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,pstate "
              "--format=csv,noheader,nounits -lms 200 2>/dev/null",
              "r");
    if (gm->stream_fp)
    {
        gm->stream_fd = fileno(gm->stream_fp);
        int flags     = fcntl(gm->stream_fd, F_GETFL, 0);
        if (flags >= 0)
        {
            fcntl(gm->stream_fd, F_SETFL, flags | O_NONBLOCK);
        }
    }
    if (pthread_create(&gm->app_thread, NULL, gpu_app_worker, gm) == 0)
    {
        gm->app_thread_started = 1;
    }
}

static void gpu_mon_poll(gpu_mon_t *gm, gpu_sample_t *out)
{
    if (gm->stream_fd >= 0)
    {
        for (;;)
        {
            size_t avail = sizeof(gm->rdbuf) - 1 - gm->rdlen;
            if (avail == 0)
            {
                gm->rdlen = 0;
                avail     = sizeof(gm->rdbuf) - 1;
            }
            ssize_t n = read(gm->stream_fd, gm->rdbuf + gm->rdlen, avail);
            if (n <= 0)
            {
                break;
            }
            gm->rdlen += (size_t) n;
            gm->rdbuf[gm->rdlen] = '\0';
            char *start          = gm->rdbuf;
            char *nl;
            while ((nl = strchr(start, '\n')) != NULL)
            {
                *nl = '\0';
                if (start[0] != '\0')
                {
                    parse_gpu_csv_line(start, &gm->cur);
                }
                start = nl + 1;
            }
            size_t rem = strlen(start);
            if (rem > 0 && start != gm->rdbuf)
            {
                memmove(gm->rdbuf, start, rem);
            }
            gm->rdlen = rem;
        }
    }
    *out = gm->cur;
}

static void gpu_mon_stop(gpu_mon_t *gm)
{
    gm->stop_flag = 1;
    if (gm->app_thread_started)
    {
        pthread_join(gm->app_thread, NULL);
        gm->app_thread_started = 0;
    }
    if (gm->stream_fp)
    {
        pclose(gm->stream_fp);
        gm->stream_fp = NULL;
        gm->stream_fd = -1;
    }
    pthread_mutex_destroy(&gm->app_mtx);
}

/* =========================================================
 * Per-CPU delta sampling (/proc/stat + cpufreq)
 * ======================================================= */

typedef struct
{
    long long user, nice, sys, idle, iowait, irq, softirq, steal;
} cpu_raw_t;

typedef struct
{
    int       has_prev;
    cpu_raw_t prev[SYSMON_MAX_CPUS];
} cpu_sampler_t;

typedef struct
{
    double total_cores;
    double user_cores;
    double sys_cores;
    double iowait_cores;
    double numa_cores[SYSMON_MAX_NUMA];
    double max_core_pct;
    int    cores_gt10;
    int    cores_gt50;
    double freq_mean_mhz;
    double freq_max_mhz;
} cpu_metrics_t;

static void sample_cpu(cpu_sampler_t *cs, const host_info_t *h, cpu_metrics_t *out)
{
    memset(out, 0, sizeof(*out));
    FILE *fp = fopen("/proc/stat", "r");
    if (fp)
    {
        char line[512];
        while (fgets(line, sizeof(line), fp))
        {
            if (strncmp(line, "cpu", 3) != 0 || !isdigit((unsigned char) line[3]))
            {
                continue;
            }
            int       idx = atoi(line + 3);
            cpu_raw_t cur = { 0 };
            if (idx < 0 || idx >= h->nb_cpus)
            {
                continue;
            }
            int nf = sscanf(line + 3, "%*d %lld %lld %lld %lld %lld %lld %lld %lld", &cur.user,
                            &cur.nice, &cur.sys, &cur.idle, &cur.iowait, &cur.irq, &cur.softirq,
                            &cur.steal);
            if (nf < 7)
            {
                continue;
            }
            if (cs->has_prev)
            {
                cpu_raw_t *p  = &cs->prev[idx];
                long long  du = (cur.user - p->user) + (cur.nice - p->nice);
                long long ds = (cur.sys - p->sys) + (cur.irq - p->irq) + (cur.softirq - p->softirq);
                long long diow  = cur.iowait - p->iowait;
                long long didle = cur.idle - p->idle;
                long long dtot  = du + ds + diow + didle + (cur.steal - p->steal);
                if (dtot > 0)
                {
                    double busy = (double) (dtot - didle - diow) / (double) dtot;
                    double uf   = (double) du / (double) dtot;
                    double sf   = (double) ds / (double) dtot;
                    double iowf = (double) diow / (double) dtot;
                    double pct  = 100.0 * busy;
                    out->total_cores += busy;
                    out->user_cores += uf;
                    out->sys_cores += sf;
                    out->iowait_cores += iowf;
                    int nn = h->cpu_to_numa[idx];
                    if (nn >= 0 && nn < SYSMON_MAX_NUMA)
                    {
                        out->numa_cores[nn] += busy;
                    }
                    if (pct > out->max_core_pct)
                    {
                        out->max_core_pct = pct;
                    }
                    if (pct > 10.0)
                    {
                        out->cores_gt10++;
                    }
                    if (pct > 50.0)
                    {
                        out->cores_gt50++;
                    }
                }
            }
            cs->prev[idx] = cur;
        }
        fclose(fp);
        cs->has_prev = 1;
    }

    int    stride = (h->nb_cpus > 64) ? (h->nb_cpus / 32) : 1;
    double fsum   = 0.0;
    double fmax   = 0.0;
    int    fcnt   = 0;
    for (int c = 0; c < h->nb_cpus; c += stride)
    {
        char path[128], buf[32];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", c);
        if (read_first_line(path, buf, sizeof(buf)) == 0)
        {
            double mhz = atof(buf) / 1000.0;
            if (mhz > 0.0)
            {
                fsum += mhz;
                if (mhz > fmax)
                {
                    fmax = mhz;
                }
                fcnt++;
            }
        }
    }
    if (fcnt > 0)
    {
        out->freq_mean_mhz = fsum / fcnt;
        out->freq_max_mhz  = fmax;
    }
}

/* =========================================================
 * Tracked Processes & Streams
 * ======================================================= */

typedef struct
{
    pid_t  pid;
    char   name[80];
    char   cmdline[128];
    char   kind[16]; /* "procinfo", "fps", "cli" */
    int    alive;
    int    seen_samples;
    double first_seen_t;
    double last_seen_t;

    /* OS CPU & scheduler totals */
    uint64_t prev_utime;
    uint64_t prev_stime;
    int      has_prev_cpu;
    double   total_user_s;
    double   total_sys_s;
    double   cpu_pct_peak;
    int      threads_max;
    long     threads_sum;
    int64_t  vmrss_peak_kb;
    int64_t  vmhwm_peak_kb;
    uint64_t base_vol_ctxt;
    uint64_t base_nonvol_ctxt;
    uint64_t base_minflt;
    uint64_t base_majflt;
    uint64_t base_migrations;
    int      has_base_sched;
    uint64_t vol_ctxt_delta;
    uint64_t nonvol_ctxt_delta;
    uint64_t minflt_delta;
    uint64_t majflt_delta;
    uint64_t migrations_delta;
    uint64_t numa_hits[SYSMON_MAX_NUMA];

    /* Closed-loop subset telemetry */
    double   cl_user_s;
    double   cl_sys_s;
    double   cl_dt_s;
    double   last_tick_cores;
    long     cl_loopcnt_start;
    long     cl_loopcnt_end;
    int      cl_has_loop;
    uint64_t cl_base_vol_ctxt;
    uint64_t cl_base_nonvol_ctxt;
    int      cl_has_sched;
    uint64_t cl_vol_ctxt_delta;
    uint64_t cl_nonvol_ctxt_delta;

    /* GPU compute allocation */
    double gpu_vram_peak_mib;
    double gpu_vram_sum_mib;
    int    gpu_samples;

    /* PROCESSINFO timing & config */
    int      has_pinfo;
    int      measure_timing;
    int      triggermode;
    int      rt_priority;
    char     trigstream[80];
    long     loopcnt;
    uint64_t missed_frames;
    uint64_t timeouts;
    int      timing_valid;
    int      timing_is_cl;
    int      timing_n;
    int64_t  exec_p50_ns;
    int64_t  exec_p95_ns;
    int64_t  exec_p99_ns;
    int64_t  exec_max_ns;
    int64_t  iter_p50_ns;
    int64_t  iter_p95_ns;
    int64_t  iter_p99_ns;
    int64_t  iter_max_ns;
    int64_t  jitter_ns;
} tracked_proc_t;

typedef struct
{
    char     name[80];
    uint8_t  datatype;
    uint8_t  naxis;
    uint32_t size[3];
    uint64_t nelement;
    uint64_t frame_bytes;
    int      location;
    uint64_t first_cnt0;
    uint64_t last_cnt0;
    uint64_t prev_cnt0;
    int      has_prev;
    double   first_t;
    double   last_t;
    double   peak_hz;
    double   peak_mb_s;
    double   cl_mb_sum;
    double   cl_hz_sum;
    int      cl_ticks;
    int64_t  last_e2e_ns;
    int      nb_readers;
    pid_t    write_pid;
} tracked_stream_t;

typedef struct
{
    double epoch_s;
    double rel_s;
    int    phase_idx;
    double total_cores;
    double sim_cores;
    double user_cores;
    double sys_cores;
    double iowait_cores;
    double numa0_cores;
    double numa1_cores;
    double max_core_pct;
    int    cores_gt10;
    int    cores_gt50;
    double freq_mean_mhz;
    double freq_max_mhz;
    double gpu_util;
    double gpu_mem_util;
    double gpu_vram_mib;
    double gpu_power_w;
    double gpu_sm_mhz;
    double gpu_mem_mhz;
    double gpu_temp_c;
    double shm_bw_mb_s;
    double shm_total_mb;
    int    active_procs;
    int    active_streams;
} tick_sample_t;

typedef struct
{
    double t_start;
    char   id[64];
    char   label[128];
} phase_mark_t;

/* Read proc stats directly for CPU ticks, memory, context switches, and NUMA threads */
static void update_proc_os_stats(tracked_proc_t    *tp,
                                 const host_info_t *h,
                                 double             clk_tck,
                                 double             dt_tick,
                                 int                in_cl)
{
    char path[128];
    snprintf(path, sizeof(path), "/proc/%d/stat", (int) tp->pid);
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        tp->alive = 0;
        return;
    }

    char     buf[1024];
    uint64_t ut = 0, st = 0, minf = 0, majf = 0;
    int64_t  rss_pages = 0;
    int      have_stat = 0;
    if (fgets(buf, sizeof(buf), fp))
    {
        char *p = strrchr(buf, ')');
        if (p && p[1] == ' ')
        {
            p += 2;
            const char *f = p;
            int         i = 0;
            while (f && *f)
            {
                if (i == 7)
                {
                    minf = strtoull(f, NULL, 10);
                }
                else if (i == 9)
                {
                    majf = strtoull(f, NULL, 10);
                }
                else if (i == 11)
                {
                    ut = strtoull(f, NULL, 10);
                }
                else if (i == 12)
                {
                    st        = strtoull(f, NULL, 10);
                    have_stat = 1;
                }
                else if (i == 21)
                {
                    rss_pages = strtoll(f, NULL, 10);
                    break;
                }
                f = strchr(f, ' ');
                if (f)
                {
                    f++;
                }
                i++;
            }
        }
    }
    fclose(fp);
    if (!have_stat)
    {
        return;
    }

    tp->alive    = 1;
    long page_kb = sysconf(_SC_PAGESIZE) / 1024;
    if (page_kb > 0)
    {
        int64_t rss_kb = rss_pages * page_kb;
        if (rss_kb > tp->vmrss_peak_kb)
        {
            tp->vmrss_peak_kb = rss_kb;
        }
    }

    if (tp->has_prev_cpu && dt_tick > 0.01)
    {
        double du = (ut >= tp->prev_utime) ? (double) (ut - tp->prev_utime) / clk_tck : 0.0;
        double ds = (st >= tp->prev_stime) ? (double) (st - tp->prev_stime) / clk_tck : 0.0;
        tp->total_user_s += du;
        tp->total_sys_s += ds;
        tp->last_tick_cores = (du + ds) / dt_tick;
        double pct          = 100.0 * tp->last_tick_cores;
        if (pct > tp->cpu_pct_peak)
        {
            tp->cpu_pct_peak = pct;
        }
        if (in_cl)
        {
            tp->cl_user_s += du;
            tp->cl_sys_s += ds;
        }
    }
    else
    {
        tp->last_tick_cores = 0.0;
    }
    tp->prev_utime   = ut;
    tp->prev_stime   = st;
    tp->has_prev_cpu = 1;

    int      th   = 1;
    int64_t  hwm  = 0;
    uint64_t vctx = 0, nvctx = 0, mig = 0;
    snprintf(path, sizeof(path), "/proc/%d/status", (int) tp->pid);
    fp = fopen(path, "r");
    if (fp)
    {
        char line[256];
        while (fgets(line, sizeof(line), fp))
        {
            if (strncmp(line, "Threads:", 8) == 0)
            {
                th = atoi(line + 8);
            }
            else if (strncmp(line, "VmHWM:", 6) == 0)
            {
                hwm = strtoll(line + 6, NULL, 10);
            }
            else if (strncmp(line, "voluntary_ctxt_switches:", 24) == 0)
            {
                vctx = strtoull(line + 24, NULL, 10);
            }
            else if (strncmp(line, "nonvoluntary_ctxt_switches:", 27) == 0)
            {
                nvctx = strtoull(line + 27, NULL, 10);
            }
        }
        fclose(fp);
    }

    if (th > tp->threads_max)
    {
        tp->threads_max = th;
    }
    tp->threads_sum += th;
    if (hwm > tp->vmhwm_peak_kb)
    {
        tp->vmhwm_peak_kb = hwm;
    }

    snprintf(path, sizeof(path), "/proc/%d/sched", (int) tp->pid);
    fp = fopen(path, "r");
    if (fp)
    {
        char line[256];
        while (fgets(line, sizeof(line), fp))
        {
            if (strncmp(line, "se.nr_migrations", 16) == 0)
            {
                char *col = strchr(line, ':');
                if (col)
                {
                    mig = strtoull(col + 1, NULL, 10);
                }
                break;
            }
        }
        fclose(fp);
    }

    if (!tp->has_base_sched)
    {
        tp->base_vol_ctxt    = vctx;
        tp->base_nonvol_ctxt = nvctx;
        tp->base_minflt      = minf;
        tp->base_majflt      = majf;
        tp->base_migrations  = mig;
        tp->has_base_sched   = 1;
    }
    else
    {
        if (vctx >= tp->base_vol_ctxt)
        {
            tp->vol_ctxt_delta = vctx - tp->base_vol_ctxt;
        }
        if (nvctx >= tp->base_nonvol_ctxt)
        {
            tp->nonvol_ctxt_delta = nvctx - tp->base_nonvol_ctxt;
        }
        if (minf >= tp->base_minflt)
        {
            tp->minflt_delta = minf - tp->base_minflt;
        }
        if (majf >= tp->base_majflt)
        {
            tp->majflt_delta = majf - tp->base_majflt;
        }
        if (mig >= tp->base_migrations)
        {
            tp->migrations_delta = mig - tp->base_migrations;
        }
    }

    if (in_cl)
    {
        if (!tp->cl_has_sched)
        {
            tp->cl_base_vol_ctxt    = vctx;
            tp->cl_base_nonvol_ctxt = nvctx;
            tp->cl_has_sched        = 1;
        }
        else
        {
            if (vctx >= tp->cl_base_vol_ctxt)
            {
                tp->cl_vol_ctxt_delta = vctx - tp->cl_base_vol_ctxt;
            }
            if (nvctx >= tp->cl_base_nonvol_ctxt)
            {
                tp->cl_nonvol_ctxt_delta = nvctx - tp->cl_base_nonvol_ctxt;
            }
        }
    }

    /* Sample thread NUMA residency from /proc/<pid>/task/<tid>/stat */
    char tdir[128];
    snprintf(tdir, sizeof(tdir), "/proc/%d/task", (int) tp->pid);
    DIR *d = opendir(tdir);
    if (d)
    {
        struct dirent *de;
        while ((de = readdir(d)) != NULL)
        {
            if (!isdigit((unsigned char) de->d_name[0]))
            {
                continue;
            }
            snprintf(path, sizeof(path), "%s/%s/stat", tdir, de->d_name);
            fp = fopen(path, "r");
            if (fp)
            {
                if (fgets(buf, sizeof(buf), fp))
                {
                    char *p = strrchr(buf, ')');
                    if (p && p[1] == ' ')
                    {
                        p += 2;
                        for (int k = 0; k < 36 && p; k++)
                        {
                            p = strchr(p, ' ');
                            if (p)
                            {
                                p++;
                            }
                        }
                        if (p)
                        {
                            int cpu = atoi(p);
                            if (cpu >= 0 && cpu < h->nb_cpus)
                            {
                                int nn = h->cpu_to_numa[cpu];
                                if (nn >= 0 && nn < SYSMON_MAX_NUMA)
                                {
                                    tp->numa_hits[nn]++;
                                }
                            }
                        }
                    }
                }
                fclose(fp);
            }
        }
        closedir(d);
    }
}

static void extract_pinfo_ring(const PROCESSINFO *pi, tracked_proc_t *tp, int in_cl)
{
    if (!pi)
    {
        return;
    }
    tp->measure_timing = pi->MeasureTiming;
    tp->missed_frames  = pi->triggermissedframe_cumul;
    tp->timeouts       = pi->trigggertimeoutcnt;
    tp->rt_priority    = pi->RT_priority;
    if (pi->MeasureTiming == 0)
    {
        return;
    }
    int nbsam = (pi->timingbuffercnt > 0) ? PROCESSINFO_NBtimer : pi->timerindex;
    if (nbsam > PROCESSINFO_NBtimer)
    {
        nbsam = PROCESSINFO_NBtimer;
    }
    int64_t exec_ns[PROCESSINFO_NBtimer];
    int64_t iter_ns[PROCESSINFO_NBtimer];
    int     nv = 0;
    for (int i = 1; i < nbsam; i++)
    {
        int64_t dt_e =
            (int64_t) (pi->texecend[i].tv_sec - pi->texecstart[i].tv_sec) * 1000000000LL +
            (int64_t) (pi->texecend[i].tv_nsec - pi->texecstart[i].tv_nsec);
        int64_t dt_i =
            (int64_t) (pi->texecstart[i].tv_sec - pi->texecstart[i - 1].tv_sec) * 1000000000LL +
            (int64_t) (pi->texecstart[i].tv_nsec - pi->texecstart[i - 1].tv_nsec);
        if (dt_e > 0 && dt_i > 0 && dt_e < 10000000000LL && dt_i < 10000000000LL)
        {
            exec_ns[nv] = dt_e;
            iter_ns[nv] = dt_i;
            nv++;
        }
    }
    if (nv < 4)
    {
        return;
    }
    /* Keep closed-loop percentiles over post/pre phase samples once observed in closed_loop */
    if (tp->timing_is_cl && !in_cl)
    {
        return;
    }
    qsort(exec_ns, (size_t) nv, sizeof(int64_t), cmp_int64);
    qsort(iter_ns, (size_t) nv, sizeof(int64_t), cmp_int64);
    tp->timing_valid = 1;
    if (in_cl)
    {
        tp->timing_is_cl = 1;
    }
    tp->timing_n    = nv;
    tp->exec_p50_ns = exec_ns[nv / 2];
    tp->exec_p95_ns = exec_ns[(nv * 95) / 100];
    tp->exec_p99_ns = exec_ns[(nv * 99) / 100];
    tp->exec_max_ns = exec_ns[nv - 1];
    tp->iter_p50_ns = iter_ns[nv / 2];
    tp->iter_p95_ns = iter_ns[(nv * 95) / 100];
    tp->iter_p99_ns = iter_ns[(nv * 99) / 100];
    tp->iter_max_ns = iter_ns[nv - 1];
    tp->jitter_ns   = tp->iter_p99_ns - tp->iter_p50_ns;
}

static void refresh_pinfo_rings(const host_info_t *h,
                                tracked_proc_t    *procs,
                                int                nb_procs,
                                int                in_cl)
{
    for (int i = 0; i < nb_procs; i++)
    {
        tracked_proc_t *tp = &procs[i];
        if (!tp->alive || !tp->has_pinfo)
        {
            continue;
        }
        char ppath[1024];
        snprintf(ppath, sizeof(ppath), "%s/proc.%s.%06d.shm", h->procdir, tp->name, (int) tp->pid);
        int          fd = -1;
        PROCESSINFO *pi = processinfo_shm_link_ro(ppath, &fd);
        if (pi != MAP_FAILED && pi != NULL)
        {
            extract_pinfo_ring(pi, tp, in_cl);
            processinfo_shm_close(pi, fd);
        }
    }
}

static tracked_proc_t *find_or_add_proc(tracked_proc_t *procs,
                                        int            *nb_procs,
                                        pid_t           pid,
                                        const char     *name,
                                        const char     *kind,
                                        double          t_rel)
{
    for (int i = 0; i < *nb_procs; i++)
    {
        if (procs[i].pid == pid)
        {
            /* Upgrade auto-discovered CLI name to canonical PROCESSINFO / FPS name */
            if (strcmp(kind, "procinfo") == 0 && strcmp(procs[i].kind, "procinfo") != 0 &&
                name[0] != '\0')
            {
                snprintf(procs[i].name, sizeof(procs[i].name), "%s", name);
                snprintf(procs[i].kind, sizeof(procs[i].kind), "%s", kind);
            }
            return &procs[i];
        }
    }
    if (*nb_procs >= SYSMON_MAX_PROCS)
    {
        return NULL;
    }
    tracked_proc_t *tp = &procs[(*nb_procs)++];
    memset(tp, 0, sizeof(*tp));
    tp->pid          = pid;
    tp->alive        = 1;
    tp->first_seen_t = t_rel;
    tp->last_seen_t  = t_rel;
    snprintf(tp->name, sizeof(tp->name), "%s", name);
    snprintf(tp->kind, sizeof(tp->kind), "%s", kind);
    char cpath[128];
    snprintf(cpath, sizeof(cpath), "/proc/%d/cmdline", (int) pid);
    int fd = open(cpath, O_RDONLY);
    if (fd >= 0)
    {
        ssize_t n = read(fd, tp->cmdline, sizeof(tp->cmdline) - 1);
        close(fd);
        if (n > 0)
        {
            for (ssize_t k = 0; k < n - 1; k++)
            {
                if (tp->cmdline[k] == '\0')
                {
                    tp->cmdline[k] = ' ';
                }
            }
            tp->cmdline[n] = '\0';
        }
    }
    return tp;
}

static void scan_extra_cli_procs(uid_t my_uid, tracked_proc_t *procs, int *nb_procs, double t_rel)
{
    DIR *d = opendir("/proc");
    if (!d)
    {
        return;
    }
    struct dirent *de;
    while ((de = readdir(d)) != NULL)
    {
        if (!isdigit((unsigned char) de->d_name[0]))
        {
            continue;
        }
        pid_t pid = (pid_t) atoi(de->d_name);
        if (pid <= 1 || pid == getpid())
        {
            continue;
        }
        struct stat st;
        char        pdir[64];
        snprintf(pdir, sizeof(pdir), "/proc/%d", (int) pid);
        if (stat(pdir, &st) != 0 || st.st_uid != my_uid)
        {
            continue;
        }
        int already = 0;
        for (int i = 0; i < *nb_procs; i++)
        {
            if (procs[i].pid == pid)
            {
                already = 1;
                break;
            }
        }
        if (already)
        {
            continue;
        }
        char comm[64], cpath[96];
        snprintf(cpath, sizeof(cpath), "%s/comm", pdir);
        if (read_first_line(cpath, comm, sizeof(comm)) != 0)
        {
            continue;
        }
        if (strncmp(comm, "cacao", 5) == 0 || strncmp(comm, "milk", 4) == 0 ||
            strncmp(comm, "csim_", 5) == 0)
        {
            if (strcmp(comm, "milk-sysmon") == 0)
            {
                continue;
            }
            (void) find_or_add_proc(procs, nb_procs, pid, comm, "cli", t_rel);
        }
    }
    closedir(d);
}

static tracked_stream_t *find_or_add_stream(tracked_stream_t *streams,
                                            int              *nb_streams,
                                            const OV_STREAM  *os,
                                            const char       *shmdir,
                                            double            t_rel)
{
    for (int i = 0; i < *nb_streams; i++)
    {
        if (strcmp(streams[i].name, os->name) == 0)
        {
            return &streams[i];
        }
    }
    if (*nb_streams >= SYSMON_MAX_STREAMS)
    {
        return NULL;
    }
    tracked_stream_t *st = &streams[(*nb_streams)++];
    memset(st, 0, sizeof(*st));
    snprintf(st->name, sizeof(st->name), "%s", os->name);
    st->datatype    = os->datatype;
    st->naxis       = os->naxis;
    st->size[0]     = os->size[0];
    st->size[1]     = os->size[1];
    st->size[2]     = os->size[2];
    st->nelement    = os->nelement;
    st->frame_bytes = dtype_bytes(os->datatype) * os->nelement;
    st->first_cnt0  = os->cnt0;
    st->last_cnt0   = os->cnt0;
    st->prev_cnt0   = os->cnt0;
    st->has_prev    = 1;
    st->first_t     = t_rel;
    st->last_t      = t_rel;
    st->location    = -1;

    char spath[PATH_MAX];
    snprintf(spath, sizeof(spath), "%s/%s.im.shm", shmdir, os->name);
    int fd = open(spath, O_RDONLY);
    if (fd >= 0)
    {
        IMAGE_METADATA md;
        if (read(fd, &md, sizeof(md)) == (ssize_t) sizeof(md))
        {
            st->location = (int) md.location;
        }
        close(fd);
    }
    return st;
}

/* =========================================================
 * Phase table reader
 * ======================================================= */

static int load_phases_file(const char *path, phase_mark_t *phases, int max_phases, double t0)
{
    if (!path || path[0] == '\0')
    {
        return 0;
    }
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        return 0;
    }
    int  n = 0;
    char line[256];
    while (n < max_phases && fgets(line, sizeof(line), fp) != NULL)
    {
        double ts     = 0.0;
        char   id[64] = "", lbl[128] = "";
        char  *t1 = strchr(line, '\t');
        if (!t1)
        {
            continue;
        }
        *t1      = '\0';
        ts       = atof(line);
        char *t2 = strchr(t1 + 1, '\t');
        if (t2)
        {
            *t2 = '\0';
            snprintf(id, sizeof(id), "%s", t1 + 1);
            snprintf(lbl, sizeof(lbl), "%s", t2 + 1);
        }
        else
        {
            snprintf(id, sizeof(id), "%s", t1 + 1);
            snprintf(lbl, sizeof(lbl), "%s", t1 + 1);
        }
        trim_str(id);
        trim_str(lbl);
        if (id[0] != '\0')
        {
            phases[n].t_start = ts - t0;
            snprintf(phases[n].id, sizeof(phases[n].id), "%s", id);
            snprintf(phases[n].label, sizeof(phases[n].label), "%s", lbl);
            n++;
        }
    }
    fclose(fp);
    return n;
}

static int resolve_phase_idx(const phase_mark_t *phases, int nb_phases, double t_rel)
{
    int idx = -1;
    for (int i = 0; i < nb_phases; i++)
    {
        if (t_rel >= phases[i].t_start)
        {
            idx = i;
        }
    }
    return idx;
}

/* =========================================================
 * CSV & JSON Export
 * ======================================================= */

static void write_csv(const char          *csv_path,
                      const tick_sample_t *ticks,
                      int                  nb_ticks,
                      const phase_mark_t  *phases,
                      int                  nb_phases)
{
    if (!csv_path || csv_path[0] == '\0')
    {
        return;
    }
    FILE *fp = fopen(csv_path, "w");
    if (!fp)
    {
        return;
    }
    fprintf(fp, "t_rel_s,phase,total_cores,sim_cores,user_cores,sys_cores,iowait_cores,"
                "numa0_cores,numa1_cores,max_core_pct,cores_gt10,cores_gt50,"
                "cpu_freq_mean_mhz,cpu_freq_max_mhz,gpu_util_pct,gpu_mem_util_pct,"
                "gpu_vram_mib,gpu_power_w,gpu_sm_mhz,gpu_mem_mhz,gpu_temp_c,"
                "shm_bw_mb_s,shm_total_mb,active_procs,active_streams\n");
    for (int i = 0; i < nb_ticks; i++)
    {
        const tick_sample_t *t = &ticks[i];
        int                  pi =
            (t->phase_idx >= 0) ? t->phase_idx : resolve_phase_idx(phases, nb_phases, t->rel_s);
        const char *pid_str = (pi >= 0 && pi < nb_phases) ? phases[pi].id : "unmarked";
        fprintf(fp,
                "%.3f,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%d,%d,"
                "%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.0f,%.0f,%.1f,%.2f,%.2f,%d,%d\n",
                t->rel_s, pid_str, safe_d(t->total_cores), safe_d(t->sim_cores),
                safe_d(t->user_cores), safe_d(t->sys_cores), safe_d(t->iowait_cores),
                safe_d(t->numa0_cores), safe_d(t->numa1_cores), safe_d(t->max_core_pct),
                t->cores_gt10, t->cores_gt50, safe_d(t->freq_mean_mhz), safe_d(t->freq_max_mhz),
                safe_d(t->gpu_util), safe_d(t->gpu_mem_util), safe_d(t->gpu_vram_mib),
                safe_d(t->gpu_power_w), safe_d(t->gpu_sm_mhz), safe_d(t->gpu_mem_mhz),
                safe_d(t->gpu_temp_c), safe_d(t->shm_bw_mb_s), safe_d(t->shm_total_mb),
                t->active_procs, t->active_streams);
    }
    fclose(fp);
}

static void write_phase_summary_json(FILE                *fp,
                                     const char          *id,
                                     const char          *label,
                                     int                  phase_filter,
                                     const tick_sample_t *ticks,
                                     int                  nb_ticks,
                                     double               total_dur)
{
    int    cnt        = 0;
    double dur        = 0.0;
    double core_s     = 0.0;
    double sim_core_s = 0.0;
    double sim_pk     = 0.0;
    double user_s     = 0.0;
    double sys_s      = 0.0;
    double numa0_s    = 0.0;
    double numa1_s    = 0.0;
    double cores_pk   = 0.0;
    double gpu_sum    = 0.0;
    double gpu_pk     = 0.0;
    double gmem_sum   = 0.0;
    double vram_pk    = 0.0;
    double pwr_sum    = 0.0;
    double pwr_pk     = 0.0;
    double energy_j   = 0.0;
    double smclk_sum  = 0.0;
    double shm_sum    = 0.0;
    double shm_pk     = 0.0;
    double t_min      = 0.0;
    double t_max      = 0.0;

    for (int i = 0; i < nb_ticks; i++)
    {
        if (phase_filter >= 0 && ticks[i].phase_idx != phase_filter)
        {
            continue;
        }
        const tick_sample_t *t = &ticks[i];
        double               dt =
            (i > 0) ? (ticks[i].rel_s - ticks[i - 1].rel_s) : ((nb_ticks > 1) ? 0.2 : total_dur);
        if (dt <= 0.0 || dt > 5.0)
        {
            dt = 0.2;
        }
        if (cnt == 0)
        {
            t_min = t->rel_s;
        }
        t_max = t->rel_s;
        cnt++;
        dur += dt;
        core_s += t->total_cores * dt;
        sim_core_s += t->sim_cores * dt;
        user_s += t->user_cores * dt;
        sys_s += t->sys_cores * dt;
        numa0_s += t->numa0_cores * dt;
        numa1_s += t->numa1_cores * dt;
        if (t->total_cores > cores_pk)
        {
            cores_pk = t->total_cores;
        }
        if (t->sim_cores > sim_pk)
        {
            sim_pk = t->sim_cores;
        }
        gpu_sum += t->gpu_util * dt;
        gmem_sum += t->gpu_mem_util * dt;
        pwr_sum += t->gpu_power_w * dt;
        energy_j += t->gpu_power_w * dt;
        smclk_sum += t->gpu_sm_mhz * dt;
        shm_sum += t->shm_bw_mb_s * dt;
        if (t->gpu_util > gpu_pk)
        {
            gpu_pk = t->gpu_util;
        }
        if (t->gpu_vram_mib > vram_pk)
        {
            vram_pk = t->gpu_vram_mib;
        }
        if (t->gpu_power_w > pwr_pk)
        {
            pwr_pk = t->gpu_power_w;
        }
        if (t->shm_bw_mb_s > shm_pk)
        {
            shm_pk = t->shm_bw_mb_s;
        }
    }
    double denom = (dur > 0.0) ? dur : 1.0;
    fprintf(fp, "    {\n");
    fprintf(fp, "      \"id\": ");
    json_escape(fp, id);
    fprintf(fp, ",\n      \"label\": ");
    json_escape(fp, label);
    fprintf(fp, ",\n      \"samples\": %d,\n", cnt);
    fprintf(fp, "      \"t_start_s\": %.3f,\n", safe_d(t_min));
    fprintf(fp, "      \"t_end_s\": %.3f,\n", safe_d(t_max));
    fprintf(fp, "      \"duration_s\": %.3f,\n", safe_d(dur));
    fprintf(fp, "      \"cpu_core_s\": %.3f,\n", safe_d(core_s));
    fprintf(fp, "      \"cpu_user_core_s\": %.3f,\n", safe_d(user_s));
    fprintf(fp, "      \"cpu_sys_core_s\": %.3f,\n", safe_d(sys_s));
    fprintf(fp, "      \"cpu_cores_mean\": %.3f,\n", safe_d(core_s / denom));
    fprintf(fp, "      \"cpu_cores_peak\": %.3f,\n", safe_d(cores_pk));
    fprintf(fp, "      \"sim_cpu_core_s\": %.3f,\n", safe_d(sim_core_s));
    fprintf(fp, "      \"sim_cores_mean\": %.3f,\n", safe_d(sim_core_s / denom));
    fprintf(fp, "      \"sim_cores_peak\": %.3f,\n", safe_d(sim_pk));
    fprintf(fp, "      \"cpu_sys_share_pct\": %.2f,\n",
            (core_s > 0.0) ? safe_d(100.0 * sys_s / core_s) : 0.0);
    fprintf(fp, "      \"numa0_cores_mean\": %.3f,\n", safe_d(numa0_s / denom));
    fprintf(fp, "      \"numa1_cores_mean\": %.3f,\n", safe_d(numa1_s / denom));
    fprintf(fp, "      \"gpu_util_mean_pct\": %.2f,\n", safe_d(gpu_sum / denom));
    fprintf(fp, "      \"gpu_util_peak_pct\": %.2f,\n", safe_d(gpu_pk));
    fprintf(fp, "      \"gpu_mem_util_mean_pct\": %.2f,\n", safe_d(gmem_sum / denom));
    fprintf(fp, "      \"gpu_vram_peak_mib\": %.1f,\n", safe_d(vram_pk));
    fprintf(fp, "      \"gpu_power_mean_w\": %.2f,\n", safe_d(pwr_sum / denom));
    fprintf(fp, "      \"gpu_power_peak_w\": %.2f,\n", safe_d(pwr_pk));
    fprintf(fp, "      \"gpu_energy_j\": %.1f,\n", safe_d(energy_j));
    fprintf(fp, "      \"gpu_sm_clock_mean_mhz\": %.1f,\n", safe_d(smclk_sum / denom));
    fprintf(fp, "      \"shm_bw_mean_mb_s\": %.2f,\n", safe_d(shm_sum / denom));
    fprintf(fp, "      \"shm_bw_peak_mb_s\": %.2f\n", safe_d(shm_pk));
    fprintf(fp, "    }");
}

static void write_json(const char             *json_path,
                       const host_info_t      *h,
                       double                  wall_total_s,
                       double                  cl_total_s,
                       const tick_sample_t    *ticks,
                       int                     nb_ticks,
                       const phase_mark_t     *phases,
                       int                     nb_phases,
                       const tracked_proc_t   *procs,
                       int                     nb_procs,
                       const tracked_stream_t *streams,
                       int                     nb_streams)
{
    FILE *fp = fopen(json_path, "w");
    if (!fp)
    {
        fprintf(stderr, "milk-sysmon: failed to write %s: %s\n", json_path, strerror(errno));
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"host\": {\n");
    fprintf(fp, "    \"hostname\": ");
    json_escape(fp, h->hostname);
    fprintf(fp, ",\n    \"kernel\": ");
    json_escape(fp, h->kernel);
    fprintf(fp, ",\n    \"cpu_model\": ");
    json_escape(fp, h->cpu_model);
    fprintf(fp, ",\n    \"nb_cpus\": %d,\n", h->nb_cpus);
    fprintf(fp, "    \"nb_numa_nodes\": %d,\n", h->nb_numa);
    fprintf(fp, "    \"numa_node0_cpus\": ");
    json_escape(fp, h->numa_cpulist[0]);
    fprintf(fp, ",\n    \"numa_node1_cpus\": ");
    json_escape(fp, (h->nb_numa > 1) ? h->numa_cpulist[1] : "");
    fprintf(fp, ",\n    \"scaling_governor\": ");
    json_escape(fp, h->governor);
    fprintf(fp, ",\n    \"cpu_max_mhz\": %.1f,\n", safe_d(h->cpu_max_mhz));
    fprintf(fp, "    \"ram_total_mb\": %.1f,\n", safe_d(h->ram_total_mb));
    fprintf(fp, "    \"ram_avail_mb\": %.1f,\n", safe_d(h->ram_avail_mb));
    fprintf(fp, "    \"shm_total_mb\": %.1f,\n", safe_d(h->shm_total_mb));
    fprintf(fp, "    \"shm_free_mb\": %.1f,\n", safe_d(h->shm_free_mb));
    fprintf(fp, "    \"perf_event_paranoid\": %d,\n", h->perf_paranoid);
    fprintf(fp, "    \"bash_version\": ");
    json_escape(fp, h->bash_version);
    fprintf(fp, ",\n    \"shmdir\": ");
    json_escape(fp, h->shmdir);
    fprintf(fp, ",\n    \"procdir\": ");
    json_escape(fp, h->procdir);
    fprintf(fp, "\n  },\n");

    fprintf(fp, "  \"gpu\": {\n");
    fprintf(fp, "    \"present\": %s,\n", h->gpu_present ? "true" : "false");
    fprintf(fp, "    \"name\": ");
    json_escape(fp, h->gpu_name);
    fprintf(fp, ",\n    \"driver\": ");
    json_escape(fp, h->gpu_driver);
    fprintf(fp, ",\n    \"pci_bus_id\": ");
    json_escape(fp, h->gpu_pci_bus);
    fprintf(fp, ",\n    \"pcie_gen\": %d,\n", h->gpu_pcie_gen);
    fprintf(fp, "    \"pcie_gen_max\": %d,\n", h->gpu_pcie_gen_max);
    fprintf(fp, "    \"pcie_width\": %d,\n", h->gpu_pcie_width);
    fprintf(fp, "    \"pcie_width_max\": %d,\n", h->gpu_pcie_width_max);
    fprintf(fp, "    \"vram_total_mib\": %.1f,\n", safe_d(h->gpu_vram_total_mib));
    fprintf(fp, "    \"power_limit_w\": %.1f,\n", safe_d(h->gpu_power_limit_w));
    fprintf(fp, "    \"numa_node\": %d\n", h->gpu_numa_node);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"run\": {\n");
    fprintf(fp, "    \"wall_duration_s\": %.3f,\n", safe_d(wall_total_s));
    fprintf(fp, "    \"closed_loop_duration_s\": %.3f,\n", safe_d(cl_total_s));
    fprintf(fp, "    \"samples\": %d\n", nb_ticks);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"phases\": [\n");
    write_phase_summary_json(fp, "overall", "Entire Monitored Run", -1, ticks, nb_ticks,
                             wall_total_s);
    for (int pi = 0; pi < nb_phases; pi++)
    {
        fprintf(fp, ",\n");
        write_phase_summary_json(fp, phases[pi].id, phases[pi].label, pi, ticks, nb_ticks,
                                 wall_total_s);
    }
    fprintf(fp, "\n  ],\n");

    fprintf(fp, "  \"processes\": [\n");
    for (int i = 0; i < nb_procs; i++)
    {
        const tracked_proc_t *p     = &procs[i];
        double                tot_s = p->total_user_s + p->total_sys_s;
        double                dur_s = p->last_seen_t - p->first_seen_t;
        if (dur_s < 0.2)
        {
            dur_s = 0.2;
        }
        double   cl_tot_s = p->cl_user_s + p->cl_sys_s;
        double   cl_denom = (cl_total_s > 0.1) ? cl_total_s : 1.0;
        uint64_t numa_tot = 0;
        for (int nn = 0; nn < h->nb_numa; nn++)
        {
            numa_tot += p->numa_hits[nn];
        }
        uint64_t n0        = p->numa_hits[0];
        uint64_t n1        = (h->nb_numa > 1) ? p->numa_hits[1] : 0;
        double   n_rem_pct = 0.0;
        if (numa_tot > 0 && h->gpu_numa_node >= 0 && h->gpu_numa_node < h->nb_numa)
        {
            n_rem_pct =
                100.0 * (double) (numa_tot - p->numa_hits[h->gpu_numa_node]) / (double) numa_tot;
        }
        long cl_loops = (p->cl_has_loop && p->cl_loopcnt_end >= p->cl_loopcnt_start)
                            ? (p->cl_loopcnt_end - p->cl_loopcnt_start)
                            : 0;
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"pid\": %d,\n", (int) p->pid);
        fprintf(fp, "      \"name\": ");
        json_escape(fp, p->name);
        fprintf(fp, ",\n      \"kind\": ");
        json_escape(fp, p->kind);
        fprintf(fp, ",\n      \"cmdline\": ");
        json_escape(fp, p->cmdline);
        fprintf(fp, ",\n      \"cpu_total_s\": %.4f,\n", safe_d(tot_s));
        fprintf(fp, "      \"cpu_user_s\": %.4f,\n", safe_d(p->total_user_s));
        fprintf(fp, "      \"cpu_sys_s\": %.4f,\n", safe_d(p->total_sys_s));
        fprintf(fp, "      \"cpu_pct_mean\": %.2f,\n", safe_d(100.0 * tot_s / dur_s));
        fprintf(fp, "      \"cpu_pct_peak\": %.2f,\n", safe_d(p->cpu_pct_peak));
        fprintf(fp, "      \"cl_cpu_s\": %.4f,\n", safe_d(cl_tot_s));
        fprintf(fp, "      \"cl_cpu_user_s\": %.4f,\n", safe_d(p->cl_user_s));
        fprintf(fp, "      \"cl_cpu_sys_s\": %.4f,\n", safe_d(p->cl_sys_s));
        fprintf(fp, "      \"cl_cpu_pct\": %.2f,\n", safe_d(100.0 * cl_tot_s / cl_denom));
        fprintf(fp, "      \"cl_loopcnt\": %ld,\n", cl_loops);
        fprintf(fp, "      \"cl_vol_ctxt\": %" PRIu64 ",\n", p->cl_vol_ctxt_delta);
        fprintf(fp, "      \"cl_nonvol_ctxt\": %" PRIu64 ",\n", p->cl_nonvol_ctxt_delta);
        fprintf(fp, "      \"threads_mean\": %.2f,\n",
                (p->seen_samples > 0) ? safe_d((double) p->threads_sum / p->seen_samples) : 1.0);
        fprintf(fp, "      \"threads_max\": %d,\n", p->threads_max);
        fprintf(fp, "      \"vmrss_peak_kb\": %" PRId64 ",\n", p->vmrss_peak_kb);
        fprintf(fp, "      \"vmhwm_peak_kb\": %" PRId64 ",\n", p->vmhwm_peak_kb);
        fprintf(fp, "      \"vol_ctxt\": %" PRIu64 ",\n", p->vol_ctxt_delta);
        fprintf(fp, "      \"nonvol_ctxt\": %" PRIu64 ",\n", p->nonvol_ctxt_delta);
        fprintf(fp, "      \"minflt\": %" PRIu64 ",\n", p->minflt_delta);
        fprintf(fp, "      \"majflt\": %" PRIu64 ",\n", p->majflt_delta);
        fprintf(fp, "      \"migrations\": %" PRIu64 ",\n", p->migrations_delta);
        fprintf(fp, "      \"numa0_hits\": %" PRIu64 ",\n", n0);
        fprintf(fp, "      \"numa1_hits\": %" PRIu64 ",\n", n1);
        fprintf(fp, "      \"numa_remote_pct\": %.2f,\n", safe_d(n_rem_pct));
        fprintf(fp, "      \"gpu_vram_peak_mib\": %.1f,\n", safe_d(p->gpu_vram_peak_mib));
        fprintf(fp, "      \"has_pinfo\": %s,\n", p->has_pinfo ? "true" : "false");
        fprintf(fp, "      \"measure_timing\": %d,\n", p->measure_timing);
        fprintf(fp, "      \"triggermode\": %d,\n", p->triggermode);
        fprintf(fp, "      \"trigstream\": ");
        json_escape(fp, p->trigstream);
        fprintf(fp, ",\n      \"rt_priority\": %d,\n", p->rt_priority);
        fprintf(fp, "      \"loopcnt\": %ld,\n", p->loopcnt);
        fprintf(fp, "      \"missed_frames\": %" PRIu64 ",\n", p->missed_frames);
        fprintf(fp, "      \"timeouts\": %" PRIu64 ",\n", p->timeouts);
        fprintf(fp, "      \"timing_valid\": %s,\n", p->timing_valid ? "true" : "false");
        fprintf(fp, "      \"timing_n\": %d,\n", p->timing_n);
        fprintf(fp, "      \"exec_p50_us\": %.2f,\n", safe_d((double) p->exec_p50_ns / 1000.0));
        fprintf(fp, "      \"exec_p95_us\": %.2f,\n", safe_d((double) p->exec_p95_ns / 1000.0));
        fprintf(fp, "      \"exec_p99_us\": %.2f,\n", safe_d((double) p->exec_p99_ns / 1000.0));
        fprintf(fp, "      \"exec_max_us\": %.2f,\n", safe_d((double) p->exec_max_ns / 1000.0));
        fprintf(fp, "      \"iter_p50_us\": %.2f,\n", safe_d((double) p->iter_p50_ns / 1000.0));
        fprintf(fp, "      \"iter_p95_us\": %.2f,\n", safe_d((double) p->iter_p95_ns / 1000.0));
        fprintf(fp, "      \"iter_p99_us\": %.2f,\n", safe_d((double) p->iter_p99_ns / 1000.0));
        fprintf(fp, "      \"iter_max_us\": %.2f,\n", safe_d((double) p->iter_max_ns / 1000.0));
        fprintf(fp, "      \"jitter_us\": %.2f\n", safe_d((double) p->jitter_ns / 1000.0));
        fprintf(fp, "    }%s\n", (i + 1 < nb_procs) ? "," : "");
    }
    fprintf(fp, "  ],\n");

    fprintf(fp, "  \"streams\": [\n");
    for (int i = 0; i < nb_streams; i++)
    {
        const tracked_stream_t *s     = &streams[i];
        uint64_t                tot_f = s->last_cnt0 - s->first_cnt0;
        double                  dur   = s->last_t - s->first_t;
        if (dur < 0.1)
        {
            dur = 0.1;
        }
        double mean_hz = (s->cl_ticks > 0) ? (s->cl_hz_sum / s->cl_ticks) : ((double) tot_f / dur);
        double mean_mb = (s->cl_ticks > 0) ? (s->cl_mb_sum / s->cl_ticks)
                                           : (((double) tot_f * s->frame_bytes) / (dur * 1e6));
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": ");
        json_escape(fp, s->name);
        fprintf(fp, ",\n      \"dtype\": ");
        json_escape(fp, dtype_str(s->datatype));
        fprintf(fp, ",\n      \"dims\": [%u, %u, %u],\n", (unsigned) s->size[0],
                (unsigned) s->size[1], (unsigned) ((s->naxis >= 3) ? s->size[2] : 1));
        fprintf(fp, "      \"frame_bytes\": %" PRIu64 ",\n", s->frame_bytes);
        fprintf(fp, "      \"location\": %d,\n", s->location);
        fprintf(fp, "      \"total_frames\": %" PRIu64 ",\n", tot_f);
        fprintf(fp, "      \"mean_hz\": %.2f,\n", safe_d(mean_hz));
        fprintf(fp, "      \"peak_hz\": %.2f,\n", safe_d(s->peak_hz));
        fprintf(fp, "      \"mean_mb_s\": %.3f,\n", safe_d(mean_mb));
        fprintf(fp, "      \"peak_mb_s\": %.3f,\n", safe_d(s->peak_mb_s));
        fprintf(fp, "      \"nb_readers\": %d,\n", s->nb_readers);
        fprintf(fp, "      \"write_pid\": %d,\n", (int) s->write_pid);
        fprintf(fp, "      \"e2e_latency_us\": %.2f\n",
                (s->last_e2e_ns > 0) ? safe_d((double) s->last_e2e_ns / 1000.0) : -1.0);
        fprintf(fp, "    }%s\n", (i + 1 < nb_streams) ? "," : "");
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    fclose(fp);
}

/* =========================================================
 * Help & CLI entry point
 * ======================================================= */

static void print_help(const char *prog, int mh_color)
{
    milk_help_banner(prog, "headless compute, GPU, NUMA, PROCESSINFO, and SHM profiler", mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s]\n\n", mh_color ? MH_CMD : "", prog, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  Continuously samples host CPU/NUMA utilization, scaling frequencies, NVIDIA GPU\n"
           "  utilization/VRAM/power/clocks, PROCESSINFO timing rings, and ImageStreamIO SHM\n"
           "  bandwidth, writing phase-annotated JSON and CSV telemetry.\n\n");
    milk_help_section("Options", mh_color);
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-o, --output FILE", mh_color ? MH_RST : "",
           "Output JSON telemetry file (default: compute_telemetry.json)");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-c, --csv FILE", mh_color ? MH_RST : "",
           "Optional output CSV time-series file");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-p, --phases-file FILE",
           mh_color ? MH_RST : "", "Optional TSV phase markers (<epoch>\\t<id>\\t<label>)");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-s, --stop-file FILE",
           mh_color ? MH_RST : "", "Stop cleanly when sentinel file appears");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-r, --rate HZ", mh_color ? MH_RST : "",
           "Sampling rate in Hz (default: 5.0)");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-d, --duration SEC", mh_color ? MH_RST : "",
           "Max run duration in seconds (default: unlimited)");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "--snapshot", mh_color ? MH_RST : "",
           "Take a single quick sample (~0.25 s) and exit");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-24s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
}

int main(int argc, char *argv[])
{
    const char *oneline = "headless CPU, GPU, NUMA, PROCESSINFO, and SHM compute profiler";
    int action = milk_help_init(argc, argv, oneline,
                                "Continuously samples host CPU/NUMA utilization, NVIDIA GPU\n"
                                "metrics, PROCESSINFO timing percentiles, and SHM bandwidth.");
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    const char *out_json    = "compute_telemetry.json";
    const char *out_csv     = "";
    const char *phases_file = "";
    const char *stop_file   = "";
    double      rate_hz     = 5.0;
    double      duration_s  = 0.0;
    int         snapshot    = 0;

    static struct option long_opts[] = {
        { "output", required_argument, 0, 'o' },
        { "csv", required_argument, 0, 'c' },
        { "phases-file", required_argument, 0, 'p' },
        { "stop-file", required_argument, 0, 's' },
        { "rate", required_argument, 0, 'r' },
        { "duration", required_argument, 0, 'd' },
        { "snapshot", no_argument, 0, 'S' },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 },
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "o:c:p:s:r:d:Sh", long_opts, NULL)) != -1)
    {
        switch (opt)
        {
        case 'o':
            out_json = optarg;
            break;
        case 'c':
            out_csv = optarg;
            break;
        case 'p':
            phases_file = optarg;
            break;
        case 's':
            stop_file = optarg;
            break;
        case 'r':
            rate_hz = atof(optarg);
            break;
        case 'd':
            duration_s = atof(optarg);
            break;
        case 'S':
            snapshot = 1;
            break;
        default:
            print_help(argv[0], 1);
            return 1;
        }
    }
    if (rate_hz <= 0.1 || rate_hz > 50.0)
    {
        rate_hz = 5.0;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sig;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    ov_dirs_init();
    host_info_t      *host  = (host_info_t *) calloc(1, sizeof(host_info_t));
    OV_MODEL         *model = (OV_MODEL *) calloc(1, sizeof(OV_MODEL));
    tracked_proc_t   *procs = (tracked_proc_t *) calloc(SYSMON_MAX_PROCS, sizeof(tracked_proc_t));
    tracked_stream_t *streams =
        (tracked_stream_t *) calloc(SYSMON_MAX_STREAMS, sizeof(tracked_stream_t));
    phase_mark_t *phases = (phase_mark_t *) calloc(SYSMON_MAX_PHASES, sizeof(phase_mark_t));
    if (!host || !model || !procs || !streams || !phases)
    {
        fprintf(stderr, "milk-sysmon: out of memory\n");
        return 1;
    }

    collect_host_info(host);
    gpu_mon_t gmon;
    gpu_mon_start(&gmon, host->gpu_present);

    int            nb_procs   = 0;
    int            nb_streams = 0;
    int            nb_phases  = 0;
    int            tick_cap   = 2048;
    int            nb_ticks   = 0;
    tick_sample_t *ticks      = (tick_sample_t *) calloc((size_t) tick_cap, sizeof(tick_sample_t));
    if (!ticks)
    {
        fprintf(stderr, "milk-sysmon: out of memory\n");
        return 1;
    }

    cpu_sampler_t csamp;
    memset(&csamp, 0, sizeof(csamp));
    cpu_metrics_t cm_dummy;
    sample_cpu(&csamp, host, &cm_dummy);

    double clk_tck   = (double) sysconf(_SC_CLK_TCK);
    double t0_epoch  = now_epoch_s();
    double t0_mono   = now_mono_s();
    double prev_mono = t0_mono;
    double cl_dur_s  = 0.0;
    uid_t  my_uid    = getuid();

    usleep(snapshot ? 250000 : (useconds_t) (1000000.0 / rate_hz));

    while (!g_stop)
    {
        if (stop_file[0] != '\0' && access(stop_file, F_OK) == 0)
        {
            break;
        }
        double now_e   = now_epoch_s();
        double now_m   = now_mono_s();
        double t_rel   = now_m - t0_mono;
        double dt_tick = now_m - prev_mono;
        prev_mono      = now_m;

        nb_phases    = load_phases_file(phases_file, phases, SYSMON_MAX_PHASES, t0_epoch);
        int cur_pidx = resolve_phase_idx(phases, nb_phases, t_rel);
        int in_cl_phase =
            (cur_pidx >= 0 && strcmp(phases[cur_pidx].id, "closed_loop_warmup") != 0 &&
             strncmp(phases[cur_pidx].id, "closed_loop", 11) == 0);
        if (in_cl_phase && dt_tick > 0.0)
        {
            cl_dur_s += dt_tick;
        }

        cpu_metrics_t cm;
        sample_cpu(&csamp, host, &cm);
        gpu_sample_t gs;
        gpu_mon_poll(&gmon, &gs);

        memset(model, 0, sizeof(OV_MODEL));
        ov_model_full_scan(model);

        /* Register & update PROCESSINFO entries */
        int act_p = 0;
        for (int i = 0; i < model->nb_procs; i++)
        {
            const OV_PROC *op = &model->procs[i];
            if (op->PID <= 0)
            {
                continue;
            }
            tracked_proc_t *tp =
                find_or_add_proc(procs, &nb_procs, op->PID, op->name, "procinfo", t_rel);
            if (!tp)
            {
                continue;
            }
            tp->has_pinfo   = 1;
            tp->loopcnt     = op->loopcnt;
            tp->triggermode = op->triggermode;
            tp->rt_priority = op->rt_priority;
            if (op->trigstreamname[0] != '\0')
            {
                snprintf(tp->trigstream, sizeof(tp->trigstream), "%s", op->trigstreamname);
            }
            if (op->active)
            {
                tp->last_seen_t = t_rel;
                tp->seen_samples++;
                act_p++;
                if (in_cl_phase)
                {
                    if (!tp->cl_has_loop)
                    {
                        tp->cl_loopcnt_start = op->loopcnt;
                        tp->cl_has_loop      = 1;
                    }
                    tp->cl_loopcnt_end = op->loopcnt;
                }
            }
        }

        /* Also catch short-lived CLI processes (cacao-cli, milk-*, etc.) */
        scan_extra_cli_procs(my_uid, procs, &nb_procs, t_rel);

        double sim_sum = 0.0;
        for (int i = 0; i < nb_procs; i++)
        {
            if (procs[i].alive)
            {
                update_proc_os_stats(&procs[i], host, clk_tck, dt_tick, in_cl_phase);
                sim_sum += procs[i].last_tick_cores;
                if (procs[i].alive && strcmp(procs[i].kind, "cli") == 0)
                {
                    procs[i].last_seen_t = t_rel;
                    procs[i].seen_samples++;
                    act_p++;
                }
            }
        }
        refresh_pinfo_rings(host, procs, nb_procs, in_cl_phase);

        /* Attribute GPU VRAM to tracked PIDs */
        pthread_mutex_lock(&gmon.app_mtx);
        for (int a = 0; a < gmon.nb_apps; a++)
        {
            const gpu_app_t *ga = &gmon.apps[a];
            tracked_proc_t  *tp =
                find_or_add_proc(procs, &nb_procs, ga->pid, ga->comm, "cli", t_rel);
            if (tp)
            {
                double m = (double) ga->mem_mib;
                if (m > tp->gpu_vram_peak_mib)
                {
                    tp->gpu_vram_peak_mib = m;
                }
                tp->gpu_vram_sum_mib += m;
                tp->gpu_samples++;
            }
        }
        pthread_mutex_unlock(&gmon.app_mtx);

        /* Track ImageStreamIO SHM streams & bandwidth */
        double shm_bw    = 0.0;
        double shm_bytes = 0.0;
        int    act_s     = 0;
        for (int i = 0; i < model->nb_streams; i++)
        {
            const OV_STREAM *os = &model->streams[i];
            if (!os->valid || os->alias)
            {
                continue;
            }
            tracked_stream_t *st =
                find_or_add_stream(streams, &nb_streams, os, host->shmdir, t_rel);
            if (!st)
            {
                continue;
            }
            st->last_t     = t_rel;
            st->nb_readers = os->nb_read_pids;
            st->write_pid  = os->write_pid;
            shm_bytes += (double) st->frame_bytes;
            if (st->has_prev && dt_tick > 0.01 && os->cnt0 >= st->prev_cnt0)
            {
                uint64_t dc  = os->cnt0 - st->prev_cnt0;
                double   hz  = (double) dc / dt_tick;
                double   mbs = ((double) dc * (double) st->frame_bytes) / (dt_tick * 1.0e6);
                shm_bw += mbs;
                if (hz > st->peak_hz)
                {
                    st->peak_hz = hz;
                }
                if (mbs > st->peak_mb_s)
                {
                    st->peak_mb_s = mbs;
                }
                if (dc > 0)
                {
                    act_s++;
                }
                if (in_cl_phase)
                {
                    st->cl_hz_sum += hz;
                    st->cl_mb_sum += mbs;
                    st->cl_ticks++;
                }
            }
            st->prev_cnt0 = os->cnt0;
            st->last_cnt0 = os->cnt0;
            st->has_prev  = 1;
            int64_t e2e   = ov_trace_end_to_end_ns(model, os);
            if (e2e > 0)
            {
                st->last_e2e_ns = e2e;
            }
        }

        if (nb_ticks >= tick_cap)
        {
            tick_cap *= 2;
            tick_sample_t *nt =
                (tick_sample_t *) realloc(ticks, (size_t) tick_cap * sizeof(tick_sample_t));
            if (!nt)
            {
                break;
            }
            ticks = nt;
        }
        tick_sample_t *tk = &ticks[nb_ticks++];
        memset(tk, 0, sizeof(*tk));
        tk->epoch_s        = now_e;
        tk->rel_s          = t_rel;
        tk->phase_idx      = cur_pidx;
        tk->total_cores    = cm.total_cores;
        tk->sim_cores      = sim_sum;
        tk->user_cores     = cm.user_cores;
        tk->sys_cores      = cm.sys_cores;
        tk->iowait_cores   = cm.iowait_cores;
        tk->numa0_cores    = cm.numa_cores[0];
        tk->numa1_cores    = (host->nb_numa > 1) ? cm.numa_cores[1] : 0.0;
        tk->max_core_pct   = cm.max_core_pct;
        tk->cores_gt10     = cm.cores_gt10;
        tk->cores_gt50     = cm.cores_gt50;
        tk->freq_mean_mhz  = cm.freq_mean_mhz;
        tk->freq_max_mhz   = cm.freq_max_mhz;
        tk->gpu_util       = gs.gpu_util;
        tk->gpu_mem_util   = gs.mem_util;
        tk->gpu_vram_mib   = gs.vram_mib;
        tk->gpu_power_w    = gs.power_w;
        tk->gpu_sm_mhz     = gs.sm_mhz;
        tk->gpu_mem_mhz    = gs.mem_mhz;
        tk->gpu_temp_c     = gs.temp_c;
        tk->shm_bw_mb_s    = shm_bw;
        tk->shm_total_mb   = shm_bytes / (1024.0 * 1024.0);
        tk->active_procs   = act_p;
        tk->active_streams = act_s;

        if (snapshot || (duration_s > 0.0 && t_rel >= duration_s))
        {
            break;
        }
        usleep((useconds_t) (1000000.0 / rate_hz));
    }

    gpu_mon_stop(&gmon);

    double wall_total_s = now_mono_s() - t0_mono;
    nb_phases           = load_phases_file(phases_file, phases, SYSMON_MAX_PHASES, t0_epoch);
    for (int i = 0; i < nb_ticks; i++)
    {
        ticks[i].phase_idx = resolve_phase_idx(phases, nb_phases, ticks[i].rel_s);
    }

    write_csv(out_csv, ticks, nb_ticks, phases, nb_phases);
    write_json(out_json, host, wall_total_s, cl_dur_s, ticks, nb_ticks, phases, nb_phases, procs,
               nb_procs, streams, nb_streams);

    ov_scan_cache_cleanup();
    free(ticks);
    free(phases);
    free(streams);
    free(procs);
    free(model);
    free(host);
    return 0;
}
