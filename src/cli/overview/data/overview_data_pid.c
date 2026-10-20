// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_pid.c
 * @brief   Process liveness and /proc/<pid>/stat values, read once per PID per scan.
 *
 * One read of /proc/<pid>/stat gives state, CPU ticks, start time and RSS. The table is reset at the
 * start of each scan (pid_cache_reset). A PID found dead is not probed again until the kernel may have
 * reused it, i.e. its PID counter (ns_last_pid) wrapped, or for OV_PID_DEAD_RECHECK_S seconds when that
 * counter cannot be read: dead stream owners and stale registry entries would otherwise cost a syscall each,
 * every scan. Dead PIDs not looked up for OV_PID_DEAD_FORGET_S seconds are dropped.
 * pid_same_process() and pid_check_zombie() always read afresh: they guard actions.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "overview_data_internal.h"
#include "overview_data_index.h"

#define OV_PID_TABLE_MAX 4096
#define OV_PID_DEAD_MAX 32768
#define OV_PID_DEAD_RECHECK_S 10.0
#define OV_PID_DEAD_FORGET_S 60.0

typedef struct
{
    pid_t           pid;
    ov_pid_status_t status;
    int             has_stat; /* the fields below were read */
    uint64_t        utime, stime, start;
    int64_t         rss_pages;
} pid_entry_t;

static pid_entry_t s_pid[OV_PID_TABLE_MAX];
static int         s_pid_nb;
static OV_INDEX_DEF(8192) s_pid_ix;

static pid_t  s_dead[OV_PID_DEAD_MAX];      /* PIDs found dead */
static double s_dead_born[OV_PID_DEAD_MAX]; /* when found dead */
static double s_dead_t[OV_PID_DEAD_MAX];    /* when last looked up */
static long   s_dead_lp[OV_PID_DEAD_MAX];   /* ns_last_pid when found dead, -1 = unknown */
static int    s_dead_nb;
static double s_scan_t;        /* start of the current scan */
static long   s_last_pid = -1; /* ns_last_pid at the start of the current scan */
static OV_INDEX_DEF(65536) s_dead_ix;

static pthread_mutex_t s_pid_mutex = PTHREAD_MUTEX_INITIALIZER;

/** now_s - monotonic seconds. */
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

/** pid_match - table entry @i is the PID in @ctx. */
static int pid_match(const void *ctx, int32_t i)
{
    return s_pid[i].pid == *(const pid_t *) ctx;
}

/** dead_match - dead-list entry @i is the PID in @ctx. */
static int dead_match(const void *ctx, int32_t i)
{
    return s_dead[i] == *(const pid_t *) ctx;
}

/** read_last_pid - the last PID the kernel allocated in our namespace, -1 if unknown. */
static long read_last_pid(void)
{
    char buf[32];
    int  fd = open("/proc/sys/kernel/ns_last_pid", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        return -1;
    }
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    buf[(n > 0) ? n : 0] = '\0';
    return (n > 0) ? strtol(buf, NULL, 10) : -1;
}

/**
 * pid_cache_reset - start a scan: forget this scan's PID reads; drop dead PIDs the kernel may have reused
 * (its PID counter wrapped) or that were not looked up recently.
 */
void pid_cache_reset(void)
{
    double now = now_s();
    long   lp  = read_last_pid();
    pthread_mutex_lock(&s_pid_mutex);
    s_scan_t   = now;
    s_last_pid = lp;
    s_pid_nb   = 0;
    ov_ix_clear(&s_pid_ix);
    int n = 0;
    for (int i = 0; i < s_dead_nb; i++)
    {
        int reusable = (lp >= 0 && s_dead_lp[i] >= 0)
                           ? (lp < s_dead_lp[i])
                           : (now - s_dead_born[i] >= OV_PID_DEAD_RECHECK_S);
        if (!reusable && now - s_dead_t[i] < OV_PID_DEAD_FORGET_S)
        {
            s_dead[n]      = s_dead[i];
            s_dead_born[n] = s_dead_born[i];
            s_dead_t[n]    = s_dead_t[i];
            s_dead_lp[n]   = s_dead_lp[i];
            n++;
        }
    }
    s_dead_nb = n;
    ov_ix_clear(&s_dead_ix);
    for (int i = 0; i < s_dead_nb; i++)
    {
        ov_ix_put(&s_dead_ix, ov_hash_u64((uint64_t) s_dead[i]), i);
    }
    pthread_mutex_unlock(&s_pid_mutex);
}

/**
 * pid_stat_fields - read /proc/<pid>/stat into @buf and return its field 3 (state), found after the
 * last ')' because the comm field may contain spaces and parentheses. NULL on failure.
 */
const char *pid_stat_fields(pid_t pid, char *buf, size_t sz)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/stat", (int) pid);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        return NULL;
    }
    ssize_t n = read(fd, buf, sz - 1);
    close(fd);
    if (n <= 0)
    {
        return NULL;
    }
    buf[n]  = '\0';
    char *p = strrchr(buf, ')');
    return (p != NULL && p[1] == ' ') ? p + 2 : NULL;
}

/**
 * pid_stat_skip - advance @p, pointing at a /proc stat field, by @n fields; NULL if too short.
 */
const char *pid_stat_skip(const char *p, int n)
{
    for (int i = 0; i < n && p != NULL; i++)
    {
        p = strchr(p, ' ');
        if (p != NULL)
        {
            p++;
        }
    }
    return p;
}

/** proc_hidden - 1 when /proc hides other users' processes (hidepid): init's stat file is not readable. */
static int proc_hidden(void)
{
    static int hidden = -1;
    if (hidden < 0)
    {
        hidden = (access("/proc/1/stat", R_OK) != 0);
    }
    return hidden;
}

/** pid_probe - one /proc read of @pid; with no stat file it is gone, unless /proc hides it (kill() decides). */
static void pid_probe(pid_t pid, pid_entry_t *e)
{
    char        buf[1024];
    const char *f3 = pid_stat_fields(pid, buf, sizeof(buf));
    memset(e, 0, sizeof(*e));
    e->pid = pid;
    if (f3 == NULL)
    {
        e->status =
            (proc_hidden() && (kill(pid, 0) == 0 || errno == EPERM)) ? OV_PID_ALIVE : OV_PID_DEAD;
        return;
    }
    e->status       = (f3[0] == 'Z') ? OV_PID_ZOMBIE : OV_PID_ALIVE;
    const char *f14 = pid_stat_skip(f3, 11);
    const char *f22 = pid_stat_skip(f14, 8);
    const char *f24 = pid_stat_skip(f22, 2);
    e->has_stat  = (f24 != NULL && sscanf(f14, "%" SCNu64 " %" SCNu64, &e->utime, &e->stime) == 2);
    e->start     = e->has_stat ? strtoull(f22, NULL, 10) : 0;
    e->rss_pages = e->has_stat ? strtoll(f24, NULL, 10) : 0;
}

/** pid_lookup - this scan's values for @pid into @out, reading /proc once per scan. */
static void pid_lookup(pid_t pid, pid_entry_t *out)
{
    uint64_t h = ov_hash_u64((uint64_t) pid);
    memset(out, 0, sizeof(*out));
    out->pid = pid;
    if (pid <= 0)
    {
        return;
    }
    pthread_mutex_lock(&s_pid_mutex);
    int i = ov_ix_find(&s_pid_ix, h, pid_match, &pid);
    int d = (i < 0) ? ov_ix_find(&s_dead_ix, h, dead_match, &pid) : -1;
    if (d >= 0)
    {
        s_dead_t[d] = s_scan_t;
        pthread_mutex_unlock(&s_pid_mutex);
        return; /* found dead and not reusable since */
    }
    if (i >= 0)
    {
        *out = s_pid[i];
        pthread_mutex_unlock(&s_pid_mutex);
        return;
    }
    pthread_mutex_unlock(&s_pid_mutex);

    pid_probe(pid, out);

    pthread_mutex_lock(&s_pid_mutex);
    if (ov_ix_find(&s_pid_ix, h, pid_match, &pid) < 0 && s_pid_nb < OV_PID_TABLE_MAX)
    {
        s_pid[s_pid_nb] = *out;
        ov_ix_put(&s_pid_ix, h, s_pid_nb++);
    }
    if (out->status == OV_PID_DEAD && s_dead_nb < OV_PID_DEAD_MAX)
    {
        s_dead[s_dead_nb]      = pid;
        s_dead_born[s_dead_nb] = s_scan_t;
        s_dead_t[s_dead_nb]    = s_scan_t;
        s_dead_lp[s_dead_nb]   = s_last_pid;
        ov_ix_put(&s_dead_ix, h, s_dead_nb++);
    }
    pthread_mutex_unlock(&s_pid_mutex);
}

/**
 * pid_get_status - Get process lifecycle status (alive, zombie, dead)
 * @pid: Process ID to query
 *
 * Return: Status enum (OV_PID_DEAD, OV_PID_ALIVE, OV_PID_ZOMBIE).
 */
ov_pid_status_t pid_get_status(pid_t pid)
{
    pid_entry_t e;
    pid_lookup(pid, &e);
    return e.status;
}

/**
 * pid_is_alive - convenience wrapper (backward compat).
 */
int pid_is_alive(pid_t pid)
{
    return (pid_get_status(pid) != OV_PID_DEAD);
}

/**
 * pid_get_rss_kb - Read RSS memory usage in KB (0 for a dead PID).
 */
int64_t pid_get_rss_kb(pid_t pid)
{
    static long page_kb;
    pid_entry_t e;
    if (page_kb == 0)
    {
        page_kb = sysconf(_SC_PAGESIZE) / 1024;
    }
    pid_lookup(pid, &e);
    return (e.status != OV_PID_DEAD && e.has_stat) ? e.rss_pages * page_kb : 0;
}

/**
 * pid_get_cpu_ticks - cumulative user and kernel CPU ticks of @pid (stat fields 14, 15).
 * Return: 0 on success, -1 on failure.
 */
int pid_get_cpu_ticks(pid_t pid, uint64_t *utime, uint64_t *stime)
{
    pid_entry_t e;
    pid_lookup(pid, &e);
    if (e.status == OV_PID_DEAD || !e.has_stat)
    {
        return -1;
    }
    *utime = e.utime;
    *stime = e.stime;
    return 0;
}

/**
 * pid_get_start_ticks - start time of @pid in clock ticks since boot (/proc stat field 22), or 0.
 */
uint64_t pid_get_start_ticks(pid_t pid)
{
    pid_entry_t e;
    pid_lookup(pid, &e);
    return (e.status != OV_PID_DEAD) ? e.start : 0;
}

/**
 * pid_same_process - 1 if @pid exists and is still the process that started at @ticks (read afresh).
 */
int pid_same_process(pid_t pid, uint64_t ticks)
{
    pid_entry_t e;
    if (pid <= 0 || ticks == 0)
    {
        return 0;
    }
    pid_probe(pid, &e);
    return e.status != OV_PID_DEAD && e.start == ticks;
}

/**
 * pid_check_zombie - 1 if /proc/<pid>/stat reports state 'Z' (read afresh).
 */
int pid_check_zombie(pid_t pid)
{
    char        buf[1024];
    const char *f3 = pid_stat_fields(pid, buf, sizeof(buf));
    return (f3 != NULL && f3[0] == 'Z');
}
