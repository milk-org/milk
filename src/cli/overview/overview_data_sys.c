#include "overview_data_internal.h"

/* =========================================================
 * Helpers
 * ========================================================= */

/**
 * pid_get_status - check PID liveness and zombie state.
 *
 * Uses a per-tick cache to avoid redundant syscalls
 * when many streams share the same PID.
 *
 * Returns OV_PID_DEAD, OV_PID_ALIVE, or OV_PID_ZOMBIE.
 */

#define PID_CACHE_MAX 512

static struct
{
    pid_t          pid;
    ov_pid_status_t status;
} s_pid_cache[PID_CACHE_MAX];

static int s_pid_cache_nb = 0;

/**
 * pid_cache_reset - clear PID cache.
 *
 * Must be called once at the start of each scan tick.
 */
void pid_cache_reset(void)
{
    s_pid_cache_nb = 0;
}

/**
 * pid_check_zombie - test if PID is zombie via /proc.
 *
 * Reads the third field of /proc/<pid>/stat. If it is
 * 'Z', the process is a zombie.
 */
int pid_check_zombie(pid_t pid)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/stat",
             (int) pid);
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return 0;
    }
    /* Skip PID and (comm), then read state */
    char state = '\0';
    if (fscanf(fp, "%*d %*s %c", &state) != 1)
    {
        state = '\0';
    }
    fclose(fp);
    return (state == 'Z');
}

/**
 * pid_get_rss_kb - Read RSS memory usage in KB.
 */
int64_t pid_get_rss_kb(pid_t pid)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/statm", (int) pid);
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return 0;
    }
    int64_t size, rss = 0;
    if (fscanf(fp, "%" SCNi64 " %" SCNi64, &size, &rss) != 2)
    {
        rss = 0;
    }
    fclose(fp);
    return (rss * sysconf(_SC_PAGESIZE)) / 1024;
}

ov_pid_status_t pid_get_status(pid_t pid)
{
    if (pid <= 0)
    {
        return OV_PID_DEAD;
    }

    /* Search cache */
    for (int i = 0; i < s_pid_cache_nb; i++)
    {
        if (s_pid_cache[i].pid == pid)
        {
            return s_pid_cache[i].status;
        }
    }

    /* Cache miss — do the syscall(s) */
    ov_pid_status_t st = OV_PID_DEAD;
    if (kill(pid, 0) == 0)
    {
        st = pid_check_zombie(pid)
             ? OV_PID_ZOMBIE : OV_PID_ALIVE;
    }

    if (s_pid_cache_nb < PID_CACHE_MAX)
    {
        s_pid_cache[s_pid_cache_nb].pid    = pid;
        s_pid_cache[s_pid_cache_nb].status = st;
        s_pid_cache_nb++;
    }

    return st;
}

/**
 * pid_is_alive - convenience wrapper (backward compat).
 */
int pid_is_alive(pid_t pid)
{
    return (pid_get_status(pid) != OV_PID_DEAD);
}

/**
 * pid_get_cpu_ticks - read cumulative CPU ticks.
 * @pid:    process ID
 * @utime:  [out] user-mode ticks
 * @stime:  [out] kernel-mode ticks
 *
 * Reads fields 14 (utime) and 15 (stime) from
 * /proc/[pid]/stat.
 *
 * Return: 0 on success, -1 on failure.
 */
int pid_get_cpu_ticks(
    pid_t    pid,
    uint64_t *utime,
    uint64_t *stime)
{
    char path[64];
    snprintf(path, sizeof(path),
             "/proc/%d/stat", (int) pid);
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return -1;
    }

    /* Skip to field 14 (utime) and 15 (stime).
     * Fields: pid (comm) state ppid pgrp session
     * tty_nr tpgid flags minflt cminflt majflt
     * cmajflt utime stime */
    int rc = fscanf(fp,
        "%*d %*s %*c %*d %*d %*d %*d %*d "
        "%*u %*u %*u %*u %*u %" SCNu64 " %" SCNu64,
        utime, stime);
    fclose(fp);

    return (rc == 2) ? 0 : -1;
}

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
    snprintf(path, sizeof(path), "/proc/%d/task", (int)pid);
    
    DIR *d = opendir(path);
    if (!d)
    {
        return 0;
    }
    
    struct dirent *dir;
    int count = 0;
    
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
                int processor = -1;
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
                            if (p) p++;
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
                    for (int i = 0; i < count; i++) {
                        if (cores[i] == processor) {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate) {
                        cores[count++] = processor;
                    }
                }
            }
        }
    }
    closedir(d);
    
    /* Sort the cores for a nicer display */
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (cores[j] < cores[i]) {
                int tmp = cores[i];
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
    if (pid <= 0 || out == NULL) return -1;
    memset(out, 0, sizeof(*out));
    
    char path[256];
    
    // 1. Read /proc/<pid>/stat for faults
    snprintf(path, sizeof(path), "/proc/%d/stat", (int)pid);
    FILE *f = fopen(path, "r");
    if (f) {
        char buf[1024];
        if (fgets(buf, sizeof(buf), f)) {
            char *p = strrchr(buf, ')');
            if (p) {
                p += 2; // skip ") "
                // p points to field 3 (state)
                // minflt is field 10, so it's 7 fields after state
                for (int i = 0; i < 7 && p; i++) {
                    p = strchr(p, ' ');
                    if (p) p++;
                }
                if (p) {
                    out->minflt = strtoul(p, &p, 10);
                    if (p) {
                        p++; // skip space to field 11
                        p = strchr(p, ' '); // skip field 11
                        if (p) {
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
    snprintf(path, sizeof(path), "/proc/%d/status", (int)pid);
    f = fopen(path, "r");
    if (f) {
        char buf[256];
        while (fgets(buf, sizeof(buf), f)) {
            if (strncmp(buf, "Threads:", 8) == 0) {
                out->threads = strtoul(buf + 8, NULL, 10);
            } else if (strncmp(buf, "voluntary_ctxt_switches:", 24) == 0) {
                out->vol_ctxt = strtoul(buf + 24, NULL, 10);
            } else if (strncmp(buf, "nonvoluntary_ctxt_switches:", 27) == 0) {
                out->nonvol_ctxt = strtoul(buf + 27, NULL, 10);
            }
        }
        fclose(f);
    }
    
    // 3. Read /proc/<pid>/sched for migrations
    snprintf(path, sizeof(path), "/proc/%d/sched", (int)pid);
    f = fopen(path, "r");
    if (f) {
        char buf[256];
        while (fgets(buf, sizeof(buf), f)) {
            if (strncmp(buf, "se.nr_migrations", 16) == 0) {
                char *p = strchr(buf, ':');
                if (p) out->migrations = strtoul(p + 1, NULL, 10);
                break;
            }
        }
        fclose(f);
    }
    
    return 0;
}

static int s_perf_pid = -1;
static int64_t s_perf_prev_loopcnt = 0;

static int s_perf_fd_inst = -1;
static int s_perf_fd_cache = -1;
static int s_perf_fd_branch = -1;
static int s_perf_fd_l1d = -1;
static int s_perf_fd_llc = -1;
static int s_perf_fd_dtlb = -1;

static uint64_t s_perf_prev_inst = 0;
static uint64_t s_perf_prev_cache = 0;
static uint64_t s_perf_prev_branch = 0;
static uint64_t s_perf_prev_l1d = 0;
static uint64_t s_perf_prev_llc = 0;
static uint64_t s_perf_prev_dtlb = 0;

static long _perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/**
 * @brief Open a perf_event file descriptor.
 *
 * Configures hardware performance counters for
 * the specified event type.
 */
static int open_perf_fd(pid_t pid, uint32_t type, uint64_t config) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = type;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 0;
    attr.exclude_hv = 1;
    attr.inherit = 1;
    int fd = (int)_perf_event_open(&attr, pid, -1, -1, 0);
    if (fd >= 0) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }
    return fd;
}

/**
 * pid_read_perf_counters - get hardware metrics via perf_event_open
 * Requires CAP_PERFMON or root, or perf_event_paranoid <= 2
 */
int pid_read_perf_counters(pid_t pid, int64_t loopcnt, ov_perf_counters_t *out)
{
    if (pid <= 0 || !out) return -1;
    
    memset(out, 0, sizeof(*out));
    
    if (s_perf_pid != pid) {
        if (s_perf_fd_inst >= 0) close(s_perf_fd_inst);
        if (s_perf_fd_cache >= 0) close(s_perf_fd_cache);
        if (s_perf_fd_branch >= 0) close(s_perf_fd_branch);
        if (s_perf_fd_l1d >= 0) close(s_perf_fd_l1d);
        if (s_perf_fd_llc >= 0) close(s_perf_fd_llc);
        if (s_perf_fd_dtlb >= 0) close(s_perf_fd_dtlb);
        
        s_perf_fd_inst = open_perf_fd(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        s_perf_fd_cache = open_perf_fd(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
        s_perf_fd_branch = open_perf_fd(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
        
        s_perf_fd_l1d = open_perf_fd(pid, PERF_TYPE_HW_CACHE, 
            (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        s_perf_fd_llc = open_perf_fd(pid, PERF_TYPE_HW_CACHE, 
            (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        s_perf_fd_dtlb = open_perf_fd(pid, PERF_TYPE_HW_CACHE, 
            (PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        
        s_perf_pid = pid;
        s_perf_prev_loopcnt = loopcnt;
        s_perf_prev_inst = 0;
        s_perf_prev_cache = 0;
        s_perf_prev_branch = 0;
        s_perf_prev_l1d = 0;
        s_perf_prev_llc = 0;
        s_perf_prev_dtlb = 0;
    }
    
    int ok = 0;
    int expected = 0;
    
    int64_t d_loops = loopcnt - s_perf_prev_loopcnt;
    if (d_loops <= 0) d_loops = 0;
    
    #define READ_FD(fd, outval, prevval, rateval) \
        do { \
            if ((fd) >= 0) { \
                expected++; \
                uint64_t val = 0; \
                if (read((fd), &val, sizeof(val)) == sizeof(val)) { \
                    (outval) = val; \
                    ok++; \
                    if (d_loops > 0 && val >= (prevval)) { \
                        (rateval) = (double)(val - (prevval)) / d_loops; \
                    } \
                    (prevval) = val; \
                } \
            } \
        } while(0)
    
    READ_FD(s_perf_fd_inst, out->instructions, s_perf_prev_inst, out->inst_per_loop);
    READ_FD(s_perf_fd_cache, out->cache_misses, s_perf_prev_cache, out->cache_miss_per_loop);
    READ_FD(s_perf_fd_branch, out->branch_misses, s_perf_prev_branch, out->branch_miss_per_loop);
    READ_FD(s_perf_fd_l1d, out->l1d_misses, s_perf_prev_l1d, out->l1d_miss_per_loop);
    READ_FD(s_perf_fd_llc, out->llc_misses, s_perf_prev_llc, out->llc_miss_per_loop);
    READ_FD(s_perf_fd_dtlb, out->dtlb_misses, s_perf_prev_dtlb, out->dtlb_miss_per_loop);
    #undef READ_FD

    if (d_loops > 0) {
        s_perf_prev_loopcnt = loopcnt;
    }
    
    return (expected > 0 && ok == expected) ? 0 : -1;
}

/**
 * ov_datatype_name - short name for data type code.
 */
const char *ov_datatype_name(uint8_t dt)
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
        return "???";
    }
}
/* suppress unused-function warning when header is
 * included but ov_datatype_name is only used by
 * the render code */
__attribute__((unused))
static const char *ov_datatype_name_ref =
    (const char *)(uintptr_t) ov_datatype_name;


