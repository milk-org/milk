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
long pid_get_rss_kb(pid_t pid)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/statm", (int) pid);
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return 0;
    }
    long size, rss = 0;
    if (fscanf(fp, "%ld %ld", &size, &rss) != 2)
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
    pid_t          pid,
    unsigned long *utime,
    unsigned long *stime)
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
        "%*u %*lu %*lu %*lu %*lu %lu %lu",
        utime, stime);
    fclose(fp);

    return (rc == 2) ? 0 : -1;
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


