// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_syscalls_linux.c
 * @brief Linux implementation of milk_syscalls.h: /proc, OFD locks, pidfd, _Fork, close_range, statfs
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include "milk_syscalls.h"

#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <linux/magic.h>
#include <malloc.h>
#include <poll.h>
#include <pwd.h>
#include <sched.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#include <unistd.h>

int milkos_self_exe(char *buf, size_t size)
{
    if (buf == NULL || size < 2)
    {
        errno = EINVAL;
        return -1;
    }
    ssize_t len = readlink("/proc/self/exe", buf, size - 1);
    if (len < 0)
    {
        return -1;
    }
    if ((size_t) len == size - 1)
    {
        errno = ENAMETOOLONG; // readlink() truncates silently
        return -1;
    }
    buf[len] = '\0';
    return 0;
}

int milkos_pipe_cloexec(int fds[2])
{
    return pipe2(fds, O_CLOEXEC);
}

void milkos_close_from(int lowfd)
{
#ifdef SYS_close_range
    if (syscall(SYS_close_range, (unsigned int) lowfd, ~0U, 0) == 0)
    {
        return;
    }
#endif
    // kernel before 5.9; close() is async-signal-safe, opendir() and sysconf() are not
    for (int fd = lowfd; fd < 65536; fd++)
    {
        close(fd);
    }
}

pid_t milkos_fork_child(void)
{
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
    return _Fork();
#else
    return fork();
#endif
}

static int ofd_lock(int fd, int cmd, short type)
{
    struct flock fl;
    int          rc;

    memset(&fl, 0, sizeof(fl)); // whole file; l_pid must be 0 for OFD locks
    fl.l_type   = type;
    fl.l_whence = SEEK_SET;
    do
    {
        rc = fcntl(fd, cmd, &fl);
    } while (rc == -1 && errno == EINTR && cmd == F_OFD_SETLKW);
    if (rc == -1 && errno == EACCES)
    {
        errno = EAGAIN; // fcntl() reports a conflicting lock with either
    }
    return rc;
}

int milkos_lock_file(int fd, int wait)
{
    return ofd_lock(fd, wait ? F_OFD_SETLKW : F_OFD_SETLK, F_WRLCK);
}

int milkos_unlock_file(int fd)
{
    return ofd_lock(fd, F_OFD_SETLK, F_UNLCK);
}

int milkos_proc_start_ticks(pid_t pid, uint64_t *ticks)
{
    char path[64];
    char buf[1024];

    snprintf(path, sizeof(path), "/proc/%d/stat", (int) pid);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        if (errno == ENOENT)
        {
            errno = ESRCH;
        }
        return -1;
    }
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0)
    {
        errno = (n == 0) ? ESRCH : errno;
        return -1;
    }
    buf[n] = '\0';

    // the command name (field 2) may hold spaces and ')': count fields after its last ')'
    char *p = strrchr(buf, ')');
    if (p != NULL && p[1] == ' ' && (p[2] == 'Z' || p[2] == 'X' || p[2] == 'x'))
    {
        errno = ESRCH; // exited, not reaped yet
        return -1;
    }
    for (int field = 3; field <= 22 && p != NULL; field++)
    {
        p = strchr(p + 1, ' ');
    }
    if (p == NULL)
    {
        errno = EIO;
        return -1;
    }
    *ticks = strtoull(p + 1, NULL, 10);
    return 0;
}

int milkos_boot_id(char *buf, size_t size)
{
    if (buf == NULL || size < 37)
    {
        errno = EINVAL;
        return -1;
    }
    int fd = open("/proc/sys/kernel/random/boot_id", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        return -1;
    }
    ssize_t n = read(fd, buf, 36);
    close(fd);
    if (n != 36)
    {
        errno = EIO;
        return -1;
    }
    buf[36] = '\0';
    return 0;
}

int milkos_pidfd_open(pid_t pid)
{
#ifdef SYS_pidfd_open
    return (int) syscall(SYS_pidfd_open, pid, 0); // close-on-exec is always set
#else
    (void) pid;
    errno = ENOSYS;
    return -1;
#endif
}

int milkos_pidfd_signal(int pidfd, int sig)
{
#ifdef SYS_pidfd_send_signal
    return (int) syscall(SYS_pidfd_send_signal, pidfd, sig, NULL, 0);
#else
    (void) pidfd;
    (void) sig;
    errno = ENOSYS;
    return -1;
#endif
}

int milkos_pidfd_wait_exit(int pidfd, int timeout_ms)
{
    struct pollfd pfd = { .fd = pidfd, .events = POLLIN, .revents = 0 };
    int           rc;

    do
    {
        rc = poll(&pfd, 1, timeout_ms); // a pidfd becomes readable when its process exits
    } while (rc == -1 && errno == EINTR);
    return (rc > 0) ? 1 : rc;
}

int milkos_on_tmpfs(const char *path)
{
    struct statfs st;

    if (statfs(path, &st) != 0)
    {
        return -1;
    }
    uint32_t type = (uint32_t) st.f_type;
    return (type == TMPFS_MAGIC || type == RAMFS_MAGIC) ? 1 : 0;
}

int milkos_service_unit(char *unit, size_t size, int *user)
{
    char  line[1024];
    char *path = NULL;
    FILE *f    = fopen("/proc/self/cgroup", "re");

    if (f == NULL)
    {
        return -1;
    }
    // cgroup v2 ("0::/path"), else the systemd hierarchy of v1 ("N:name=systemd:/path")
    while (fgets(line, sizeof(line), f) != NULL)
    {
        line[strcspn(line, "\n")] = '\0';
        if (strncmp(line, "0::", 3) == 0)
        {
            path = line + 3;
            break;
        }
        char *p = strstr(line, ":name=systemd:");
        if (p != NULL)
        {
            path = p + 14;
            break;
        }
    }
    fclose(f);
    // the innermost service: a unit may make cgroups of its own below its service (Delegate=)
    for (char *slash = (path != NULL) ? strrchr(path, '/') : NULL; slash != NULL;
         slash       = strrchr(path, '/'))
    {
        size_t len = strlen(slash + 1);
        if (len > 8 && strcmp(slash + 1 + len - 8, ".service") == 0 &&
            strncmp(slash + 1, "user@", 5) != 0)
        {
            snprintf(unit, size, "%s", slash + 1);
            *slash = '\0';
            *user  = (strstr(path, "/user@") != NULL);
            return 0;
        }
        *slash = '\0';
    }
    return -1;
}

int milkos_user_ids(uid_t uid, gid_t *gid, gid_t *groups, int max)
{
    struct passwd  pw;
    struct passwd *found = NULL;
    char           buf[4096];
    int            n = max;

    if (getpwuid_r(uid, &pw, buf, sizeof(buf), &found) != 0 || found == NULL)
    {
        errno = ENOENT;
        return -1;
    }
    if (getgrouplist(pw.pw_name, pw.pw_gid, groups, &n) == -1)
    {
        errno = ERANGE;
        return -1;
    }
    *gid = pw.pw_gid;
    return n;
}

int milkos_set_ids(uid_t uid, gid_t gid, const gid_t *groups, int ngroups)
{
    if (setgroups((size_t) ngroups, groups) != 0 || setgid(gid) != 0 || setuid(uid) != 0)
    {
        return -1;
    }
    if (uid != 0 && setuid(0) == 0)
    {
        errno = EPERM;
        return -1;
    }
    return 0;
}

int milkos_malloc_keep(void)
{
#if defined(__GLIBC__)
    const char *tun = getenv("GLIBC_TUNABLES");
    if (tun != NULL && strstr(tun, "glibc.malloc.") != NULL)
    {
        return 1;
    }
    /* fixed thresholds: glibc no longer raises them as blocks are freed, which made the cost of a
     * repeated allocation depend on the heap's history (trim, then fault the pages in again) */
    if (mallopt(M_MMAP_THRESHOLD, 32 << 20) != 1 || mallopt(M_TRIM_THRESHOLD, 256 << 20) != 1)
    {
        return -1;
    }
    return 0;
#else
    return -1;
#endif
}

/**
 * parse_cpulist - Parse a Linux sysfs CPU list file (e.g. "0-15\n") into @out
 * @path: Sysfs file path
 * @out:  Output CPU affinity set
 *
 * Return: 0 when at least one CPU was parsed, -1 otherwise
 */
static int parse_cpulist(const char *path, cpu_set_t *out)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return -1;
    }
    char buf[256];
    if (fgets(buf, sizeof(buf), fp) == NULL)
    {
        fclose(fp);
        return -1;
    }
    fclose(fp);

    CPU_ZERO(out);
    const char *p = buf;
    while (*p != '\0' && *p != '\n')
    {
        char *end = NULL;
        long  lo  = strtol(p, &end, 10);
        if (end == p || lo < 0 || lo >= CPU_SETSIZE)
        {
            break;
        }
        long hi = lo;
        p       = end;
        if (*p == '-')
        {
            p++;
            hi = strtol(p, &end, 10);
            if (end == p || hi < lo)
            {
                break;
            }
            if (hi >= CPU_SETSIZE)
            {
                hi = CPU_SETSIZE - 1;
            }
            p = end;
        }
        for (long c = lo; c <= hi; c++)
        {
            CPU_SET((int) c, out);
        }
        if (*p == ',')
        {
            p++;
        }
    }
    return (CPU_COUNT(out) > 0) ? 0 : -1;
}

/**
 * read_sysfs_int - Read a single decimal integer from @path, or @defval on failure
 * @path:   Sysfs file path
 * @defval: Fallback integer value
 *
 * Return: Parsed integer or @defval
 */
static int read_sysfs_int(const char *path, int defval)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return defval;
    }
    int val = defval;
    if (fscanf(fp, "%d", &val) != 1)
    {
        val = defval;
    }
    fclose(fp);
    return val;
}

/**
 * milkos_pcore_count - Count the distinct physical P-cores in this process's CPU affinity
 *
 * Intersects `sched_getaffinity()` with `/sys/devices/cpu_core/cpus` when present (so hybrid
 * E-cores are not counted) and counts distinct `(physical_package_id, core_id)` pairs. The
 * process affinity mask is only read, never changed.
 *
 * Return: Number of physical P-cores available ($\ge 1$)
 */
int milkos_pcore_count(void)
{
    cpu_set_t aff;
    CPU_ZERO(&aff);
    if (sched_getaffinity(0, sizeof(aff), &aff) != 0 || CPU_COUNT(&aff) <= 0)
    {
        return 1;
    }

    cpu_set_t cand = aff;
    cpu_set_t pcores;
    if (parse_cpulist("/sys/devices/cpu_core/cpus", &pcores) == 0)
    {
        cpu_set_t inter;
        CPU_AND(&inter, &aff, &pcores);
        int n_inter = CPU_COUNT(&inter);
        if (n_inter > 0)
        {
            cand = inter;
        }
    }

    struct
    {
        int pkg;
        int core;
    } seen[256];
    int nseen = 0;

    for (int cpu = 0; cpu < CPU_SETSIZE && nseen < 256; cpu++)
    {
        if (!CPU_ISSET(cpu, &cand))
        {
            continue;
        }
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        int core = read_sysfs_int(path, cpu);
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",
                 cpu);
        int pkg = read_sysfs_int(path, 0);

        int dup = 0;
        for (int i = 0; i < nseen; i++)
        {
            if (seen[i].pkg == pkg && seen[i].core == core)
            {
                dup = 1;
                break;
            }
        }
        if (!dup)
        {
            seen[nseen].pkg  = pkg;
            seen[nseen].core = core;
            nseen++;
        }
    }

    return (nseen > 0) ? nseen : CPU_COUNT(&cand);
}

int milkos_spawn_closefrom(posix_spawn_file_actions_t *fa, int lowfd)
{
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
    return posix_spawn_file_actions_addclosefrom_np(fa, lowfd);
#else
    (void) fa;
    (void) lowfd;
    return ENOSYS;
#endif
}
