// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_detach.c
 * @brief Detached processes (milk_detach.h); the Linux-only calls come from milk_os
 */

#include "milk_detach.h"
#include "milk_os.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static long detach_now_ms(void)
{
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long) t.tv_sec * 1000L + t.tv_nsec / 1000000L;
}

int milkdetach_id_of(pid_t pid, MILKDETACH_ID *id)
{
    memset(id, 0, sizeof(*id));
    id->pid = pid;
    if (milkos_proc_start_ticks(pid, &id->start_ticks) != 0)
    {
        return -1;
    }
    if (milkos_boot_id(id->boot_id, sizeof(id->boot_id)) != 0)
    {
        id->boot_id[0] = '\0';
    }
    return 0;
}

int milkdetach_id_alive(const MILKDETACH_ID *id)
{
    char     boot[37];
    uint64_t ticks;

    if (id->pid <= 0)
    {
        return 0;
    }
    if (id->boot_id[0] != '\0' && milkos_boot_id(boot, sizeof(boot)) == 0 &&
        strcmp(boot, id->boot_id) != 0)
    {
        return 0; // recorded before a reboot
    }
    return milkos_proc_start_ticks(id->pid, &ticks) == 0 && ticks == id->start_ticks;
}

/* The grandchild below calls async-signal-safe functions only: its parent may be multithreaded. */

static size_t detach_utoa(char *buf, unsigned long v)
{
    char   tmp[24];
    size_t n   = 0;
    size_t len = 0;

    do
    {
        tmp[n++] = (char) ('0' + v % 10);
        v /= 10;
    } while (v != 0);
    while (n > 0)
    {
        buf[len++] = tmp[--n];
    }
    return len;
}

static size_t detach_cat(char *buf, size_t n, size_t size, const char *s)
{
    size_t len = strlen(s);

    if (n + len >= size)
    {
        len = size - n - 1;
    }
    memcpy(buf + n, s, len);
    return n + len;
}

static void detach_write_pid(int fd)
{
    char   line[32];
    size_t n = detach_cat(line, 0, sizeof(line), "MAINPID=");

    n += detach_utoa(line + n, (unsigned long) getpid());
    line[n++] = '\n';
    (void) !write(fd, line, n);
}

static void detach_fail(int fd, int err, const char *what)
{
    char   line[160];
    size_t n = detach_cat(line, 0, sizeof(line), "ERRNO=");

    n += detach_utoa(line + n, (unsigned long) err);
    n         = detach_cat(line, n, sizeof(line), "\nSTATUS=");
    n         = detach_cat(line, n, sizeof(line) - 1, what);
    line[n++] = '\n';
    (void) !write(fd, line, n);
    _exit(127);
}

static void detach_exec(const MILKDETACH_SPEC *s, int wr)
{
    struct sigaction dfl;
    sigset_t         none;
    int              nfd;
    int              ofd = -1;
    int              pfd = -1;
    int              nul;

    // above 9 first, so that placing 0-3 cannot overwrite them
    if ((nfd = fcntl(wr, F_DUPFD, 10)) < 0)
    {
        detach_fail(wr, errno, "dup");
    }
    if (s->out_fd >= 0 && (ofd = fcntl(s->out_fd, F_DUPFD, 10)) < 0)
    {
        detach_fail(nfd, errno, "dup");
    }
    if (s->pass_fd > 2 && (pfd = fcntl(s->pass_fd, F_DUPFD, 10)) < 0)
    {
        detach_fail(nfd, errno, "dup");
    }
    if ((nul = open("/dev/null", O_RDWR)) < 0)
    {
        detach_fail(nfd, errno, "open /dev/null");
    }
    if (dup2(nul, STDIN_FILENO) < 0 || dup2(ofd >= 0 ? ofd : nul, STDOUT_FILENO) < 0 ||
        dup2(ofd >= 0 ? ofd : nul, STDERR_FILENO) < 0 || dup2(nfd, MILKDETACH_NOTIFY_FD) < 0 ||
        (pfd >= 0 && dup2(pfd, MILKDETACH_PASS_FD) < 0))
    {
        detach_fail(nfd, errno, "dup2");
    }
    milkos_close_from((pfd >= 0) ? MILKDETACH_PASS_FD + 1 : MILKDETACH_NOTIFY_FD + 1);
    setpgid(0, 0);

    // ignored or blocked signals survive exec: a caller's must not stop the process's SIGINT
    memset(&dfl, 0, sizeof(dfl));
    dfl.sa_handler = SIG_DFL;
    sigemptyset(&dfl.sa_mask);
    for (int sig = 1; sig < 65; sig++)
    {
        sigaction(sig, &dfl, NULL); // EINVAL for SIGKILL, SIGSTOP and reserved numbers
    }
    sigemptyset(&none);
    sigprocmask(SIG_SETMASK, &none, NULL);

    if (s->drop && milkos_set_ids(s->uid, s->gid, s->groups, s->ngroups) != 0)
    {
        detach_fail(MILKDETACH_NOTIFY_FD, errno, "dropping privileges");
    }
    if (s->workdir != NULL && chdir(s->workdir) != 0)
    {
        detach_fail(MILKDETACH_NOTIFY_FD, errno, "chdir to the work directory");
    }
    detach_write_pid(MILKDETACH_NOTIFY_FD);
    if (s->no_notify)
    {
        fcntl(MILKDETACH_NOTIFY_FD, F_SETFD, FD_CLOEXEC);
    }
    execve(s->path, s->argv, s->envp);
    detach_fail(MILKDETACH_NOTIFY_FD, errno, "execve");
}

int milkdetach_start(const MILKDETACH_SPEC *spec)
{
    int   pfd[2];
    int   st = 0;
    pid_t child;

    if (milkos_pipe_cloexec(pfd) != 0)
    {
        return -1;
    }
    child = milkos_fork_child();
    if (child < 0)
    {
        int err = errno;
        close(pfd[0]);
        close(pfd[1]);
        errno = err;
        return -1;
    }
    if (child == 0)
    {
        pid_t grandchild;

        if (setsid() < 0)
        {
            _exit(127);
        }
        grandchild = milkos_fork_child();
        if (grandchild == 0)
        {
            detach_exec(spec, pfd[1]);
        }
        _exit(grandchild < 0 ? 127 : 0);
    }
    close(pfd[1]);
    while (waitpid(child, &st, 0) < 0)
    {
        if (errno != EINTR)
        {
            st = 0; // ECHILD: the caller ignores SIGCHLD or reaped it; the pipe tells
            break;
        }
    }
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
    {
        close(pfd[0]);
        errno = EAGAIN;
        return -1;
    }
    return pfd[0];
}

/* Return 1 on READY=1 */
static int detach_line(const char *l, MILKDETACH_READY *r)
{
    if (strncmp(l, "MAINPID=", 8) == 0)
    {
        r->pid = (pid_t) atol(l + 8);
    }
    else if (strncmp(l, "STATUS=", 7) == 0)
    {
        snprintf(r->status, sizeof(r->status), "%s", l + 7);
    }
    else if (strncmp(l, "ERRNO=", 6) == 0)
    {
        r->err = atoi(l + 6);
    }
    return strcmp(l, "READY=1") == 0;
}

int milkdetach_wait_ready(int fd, int timeout_ms, MILKDETACH_READY *r)
{
    char   buf[1024];
    size_t len = 0;
    long   end = detach_now_ms() + timeout_ms;

    r->pid       = -1;
    r->err       = 0;
    r->status[0] = '\0';
    for (;;)
    {
        struct pollfd p    = { .fd = fd, .events = POLLIN, .revents = 0 };
        int           wait = -1;

        if (timeout_ms >= 0)
        {
            long left = end - detach_now_ms();
            wait      = (left > 0) ? (int) left : 0;
        }
        int rc = poll(&p, 1, wait);
        if (rc < 0 && errno == EINTR)
        {
            continue;
        }
        if (rc == 0)
        {
            return MILKDETACH_TIMEOUT;
        }
        ssize_t n = (rc < 0) ? -1 : read(fd, buf + len, sizeof(buf) - 1 - len);
        if (n < 0 && errno == EINTR)
        {
            continue;
        }
        if (n <= 0)
        {
            return (r->err != 0) ? MILKDETACH_FAILED : MILKDETACH_DIED;
        }
        len += (size_t) n;
        buf[len] = '\0';

        char *line = buf;
        char *nl;
        while ((nl = strchr(line, '\n')) != NULL)
        {
            *nl = '\0';
            if (detach_line(line, r))
            {
                return (r->err != 0) ? MILKDETACH_FAILED : MILKDETACH_OK;
            }
            line = nl + 1;
        }
        len = (size_t) (buf + len - line);
        memmove(buf, line, len);
        if (len == sizeof(buf) - 1)
        {
            len = 0; // a line longer than the buffer: dropped
        }
    }
}

int milkdetach_notify_fd(const char *envname)
{
    const char *v = getenv(envname);
    struct stat st;
    int         fd;

    if (v == NULL || v[0] == '\0')
    {
        return -1;
    }
    fd = atoi(v);
    unsetenv(envname); // programs this process runs are not detached processes
    if (fd < 0 || fstat(fd, &st) != 0 || !S_ISFIFO(st.st_mode) ||
        fcntl(fd, F_SETFD, FD_CLOEXEC) == -1)
    {
        return -1;
    }
    return fd;
}

int milkdetach_notify(int fd, const char *msg)
{
    sigset_t pipe_set;
    sigset_t old;
    size_t   len = strlen(msg);
    size_t   off = 0;
    int      rc  = 0;

    if (fd < 0)
    {
        return -1;
    }
    // a starter that gave up has closed the pipe: EPIPE, not a SIGPIPE that ends the process
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &pipe_set, &old);
    while (off < len)
    {
        ssize_t n = write(fd, msg + off, len - off);
        if (n < 0 && errno == EINTR)
        {
            continue;
        }
        if (n < 0)
        {
            int             err  = errno;
            struct timespec zero = { 0, 0 };
            if (err == EPIPE && !sigismember(&old, SIGPIPE))
            {
                sigtimedwait(&pipe_set, NULL, &zero);
            }
            errno = err;
            rc    = -1;
            break;
        }
        off += (size_t) n;
    }
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    return rc;
}

static int detach_wait_gone(const MILKDETACH_ID *id, int pfd, int ms)
{
    if (pfd >= 0)
    {
        int rc = milkos_pidfd_wait_exit(pfd, ms);
        if (rc >= 0)
        {
            return rc;
        }
    }
    long end = detach_now_ms() + ms;
    while (milkdetach_id_alive(id))
    {
        struct timespec t = { 0, 10000000L };
        if (detach_now_ms() >= end)
        {
            return 0;
        }
        nanosleep(&t, NULL);
    }
    return 1;
}

static void detach_signal(const MILKDETACH_ID *id, int pfd, pid_t pgid, int sig)
{
    if (pgid == id->pid && milkdetach_id_alive(id))
    {
        kill(-pgid, sig); // the leader is alive, so its group id cannot have been reused
    }
    else if (pfd >= 0)
    {
        milkos_pidfd_signal(pfd, sig);
    }
    else if (milkdetach_id_alive(id))
    {
        kill(id->pid, sig);
    }
}

int milkdetach_stop(const MILKDETACH_ID *id, pid_t pgid, int first_ms, int t2_ms, int t3_ms)
{
    static const int sigs[3]  = { SIGINT, SIGTERM, SIGKILL };
    const int        waits[3] = { t2_ms, t3_ms, 5000 };
    int              pfd      = milkos_pidfd_open(id->pid);
    int              step     = -1;

    // checked after opening the pidfd, so that it refers to this very process
    if (!milkdetach_id_alive(id) || (first_ms > 0 && detach_wait_gone(id, pfd, first_ms)))
    {
        step = 0;
    }
    for (int k = 0; k < 3 && step < 0; k++)
    {
        detach_signal(id, pfd, pgid, sigs[k]);
        if (detach_wait_gone(id, pfd, waits[k]))
        {
            step = k + 1;
        }
    }
    if (pfd >= 0)
    {
        close(pfd);
    }
    return step;
}

const char *milkdetach_stop_step_name(int step)
{
    static const char *const names[4] = { "request", "SIGINT", "SIGTERM", "SIGKILL" };

    return (step >= 0 && step < 4) ? names[step] : "still running";
}

int milkdetach_open_output(const char *path, int rotate)
{
    if (rotate)
    {
        char old[4096];
        if (snprintf(old, sizeof(old), "%s.1", path) < (int) sizeof(old))
        {
            (void) rename(path, old); // ENOENT the first time
        }
    }
    return open(path, O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0644);
}

int milkdetach_output_rotate_if(const char *path, long long max_bytes)
{
    struct stat out;
    struct stat cur;

    if (max_bytes <= 0 || fstat(STDOUT_FILENO, &out) != 0 || !S_ISREG(out.st_mode) ||
        (long long) out.st_size <= max_bytes)
    {
        return 0;
    }
    if (stat(path, &cur) != 0 || cur.st_dev != out.st_dev || cur.st_ino != out.st_ino)
    {
        return 0; // stdout is not that file
    }
    fflush(stdout);
    int fd = milkdetach_open_output(path, 1);
    if (fd < 0)
    {
        return -1;
    }
    dup2(fd, STDOUT_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);
    return 1;
}
