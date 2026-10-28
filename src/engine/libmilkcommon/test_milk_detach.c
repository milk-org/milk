// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkcommon-detach: a detached process is nobody's child and holds only 0-3 (and the descriptor
 * it is given), the ready pipe reports ready, exec failures and early deaths (or only exec, without a
 * ready pipe), the stop ladder escalates, outputs rotate. */

#include "milk_detach.h"
#include "milk_os.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int  failures = 0;
static char dir[]    = "/tmp/test-detach-XXXXXX";

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

extern char **environ;

/* Start "/bin/sh -c script"; return the ready pipe */
static int start_sh(const char *script, int out_fd)
{
    char           *argv[] = { (char *) "sh", (char *) "-c", (char *) script, NULL };
    MILKDETACH_SPEC sp     = {
        .path = "/bin/sh", .argv = argv, .envp = environ, .workdir = dir, .out_fd = out_fd
    };
    return milkdetach_start(&sp);
}

static int count_fds(pid_t pid, int *has_above_2)
{
    char path[64];
    int  n = 0;

    *has_above_2 = 0;
    for (int fd = 0; fd < 1024; fd++)
    {
        struct stat st;
        snprintf(path, sizeof(path), "/proc/%d/fd/%d", (int) pid, fd);
        if (lstat(path, &st) == 0)
        {
            n++;
            *has_above_2 |= (fd > 2);
        }
    }
    return n;
}

static void test_ready_and_hygiene(void)
{
    MILKDETACH_READY r;
    MILKDETACH_ID    id;
    struct sigaction ign = { .sa_handler = SIG_IGN };
    struct sigaction old;
    sigset_t         block;
    sigset_t         oldmask;
    int              above;

    // a descriptor without close-on-exec, and SIGINT ignored and blocked: none of it reaches the process
    int leak = open("/dev/null", O_RDONLY);
    sigaction(SIGINT, &ign, &old);
    sigemptyset(&block);
    sigaddset(&block, SIGINT);
    sigprocmask(SIG_BLOCK, &block, &oldmask);

    int fd = start_sh("printf 'STATUS=hello\\nREADY=1\\n' >&3; exec 3>&-; exec sleep 30", -1);
    CHECK(fd >= 0);
    CHECK(milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_OK);
    CHECK(r.pid > 0 && strcmp(r.status, "hello") == 0);
    close(fd);
    sigprocmask(SIG_SETMASK, &oldmask, NULL);
    sigaction(SIGINT, &old, NULL);
    close(leak);

    struct timespec t = { 0, 100000000L };
    nanosleep(&t, NULL); // the exec of sleep
    CHECK(waitpid(r.pid, NULL, WNOHANG) == -1 && errno == ECHILD);
    CHECK(milkdetach_id_of(r.pid, &id) == 0 && milkdetach_id_alive(&id));
    CHECK(getpgid(r.pid) == r.pid && getsid(r.pid) != getsid(0));
    CHECK(count_fds(r.pid, &above) == 3 && above == 0);

    CHECK(milkdetach_stop(&id, r.pid, 0, 2000, 2000) ==
          1); // SIGINT works despite the caller's mask
    CHECK(!milkdetach_id_alive(&id));
}

static void test_failures(void)
{
    MILKDETACH_READY r;
    char            *argv[] = { (char *) "nope", NULL };
    MILKDETACH_SPEC  sp     = {
        .path = "/nonexistent/nope", .argv = argv, .envp = environ, .out_fd = -1
    };

    int fd = milkdetach_start(&sp);
    CHECK(fd >= 0);
    CHECK(milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_FAILED);
    CHECK(r.err == ENOENT && r.pid > 0 && strstr(r.status, "execve") != NULL);
    close(fd);

    sp.path    = "/bin/sh";
    sp.workdir = "/nonexistent";
    fd         = milkdetach_start(&sp);
    CHECK(fd >= 0 && milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_FAILED && r.err == ENOENT);
    CHECK(strstr(r.status, "chdir") != NULL);
    close(fd);

    fd = start_sh("exit 3", -1);
    CHECK(fd >= 0 && milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_DIED);
    close(fd);

    fd = start_sh("exec 3>&-; sleep 30", -1); // closes the pipe without READY: same as dying
    CHECK(fd >= 0 && milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_DIED && r.pid > 0);
    close(fd);
    kill(-r.pid, SIGKILL);
}

static void test_timeout_and_ladder(void)
{
    MILKDETACH_READY r;
    MILKDETACH_ID    id;
    char             pidfile[256];
    char             script[512];

    int fd = start_sh("exec sleep 30", -1);
    CHECK(fd >= 0 && milkdetach_wait_ready(fd, 200, &r) == MILKDETACH_TIMEOUT && r.pid > 0);
    close(fd);
    CHECK(milkdetach_id_of(r.pid, &id) == 0 && milkdetach_stop(&id, r.pid, 0, 2000, 2000) == 1);

    // ignores SIGINT and SIGTERM; its child dies with it: signals go to the group
    snprintf(pidfile, sizeof(pidfile), "%s/child.pid", dir);
    snprintf(script, sizeof(script),
             "trap '' INT TERM; sleep 30 & echo $! > %s; printf 'READY=1\\n' >&3; exec 3>&-; wait",
             pidfile);
    fd = start_sh(script, -1);
    CHECK(fd >= 0 && milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_OK);
    close(fd);
    CHECK(milkdetach_id_of(r.pid, &id) == 0);
    CHECK(milkdetach_stop(&id, r.pid, 100, 200, 200) == 3);

    FILE *f     = fopen(pidfile, "r");
    int   child = 0;
    CHECK(f != NULL && fscanf(f, "%d", &child) == 1);
    if (f != NULL)
    {
        fclose(f);
    }
    uint64_t ticks;
    for (int i = 0; i < 100 && milkos_proc_start_ticks(child, &ticks) == 0; i++)
    {
        struct timespec t = { 0, 10000000L };
        nanosleep(&t, NULL);
    }
    CHECK(child > 0 && milkos_proc_start_ticks(child, &ticks) == -1);
    unlink(pidfile);
}

static void test_identity(void)
{
    MILKDETACH_ID id;

    CHECK(milkdetach_id_of(getpid(), &id) == 0 && milkdetach_id_alive(&id));
    pid_t pid = fork();
    if (pid == 0)
    {
        _exit(0);
    }
    MILKDETACH_ID cid;
    CHECK(milkdetach_id_of(pid, &cid) == 0 || errno == ESRCH);
    struct timespec t = { 0, 100000000L };
    nanosleep(&t, NULL);
    CHECK(!milkdetach_id_alive(&cid)); // a zombie is gone
    waitpid(pid, NULL, 0);

    id.start_ticks++;
    CHECK(!milkdetach_id_alive(&id)); // same pid, other process
    id.start_ticks--;
    id.boot_id[0] ^= 1;
    CHECK(!milkdetach_id_alive(&id)); // earlier boot
}

static void test_output(void)
{
    char             path[256];
    char             old[300];
    struct stat      st;
    MILKDETACH_READY r;

    snprintf(path, sizeof(path), "%s/run.out", dir);
    snprintf(old, sizeof(old), "%s.1", path);
    int fd = milkdetach_open_output(path, 1);
    CHECK(fd >= 0 && (fcntl(fd, F_GETFD) & FD_CLOEXEC));
    CHECK(write(fd, "first\n", 6) == 6);

    // the process's output goes to the file
    int rfd = start_sh("echo out; echo err >&2; printf 'READY=1\\n' >&3", fd);
    close(fd);
    CHECK(rfd >= 0 && milkdetach_wait_ready(rfd, 5000, &r) == MILKDETACH_OK);
    close(rfd);
    struct timespec t = { 0, 100000000L };
    nanosleep(&t, NULL);
    CHECK(stat(path, &st) == 0 && st.st_size == 14);

    fd = milkdetach_open_output(path, 1);
    CHECK(fd >= 0 && stat(old, &st) == 0 && st.st_size == 14);
    CHECK(stat(path, &st) == 0 && st.st_size == 0);

    // rotation from the writer: only when stdout is that file and over the cap (it moves stderr too)
    int saved     = dup(STDOUT_FILENO);
    int saved_err = dup(STDERR_FILENO);
    fflush(stdout);
    dup2(fd, STDOUT_FILENO);
    close(fd);
    printf("0123456789\n");
    fflush(stdout);
    int r1 = milkdetach_output_rotate_if(path, 100);
    int r2 = milkdetach_output_rotate_if(path, 5);
    printf("after\n");
    fflush(stdout);
    dup2(saved, STDOUT_FILENO);
    dup2(saved_err, STDERR_FILENO);
    close(saved);
    close(saved_err);
    CHECK(r1 == 0 && r2 == 1);
    CHECK(stat(old, &st) == 0 && st.st_size == 11);
    CHECK(stat(path, &st) == 0 && st.st_size == 6);
    unlink(path);
    unlink(old);
}

static void test_notify(void)
{
    int  p[2];
    char buf[64];

    CHECK(pipe(p) == 0);
    char num[16];
    snprintf(num, sizeof(num), "%d", p[1]);
    setenv("TEST_DETACH_FD", num, 1);
    int fd = milkdetach_notify_fd("TEST_DETACH_FD");
    CHECK(fd == p[1] && getenv("TEST_DETACH_FD") == NULL && (fcntl(fd, F_GETFD) & FD_CLOEXEC));
    CHECK(milkdetach_notify(fd, "READY=1\n") == 0);
    CHECK(read(p[0], buf, sizeof(buf)) == 8 && memcmp(buf, "READY=1\n", 8) == 0);
    close(p[0]);
    CHECK(milkdetach_notify(fd, "STATUS=late\n") == -1 && errno == EPIPE); // and no SIGPIPE
    close(fd);

    int nul = open("/dev/null", O_RDONLY);
    snprintf(num, sizeof(num), "%d", nul);
    setenv("TEST_DETACH_FD", num, 1); // not a pipe (fd 0 may be a pipe under CI runners)
    CHECK(milkdetach_notify_fd("TEST_DETACH_FD") == -1);
    close(nul);
}

/* no_notify: the pipe closes at exec, so end of file is success; a failed exec still reports */
static void test_no_notify(void)
{
    MILKDETACH_READY r;
    MILKDETACH_ID    id;
    char            *argv[] = { (char *) "sleep", (char *) "30", NULL };
    MILKDETACH_SPEC  sp     = {
        .path = "/bin/sleep", .argv = argv, .envp = environ, .out_fd = -1, .no_notify = 1
    };
    int above;

    int fd = milkdetach_start(&sp);
    CHECK(fd >= 0 && milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_DIED);
    CHECK(r.pid > 0 && r.err == 0);
    close(fd);
    // the program's loader opens libraries for a moment; a leaked descriptor would be in every look
    struct timespec t     = { 0, 10000000L };
    int             clean = 0;
    for (int i = 0; i < 200 && !clean; i++)
    {
        clean = (count_fds(r.pid, &above) == 3 && above == 0);
        if (!clean)
        {
            nanosleep(&t, NULL);
        }
    }
    CHECK(milkdetach_id_of(r.pid, &id) == 0 && clean);
    CHECK(milkdetach_stop(&id, r.pid, 0, 2000, 2000) == 1);

    sp.path = "/nonexistent/nope";
    fd      = milkdetach_start(&sp);
    CHECK(fd >= 0 && milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_FAILED && r.err == ENOENT);
    close(fd);
}

/* pass_fd: the process reads what its starter left in the descriptor it is given */
static void test_pass_fd(void)
{
    MILKDETACH_READY r;
    MILKDETACH_ID    id;
    int              p[2];
    char *argv[]       = { (char *) "sh", (char *) "-c",
                           (char *) "read l <&4 && [ \"$l\" = hello ] && printf 'READY=1\\n' >&3; "
                                    "exec 3>&-; exec sleep 30",
                           NULL };
    MILKDETACH_SPEC sp = {
        .path = "/bin/sh", .argv = argv, .envp = environ, .workdir = dir, .out_fd = -1
    };

    CHECK(pipe(p) == 0 && write(p[1], "hello\n", 6) == 6);
    sp.pass_fd = p[0];
    int fd     = milkdetach_start(&sp);
    close(p[0]);
    close(p[1]);
    CHECK(fd >= 0 && milkdetach_wait_ready(fd, 5000, &r) == MILKDETACH_OK);
    close(fd);
    CHECK(milkdetach_id_of(r.pid, &id) == 0 && milkdetach_stop(&id, r.pid, 0, 2000, 2000) == 1);
}

int main(void)
{
    if (mkdtemp(dir) == NULL)
    {
        perror("mkdtemp");
        return 1;
    }
    test_ready_and_hygiene();
    test_failures();
    test_timeout_and_ladder();
    test_identity();
    test_output();
    test_notify();
    test_no_notify();
    test_pass_fd();
    rmdir(dir);

    if (failures > 0)
    {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all ok\n");
    return 0;
}
