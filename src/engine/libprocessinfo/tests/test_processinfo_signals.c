// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest processinfo_unit_signals: a fault ends the process as CRASHED instead of re-faulting forever,
 * the previous handler still gets it, the crash record names the signal, SIGTERM ends a loop, and
 * RestoreSignals() undoes CatchSignals(). */

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "processinfo.h"
#include "processinfo_signals.h"
#include "processtools.h"

static int  failures  = 0;
static char procdir[] = "/tmp/test-pisig-XXXXXX";

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

static PROCESSINFO *child_setup(void)
{
    prctl(PR_SET_DUMPABLE, 0); // no core files from the crashing children
    PROCESSINFO *pi =
        processinfo_setup("pisig", "signal test", "start", __func__, __FILE__, __LINE__);
    if (pi == NULL)
    {
        _exit(3);
    }
    processinfo_CatchSignals();
    return pi;
}

static void child_segv(void)
{
    child_setup();
    volatile int *volatile p = NULL;
    *p = 1; // a real fault: returning from the handler would run this store again, forever
    _exit(1);
}

static void prev_handler(int sig)
{
    (void) sig;
    _exit(42);
}

static void child_segv_chained(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = prev_handler; // like milk-cli's crash report
    sigaction(SIGSEGV, &sa, NULL);
    child_setup();
    volatile int *volatile p = NULL;
    *p                       = 1;
    _exit(1);
}

static char crash_info[256];
static char crash_out[256];

static void child_segv_recorded(void)
{
    int info = open(crash_info, O_WRONLY | O_CREAT | O_APPEND, 0644);
    int out  = open(crash_out, O_WRONLY | O_CREAT | O_APPEND, 0644);
    processinfo_set_crash_record(info, out);
    child_segv();
}

static void child_abort(void)
{
    child_setup();
    abort();
}

static void child_term(void)
{
    PROCESSINFO *pi = child_setup();
    raise(SIGTERM);
    _exit((processinfo_exit_requested(pi) && processinfo_loopstep(pi) == 0) ? 0 : 1);
}

/* Fork @child; wait up to 5 s, since the old handler made a faulting child spin forever. */
static int run_child(void (*child)(void), pid_t *pid)
{
    *pid = fork();
    if (*pid == 0)
    {
        child();
        _exit(0);
    }
    for (int i = 0; i < 500; i++)
    {
        int st;
        if (waitpid(*pid, &st, WNOHANG) == *pid)
        {
            return st;
        }
        usleep(10000);
    }
    kill(*pid, SIGKILL);
    waitpid(*pid, NULL, 0);
    return -1;
}

/* contents of @path (removed), or "" */
static const char *slurp(const char *path)
{
    static char buf[256];
    FILE       *f = fopen(path, "r");
    size_t      n = 0;

    if (f != NULL)
    {
        n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
    }
    buf[n] = '\0';
    unlink(path);
    return buf;
}

/* loopstat and status message a dead child left in its processinfo file; removes the file */
static int child_record(pid_t pid, char *msg, size_t size)
{
    char path[256];
    int  fd;

    snprintf(path, sizeof(path), "%s/proc.pisig.%06d.shm", procdir, (int) pid);
    PROCESSINFO *pi = processinfo_shm_link(path, &fd);
    if (pi == MAP_FAILED)
    {
        return -1;
    }
    int loopstat = pi->loopstat;
    snprintf(msg, size, "%s", pi->statusmsg);
    processinfo_shm_close(pi, fd);
    unlink(path);
    return loopstat;
}

static void on_int(int sig)
{
    (void) sig;
}

static void test_restore(void)
{
    struct sigaction sa;
    struct sigaction cur;

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_int;
    sigaction(SIGINT, &sa, NULL);

    processinfo_signal_INT = 1; // left over from an earlier loop of this process
    processinfo_CatchSignals();
    CHECK(processinfo_signal_INT == 0);
    processinfo_CatchSignals();
    processinfo_RestoreSignals();
    sigaction(SIGINT, NULL, &cur);
    CHECK(cur.sa_handler != on_int);
    processinfo_RestoreSignals();
    sigaction(SIGINT, NULL, &cur);
    CHECK(cur.sa_handler == on_int);
}

static void test_exit_requested(void)
{
    static PROCESSINFO pi;

    CHECK(!processinfo_exit_requested(&pi));
    pi.CTRLval = PROCESSINFO_CTRLVAL_EXIT;
    CHECK(processinfo_exit_requested(&pi));
}

int main(void)
{
    char  msg[STRINGMAXLEN_PROCESSINFO_STATUSMSG];
    pid_t pid;
    int   st;

    if (mkdtemp(procdir) == NULL)
    {
        perror("mkdtemp");
        return 1;
    }
    setenv("MILK_PROC_DIR", procdir, 1);

    test_exit_requested();
    test_restore();

    st = run_child(child_segv, &pid);
    CHECK(st != -1 && WIFSIGNALED(st) && WTERMSIG(st) == SIGSEGV);
    CHECK(child_record(pid, msg, sizeof(msg)) == PROCESSINFO_LOOPSTAT_CRASHED);
    CHECK(strcmp(msg, "SIGSEGV") == 0);

    snprintf(crash_info, sizeof(crash_info), "%s/run.info", procdir);
    snprintf(crash_out, sizeof(crash_out), "%s/run.out", procdir);
    st = run_child(child_segv_recorded, &pid);
    CHECK(st != -1 && WIFSIGNALED(st) && WTERMSIG(st) == SIGSEGV);
    CHECK(child_record(pid, msg, sizeof(msg)) == PROCESSINFO_LOOPSTAT_CRASHED);
    CHECK(strcmp(slurp(crash_info), "signal=SIGSEGV\n") == 0);
    CHECK(strcmp(slurp(crash_out), "=== killed by SIGSEGV\n") == 0);

    st = run_child(child_segv_chained, &pid);
    CHECK(st != -1 && WIFEXITED(st) && WEXITSTATUS(st) == 42);
    CHECK(child_record(pid, msg, sizeof(msg)) == PROCESSINFO_LOOPSTAT_CRASHED);

    st = run_child(child_abort, &pid);
    CHECK(st != -1 && WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT);
    CHECK(child_record(pid, msg, sizeof(msg)) == PROCESSINFO_LOOPSTAT_CRASHED);
    CHECK(strcmp(msg, "SIGABRT") == 0);

    st = run_child(child_term, &pid);
    CHECK(st != -1 && WIFEXITED(st) && WEXITSTATUS(st) == 0);
    child_record(pid, msg, sizeof(msg));

    char list[256];
    snprintf(list, sizeof(list), "%s/processinfo.list.shm", procdir);
    unlink(list);
    rmdir(procdir);

    if (failures > 0)
    {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all ok\n");
    return 0;
}
