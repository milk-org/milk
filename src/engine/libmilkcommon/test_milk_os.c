// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest: milkos_* return what they promise; OFD locks conflict within one process, unlike fcntl locks. */

#include "milk_os.h"

#include <errno.h>
#include <fcntl.h>
#if defined(__GLIBC__)
#    include <malloc.h>
#endif
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            return 1;                                                    \
        }                                                                \
    } while (0)

int main(void)
{
    char     path[4096];
    char     boot[37];
    int      p[2];
    uint64_t ticks = 0;
    gid_t    gid;
    gid_t    groups[256];
    int      st;

    CHECK(milkos_self_exe(path, sizeof(path)) == 0);
    CHECK(strstr(path, "test-milk-os") != NULL);
    CHECK(milkos_self_exe(path, 4) == -1 && errno == ENAMETOOLONG);

    CHECK(milkos_pipe_cloexec(p) == 0);
    CHECK((fcntl(p[0], F_GETFD) & FD_CLOEXEC) && (fcntl(p[1], F_GETFD) & FD_CLOEXEC));
    close(p[0]);
    close(p[1]);

    // a child forked again with milkos_fork_child() keeps no descriptor above 2
    CHECK(dup2(STDOUT_FILENO, 57) == 57);
    pid_t pid = fork();
    if (pid == 0)
    {
        pid_t inner = milkos_fork_child();
        if (inner == 0)
        {
            milkos_close_from(3);
            _exit(fcntl(57, F_GETFD) == -1 && errno == EBADF ? 0 : 1);
        }
        _exit((inner > 0 && waitpid(inner, &st, 0) == inner && WIFEXITED(st)) ? WEXITSTATUS(st)
                                                                              : 2);
    }
    CHECK(pid > 0 && waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == 0);
    close(57);

    // two open file descriptions of one file: the second lock fails while the first is held
    char lockname[] = "/tmp/test-milk-os-XXXXXX";
    int  fd1        = mkstemp(lockname);
    int  fd2        = open(lockname, O_RDWR | O_CLOEXEC);
    CHECK(fd1 >= 0 && fd2 >= 0);
    unlink(lockname);
    CHECK(milkos_lock_file(fd1, 0) == 0);
    CHECK(milkos_lock_file(fd2, 0) == -1 && errno == EAGAIN);
    CHECK(milkos_unlock_file(fd1) == 0);
    CHECK(milkos_lock_file(fd2, 1) == 0);
    close(fd1);
    close(fd2);

    CHECK(milkos_proc_start_ticks(getpid(), &ticks) == 0 && ticks > 0);
    CHECK(milkos_boot_id(boot, sizeof(boot)) == 0 && strlen(boot) == 36);

    // a pidfd waits for and signals exactly its process; a reaped pid is gone
    pid = fork();
    if (pid == 0)
    {
        pause();
        _exit(0);
    }
    int pfd = milkos_pidfd_open(pid);
    CHECK(pfd >= 0);
    CHECK(milkos_pidfd_wait_exit(pfd, 20) == 0);
    CHECK(milkos_pidfd_signal(pfd, SIGTERM) == 0);
    CHECK(milkos_pidfd_wait_exit(pfd, 5000) == 1);
    CHECK(waitpid(pid, &st, 0) == pid && WIFSIGNALED(st) && WTERMSIG(st) == SIGTERM);
    close(pfd);
    CHECK(milkos_proc_start_ticks(pid, &ticks) == -1 && errno == ESRCH);

    CHECK(milkos_on_tmpfs("/proc") == 0);
    CHECK(milkos_on_tmpfs("/no/such/dir") == -1);
    if (access("/dev/shm", F_OK) == 0)
    {
        CHECK(milkos_on_tmpfs("/dev/shm") == 1);
    }

    int n = milkos_user_ids(getuid(), &gid, groups, 256);
    CHECK(n >= 1);
    int has_gid = 0;
    for (int i = 0; i < n; i++)
    {
        has_gid |= (groups[i] == gid);
    }
    CHECK(has_gid);

#if defined(__GLIBC__)
    /* an 8 MiB block comes from the heap, not from an mmap, and freeing it does not shrink the heap */
    int keep = milkos_malloc_keep();
    CHECK(keep == 0 || keep == 1);
    if (keep == 0)
    {
        struct mallinfo2 m0 = mallinfo2();
        char *volatile big  = malloc(8 << 20); // volatile: the compiler may drop an unused block
        CHECK(big != NULL);
        memset(big, 1, 8 << 20);
        struct mallinfo2 m1 = mallinfo2();
        CHECK(m1.hblkhd == m0.hblkhd && m1.uordblks >= m0.uordblks + (8 << 20));
        free(big);
        CHECK(mallinfo2().arena == m1.arena);
    }
#endif

    printf("all ok\n");
    return 0;
}
