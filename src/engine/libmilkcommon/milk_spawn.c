// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_spawn.c
 * @brief Shell-free process spawning: strings from files or shm never reach /bin/sh
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include "milk_spawn.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

/**
 * milkspawn_start - start argv[0] (PATH lookup, no shell) with stdin on /dev/null and no fd above 2.
 * @out_fd: fd for the child's stdout, or -1 for /dev/null; @err_fd: same for stderr.
 * Return: child pid, or -1 with errno set (ENOENT when argv[0] is not found).
 */
pid_t milkspawn_start(const char *const argv[], int out_fd, int err_fd)
{
    posix_spawn_file_actions_t fa;
    posix_spawnattr_t          at;
    sigset_t                   sigs;
    pid_t                      pid = -1;
    int                        rc;

    if (posix_spawn_file_actions_init(&fa) != 0)
    {
        return -1;
    }
    if (posix_spawnattr_init(&at) != 0)
    {
        posix_spawn_file_actions_destroy(&fa);
        return -1;
    }
    posix_spawn_file_actions_addopen(&fa, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (out_fd < 0)
    {
        posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    }
    else if (out_fd != STDOUT_FILENO)
    {
        posix_spawn_file_actions_adddup2(&fa, out_fd, STDOUT_FILENO);
    }
    if (err_fd < 0)
    {
        posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    }
    else if (err_fd != STDERR_FILENO)
    {
        posix_spawn_file_actions_adddup2(&fa, err_fd, STDERR_FILENO);
    }
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
    posix_spawn_file_actions_addclosefrom_np(&fa, STDERR_FILENO + 1);
#endif
    /* children start with an empty mask and default INT/QUIT/PIPE, even if the caller ignores them */
    sigemptyset(&sigs);
    posix_spawnattr_setsigmask(&at, &sigs);
    sigaddset(&sigs, SIGINT);
    sigaddset(&sigs, SIGQUIT);
    sigaddset(&sigs, SIGPIPE);
    posix_spawnattr_setsigdefault(&at, &sigs);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);

    /* posix_spawnp() does not modify argv; the cast only drops const */
    rc = posix_spawnp(&pid, argv[0], &fa, &at, (char *const *) argv, environ);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0)
    {
        errno = rc;
        return -1;
    }
    return pid;
}

/**
 * milkspawn_wait - reap @pid, retrying on EINTR.
 * Return: exit status, 128 + signal number if it was killed, or -1 if waitpid() failed.
 */
int milkspawn_wait(pid_t pid)
{
    int st;

    while (waitpid(pid, &st, 0) < 0)
    {
        if (errno != EINTR)
        {
            return -1;
        }
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

/**
 * milkspawn_run - milkspawn_start() then milkspawn_wait().
 * Return: exit status as milkspawn_wait(), or -1 (errno set) if the child could not be started.
 */
int milkspawn_run(const char *const argv[], int out_fd, int err_fd)
{
    pid_t pid = milkspawn_start(argv, out_fd, err_fd);

    return (pid < 0) ? -1 : milkspawn_wait(pid);
}
