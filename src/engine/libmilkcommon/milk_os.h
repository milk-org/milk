// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_os.h
 * @brief The OS services milk needs beyond strict POSIX, behind one interface (milk_os_linux.c)
 */

#ifndef MILK_OS_H
#define MILK_OS_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

/** milkos_self_exe - absolute path of the running executable. Return: 0, or -1 with errno set. */
int milkos_self_exe(char *buf, size_t size);

/** milkos_pipe_cloexec - pipe() with both ends close-on-exec, set atomically. Return: 0 or -1. */
int milkos_pipe_cloexec(int fds[2]);

/** milkos_close_from - close every descriptor >= @lowfd. Async-signal-safe, for a forked child. */
void milkos_close_from(int lowfd);

/** milkos_fork_child - fork that is async-signal-safe (no atfork handlers), to fork in a child. */
pid_t milkos_fork_child(void);

/** milkos_lock_file - exclusive OFD lock on the whole file. Return: 0, or -1 (EAGAIN: held). */
int milkos_lock_file(int fd, int wait);

/** milkos_unlock_file - release a milkos_lock_file() lock. Return: 0 or -1. */
int milkos_unlock_file(int fd);

/** milkos_proc_start_ticks - start of @pid in clock ticks since boot; a zombie counts as gone. Return: 0, -1 (ESRCH). */
int milkos_proc_start_ticks(pid_t pid, uint64_t *ticks);

/** milkos_boot_id - identifier of the current boot, 36 characters. Return: 0 or -1. */
int milkos_boot_id(char *buf, size_t size);

/** milkos_pidfd_open - descriptor on process @pid, immune to pid reuse. Return: fd or -1. */
int milkos_pidfd_open(pid_t pid);

/** milkos_pidfd_signal - send @sig to the process behind @pidfd. Return: 0 or -1. */
int milkos_pidfd_signal(int pidfd, int sig);

/** milkos_pidfd_wait_exit - wait @timeout_ms (-1: forever). Return: 1 exited, 0 timeout, -1. */
int milkos_pidfd_wait_exit(int pidfd, int timeout_ms);

/** milkos_on_tmpfs - 1 if @path is on tmpfs or ramfs, 0 if not, -1 on error. */
int milkos_on_tmpfs(const char *path);

/**
 * milkos_service_unit - the systemd service whose cgroup holds this process (/proc/self/cgroup), and
 * *@user 1 if a user manager runs it. Return: 0, or -1 if the process is in no service.
 */
int milkos_service_unit(char *unit, size_t size, int *user);

/**
 * milkos_user_ids - primary group and supplementary groups of user @uid, for dropping to it.
 * Return: number of groups written to @groups, or -1 (unknown user, or more than @max groups).
 */
int milkos_user_ids(uid_t uid, gid_t *gid, gid_t *groups, int max);

/**
 * milkos_set_ids - become user @uid, group @gid with @ngroups supplementary @groups, for good.
 * Async-signal-safe in a single-threaded child. Return: 0, or -1 with errno set (EPERM if root is still regainable).
 */
int milkos_set_ids(uid_t uid, gid_t gid, const gid_t *groups, int ngroups);

/**
 * milkos_malloc_keep - keep freed memory for reuse: blocks up to 32 MiB come from the heap, which is not
 * trimmed while less than 256 MiB of it is free, so a buffer freed and allocated again in a loop is not
 * faulted in again. Left to the user when GLIBC_TUNABLES sets malloc tunables.
 * Return: 0, 1 if left to GLIBC_TUNABLES, -1 where the C library cannot do it.
 */
int milkos_malloc_keep(void);

#endif
