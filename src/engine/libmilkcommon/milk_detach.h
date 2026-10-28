// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_detach.h
 * @brief Detached processes: start with a ready pipe, identity, output files, signal ladder
 *
 * Knows nothing about FPS: units (libcomputeunit) and later milk-cli servers build on it.
 * The ready pipe carries newline-terminated sd_notify lines: MAINPID=, STATUS=, READY=1, ERRNO=.
 */

#ifndef MILK_DETACH_H
#define MILK_DETACH_H

#include <stdint.h>
#include <sys/types.h>

/* Descriptor of the ready pipe in a detached process, and of the descriptor it is given (pass_fd) */
#define MILKDETACH_NOTIFY_FD 3
#define MILKDETACH_PASS_FD 4

/* A process as (pid, start time, boot): a pid alone can be reused */
typedef struct
{
    pid_t    pid;
    uint64_t start_ticks;
    char     boot_id[37];
} MILKDETACH_ID;

/* What milkdetach_start() runs */
typedef struct
{
    const char  *path;    /* absolute path of the executable */
    char *const *argv;    /* NULL-terminated */
    char *const *envp;    /* NULL-terminated, built before the call */
    const char  *workdir; /* NULL: keep the caller's */
    int          out_fd;  /* stdout and stderr of the process, -1: /dev/null */
    int   no_notify;      /* the process gets no ready pipe: end of file means execve() succeeded */
    int   pass_fd;        /* given to the process as MILKDETACH_PASS_FD, if > 2 */
    int   drop;           /* become uid/gid/groups before exec */
    uid_t uid;
    gid_t gid;
    const gid_t *groups;
    int          ngroups;
} MILKDETACH_SPEC;

/* What the ready pipe said */
typedef struct
{
    pid_t pid;         /* MAINPID=, -1 if not received */
    int   err;         /* ERRNO=, 0 if not received */
    char  status[256]; /* last STATUS= */
} MILKDETACH_READY;

enum
{
    MILKDETACH_OK = 0, /* READY=1 */
    MILKDETACH_FAILED, /* ERRNO= received */
    MILKDETACH_DIED,   /* end of file before READY=1 */
    MILKDETACH_TIMEOUT
};

/** milkdetach_id_of - identity of the running process @pid. Return: 0, or -1 (ESRCH). */
int milkdetach_id_of(pid_t pid, MILKDETACH_ID *id);

/** milkdetach_id_alive - 1 if the process @id still runs (same boot, same start), else 0. */
int milkdetach_id_alive(const MILKDETACH_ID *id);

/**
 * milkdetach_start - run @spec in a new session and process group, reparented to init (double fork).
 * The process gets stdin from /dev/null, stdout and stderr on @spec->out_fd, the write end of the
 * ready pipe on MILKDETACH_NOTIFY_FD, @spec->pass_fd on MILKDETACH_PASS_FD, and no other descriptor.
 * Safe from a multithreaded caller.
 * Return: read end of the ready pipe (close-on-exec), or -1 with errno set.
 */
int milkdetach_start(const MILKDETACH_SPEC *spec);

/**
 * milkdetach_wait_ready - read the ready pipe @fd until READY=1, end of file or @timeout_ms (-1: none).
 * Return: MILKDETACH_OK, _FAILED, _DIED or _TIMEOUT; @r holds what was received.
 */
int milkdetach_wait_ready(int fd, int timeout_ms, MILKDETACH_READY *r);

/** milkdetach_notify_fd - ready pipe named by environment variable @envname, made close-on-exec; unsets it. -1 if none. */
int milkdetach_notify_fd(const char *envname);

/** milkdetach_notify - write the lines @msg (each ending in '\n') to ready pipe @fd. Return: 0 or -1. */
int milkdetach_notify(int fd, const char *msg);

/**
 * milkdetach_stop - end process @id: wait @first_ms for it to exit on its own (after a request), then
 * SIGINT, wait @t2_ms, SIGTERM, wait @t3_ms, SIGKILL. Signals go to the process group when @pgid is
 * the process's own pid, else to the process only.
 * Return: 0 gone without a signal, 1-3 the signal that ended it (INT, TERM, KILL), -1 still running.
 */
int milkdetach_stop(const MILKDETACH_ID *id, pid_t pgid, int first_ms, int t2_ms, int t3_ms);

/** milkdetach_stop_step_name - "request", "SIGINT", "SIGTERM", "SIGKILL" for milkdetach_stop()'s result. */
const char *milkdetach_stop_step_name(int step);

/**
 * milkdetach_open_output - open output file @path for appending (O_NOFOLLOW, close-on-exec, 0644);
 * with @rotate, first rename an existing @path to @path.1. Return: descriptor or -1.
 */
int milkdetach_open_output(const char *path, int rotate);

/**
 * milkdetach_output_rotate_if - when stdout is a file over @max_bytes, rotate @path to @path.1 and
 * point stdout and stderr to a fresh @path. Return: 1 rotated, 0 not needed, -1 error.
 */
int milkdetach_output_rotate_if(const char *path, long long max_bytes);

#endif
