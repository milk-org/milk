// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_unit_state.h
 * @brief   Unit directory, run/conf identity records, unit lock, and the one definition of a unit's state
 *
 * A unit's files live in <dir>/<name>.fps.d/, <dir> being MILK_UNIT_DIR or the shm directory:
 * env (environment snapshot), run.out/conf.out (output, with a .1), run.info/conf.info (KEY=VALUE
 * records of the last process of each role: identity at start, exit or stop lines at the end), and
 * systemd ("<role>=<unit> user|system" for the roles a systemd service runs, U17).
 * Starting and stopping live in libcomputeunit (fps_unit.h).
 */

#ifndef FPS_UNIT_STATE_H
#define FPS_UNIT_STATE_H

#include <stddef.h>
#include <sys/types.h>

#include "fps_types.h"
#include "milk_detach.h"

typedef enum
{
    FPS_UNIT_RUN  = 0,
    FPS_UNIT_CONF = 1
} FPS_UNIT_ROLE;

typedef enum
{
    FPS_UNIT_STOPPED = 0, /* no process, no record of one */
    FPS_UNIT_STARTING,    /* alive, loop not active yet */
    FPS_UNIT_RUNNING,     /* alive, loop active (STATUS_RUNLOOP, or SIGNAL_CONFRUN for conf) */
    FPS_UNIT_STOPPING,    /* alive, stop requested */
    FPS_UNIT_EXITED,      /* last process exited with status 0, or was stopped */
    FPS_UNIT_FAILED       /* last process: non-zero exit, fatal signal, or no exit record */
} FPS_UNIT_STATE;

/* One <role>.info record */
typedef struct
{
    MILKDETACH_ID id; /* pid 0: no record */
    pid_t         pgid;
    double        started; /* s since the epoch */
    char          session[64];
    char          starter[128];
    char          procdir[512];
    int           inprocess; /* a run inside milk-cli: stop only requests (U16) */
    int           has_exit;
    int           exit_code; /* exit=N */
    char          signame[16];
    int           stop_requested;
    char          stop[16]; /* how a stop ended it: request, SIGINT, SIGTERM, SIGKILL */
} FPS_UNIT_INFO;

typedef struct
{
    FPS_UNIT_STATE state;
    pid_t          pid;    /* alive process, else the last one (0: none) */
    double         uptime; /* s, while alive */
    char           last[96];
    char           session[64];
    char      manager[144]; /* "<unit> (user)" or "<unit> (system)" when systemd runs the role */
    long long out_size;     /* bytes in <role>.out, -1: none */
} FPS_UNIT_STATUS;

/* fps_unit_status_fps() flags */
#define FPS_UNIT_REPAIR 0x1 /* clear stale run bits of a dead process when the unit lock is free */
#define FPS_UNIT_LOCKED 0x2 /* the caller holds the unit lock: repair without taking it */

/** fps_unit_role_name - "run" or "conf". */
const char *fps_unit_role_name(FPS_UNIT_ROLE role);

/** fps_unit_state_name - "stopped", "starting", "running", "stopping", "exited", "failed". */
const char *fps_unit_state_name(FPS_UNIT_STATE state);

/** fps_unit_state_alive - 1 for the states with a live process. */
static inline int fps_unit_state_alive(FPS_UNIT_STATE s)
{
    return s == FPS_UNIT_STARTING || s == FPS_UNIT_RUNNING || s == FPS_UNIT_STOPPING;
}

/** fps_unit_base_dir - MILK_UNIT_DIR, else the FPS shm directory. Return: 0 or -1. */
int fps_unit_base_dir(char *buf, size_t size);

/** fps_unit_path - <base>/<name>.fps.d, or a @file in it. Return: 0, or -1 if it does not fit. */
int fps_unit_path(const char *name, const char *file, char *buf, size_t size);

/** fps_unit_role_path - <role>.<ext> in the unit directory, e.g. run.out, conf.info. Return: 0 or -1. */
int fps_unit_role_path(const char   *name,
                       FPS_UNIT_ROLE role,
                       const char   *ext,
                       char         *buf,
                       size_t        size);

/** fps_unit_dir_create - create the unit directory. Return: 0, or -1 with errno set. */
int fps_unit_dir_create(const char *name);

/** fps_unit_dir_remove - remove the unit directory and its files. Return: 0, or -1. */
int fps_unit_dir_remove(const char *name);

/** fps_unit_info_read - last <role>.info record. Return: 0, or -1 if there is none. */
int fps_unit_info_read(const char *name, FPS_UNIT_ROLE role, FPS_UNIT_INFO *info);

/**
 * fps_unit_info_create - start a new <role>.info with the identity of the calling process.
 * Return: the file, open for appending (close-on-exec), or -1.
 */
int fps_unit_info_create(const char *name, FPS_UNIT_ROLE role, const char *starter, int inprocess);

/** fps_unit_info_append - append the line @line (no newline) to <role>.info. Return: 0 or -1. */
int fps_unit_info_append(const char *name, FPS_UNIT_ROLE role, const char *line);

/**
 * fps_unit_systemd_read - the systemd service that runs @role of @name, and *@user 1 if a user manager
 * runs it. Return: 1, or 0 if milk starts and stops @role itself.
 */
int fps_unit_systemd_read(const char *name, FPS_UNIT_ROLE role, char *unit, size_t size, int *user);

/** fps_unit_systemd_write - record that systemd service @unit (a user manager's if @user) runs @role of @name. Return: 0 or -1. */
int fps_unit_systemd_write(const char *name, FPS_UNIT_ROLE role, const char *unit, int user);

/** fps_unit_tail_offset - where the last @nlines lines of file @fd (size @end) start; 0 if fewer, or @nlines <= 0. */
off_t fps_unit_tail_offset(int fd, off_t end, long nlines);

/**
 * fps_unit_out_tail - the last @nlines lines of @role's output, NUL-terminated: at most @size - 1 bytes,
 * so the oldest of them may be cut. Return: bytes in @buf, or -1 if there is no output file.
 */
int fps_unit_out_tail(const char *name, FPS_UNIT_ROLE role, long nlines, char *buf, size_t size);

/** fps_unit_proc_path - <procdir>/proc.<name>.<pid>.shm of the process in @info. Return: 0 or -1. */
int fps_unit_proc_path(const char *name, const FPS_UNIT_INFO *info, char *buf, size_t size);

/**
 * fps_unit_lock - take the unit lock: an OFD lock on <name>.fps.shm, through a descriptor of its own.
 * Return: descriptor holding the lock, or -1 (errno: ENOENT no FPS, EACCES not the owner, EAGAIN held).
 */
int fps_unit_lock(const char *name, int wait);

/** fps_unit_unlock - release fps_unit_lock(). */
void fps_unit_unlock(int fd);

/** fps_unit_status_fps - state of @role of the connected @fps; see FPS_UNIT_REPAIR. Return: 0. */
int fps_unit_status_fps(FPS *fps, FPS_UNIT_ROLE role, FPS_UNIT_STATUS *st, int flags);

/** fps_unit_status - fps_unit_status_fps() with repair, by name. Return: 0, or -1 if the FPS does not exist. */
int fps_unit_status(const char *name, FPS_UNIT_ROLE role, FPS_UNIT_STATUS *st);

#endif
