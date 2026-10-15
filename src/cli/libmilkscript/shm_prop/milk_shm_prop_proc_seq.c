// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop_proc_seq.c
 * @brief Processinfo (`@p.`) and FPS Sequencer (`@q.`/`@seq.`) entity handlers.
 *
 * Role & Goal:
 *   Implements shared-memory telemetry reading, control-value (`ctrlval`)
 *   writing, and liveness/running checks for `PROCESSINFO` (`proc.*.shm`)
 *   and `MILKSEQ_STATE` (`milkseq.*.shm`) structures.
 */

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "CLIcore.h"
#include "fpsseq.h"
#include "milk_shm_prop_internal.h"
#include "processinfo.h"

extern PROCESSINFO *processinfo_shm_link(const char *pname, int *fd);
extern errno_t      processinfo_procdirname(char *procdname);
extern int          processinfo_shm_list_create(void);

/**
 * @brief Look up an active process PID by name in `pinfolist`.
 *
 * @param name Process name in `pinfolist->pnamearray`
 * @return Positive `pid_t` if found, 0 otherwise
 */
static pid_t find_active_proc_pid(const char *name)
{
    if (pinfolist == NULL)
    {
        processinfo_shm_list_create();
    }
    if (pinfolist == NULL)
    {
        return 0;
    }
    for (int pidx = 0; pidx < PROCESSINFOLISTSIZE; pidx++)
    {
        if (pinfolist->active[pidx] && strcmp(pinfolist->pnamearray[pidx], name) == 0)
        {
            return pinfolist->PIDarray[pidx];
        }
    }
    return 0;
}

/**
 * @brief Map the `PROCESSINFO` SHM segment for a given PID.
 *
 * @param pid    Target process ID
 * @param fd_out Receives the open file descriptor (must be closed after unmap)
 * @return Mapped `PROCESSINFO *`, or NULL on failure
 */
static PROCESSINFO *map_proc_shm_by_pid(pid_t pid, int *fd_out)
{
    *fd_out = -1;
    if (pid <= 0)
    {
        return NULL;
    }
    char pfname[STRINGMAXLEN_FULLFILENAME];
    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);
    snprintf(pfname, sizeof(pfname), "%s/proc.%d.shm", procdname, (int) pid);
    PROCESSINFO *pi = processinfo_shm_link(pfname, fd_out);
    if (pi == MAP_FAILED || pi == NULL)
    {
        if (*fd_out >= 0)
        {
            close(*fd_out);
            *fd_out = -1;
        }
        return NULL;
    }
    return pi;
}

/**
 * @brief Read a processinfo telemetry property into @out.
 *
 * @param name  Process name in `pinfolist`
 * @param prop  Property key (pid, loopstat, loopcnt, loopfreq, exectime, etc.)
 * @param out   Output buffer
 * @param outsz Size of @out in bytes
 * @return 0 on success, -1 if process or property not found
 */
int milk_shm_prop_get_proc(const char *name, const char *prop, char *out, size_t outsz)
{
    pid_t        found_pid = find_active_proc_pid(name);
    int          pfd       = -1;
    PROCESSINFO *pi        = map_proc_shm_by_pid(found_pid, &pfd);
    if (pi == NULL)
    {
        return -1;
    }

    int found = 1;
    if (strcmp(prop, "pid") == 0)
    {
        snprintf(out, outsz, "%d", (int) pi->PID);
    }
    else if (strcmp(prop, "loopstat") == 0)
    {
        snprintf(out, outsz, "%d", pi->loopstat);
    }
    else if (strcmp(prop, "loopcnt") == 0)
    {
        snprintf(out, outsz, "%ld", pi->loopcnt);
    }
    else if (strcmp(prop, "loopfreq") == 0)
    {
        double hz = (pi->dtmedian_iter_ns > 0) ? (1.0e9 / (double) pi->dtmedian_iter_ns) : 0.0;
        snprintf(out, outsz, "%.1f", hz);
    }
    else if (strcmp(prop, "exectime") == 0)
    {
        snprintf(out, outsz, "%.1f", (double) pi->dtmedian_exec_ns / 1000.0);
    }
    else if (strcmp(prop, "rtprio") == 0)
    {
        snprintf(out, outsz, "%d", pi->RT_priority);
    }
    else if (strcmp(prop, "ctrlval") == 0)
    {
        snprintf(out, outsz, "%d", pi->CTRLval);
    }
    else if (strcmp(prop, "trigmode") == 0)
    {
        snprintf(out, outsz, "%d", pi->triggermode);
    }
    else if (strcmp(prop, "statusmsg") == 0)
    {
        snprintf(out, outsz, "%s", pi->statusmsg);
    }
    else if (strcmp(prop, "tmux") == 0)
    {
        snprintf(out, outsz, "%s", pi->tmuxname);
    }
    else if (strcmp(prop, "description") == 0)
    {
        snprintf(out, outsz, "%s", pi->description);
    }
    else if (strcmp(prop, "missedframes") == 0)
    {
        snprintf(out, outsz, "%lu", (unsigned long) pi->triggermissedframe_cumul);
    }
    else
    {
        found = 0;
    }

    munmap(pi, sizeof(PROCESSINFO));
    close(pfd);
    return found ? 0 : -1;
}

/**
 * @brief Write `ctrlval` on an active processinfo entry (`@p.<name>.ctrlval`).
 *
 * Accepts symbolic names ("run", "pause", "step", "stop", "exit") or integer
 * control codes.
 *
 * @param name Process name in `pinfolist`
 * @param prop Property key (must be "ctrlval")
 * @param val  Control value string
 * @return 0 on success, -1 on error
 */
int milk_shm_prop_set_proc(const char *name, const char *prop, const char *val)
{
    if (strcmp(prop, "ctrlval") != 0)
    {
        return -1;
    }
    int ctrlval_int = atoi(val);
    if (strcmp(val, "run") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_RUN;
    }
    else if (strcmp(val, "pause") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_PAUSE;
    }
    else if (strcmp(val, "step") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_INCR;
    }
    else if (strcmp(val, "stop") == 0 || strcmp(val, "exit") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_EXIT;
    }

    pid_t        found_pid = find_active_proc_pid(name);
    int          pfd       = -1;
    PROCESSINFO *pi_shm    = map_proc_shm_by_pid(found_pid, &pfd);
    if (pi_shm == NULL)
    {
        return -1;
    }
    pi_shm->CTRLval = ctrlval_int;
    munmap(pi_shm, sizeof(PROCESSINFO));
    close(pfd);
    return 0;
}

/**
 * @brief Check whether a named processinfo entry exists (and is running).
 *
 * @param name          Process name in `pinfolist`
 * @param check_running If non-zero, check `loopstat == 1 && CTRLval == 0`
 * @return 1 if exists/running, 0 otherwise
 */
int milk_shm_prop_exists_proc(const char *name, int check_running)
{
    pid_t found_pid = find_active_proc_pid(name);
    if (found_pid <= 0 || kill(found_pid, 0) != 0)
    {
        return 0;
    }
    if (!check_running)
    {
        return 1;
    }
    int          pfd    = -1;
    PROCESSINFO *pi_shm = map_proc_shm_by_pid(found_pid, &pfd);
    if (pi_shm == NULL)
    {
        return 0;
    }
    int running = (pi_shm->loopstat == 1 && pi_shm->CTRLval == 0) ? 1 : 0;
    munmap(pi_shm, sizeof(PROCESSINFO));
    close(pfd);
    return running;
}

/**
 * @brief Read an FPS sequencer property (`@q.<name>.<prop>`).
 *
 * @param name  Sequencer name
 * @param prop  Property key (status, tasks, errors, pid, completed)
 * @param out   Output buffer
 * @param outsz Size of @out in bytes
 * @return 0 on success, -1 if sequencer or property not found
 */
int milk_shm_prop_get_seq(const char *name, const char *prop, char *out, size_t outsz)
{
    MILKSEQ_STATE *seqst = milkseq_connect(name);
    if (seqst == NULL)
    {
        return -1;
    }

    int found = 1;
    if (strcmp(prop, "status") == 0)
    {
        const char *s  = "IDLE";
        uint32_t    st = seqst->status;
        if (st & MILKSEQ_STATUS_ERROR)
        {
            s = "ERROR";
        }
        else if (st & MILKSEQ_STATUS_STOPPING)
        {
            s = "STOPPING";
        }
        else if (st & MILKSEQ_STATUS_RUNNING)
        {
            s = "RUNNING";
        }
        snprintf(out, outsz, "%s", s);
    }
    else if (strcmp(prop, "tasks") == 0)
    {
        snprintf(out, outsz, "%u", seqst->NBtasks_active);
    }
    else if (strcmp(prop, "errors") == 0)
    {
        snprintf(out, outsz, "%u", seqst->error_count);
    }
    else if (strcmp(prop, "pid") == 0)
    {
        snprintf(out, outsz, "%d", (int) seqst->pid);
    }
    else if (strcmp(prop, "completed") == 0)
    {
        snprintf(out, outsz, "%u", seqst->NBtasks_completed);
    }
    else
    {
        found = 0;
    }

    milkseq_disconnect(seqst);
    return found ? 0 : -1;
}

/**
 * @brief Check whether a named FPS sequencer exists (and is actively running).
 *
 * @param name          Sequencer name
 * @param check_running If non-zero, check `MILKSEQ_STATUS_RUNNING`
 * @return 1 if exists/running, 0 otherwise
 */
int milk_shm_prop_exists_seq(const char *name, int check_running)
{
    MILKSEQ_STATE *seqst = milkseq_connect(name);
    if (seqst == NULL)
    {
        return 0;
    }
    int running = check_running ? ((seqst->status & MILKSEQ_STATUS_RUNNING) ? 1 : 0) : 1;
    milkseq_disconnect(seqst);
    return running;
}
