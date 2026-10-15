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

#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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
    if (strcmp(prop, "pid") == 0 || strcmp(prop, "PID") == 0)
    {
        snprintf(out, outsz, "%d", (int) pi->PID);
    }
    else if (strcmp(prop, "loopstat") == 0)
    {
        snprintf(out, outsz, "%d", pi->loopstat);
    }
    else if (strcmp(prop, "status") == 0)
    {
        const char *st = "OTHER";
        switch (pi->loopstat)
        {
        case PROCESSINFO_LOOPSTAT_INIT:
            st = "INIT";
            break;
        case PROCESSINFO_LOOPSTAT_ACTIVE:
            st = "ACTIVE";
            break;
        case PROCESSINFO_LOOPSTAT_PAUSE:
            st = "PAUSED";
            break;
        case PROCESSINFO_LOOPSTAT_STOP:
            st = "STOPPED";
            break;
        case PROCESSINFO_LOOPSTAT_ERROR:
            st = "ERROR";
            break;
        case PROCESSINFO_LOOPSTAT_SPIN:
            st = "SPINNING";
            break;
        case PROCESSINFO_LOOPSTAT_CRASHED:
            st = "CRASHED";
            break;
        default:
            break;
        }
        snprintf(out, outsz, "%s", st);
    }
    else if (strcmp(prop, "loopcnt") == 0 || strcmp(prop, "cnt0") == 0)
    {
        snprintf(out, outsz, "%ld", pi->loopcnt);
    }
    else if (strcmp(prop, "loopcntMax") == 0)
    {
        snprintf(out, outsz, "%ld", pi->loopcntMax);
    }
    else if (strcmp(prop, "loopfreq") == 0)
    {
        double hz = (pi->dtmedian_iter_ns > 0) ? (1.0e9 / (double) pi->dtmedian_iter_ns) : 0.0;
        snprintf(out, outsz, "%.1f", hz);
    }
    else if (strcmp(prop, "dt_median_ns") == 0 || strcmp(prop, "dtmedian_iter_ns") == 0)
    {
        snprintf(out, outsz, "%ld", pi->dtmedian_iter_ns);
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
    else if (strcmp(prop, "triggerstream") == 0 || strcmp(prop, "triggerstreamname") == 0)
    {
        snprintf(out, outsz, "%s", pi->triggerstreamname);
    }
    else if (strcmp(prop, "statusmsg") == 0 || strcmp(prop, "msg") == 0)
    {
        snprintf(out, outsz, "%s", pi->statusmsg);
    }
    else if (strcmp(prop, "tmux") == 0 || strcmp(prop, "tmuxname") == 0)
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
    else if (strcmp(prop, "list") == 0 || strcmp(prop, "*") == 0)
    {
        snprintf(out, outsz,
                 "pid status loopstat loopcnt loopcntMax loopfreq dt_median_ns exectime rtprio "
                 "ctrlval trigmode triggerstream statusmsg tmux description missedframes");
    }
    else if (strcmp(prop, "kv") == 0)
    {
        const char *st = "OTHER";
        switch (pi->loopstat)
        {
        case PROCESSINFO_LOOPSTAT_INIT:
            st = "INIT";
            break;
        case PROCESSINFO_LOOPSTAT_ACTIVE:
            st = "ACTIVE";
            break;
        case PROCESSINFO_LOOPSTAT_PAUSE:
            st = "PAUSED";
            break;
        case PROCESSINFO_LOOPSTAT_STOP:
            st = "STOPPED";
            break;
        case PROCESSINFO_LOOPSTAT_ERROR:
            st = "ERROR";
            break;
        case PROCESSINFO_LOOPSTAT_SPIN:
            st = "SPINNING";
            break;
        case PROCESSINFO_LOOPSTAT_CRASHED:
            st = "CRASHED";
            break;
        default:
            break;
        }
        double hz = (pi->dtmedian_iter_ns > 0) ? (1.0e9 / (double) pi->dtmedian_iter_ns) : 0.0;
        snprintf(out, outsz,
                 "pid\t%d\nPID\t%d\nstatus\t%s\nloopstat\t%d\nloopcnt\t%ld\ncnt0\t%ld\n"
                 "loopcntMax\t%ld\nloopfreq\t%.1f\ndt_median_ns\t%ld\nexectime\t%.1f\n"
                 "rtprio\t%d\nctrlval\t%d\ntrigmode\t%d\ntriggerstream\t%s\nstatusmsg\t%s\n"
                 "msg\t%s\ntmux\t%s\ndescription\t%s\nmissedframes\t%lu",
                 (int) pi->PID, (int) pi->PID, st, pi->loopstat, pi->loopcnt, pi->loopcnt,
                 pi->loopcntMax, hz, pi->dtmedian_iter_ns, (double) pi->dtmedian_exec_ns / 1000.0,
                 pi->RT_priority, pi->CTRLval, pi->triggermode, pi->triggerstreamname,
                 pi->statusmsg, pi->statusmsg, pi->tmuxname, pi->description,
                 (unsigned long) pi->triggermissedframe_cumul);
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
    if (strcmp(prop, "ctrlval") != 0 && strcmp(prop, "CTRLval") != 0)
    {
        return -1;
    }
    int ctrlval_int = atoi(val);
    if (strcasecmp(val, "run") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_RUN;
    }
    else if (strcasecmp(val, "pause") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_PAUSE;
    }
    else if (strcasecmp(val, "step") == 0 || strcasecmp(val, "incr") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_INCR;
    }
    else if (strcasecmp(val, "stop") == 0 || strcasecmp(val, "exit") == 0)
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
 * @param prop  Property key (status, state, tasks, active_step, completed, step_count, progress, errors, pid, name)
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
    if (strcmp(prop, "status") == 0 || strcmp(prop, "state") == 0)
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
    else if (strcmp(prop, "tasks") == 0 || strcmp(prop, "active_step") == 0)
    {
        snprintf(out, outsz, "%u", seqst->NBtasks_active);
    }
    else if (strcmp(prop, "step_count") == 0)
    {
        snprintf(out, outsz, "%u", seqst->NBtasks_active + seqst->NBtasks_completed);
    }
    else if (strcmp(prop, "progress") == 0)
    {
        uint32_t tot = seqst->NBtasks_active + seqst->NBtasks_completed;
        snprintf(out, outsz, "%u/%u", seqst->NBtasks_completed, tot);
    }
    else if (strcmp(prop, "errors") == 0)
    {
        snprintf(out, outsz, "%u", seqst->error_count);
    }
    else if (strcmp(prop, "last_error") == 0)
    {
        snprintf(out, outsz, "%s", seqst->last_error);
    }
    else if (strcmp(prop, "pid") == 0 || strcmp(prop, "PID") == 0)
    {
        snprintf(out, outsz, "%d", (int) seqst->pid);
    }
    else if (strcmp(prop, "completed") == 0)
    {
        snprintf(out, outsz, "%u", seqst->NBtasks_completed);
    }
    else if (strcmp(prop, "name") == 0)
    {
        snprintf(out, outsz, "%s", seqst->name);
    }
    else if (strcmp(prop, "script") == 0 || strcmp(prop, "script_path") == 0)
    {
        snprintf(out, outsz, "%s", seqst->script_path);
    }
    else if (strcmp(prop, "list") == 0 || strcmp(prop, "*") == 0)
    {
        snprintf(out, outsz,
                 "status state tasks active_step completed step_count progress errors last_error "
                 "pid name script");
    }
    else if (strcmp(prop, "kv") == 0)
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
        uint32_t tot = seqst->NBtasks_active + seqst->NBtasks_completed;
        snprintf(out, outsz,
                 "status\t%s\nstate\t%s\ntasks\t%u\nactive_step\t%u\ncompleted\t%u\n"
                 "step_count\t%u\nprogress\t%u/%u\nerrors\t%u\nlast_error\t%s\n"
                 "pid\t%d\nPID\t%d\nname\t%s\nscript\t%s",
                 s, s, seqst->NBtasks_active, seqst->NBtasks_active, seqst->NBtasks_completed, tot,
                 seqst->NBtasks_completed, tot, seqst->error_count, seqst->last_error,
                 (int) seqst->pid, (int) seqst->pid, seqst->name, seqst->script_path);
    }
    else
    {
        found = 0;
    }

    milkseq_disconnect(seqst);
    return found ? 0 : -1;
}

/**
 * @brief Control or submit commands to an active FPS sequencer (`@q.<name>.<prop>`).
 *
 * Supports:
 * - `action` / `status` / `state`: `pause`, `resume` (`run`), `abort` (`stop`, `exit`)
 * - `submit` / `fifo`: write a sequencer command line directly to its FIFO
 *
 * @param name Sequencer name
 * @param prop Property key (`action`, `status`, `state`, `submit`, `fifo`)
 * @param val  Action or command string
 * @return 0 on success, -1 on error
 */
int milk_shm_prop_set_seq(const char *name, const char *prop, const char *val)
{
    if (name == NULL || prop == NULL || val == NULL)
    {
        return -1;
    }
    MILKSEQ_STATE *seqst = milkseq_connect(name);
    if (seqst == NULL)
    {
        return -1;
    }

    int rc = -1;
    if (strcmp(prop, "action") == 0 || strcmp(prop, "status") == 0 || strcmp(prop, "state") == 0)
    {
        if (strcasecmp(val, "pause") == 0 || strcasecmp(val, "idle") == 0)
        {
            if (seqst->pid > 0 && seqst->pid != getpid())
            {
                kill(seqst->pid, SIGSTOP);
            }
            seqst->status = MILKSEQ_STATUS_IDLE;
            rc            = 0;
        }
        else if (strcasecmp(val, "resume") == 0 || strcasecmp(val, "run") == 0 ||
                 strcasecmp(val, "running") == 0)
        {
            if (seqst->pid > 0 && seqst->pid != getpid())
            {
                kill(seqst->pid, SIGCONT);
            }
            seqst->status = MILKSEQ_STATUS_RUNNING;
            rc            = 0;
        }
        else if (strcasecmp(val, "abort") == 0 || strcasecmp(val, "stop") == 0 ||
                 strcasecmp(val, "exit") == 0)
        {
            seqst->status = MILKSEQ_STATUS_STOPPING;
            if (seqst->fifo_path[0] != '\0')
            {
                int fd = open(seqst->fifo_path, O_WRONLY | O_NONBLOCK);
                if (fd >= 0)
                {
                    ssize_t _ = write(fd, "exit\n", 5);
                    (void) _;
                    close(fd);
                }
            }
            rc = 0;
        }
    }
    else if (strcmp(prop, "submit") == 0 || strcmp(prop, "fifo") == 0)
    {
        if (seqst->fifo_path[0] != '\0')
        {
            int fd = open(seqst->fifo_path, O_WRONLY | O_NONBLOCK);
            if (fd >= 0)
            {
                char line[512];
                int  n = snprintf(line, sizeof(line), "%s\n", val);
                if (n > 0 && write(fd, line, (size_t) n) > 0)
                {
                    rc = 0;
                }
                close(fd);
            }
        }
    }

    milkseq_disconnect(seqst);
    return rc;
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
