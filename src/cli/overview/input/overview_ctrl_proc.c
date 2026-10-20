// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_ctrl_proc.c
 * @brief Control-mode actions for processes (kill, pause, reset, remove, inspect)
 */

#include "overview_input_internal.h"
#include "overview_data_internal.h"
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 * plist_map_rw - map the existing process registry read-write, only for one control action.
 * Return: the registry, or NULL (nothing is created when it does not exist).
 */
static PROCESSINFOLIST *plist_map_rw(ov_shm_map_t *m)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/processinfo.list.shm", ov_get_procdir());
    if (ov_shm_map(m, path, 1, sizeof(PROCESSINFOLIST)) != 0)
    {
        return NULL;
    }
    return (PROCESSINFOLIST *) m->addr;
}

/**
 * plist_reap_dead - settle registry slots whose PID no longer exists (ESRCH only): free the slot when
 * its proc file is gone, else record STOPPED (loopstat STOP) or CRASHED. Return: slots changed, -1.
 */
static int plist_reap_dead(void)
{
    ov_shm_map_t     m;
    PROCESSINFOLIST *pl = plist_map_rw(&m);
    int              n  = 0;
    if (pl == NULL)
    {
        return -1;
    }
    for (long i = 0; i < PROCESSINFOLISTSIZE; i++)
    {
        pid_t pid = pl->PIDarray[i];
        if (pl->active[i] == 0 || pid <= 0 || kill(pid, 0) == 0 || errno != ESRCH)
        {
            continue;
        }
        char         fpath[PATH_MAX];
        ov_shm_map_t pm;
        snprintf(fpath, sizeof(fpath), "%s/proc.%.*s.%06d.shm", ov_get_procdir(),
                 (int) sizeof(pl->pnamearray[i]), pl->pnamearray[i], (int) pid);
        int newstate = 0;
        if (ov_shm_map(&pm, fpath, 0, sizeof(PROCESSINFO)) == 0)
        {
            newstate =
                (((const PROCESSINFO *) pm.addr)->loopstat == PROCESSINFO_LOOPSTAT_STOP) ? 2 : 3;
            ov_shm_unmap(&pm);
        }
        if (pl->active[i] != newstate)
        {
            pl->active[i] = newstate;
            n++;
        }
    }
    ov_shm_unmap(&m);
    return n;
}

/**
 * pid_is_stopped - check if a process is in 'T' state.
 * @pid: process PID
 *
 * Reads /proc/[pid]/stat and returns 1 if the state
 * character is 'T' (stopped), 0 otherwise.
 */
int pid_is_stopped(pid_t pid)
{
    if (pid <= 0)
    {
        return 0;
    }
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return 0;
    }
    int  p;
    char comm[256];
    char state = '?';
    if (fscanf(fp, "%d %s %c", &p, comm, &state) != 3)
    {
        state = '?';
    }
    fclose(fp);
    return (state == 'T');
}

/**
 * ov_ctrl_proc_kill - send SIGTERM to a process.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_kill(const OV_PROC *p, OV_CMDLOG *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }

    int rc = kill(p->PID, SIGTERM);
    if (log != NULL)
    {
        ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                       "💀 Process \"%s\" (PID %d) — SIGTERM", p->name, p->PID);
    }
}

/**
 * ov_ctrl_proc_sigkill - send SIGKILL to a process.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_sigkill(const OV_PROC *p, OV_CMDLOG *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }
    int rc = kill(p->PID, SIGKILL);
    if (log != NULL)
    {
        ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                       "Process \"%s\" (PID %d) — SIGKILL", p->name, p->PID);
    }
}

/**
 * ov_ctrl_proc_set_ctrlval - mutate process CTRLval.
 * @p:   process model entry
 * @val: new value (-1 to toggle between 0 and 1)
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_set_ctrlval(const OV_PROC *p, int val, OV_CMDLOG *log)
{
    if (p == NULL || p->PID <= 0 || !p->valid)
    {
        return;
    }

    char fname[1024];
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm", ov_get_procdir(), p->name, (int) p->PID);

    int fd = open(fname, O_RDWR);
    if (fd < 0)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Process \"%s\" — ctrl failed (open)", p->name);
        }
        return;
    }

    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size < (off_t) sizeof(PROCESSINFO))
    {
        close(fd);
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Process \"%s\" — ctrl failed (stat)", p->name);
        }
        return;
    }

    PROCESSINFO *pinfo =
        (PROCESSINFO *) mmap(NULL, sizeof(PROCESSINFO), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pinfo == MAP_FAILED)
    {
        close(fd);
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Process \"%s\" — ctrl failed (mmap)", p->name);
        }
        return;
    }

    int old_val    = pinfo->CTRLval;
    int new_val    = (val == -1) ? (old_val == 0 ? 1 : 0) : val;
    pinfo->CTRLval = new_val;

    munmap(pinfo, sizeof(PROCESSINFO));
    close(fd);

    if (log != NULL)
    {
        const char *action;
        const char *emoji = "⚡";
        if (new_val == 0)
        {
            action = "Resume";
            emoji  = "⏯️";
        }
        else if (new_val == 1)
        {
            action = "Pause";
            emoji  = "⏸️";
        }
        else if (new_val == 2)
        {
            action = "Step";
            emoji  = "⏭️";
        }
        else if (new_val == 3)
        {
            action = "Exit request";
            emoji  = "⏹️";
        }
        else
        {
            action = "CTRLval updated";
        }

        ov_cmdlog_push(log, OV_CMDLOG_OK, "%s Process \"%s\" — %s", emoji, p->name, action);
    }
}

/**
 * ov_ctrl_proc_zero_counters - reset process loopcnt.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_zero_counters(const OV_PROC *p, OV_CMDLOG *log)
{
    if (p == NULL || p->PID <= 0 || !p->valid)
    {
        return;
    }

    char fname[1024];
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm", ov_get_procdir(), p->name, (int) p->PID);

    int fd = open(fname, O_RDWR);
    if (fd < 0)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Process \"%s\" — zero failed (open)", p->name);
        }
        return;
    }

    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size < (off_t) sizeof(PROCESSINFO))
    {
        close(fd);
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Process \"%s\" — zero failed (stat)", p->name);
        }
        return;
    }

    PROCESSINFO *pinfo =
        (PROCESSINFO *) mmap(NULL, sizeof(PROCESSINFO), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pinfo == MAP_FAILED)
    {
        close(fd);
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Process \"%s\" — zero failed (mmap)", p->name);
        }
        return;
    }

    pinfo->loopcnt = 0;

    munmap(pinfo, sizeof(PROCESSINFO));
    close(fd);

    if (log != NULL)
    {
        ov_cmdlog_push(log, OV_CMDLOG_OK, "0️⃣ Process \"%s\" — Counters zeroed", p->name);
    }
}

/**
 * ov_ctrl_proc_remove - remove a single process from shm.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_remove(const OV_PROC *p, OV_CMDLOG *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }

    if (kill(p->PID, 0) == 0 || errno == EPERM)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Process \"%s\" (PID %d) is still alive",
                           p->name, p->PID);
        }
        return;
    }

    char fname[1024];
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm", ov_get_procdir(), p->name, (int) p->PID);

    int rc        = unlink(fname);
    int file_gone = (rc == 0 || errno == ENOENT);

    int              deactivated = 0;
    ov_shm_map_t     lm;
    PROCESSINFOLIST *pl = plist_map_rw(&lm);
    if (pl != NULL)
    {
        for (long i = 0; i < PROCESSINFOLISTSIZE; i++)
        {
            if (pl->active[i] != 0 && pl->PIDarray[i] == p->PID)
            {
                pl->active[i] = 0;
                deactivated   = 1;
                break;
            }
        }
        ov_shm_unmap(&lm);
    }

    if (log != NULL)
    {
        if (deactivated || file_gone)
        {
            ov_cmdlog_push(log, OV_CMDLOG_OK, "🗑 Process \"%s\" (PID %d) entry removed", p->name,
                           p->PID);
        }
        else
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "failed to remove entry for process \"%s\"",
                           p->name);
        }
    }
}

/**
 * ov_ctrl_proc_pause_toggle - toggle SIGSTOP/SIGCONT.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_pause_toggle(const OV_PROC *p, OV_CMDLOG *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }
    int stopped = pid_is_stopped(p->PID);
    int sig     = stopped ? SIGCONT : SIGSTOP;
    int rc      = kill(p->PID, sig);
    if (log != NULL)
    {
        ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                       "%s Process \"%s\" (PID %d) — %s", stopped ? "⏯️" : "⏸️", p->name, p->PID,
                       stopped ? "resumed" : "paused");
    }
}

/**
 * ov_ctrl_procs_cleanup - remove crashed/stopped processes
 * @log: command log (may be NULL)
 */
void ov_ctrl_procs_cleanup(OV_CMDLOG *log)
{
    /* Silently remove crashed/stopped procinfo entries */
    int rc     = system("milk-procinfo-rm -c </dev/null >/dev/null 2>&1");
    int reaped = plist_reap_dead();
    if (log != NULL)
    {
        ov_cmdlog_push(log, (rc == 0 && reaped >= 0) ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                       "Process cleanup: stale files removed, %d registry slot(s) settled",
                       reaped > 0 ? reaped : 0);
    }
}

/**
 * ov_ctrl_inspect_item - spawn an interactive detailed view
 * @panel: the active panel type
 * @item:  pointer to the selected item (OV_STREAM, OV_PROC, or OV_FPS)
 */
void ov_ctrl_inspect_item(ov_focus_t panel, const void *item)
{
    if (item == NULL)
    {
        return;
    }

    char cmd[512] = { 0 };

    if (panel == OV_FOCUS_STREAMS)
    {
        const OV_STREAM *s = (const OV_STREAM *) item;
        snprintf(cmd, sizeof(cmd), "milk-stream-info %s", s->name);
    }
    else if (panel == OV_FOCUS_PROCS)
    {
        const OV_PROC *p = (const OV_PROC *) item;
        snprintf(cmd, sizeof(cmd), "milk-procinfo-info %s", p->name);
    }
    else if (panel == OV_FOCUS_FPS)
    {
        const OV_FPS *f = (const OV_FPS *) item;
        snprintf(cmd, sizeof(cmd), "milk-fps-info %s", f->name);
    }
    else
    {
        return;
    }

    /* Suspend TUI */
    ov_raw_mode_exit();
    int rc_clear = system("clear");
    (void) rc_clear;

    /* Show the diagnostic output */
    int rc_cmd = system(cmd);
    (void) rc_cmd;

    /* Prompt the user to return */
    printf("\n\033[1;36m"
           "--- Press ENTER to return to dashboard ---"
           "\033[0m\n");
    fflush(stdout);

    /* Wait for ENTER (or any key) */
    {
        char buf[4];
        if (read(STDIN_FILENO, buf, sizeof(buf)) < 0)
        {
            /* Ignore read failure when waiting for user acknowledgment */
        }
    }

    /* Resume TUI */
    ov_raw_mode_enter();
    ov_buf_force_clear();
}
