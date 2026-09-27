// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_ctrl_proc.c
 * @brief Control-mode actions for processes (kill, pause, reset, remove, inspect)
 */

#include "overview_input_internal.h"
#include "processinfo_shm_list_create.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm", ov_get_shmdir(), p->name, (int) p->PID);

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
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm", ov_get_shmdir(), p->name, (int) p->PID);

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
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm", ov_get_shmdir(), p->name, (int) p->PID);

    int rc        = unlink(fname);
    int file_gone = (rc == 0 || errno == ENOENT);

    int deactivated = 0;
    if (pinfolist == NULL)
    {
        long pindex_unused;
        processinfo_shm_list_create(&pindex_unused);
    }

    if (pinfolist != NULL)
    {
        for (long i = 0; i < PROCESSINFOLISTSIZE; i++)
        {
            if (pinfolist->PIDarray[i] == p->PID)
            {
                pinfolist->active[i] = 0;
                deactivated          = 1;
                break;
            }
        }
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
    int rc = system("milk-procinfo-rm -c </dev/null >/dev/null 2>&1");
    if (log != NULL)
    {
        ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                       "🧹 Process cleanup requested");
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
