// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_ctrl_proc.c
 * @brief Control-mode actions for processes (kill, pause, reset, remove, inspect)
 */

#include "overview_input_internal.h"
#include "overview_data_internal.h"
#include "fps_unit_state.h"
#include "milk_spawn.h"
#include "overview_ctrl.h"
#include "processinfo_list.h"
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
 * plist_map - map the existing process registry read-only (slots are changed through processinfo_list_*).
 * Return: the registry, or NULL (nothing is created when it does not exist).
 */
static const PROCESSINFOLIST *plist_map(ov_shm_map_t *m)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/processinfo.list.shm", ov_get_procdir());
    if (ov_shm_map(m, path, 0, sizeof(PROCESSINFOLIST)) != 0)
    {
        return NULL;
    }
    return (const PROCESSINFOLIST *) m->addr;
}

/**
 * proc_shm_path - path of @p's proc file, built from its registry slot as the scanner does.
 * Return: 0, or -1 when the slot no longer holds @p's PID.
 */
static int proc_shm_path(const OV_PROC *p, char *buf, size_t len)
{
    ov_shm_map_t           m;
    const PROCESSINFOLIST *pl = plist_map(&m);
    int                    rc = -1;
    if (pl != NULL && p->pindex >= 0 && p->pindex < PROCESSINFOLISTSIZE &&
        pl->PIDarray[p->pindex] == p->PID)
    {
        snprintf(buf, len, "%s/proc.%.*s.%06d.shm", ov_get_procdir(),
                 (int) sizeof(pl->pnamearray[0]), pl->pnamearray[p->pindex], (int) p->PID);
        rc = 0;
    }
    ov_shm_unmap(&m);
    return rc;
}

/**
 * proc_map_rw - map @p's proc file read-write for one action. Return: the PROCESSINFO or NULL
 * (reason logged).
 */
static PROCESSINFO *proc_map_rw(const OV_PROC *p, ov_shm_map_t *m, const char *what, OV_CMDLOG *log)
{
    char path[PATH_MAX];
    if (proc_shm_path(p, path, sizeof(path)) != 0)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                       "Process \"%s\" (PID %d): %s refused, registry entry changed", p->name,
                       (int) p->PID, what);
        return NULL;
    }
    if (ov_shm_map(m, path, 1, sizeof(PROCESSINFO)) != 0)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Process \"%s\" (PID %d): %s failed: %s", p->name,
                       (int) p->PID, what, strerror(errno));
        return NULL;
    }
    return (PROCESSINFO *) m->addr;
}

/**
 * proc_signal - send @sig to @p only if it is running and its PID still names the same process.
 */
static void proc_signal(const OV_PROC *p, int sig, const char *signame, OV_CMDLOG *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }
    if (!p->active)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Process \"%s\" (PID %d) is not running; not signalled",
                       p->name, (int) p->PID);
        return;
    }
    if (!pid_same_process(p->PID, p->start_ticks))
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Process \"%s\": PID %d was reused; not signalled",
                       p->name, (int) p->PID);
        return;
    }
    int rc = kill(p->PID, sig);
    ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "Process \"%s\" (PID %d): %s %s",
                   p->name, (int) p->PID, signame, rc == 0 ? "sent" : strerror(errno));
}

/**
 * ov_ctrl_proc_kill - send SIGTERM to a process (refused for a reused PID).
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_kill(const OV_PROC *p, OV_CMDLOG *log)
{
    proc_signal(p, SIGTERM, "SIGTERM", log);
}

/**
 * ov_ctrl_proc_sigkill - send SIGKILL to a process (refused for a reused PID).
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_sigkill(const OV_PROC *p, OV_CMDLOG *log)
{
    proc_signal(p, SIGKILL, "SIGKILL", log);
}

/**
 * proc_unit - the unit whose run process @p is, if its run.info records this very process. Return: 1.
 */
static int proc_unit(const OV_PROC *p, char *name, size_t size)
{
    FPS_UNIT_INFO info;

    if (p->name[0] == '\0' || fps_unit_info_read(p->name, FPS_UNIT_RUN, &info) != 0 ||
        info.inprocess || info.id.pid != p->PID ||
        (p->start_ticks != 0 && info.id.start_ticks != p->start_ticks))
    {
        return 0;
    }
    snprintf(name, size, "%s", p->name);
    return 1;
}

/**
 * ov_ctrl_proc_set_ctrlval - mutate process CTRLval.
 * @p:   process model entry
 * @val: new value (-1 to toggle between 0 and 1)
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_set_ctrlval(const OV_PROC *p, int val, OV_CMDLOG *log)
{
    static const char *const names[] = { "Resume", "Pause", "Step", "Exit request" };
    ov_shm_map_t             m;
    char                     unit[STRINGMAXLEN_FPS_NAME];
    if (p == NULL || p->PID <= 0 || !p->valid)
    {
        return;
    }
    if (val == 3 && proc_unit(p, unit, sizeof(unit)))
    {
        ov_ctrl_unit_act(unit, "runstop", log); // the unit's stop: request, then signals (U6, U11)
        return;
    }
    PROCESSINFO *pinfo = proc_map_rw(p, &m, "CTRLval", log);
    if (pinfo == NULL)
    {
        return;
    }
    int new_val    = (val == -1) ? (pinfo->CTRLval == 0 ? 1 : 0) : val;
    pinfo->CTRLval = new_val;
    ov_shm_unmap(&m);
    ov_cmdlog_push(log, OV_CMDLOG_OK, "Process \"%s\": %s", p->name,
                   (new_val >= 0 && new_val <= 3) ? names[new_val] : "CTRLval updated");
}

/**
 * ov_ctrl_proc_zero_counters - reset process loopcnt.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_zero_counters(const OV_PROC *p, OV_CMDLOG *log)
{
    ov_shm_map_t m;
    if (p == NULL || p->PID <= 0 || !p->valid)
    {
        return;
    }
    PROCESSINFO *pinfo = proc_map_rw(p, &m, "zero counters", log);
    if (pinfo == NULL)
    {
        return;
    }
    pinfo->loopcnt = 0;
    ov_shm_unmap(&m);
    ov_cmdlog_push(log, OV_CMDLOG_OK, "Process \"%s\": counters zeroed", p->name);
}

/**
 * ov_ctrl_proc_remove - remove the registry entry and proc file of a process that is gone.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_remove(const OV_PROC *p, OV_CMDLOG *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }
    if ((kill(p->PID, 0) == 0 || errno == EPERM) &&
        (p->start_ticks == 0 || pid_same_process(p->PID, p->start_ticks)))
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Process \"%s\" (PID %d) is still alive", p->name,
                       (int) p->PID);
        return;
    }

    char path[PATH_MAX];
    if (proc_shm_path(p, path, sizeof(path)) != 0)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Process \"%s\": registry entry changed; not removed",
                       p->name);
        return;
    }
    int file_gone   = (unlink(path) == 0 || errno == ENOENT);
    int deactivated = (processinfo_list_clear_slot(p->pindex, p->PID) == 1);

    ov_cmdlog_push(log, (deactivated && file_gone) ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                   "Process \"%s\" (PID %d): entry %s", p->name, (int) p->PID,
                   (deactivated && file_gone) ? "removed" : "not fully removed");
}

/**
 * ov_ctrl_procs_cleanup - remove crashed/stopped processes
 * @log: command log (may be NULL)
 */
void ov_ctrl_procs_cleanup(OV_CMDLOG *log)
{
    static const char *const argv[] = { "milk-procinfo-rm", "-c", NULL };
    int                      rc     = milkspawn_run(argv, -1, -1);
    int                      err    = errno;
    int                      reaped = processinfo_list_reap(0);
    if (rc < 0)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Process cleanup failed: milk-procinfo-rm: %s",
                       strerror(err));
    }
    else if (rc != 0)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Process cleanup failed: milk-procinfo-rm exit %d", rc);
    }
    else
    {
        ov_cmdlog_push(log, reaped >= 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
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

    /* argv only: names come from shm file names and must never reach a shell */
    char        pid_str[16];
    const char *argv[5] = { NULL };
    if (panel == OV_FOCUS_STREAMS)
    {
        argv[0] = "milk-stream-info";
        argv[1] = "--";
        argv[2] = ((const OV_STREAM *) item)->name;
    }
    else if (panel == OV_FOCUS_PROCS)
    {
        snprintf(pid_str, sizeof(pid_str), "%d", (int) ((const OV_PROC *) item)->PID);
        argv[0] = "milk-procinfo-info";
        argv[1] = "--pid";
        argv[2] = pid_str;
    }
    else if (panel == OV_FOCUS_FPS)
    {
        argv[0] = "milk-fps-info";
        argv[1] = "--";
        argv[2] = ((const OV_FPS *) item)->name;
    }
    else
    {
        return;
    }

    /* Suspend TUI */
    ov_raw_mode_exit();
    fputs("\033[H\033[2J", stdout);
    fflush(stdout);

    /* Show the diagnostic output; ^C stops the tool, not milk-CTRL (as system() did) */
    struct sigaction ign, old_int, old_quit;
    memset(&ign, 0, sizeof(ign));
    ign.sa_handler = SIG_IGN;
    sigaction(SIGINT, &ign, &old_int);
    sigaction(SIGQUIT, &ign, &old_quit);
    int rc = milkspawn_run(argv, STDOUT_FILENO, STDERR_FILENO);
    if (rc < 0)
    {
        printf("\n%s: %s\n", argv[0], strerror(errno));
    }
    else if (rc > 0)
    {
        printf("\n%s exited with status %d\n", argv[0], rc);
    }
    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGQUIT, &old_quit, NULL);

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
