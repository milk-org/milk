// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop_proc.c
 * @brief Processinfo (`@p.`) entity handlers.
 *
 * Role & Goal:
 *   Implements shared-memory telemetry reading, control-value (`ctrlval`)
 *   writing, and liveness/running checks for `PROCESSINFO` (`proc.*.shm`) structures.
 */

#include <dirent.h>
#include <errno.h>
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
#include "processinfo_timing.h"
#include "milk_shm_prop_internal.h"
#include "processinfo.h"
#include "processtools_trigger.h"

extern PROCESSINFO *processinfo_shm_link(const char *pname, int *fd);
extern errno_t      processinfo_procdirname(char *procdname);

/**
 * get_procdir_local - Resolve the directory holding processinfo shared-memory files: the one the
 * processes write to, from processinfo_procdirname() (MILK_PROC_DIR, else the default directory)
 * @pdname: Output buffer receiving the resolved directory path
 * @pdsz:   Capacity of @pdname in bytes
 *
 * Return: void
 */
static void get_procdir_local(char *pdname, size_t pdsz)
{
    char dir[STRINGMAXLEN_FULLFILENAME];
    processinfo_procdirname(dir);
    snprintf(pdname, pdsz, "%s", dir);
}

static char s_pinfolist_dir[256] = "";
static int  s_pinfolist_fd       = -1;

/**
 * get_pinfolist_local - Map and cache the global PROCESSINFOLIST shared-memory table
 *
 * Resolves the active processinfo directory and maps `processinfo.list.shm` if not
 * already mapped (or remaps it if the directory changed), updating the global
 * `pinfolist` pointer.
 *
 * Return: Pointer to the mapped `PROCESSINFOLIST`, or NULL if unavailable.
 */
static PROCESSINFOLIST *get_pinfolist_local(void)
{
    char pdname[256];
    get_procdir_local(pdname, sizeof(pdname));
    if (pinfolist != NULL && pinfolist != MAP_FAILED && strcmp(s_pinfolist_dir, pdname) == 0)
    {
        return pinfolist;
    }
    if (pinfolist != NULL && pinfolist != MAP_FAILED)
    {
        munmap(pinfolist, sizeof(PROCESSINFOLIST));
        pinfolist = NULL;
    }
    if (s_pinfolist_fd >= 0)
    {
        close(s_pinfolist_fd);
        s_pinfolist_fd = -1;
    }
    snprintf(s_pinfolist_dir, sizeof(s_pinfolist_dir), "%s", pdname);
    char lfn[512];
    snprintf(lfn, sizeof(lfn), "%s/processinfo.list.shm", pdname);
    if (access(lfn, R_OK) == 0)
    {
        int lfd   = -1;
        pinfolist = (PROCESSINFOLIST *) processinfo_shm_link(lfn, &lfd);
        if (pinfolist == MAP_FAILED)
        {
            if (lfd >= 0)
            {
                close(lfd);
            }
            pinfolist = NULL;
        }
        else
        {
            s_pinfolist_fd = lfd;
        }
    }
    return pinfolist;
}

/**
 * @brief Look up an active process PID by name in `pinfolist` or SHM directory.
 *
 * @param name Process name in `pinfolist->pnamearray`
 * @return Positive `pid_t` if found, 0 otherwise
 */
static pid_t find_active_proc_pid(const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return 0;
    }
    PROCESSINFOLIST *plist = get_pinfolist_local();
    pid_t            any_p = 0;
    if (plist != NULL)
    {
        for (int pidx = 0; pidx < PROCESSINFOLISTSIZE; pidx++)
        {
            if (plist->active[pidx] && strcmp(plist->pnamearray[pidx], name) == 0 &&
                plist->PIDarray[pidx] > 0)
            {
                pid_t p = plist->PIDarray[pidx];
                if (kill(p, 0) == 0)
                {
                    return p;
                }
                if (any_p == 0)
                {
                    any_p = p;
                }
            }
        }
    }
    if (any_p > 0)
    {
        return any_p;
    }

    char pdname[256];
    get_procdir_local(pdname, sizeof(pdname));
    DIR *dp = opendir(pdname);
    if (dp != NULL)
    {
        size_t         nlen = strlen(name);
        struct dirent *de;
        while ((de = readdir(dp)) != NULL)
        {
            size_t dlen = strlen(de->d_name);
            if (dlen > nlen + 9 && strncmp(de->d_name, "proc.", 5) == 0 &&
                strncmp(de->d_name + 5, name, nlen) == 0 && de->d_name[5 + nlen] == '.' &&
                strcmp(de->d_name + dlen - 4, ".shm") == 0)
            {
                pid_t p = (pid_t) strtol(de->d_name + 6 + nlen, NULL, 10);
                if (p > 0)
                {
                    if (kill(p, 0) == 0)
                    {
                        closedir(dp);
                        return p;
                    }
                    if (any_p == 0)
                    {
                        any_p = p;
                    }
                }
            }
        }
        closedir(dp);
    }
    return any_p;
}

/**
 * @brief Map the `PROCESSINFO` SHM segment for a given name and PID.
 *
 * @param name   Process name (may be NULL)
 * @param pid    Target process ID
 * @param fd_out Receives the open file descriptor (must be closed after unmap)
 * @return Mapped `PROCESSINFO *`, or NULL on failure
 */
static PROCESSINFO *map_proc_shm(const char *name, pid_t pid, int *fd_out)
{
    *fd_out = -1;
    if (pid <= 0)
    {
        return NULL;
    }
    char pdname[256];
    get_procdir_local(pdname, sizeof(pdname));

    char pfname[512];
    if (name != NULL && name[0] != '\0')
    {
        snprintf(pfname, sizeof(pfname), "%s/proc.%s.%06d.shm", pdname, name, (int) pid);
        if (access(pfname, R_OK) == 0)
        {
            PROCESSINFO *pi = processinfo_shm_link(pfname, fd_out);
            if (pi != MAP_FAILED && pi != NULL)
            {
                return pi;
            }
            if (*fd_out >= 0)
            {
                close(*fd_out);
                *fd_out = -1;
            }
        }
        snprintf(pfname, sizeof(pfname), "%s/proc.%s.%d.shm", pdname, name, (int) pid);
        if (access(pfname, R_OK) == 0)
        {
            PROCESSINFO *pi = processinfo_shm_link(pfname, fd_out);
            if (pi != MAP_FAILED && pi != NULL)
            {
                return pi;
            }
            if (*fd_out >= 0)
            {
                close(*fd_out);
                *fd_out = -1;
            }
        }
    }

    snprintf(pfname, sizeof(pfname), "%s/proc.%d.shm", pdname, (int) pid);
    if (access(pfname, R_OK) == 0)
    {
        PROCESSINFO *pi = processinfo_shm_link(pfname, fd_out);
        if (pi != MAP_FAILED && pi != NULL)
        {
            return pi;
        }
        if (*fd_out >= 0)
        {
            close(*fd_out);
            *fd_out = -1;
        }
    }
    return NULL;
}

/**
 * proc_status_str - Map a PROCESSINFO state and OS process liveness to a status label
 * @pi:  Mapped `PROCESSINFO` structure
 * @pid: Fallback PID to check if `pi->PID` is unset
 *
 * Checks whether the target process is still alive via `kill(pid, 0)` and combines
 * `loopstat` and `CTRLval` into a human-readable state string (`"ACTIVE"`, `"PAUSED"`,
 * `"STOPPED"`, `"CRASHED"`, `"INIT"`, `"ERROR"`, `"SPINNING"`, or `"OTHER"`).
 *
 * Return: Static status string describing the process state.
 */
static const char *proc_status_str(const PROCESSINFO *pi, pid_t pid)
{
    pid_t check_pid = (pi->PID > 0) ? pi->PID : pid;
    if (check_pid > 0 && kill(check_pid, 0) != 0 && errno == ESRCH)
    {
        if (pi->loopstat == PROCESSINFO_LOOPSTAT_STOP || pi->CTRLval == PROCESSINFO_CTRLVAL_EXIT)
        {
            return "STOPPED";
        }
        return "CRASHED";
    }
    if (pi->loopstat == PROCESSINFO_LOOPSTAT_STOP || pi->CTRLval == PROCESSINFO_CTRLVAL_EXIT)
    {
        return "STOPPED";
    }
    if (pi->loopstat == PROCESSINFO_LOOPSTAT_PAUSE || pi->CTRLval == PROCESSINFO_CTRLVAL_PAUSE)
    {
        return "PAUSED";
    }
    switch (pi->loopstat)
    {
    case PROCESSINFO_LOOPSTAT_INIT:
        return "INIT";
    case PROCESSINFO_LOOPSTAT_ACTIVE:
        return "ACTIVE";
    case PROCESSINFO_LOOPSTAT_PAUSE:
        return "PAUSED";
    case PROCESSINFO_LOOPSTAT_STOP:
        return "STOPPED";
    case PROCESSINFO_LOOPSTAT_ERROR:
        return "ERROR";
    case PROCESSINFO_LOOPSTAT_SPIN:
        return "SPINNING";
    case PROCESSINFO_LOOPSTAT_CRASHED:
        return "CRASHED";
    default:
        return "OTHER";
    }
}

/**
 * proc_ctrl_str - Convert a numeric PROCESSINFO control code to its symbolic name
 * @ctrlval: Process control value (`PROCESSINFO_CTRLVAL_*`)
 *
 * Translates integer control values (`0..3`) into `"run"`, `"pause"`, `"step"`, or
 * `"stop"`.
 *
 * Return: Static symbolic control name, or `"unknown"`.
 */
static const char *proc_ctrl_str(int ctrlval)
{
    switch (ctrlval)
    {
    case PROCESSINFO_CTRLVAL_RUN:
        return "run";
    case PROCESSINFO_CTRLVAL_PAUSE:
        return "pause";
    case PROCESSINFO_CTRLVAL_INCR:
        return "step";
    case PROCESSINFO_CTRLVAL_EXIT:
        return "stop";
    default:
        return "unknown";
    }
}

static const char *const trigmode_names[] = {
    [PROCESSINFO_TRIGGERMODE_IMMEDIATE]               = "immediate",
    [PROCESSINFO_TRIGGERMODE_CNT0]                    = "cnt0",
    [PROCESSINFO_TRIGGERMODE_CNT1]                    = "cnt1",
    [PROCESSINFO_TRIGGERMODE_SEMAPHORE]               = "semaphore",
    [PROCESSINFO_TRIGGERMODE_DELAY]                   = "delay",
    [PROCESSINFO_TRIGGERMODE_SEMAPHORE_PROP_TIMEOUTS] = "semaphore-timeout",
    [PROCESSINFO_TRIGGERMODE_CNT2]                    = "cnt2",
};
#define NB_TRIGMODES ((long) (sizeof(trigmode_names) / sizeof(trigmode_names[0])))

const char *milk_trigmode_name(long mode)
{
    return (mode >= 0 && mode < NB_TRIGMODES) ? trigmode_names[mode] : NULL;
}

int milk_trigmode_parse(const char *s)
{
    for (long m = 0; m < NB_TRIGMODES; m++)
    {
        if (strcasecmp(s, trigmode_names[m]) == 0)
        {
            return (int) m;
        }
    }
    char *end = NULL;
    long  m   = strtol(s, &end, 10);
    return (end != s && *end == '\0' && milk_trigmode_name(m) != NULL) ? (int) m : -1;
}

/* the loop timing properties, in their --json order; dtmedian_iter_ns is an alias of dt_median_ns */
static const char *const timing_props[] = { "loopfreq",     "hz",        "dt",       "dtmax",
                                            "dt_median_ns", "dt_max_ns", "exectime", "exec_max",
                                            "jitter",       NULL };

/**
 * proc_timing_fmt() - write loop timing property @prop of @pi into @out, computed from its timing ring (a
 * unit does not publish medians itself): Hz for loopfreq/hz, ns for the _ns ones, µs for the others.
 * Return: 0, or -1 if @prop is not a timing property.
 */
static int proc_timing_fmt(const PROCESSINFO *pi, const char *prop, char *out, size_t outsz)
{
    static const struct
    {
        const char *name;
        int         idx;  /* into ts */
        int         unit; /* 0: Hz, 1: ns, 2: µs */
    } tab[] = { { "loopfreq", 0, 0 },  { "hz", 0, 0 },           { "dt", 0, 2 },
                { "dtmax", 1, 2 },     { "dt_median_ns", 0, 1 }, { "dtmedian_iter_ns", 0, 1 },
                { "dt_max_ns", 1, 1 }, { "exectime", 2, 2 },     { "exec_max", 3, 2 },
                { "jitter", 4, 2 } };

    for (size_t i = 0; i < sizeof(tab) / sizeof(tab[0]); i++)
    {
        if (strcmp(prop, tab[i].name) != 0)
        {
            continue;
        }
        /* iteration median and max, execution median and max, jitter, in ns; -1: no samples */
        int64_t ts[5];
        processinfo_timing_stats(pi, &ts[0], &ts[1], &ts[2], &ts[3], &ts[4]);
        int64_t v = ts[tab[i].idx];
        if (tab[i].unit == 0)
        {
            snprintf(out, outsz, "%.1f", (v > 0) ? 1.0e9 / (double) v : 0.0);
        }
        else if (tab[i].unit == 1)
        {
            snprintf(out, outsz, "%ld", (long) v);
        }
        else
        {
            snprintf(out, outsz, "%.1f", (v >= 0) ? (double) v / 1000.0 : 0.0);
        }
        return 0;
    }
    return -1;
}

/**
 * @brief Read a processinfo telemetry property into @out.
 *
 * @param name  Process name in `pinfolist`
 * @param prop  Property key (pid, status, state, ctrl, loopstat, loopcnt, loopfreq, etc.)
 * @param out   Output buffer
 * @param outsz Size of @out in bytes
 * @return 0 on success, -1 if process or property not found
 */
int milk_shm_prop_get_proc(const char *name, const char *prop, char *out, size_t outsz)
{
    pid_t        found_pid = find_active_proc_pid(name);
    int          pfd       = -1;
    PROCESSINFO *pi        = map_proc_shm(name, found_pid, &pfd);
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
    else if (strcmp(prop, "status") == 0 || strcmp(prop, "state") == 0)
    {
        snprintf(out, outsz, "%s", proc_status_str(pi, found_pid));
    }
    else if (strcmp(prop, "ctrl") == 0)
    {
        snprintf(out, outsz, "%s", proc_ctrl_str(pi->CTRLval));
    }
    else if (strcmp(prop, "loopcnt") == 0 || strcmp(prop, "cnt0") == 0)
    {
        snprintf(out, outsz, "%ld", pi->loopcnt);
    }
    else if (strcmp(prop, "loopcntMax") == 0)
    {
        snprintf(out, outsz, "%ld", pi->loopcntMax);
    }
    else if (proc_timing_fmt(pi, prop, out, outsz) == 0)
    {
        /* a loop timing property, written */
    }
    else if (strcmp(prop, "rtprio") == 0)
    {
        snprintf(out, outsz, "%d", pi->RT_priority);
    }
    else if (strcmp(prop, "ctrlval") == 0 || strcmp(prop, "CTRLval") == 0)
    {
        snprintf(out, outsz, "%d", pi->CTRLval);
    }
    else if (strcmp(prop, "trigmode") == 0)
    {
        const char *tm = milk_trigmode_name(pi->triggermode);
        if (tm != NULL)
        {
            snprintf(out, outsz, "%s", tm);
        }
        else
        {
            snprintf(out, outsz, "%d", pi->triggermode);
        }
    }
    else if (strcmp(prop, "triggerstream") == 0 || strcmp(prop, "triggerstreamname") == 0)
    {
        snprintf(out, outsz, "%s", pi->triggerstreamname);
    }
    else if (strcmp(prop, "statusmsg") == 0 || strcmp(prop, "msg") == 0)
    {
        snprintf(out, outsz, "%s", pi->statusmsg);
    }
    else if (strcmp(prop, "unit") == 0)
    {
        snprintf(out, outsz, "%s", pi->unitname);
    }
    else if (strcmp(prop, "tmux") == 0 || strcmp(prop, "tmuxname") == 0)
    {
        fprintf(stderr, "p.%s.%s was renamed p.%s.unit: units have no tmux session\n", name, prop,
                name);
        found = 0;
    }
    else if (strcmp(prop, "description") == 0)
    {
        snprintf(out, outsz, "%s", pi->description);
    }
    else if (strcmp(prop, "timeouts") == 0)
    {
        snprintf(out, outsz, "%lu", (unsigned long) pi->trigggertimeoutcnt);
    }
    else if (strcmp(prop, "missedframes") == 0)
    {
        snprintf(out, outsz, "%lu", (unsigned long) pi->triggermissedframe_cumul);
    }
    else if (strcmp(prop, "list") == 0 || strcmp(prop, "*") == 0)
    {
        snprintf(out, outsz,
                 "pid status state ctrl ctrlval loopstat loopcnt loopcntMax loopfreq hz dt dtmax "
                 "dt_median_ns dt_max_ns exectime exec_max jitter rtprio trigmode triggerstream "
                 "statusmsg unit description missedframes timeouts");
    }
    else if (strcmp(prop, "kv") == 0)
    {
        const char *st = proc_status_str(pi, found_pid);
        const char *ct = proc_ctrl_str(pi->CTRLval);
        const char *tm = milk_trigmode_name(pi->triggermode);
        char        tmnum[16];
        snprintf(tmnum, sizeof(tmnum), "%d", pi->triggermode);
        int pos = snprintf(
            out, outsz,
            "pid\t%d\nPID\t%d\nstatus\t%s\nstate\t%s\nctrl\t%s\nloopstat\t%d\n"
            "loopcnt\t%ld\ncnt0\t%ld\nloopcntMax\t%ld\nrtprio\t%d\nctrlval\t%d\n"
            "trigmode\t%s\ntriggerstream\t%s\nstatusmsg\t%s\nmsg\t%s\nunit\t%s\n"
            "description\t%s\nmissedframes\t%lu\ntimeouts\t%lu",
            (int) pi->PID, (int) pi->PID, st, st, ct, pi->loopstat, pi->loopcnt, pi->loopcnt,
            pi->loopcntMax, pi->RT_priority, pi->CTRLval, (tm != NULL) ? tm : tmnum,
            pi->triggerstreamname, pi->statusmsg, pi->statusmsg, pi->unitname, pi->description,
            (unsigned long) pi->triggermissedframe_cumul, (unsigned long) pi->trigggertimeoutcnt);
        for (int i = 0; timing_props[i] != NULL && pos > 0 && (size_t) pos < outsz; i++)
        {
            char v[32];
            proc_timing_fmt(pi, timing_props[i], v, sizeof(v));
            int n = snprintf(out + pos, outsz - (size_t) pos, "\n%s\t%s", timing_props[i], v);
            if (n < 0)
            {
                break;
            }
            pos += n;
        }
    }
    else
    {
        found = 0;
    }

    munmap(pi, sizeof(PROCESSINFO));
    close(pfd);
    return found ? 0 : -1;
}

/** parse_bool() - 1 for 1/on/true/yes/empty, 0 for 0/off/false/no, -1 otherwise. */
static int parse_bool(const char *v)
{
    if (v == NULL || v[0] == '\0' || strcmp(v, "1") == 0 || strcasecmp(v, "on") == 0 ||
        strcasecmp(v, "true") == 0 || strcasecmp(v, "yes") == 0)
    {
        return 1;
    }
    if (strcmp(v, "0") == 0 || strcasecmp(v, "off") == 0 || strcasecmp(v, "false") == 0 ||
        strcasecmp(v, "no") == 0)
    {
        return 0;
    }
    return -1;
}

/**
 * milk_shm_prop_set_proc - Write process control state on an active processinfo entry
 * @name: Process name in `pinfolist` or `/dev/shm/proc.*.shm`
 * @prop: Control property or action shorthand (`ctrl`, `run`, `pause`, `step`, `stop`, etc.)
 * @val:  Control value string (symbolic action, boolean `1`/`0`, or integer code `0..3`)
 *
 * Resolves @prop and @val into a `PROCESSINFO_CTRLVAL_*` code, maps the target
 * process's `PROCESSINFO` segment, and updates its `CTRLval` field.
 *
 * Return: 0 on success, -1 on invalid arguments or missing process.
 */
int milk_shm_prop_set_proc(const char *name, const char *prop, const char *val)
{
    if (name == NULL || prop == NULL)
    {
        return -1;
    }

    /* action shorthands take a boolean: pause=0 resumes, run=0 pauses, step/stop=0 do nothing */
    int is_run   = (strcasecmp(prop, "run") == 0 || strcasecmp(prop, "resume") == 0);
    int is_pause = (strcasecmp(prop, "pause") == 0);
    int is_other = (strcasecmp(prop, "step") == 0 || strcasecmp(prop, "incr") == 0 ||
                    strcasecmp(prop, "stop") == 0 || strcasecmp(prop, "exit") == 0 ||
                    strcasecmp(prop, "kill") == 0);
    if (is_run || is_pause || is_other)
    {
        int on = parse_bool(val);
        if (on < 0)
        {
            fprintf(stderr, "ERROR: p.%s.%s expects 1/0 (got '%s')\n", name, prop, val);
            return -1;
        }
        if (on == 0)
        {
            if (is_other)
            {
                return 0;
            }
            prop = is_pause ? "run" : "pause";
        }
    }

    int ctrlval_int = -1;
    if (strcasecmp(prop, "run") == 0 || strcasecmp(prop, "resume") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_RUN;
    }
    else if (strcasecmp(prop, "pause") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_PAUSE;
    }
    else if (strcasecmp(prop, "step") == 0 || strcasecmp(prop, "incr") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_INCR;
    }
    else if (strcasecmp(prop, "stop") == 0 || strcasecmp(prop, "exit") == 0 ||
             strcasecmp(prop, "kill") == 0)
    {
        ctrlval_int = PROCESSINFO_CTRLVAL_EXIT;
    }
    else if (strcmp(prop, "ctrl") == 0 || strcmp(prop, "ctrlval") == 0 ||
             strcmp(prop, "CTRLval") == 0 || strcmp(prop, "action") == 0 ||
             strcmp(prop, "status") == 0 || strcmp(prop, "state") == 0)
    {
        if (val == NULL || val[0] == '\0')
        {
            return -1;
        }
        if (strcasecmp(val, "run") == 0 || strcasecmp(val, "resume") == 0 ||
            strcasecmp(val, "active") == 0 || strcmp(val, "0") == 0)
        {
            ctrlval_int = PROCESSINFO_CTRLVAL_RUN;
        }
        else if (strcasecmp(val, "pause") == 0 || strcasecmp(val, "paused") == 0 ||
                 strcmp(val, "1") == 0)
        {
            ctrlval_int = PROCESSINFO_CTRLVAL_PAUSE;
        }
        else if (strcasecmp(val, "step") == 0 || strcasecmp(val, "incr") == 0 ||
                 strcmp(val, "2") == 0)
        {
            ctrlval_int = PROCESSINFO_CTRLVAL_INCR;
        }
        else if (strcasecmp(val, "stop") == 0 || strcasecmp(val, "stopped") == 0 ||
                 strcasecmp(val, "exit") == 0 || strcasecmp(val, "kill") == 0 ||
                 strcmp(val, "3") == 0)
        {
            ctrlval_int = PROCESSINFO_CTRLVAL_EXIT;
        }
    }

    if (ctrlval_int < 0)
    {
        fprintf(stderr, "ERROR: p.%s.%s: invalid value '%s' (run, pause, step, stop)\n", name, prop,
                (val != NULL) ? val : "");
        return -1;
    }

    pid_t        found_pid = find_active_proc_pid(name);
    int          pfd       = -1;
    PROCESSINFO *pi_shm    = map_proc_shm(name, found_pid, &pfd);
    if (pi_shm == NULL)
    {
        fprintf(stderr, "ERROR: p.%s: no such process\n", name);
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
 * @param name          Process name in `pinfolist` (if empty, ensures `pinfolist` is mapped)
 * @param check_running If non-zero, check `loopstat == 1 && CTRLval == 0`
 * @return 1 if exists/running, 0 otherwise
 */
int milk_shm_prop_exists_proc(const char *name, int check_running)
{
    if (name == NULL || name[0] == '\0')
    {
        (void) get_pinfolist_local();
        return 0;
    }
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
    PROCESSINFO *pi_shm = map_proc_shm(name, found_pid, &pfd);
    if (pi_shm == NULL)
    {
        return 0;
    }
    int running = (pi_shm->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE &&
                   pi_shm->CTRLval == PROCESSINFO_CTRLVAL_RUN)
                      ? 1
                      : 0;
    munmap(pi_shm, sizeof(PROCESSINFO));
    close(pfd);
    return running;
}
