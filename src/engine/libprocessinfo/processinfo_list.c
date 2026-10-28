// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_list.c
 * @brief Settle registry slots of processes that are gone: the logic milk-CTRL, procCTRL and the tools share.
 *
 * Slot states (PROCESSINFOLIST.active): 0 free, 1 active, 2 stopped, 3 crashed. Writes are not locked against
 * a process claiming a slot (processinfo_shm_list_create): only slots of dead processes are changed.
 */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "processinfo.h"
#include "processinfo_list.h"
#include "processinfo_procdirname.h"

/* Map @path (at least @size bytes) read-write or read-only; NULL if absent or too small. Nothing is created. */
static void *map_file(const char *path, size_t size, int rw)
{
    int         fd = open(path, (rw ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NONBLOCK);
    struct stat st;
    void       *p = NULL;
    if (fd < 0)
    {
        return NULL;
    }
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && (size_t) st.st_size >= size)
    {
        p = mmap(NULL, size, rw ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
        p = (p == MAP_FAILED) ? NULL : p;
    }
    close(fd);
    return p;
}

static PROCESSINFOLIST *map_list(int rw)
{
    char dir[PATH_MAX], path[PATH_MAX + 32];
    processinfo_procdirname(dir);
    snprintf(path, sizeof(path), "%s/processinfo.list.shm", dir);
    return (PROCESSINFOLIST *) map_file(path, sizeof(PROCESSINFOLIST), rw);
}

/* Seconds since the epoch at which @pid started, -1 if unknown */
static double pid_start_s(pid_t pid)
{
    static double btime = -1;
    char          buf[1024], path[64];
    if (btime < 0)
    {
        FILE *f = fopen("/proc/stat", "re");
        while (f != NULL && fgets(buf, sizeof(buf), f) != NULL)
        {
            if (strncmp(buf, "btime ", 6) == 0)
            {
                btime = strtod(buf + 6, NULL);
            }
        }
        if (f != NULL)
        {
            fclose(f);
        }
    }
    snprintf(path, sizeof(path), "/proc/%d/stat", (int) pid);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0 || btime < 0)
    {
        if (fd >= 0)
        {
            close(fd);
        }
        return -1;
    }
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    buf[(n > 0) ? n : 0] = '\0';
    char *p              = strrchr(buf, ')'); /* comm may contain spaces */
    for (int field = 2; p != NULL && field < 22; field++)
    {
        p = strchr(p + 1, ' ');
    }
    return (p == NULL) ? -1 : btime + strtod(p + 1, NULL) / (double) sysconf(_SC_CLK_TCK);
}

/**
 * processinfo_pid_gone - 1 if @pid no longer runs the process registered at @createtime (s since the epoch,
 * 0 = unknown): it does not exist (ESRCH), or it started after @createtime (PID reused). 0 otherwise.
 */
int processinfo_pid_gone(pid_t pid, double createtime)
{
    if (pid <= 0)
    {
        return 1;
    }
    if (kill(pid, 0) != 0 && errno == ESRCH)
    {
        return 1;
    }
    double start = (createtime > 0) ? pid_start_s(pid) : -1;
    return start > createtime + 1.0; /* 1 s slack: btime is rounded */
}

/**
 * processinfo_list_reap - settle the slots of processes that are gone: free them when their proc file is gone
 * (or with PROCESSINFO_REAP_FREE_SLOTS), else mark them stopped or crashed from the file's loop state;
 * PROCESSINFO_REAP_UNLINK removes the proc files first. Return: slots changed, -1 without a registry.
 */
int processinfo_list_reap(unsigned flags)
{
    PROCESSINFOLIST *pl = map_list(1);
    char             dir[PATH_MAX], path[PATH_MAX + 128];
    int              n = 0;
    if (pl == NULL)
    {
        return -1;
    }
    processinfo_procdirname(dir);
    for (long i = 0; i < PROCESSINFOLISTSIZE; i++)
    {
        pid_t pid = pl->PIDarray[i];
        if (pl->active[i] == 0 || !processinfo_pid_gone(pid, pl->createtime[i]))
        {
            continue;
        }
        snprintf(path, sizeof(path), "%s/proc.%.*s.%06d.shm", dir, (int) sizeof(pl->pnamearray[i]),
                 pl->pnamearray[i], (int) pid);
        if (flags & PROCESSINFO_REAP_UNLINK)
        {
            unlink(path);
        }
        int                newstate = 0;
        const PROCESSINFO *pi = (flags & (PROCESSINFO_REAP_FREE_SLOTS | PROCESSINFO_REAP_UNLINK))
                                    ? NULL
                                    : (const PROCESSINFO *) map_file(path, sizeof(PROCESSINFO), 0);
        if (pi != NULL)
        {
            newstate = (pi->loopstat == PROCESSINFO_LOOPSTAT_STOP) ? 2 : 3;
            munmap((void *) pi, sizeof(PROCESSINFO));
        }
        if (pl->active[i] != newstate)
        {
            pl->active[i] = newstate;
            n++;
        }
    }
    munmap(pl, sizeof(PROCESSINFOLIST));
    return n;
}

/**
 * processinfo_list_clear_slot - free registry slot @pindex if it still holds @pid.
 * Return: 1 freed, 0 the slot holds another process, -1 no registry or bad index.
 */
int processinfo_list_clear_slot(long pindex, pid_t pid)
{
    if (pindex < 0 || pindex >= PROCESSINFOLISTSIZE)
    {
        return -1;
    }
    PROCESSINFOLIST *pl = map_list(1);
    if (pl == NULL)
    {
        return -1;
    }
    int rc = (pl->PIDarray[pindex] == pid);
    if (rc)
    {
        pl->active[pindex] = 0;
    }
    munmap(pl, sizeof(PROCESSINFOLIST));
    return rc;
}
