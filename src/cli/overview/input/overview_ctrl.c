// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_ctrl.c
 * @brief Control-mode actions for FPS and shared memory streams
 *
 * Unit starts, stops and erases run as milk-fps-<op> tools (fps_unit_job_*) that the event loop polls
 * (ov_ctrl_jobs_poll()), so the screen never waits for a unit.
 */

#include "overview_input_internal.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"

#undef STRINGMAXLEN_DIRNAME
#undef STRINGMAXLEN_FULLFILENAME
#undef STRINGMAXLEN_COMMAND
#undef PRINT_ERROR
#include "fps_types.h"
#include "fps_unit.h"

#define OV_CTRL_JOBS 64 /* unit actions running at once (a batch starts one per mark) */

static FPS_UNIT_JOB s_jobs[OV_CTRL_JOBS];


/** ov_ctrl_name_ok - 1 if @n is an FPS name the milk-fps tools take as a name: [A-Za-z0-9_-], no leading '-'. */
static int ov_ctrl_name_ok(const char *n, const char *what, OV_CMDLOG *log)
{
    size_t len = strlen(n);
    int    ok  = len > 0 && n[0] != '-';

    for (size_t i = 0; ok && i < len; i++)
    {
        ok = isalnum((unsigned char) n[i]) || n[i] == '_' || n[i] == '-';
    }
    if (!ok)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS %s refused: name outside [A-Za-z0-9_-]", what);
    }
    return ok;
}

int ov_ctrl_unit_act(const char *name, const char *op, OV_CMDLOG *log)
{
    int slot = -1;

    if (!ov_ctrl_name_ok(name, op, log))
    {
        return -1;
    }
    for (int i = 0; i < OV_CTRL_JOBS; i++)
    {
        if (s_jobs[i].pid > 0 && strcmp(s_jobs[i].name, name) == 0)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\": %s refused, %s still running", name,
                           op, s_jobs[i].op);
            return -1;
        }
        if (s_jobs[i].pid <= 0 && slot < 0)
        {
            slot = i;
        }
    }
    if (slot < 0)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\": %s refused, %d unit actions running", name,
                       op, OV_CTRL_JOBS);
        return -1;
    }
    if (fps_unit_job_start(&s_jobs[slot], name, op) != 0)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\": cannot run milk-fps-%s: %s", name, op,
                       strerror(errno));
        return -1;
    }
    ov_cmdlog_push(log, OV_CMDLOG_INFO, "FPS \"%s\": %s ...", name, op);
    return 0;
}

int ov_ctrl_jobs_poll(OV_CMDLOG *log)
{
    int ended = 0;

    for (int i = 0; i < OV_CTRL_JOBS; i++)
    {
        if (s_jobs[i].pid <= 0)
        {
            continue;
        }
        int rc = fps_unit_job_poll(&s_jobs[i]);
        if (rc < 0)
        {
            continue;
        }
        ended = 1;
        if (s_jobs[i].msg[0] != '\0') // the tool's summary line: "loop run: ready after ..."
        {
            ov_cmdlog_push(log, (rc == 0) ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "%s", s_jobs[i].msg);
        }
        else
        {
            ov_cmdlog_push(log, (rc == 0) ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                           "FPS \"%s\": %s %s (status %d)", s_jobs[i].name, s_jobs[i].op,
                           (rc == 0) ? "done" : "failed", rc);
        }
    }
    if (ended)
    {
        ov_scan_force_update();
    }
    return ended;
}

/**
 * ov_ctrl_fps_run_toggle - start or stop the run process of the unit (milk-fps-runstart / runstop).
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_run_toggle(const OV_FPS *f, OV_CMDLOG *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }
    ov_ctrl_unit_act(
        f->name, fps_unit_state_alive((FPS_UNIT_STATE) f->run_state) ? "runstop" : "runstart", log);
}

/**
 * ov_ctrl_fps_conf_toggle - start or stop the conf process of the unit (milk-fps-confstart / confstop).
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_conf_toggle(const OV_FPS *f, OV_CMDLOG *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }
    ov_ctrl_unit_act(
        f->name, fps_unit_state_alive((FPS_UNIT_STATE) f->conf_state) ? "confstop" : "confstart",
        log);
}

/**
 * ov_ctrl_stream_delete - destroy a shared memory stream.
 * @s:   stream model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_stream_delete(const OV_STREAM *s, OV_CMDLOG *log)
{
    if (s == NULL || !s->valid)
    {
        return;
    }

    /* The file must still be the stream the user selected (same inode) */
    char        fullpath[512];
    struct stat st;
    ImageStreamIO_filename(fullpath, sizeof(fullpath), s->name);
    if (s->inode != 0 && (stat(fullpath, &st) != 0 || (uint64_t) st.st_ino != (uint64_t) s->inode))
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Stream \"%s\" was re-created or removed; not deleted",
                       s->name);
        return;
    }

    /* Evict from scanner cache before destroying semaphores so scan thread doesn't poll it */
    scache_evict_by_name(s->name);

    IMAGE im;
    memset(&im, 0, sizeof(im));

    if (ImageStreamIO_read_sharedmem_image_toIMAGE(s->name, &im) != 0)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Stream \"%s\" — delete failed (open)", s->name);
        }
        return;
    }

    /* Destroy semaphores */
    for (int si = 0; si < im.md->sem; si++)
    {
        sem_destroy(im.semptr[si]);
    }

    /* Close (unmap + close fd) */
    ImageStreamIO_closeIm(&im);

    if (unlink(fullpath) != 0)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "Stream \"%s\" — delete failed (unlink)", s->name);
        }
        return;
    }

    /* Trigger immediate scan to refresh model */
    ov_scan_force_update();

    if (log != NULL)
    {
        ov_cmdlog_push(log, OV_CMDLOG_OK, "Stream \"%s\" — deleted", s->name);
    }
}

/**
 * ov_ctrl_fps_signal_pid - send signal to FPS PIDs.
 * @f:   FPS model entry
 * @sig: signal number
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_signal_pid(const OV_FPS *f, int sig, OV_CMDLOG *log)
{
    if (f == NULL)
    {
        return;
    }
    const char *signame = (sig == SIGTERM) ? "SIGTERM" : (sig == SIGKILL) ? "SIGKILL" : "signal";
    const struct
    {
        const char *role;
        int         alive;
        pid_t       pid;
        uint64_t    ticks;
    } t[2]   = { { "run", f->run_alive, f->runpid, f->run_start_ticks },
                 { "conf", f->conf_alive, f->confpid, f->conf_start_ticks } };
    int sent = 0;
    for (int i = 0; i < 2; i++)
    {
        if (!t[i].alive || t[i].pid <= 0)
        {
            continue;
        }
        if (!pid_same_process(t[i].pid, t[i].ticks))
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\": %s PID %d was reused; not signalled",
                           f->name, t[i].role, (int) t[i].pid);
            continue;
        }
        sent += (kill(t[i].pid, sig) == 0);
    }
    ov_cmdlog_push(log, sent ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "FPS \"%s\": %s %s", f->name, signame,
                   sent ? "sent" : "not sent (no live conf/run process)");
}

/**
 * ov_ctrl_fps_remove - stop conf and run, then remove the FPS and its unit directory (milk-fps-rm -f).
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_remove(const OV_FPS *f, OV_CMDLOG *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }
    ov_ctrl_unit_act(f->name, "rm", log);
}
