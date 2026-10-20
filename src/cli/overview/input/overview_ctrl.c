// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_ctrl.c
 * @brief Control-mode actions for FPS and shared memory streams
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
#include "fps_CONFstart.h"
#include "fps_CONFstop.h"
#include "fps_RUNstart.h"
#include "fps_RUNstop.h"
#include "fps_FPSremove.h"
#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_isvalid.h"
#include "milk_spawn.h"


/**
 * ov_ctrl_fps_action - run an action on an FPS.
 * @fps_name: name of the FPS
 * @action:   function pointer to the action to execute
 *
 * Connects to the FPS, executes the action, and disconnects.
 *
 * Return: 0 on success, -1 on failure
 */
static int ov_ctrl_fps_action(const char *fps_name, errno_t (*action)(FPS *))
{
    if (fps_name == NULL || action == NULL)
    {
        return -1;
    }

    FPS fps;
    memset(&fps, 0, sizeof(fps));

    long rc = fps_connect(fps_name, &fps, FPSCONNECT_SIMPLE);
    if (rc == -1)
    {
        return -1;
    }

    errno_t arc = action(&fps);

    fps_disconnect(&fps);

    return (arc == RETURN_SUCCESS) ? 0 : -1;
}

/**
 * ov_ctrl_stderr_mute - send stderr to /dev/null while libfps runs its tmux commands.
 * Return: saved stderr fd for ov_ctrl_stderr_restore(), or -1 if stderr was left alone.
 */
static int ov_ctrl_stderr_mute(void)
{
    int saved   = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    int devnull = open("/dev/null", O_WRONLY | O_CLOEXEC);

    if (saved < 0 || devnull < 0 || dup2(devnull, STDERR_FILENO) < 0)
    {
        if (saved >= 0)
        {
            close(saved);
        }
        saved = -1;
    }
    if (devnull >= 0)
    {
        close(devnull);
    }
    return saved;
}

/**
 * ov_ctrl_stderr_restore - undo ov_ctrl_stderr_mute().
 * @saved: its return value
 */
static void ov_ctrl_stderr_restore(int saved)
{
    if (saved >= 0)
    {
        dup2(saved, STDERR_FILENO);
        close(saved);
    }
}

/**
 * ov_ctrl_tmux_probe - tell how tmux resolves the bare "-t @name" target that libfps uses.
 * Return: 1 if session =@name exists, 0 if @name matches no session, -1 if it would hit another session.
 */
static int ov_ctrl_tmux_probe(const char *name)
{
    char        exact[STRINGMAXLEN_FPS_NAME + 1];
    const char *argv[] = { "tmux", "has-session", "-t", exact, NULL };

    snprintf(exact, sizeof(exact), "=%s", name);
    if (milkspawn_run(argv, -1, -1) == 0)
    {
        return 1;
    }
    argv[3] = name;
    return (milkspawn_run(argv, -1, -1) == 0) ? -1 : 0;
}

/**
 * ov_ctrl_fps_guard - vet @fps before libfps pastes its strings into sh and bare tmux targets.
 * @f:    selected FPS row
 * @fps:  connected FPS
 * @what: action name for the log
 * @log:  command log (may be NULL)
 *
 * Return: ov_ctrl_tmux_probe() result (1 or 0) if the action may go ahead, -1 if refused (reason logged).
 */
static int ov_ctrl_fps_guard(const OV_FPS *f, FPS *fps, const char *what, OV_CMDLOG *log)
{
    const char *n   = fps->md->name;
    size_t      len = strnlen(n, sizeof(fps->md->name));
    int ok = len > 0 && len < sizeof(fps->md->name) && n[0] != '-' && strcmp(n, f->name) == 0;
    struct stat st;

    for (size_t i = 0; ok && i < len; i++)
    {
        ok = isalnum((unsigned char) n[i]) || n[i] == '_' || n[i] == '-';
    }
    if (!ok)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                       "FPS %s refused: name outside [A-Za-z0-9_-] or not the file name", what);
        return -1;
    }
    if (fstat(fps->SMfd, &st) != 0 || st.st_uid != geteuid())
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\": %s refused, shm file not owned by you", n,
                       what);
        return -1;
    }
    int ts = ov_ctrl_tmux_probe(n);
    if (ts < 0)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                       "FPS \"%s\": %s refused, tmux name matches another session", n, what);
    }
    return ts;
}

/**
 * ov_ctrl_fps_run_toggle - start or stop the FPS run process.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_run_toggle(const OV_FPS *f, OV_CMDLOG *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }

    FPS fps;
    memset(&fps, 0, sizeof(fps));
    if (fps_connect(f->name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\": connect failed", f->name);
        return;
    }

    const char *what = f->run_alive ? "RUNstop" : "RUNstart";
    int         ts   = ov_ctrl_fps_guard(f, &fps, what, log);

    if (ts == 0 && f->run_alive)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                       "FPS \"%s\": no tmux session, RUNstop not sent (k: SIGTERM)", f->name);
    }
    else if (ts >= 0 && !f->run_alive &&
             !(fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK))
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                       "FPS \"%s\": RUNstart not sent, conf check not OK (%u errors)", f->name,
                       (unsigned) fps.md->conferrcnt);
    }
    else if (ts >= 0)
    {
        int     saved = ov_ctrl_stderr_mute();
        errno_t rc =
            f->run_alive ? functionparameter_RUNstop(&fps) : functionparameter_RUNstart(&fps);

        ov_ctrl_stderr_restore(saved);
        ov_cmdlog_push(log, rc == RETURN_SUCCESS ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                       "FPS \"%s\": %s %s", f->name, what,
                       rc == RETURN_SUCCESS ? "sent" : "failed");
    }
    fps_disconnect(&fps);
}

/**
 * ov_ctrl_fps_conf_toggle - start or stop the FPS conf process.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_conf_toggle(const OV_FPS *f, OV_CMDLOG *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }
    if (f->conf_alive)
    {
        int rc = ov_ctrl_fps_action(f->name, functionparameter_CONFstop);
        ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "FPS \"%s\": CONFstop %s",
                       f->name, rc == 0 ? "requested" : "failed");
        return;
    }

    FPS fps;
    memset(&fps, 0, sizeof(fps));
    if (fps_connect(f->name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\": connect failed", f->name);
        return;
    }
    if (ov_ctrl_fps_guard(f, &fps, "CONFstart", log) >= 0)
    {
        int     saved = ov_ctrl_stderr_mute();
        errno_t rc    = functionparameter_CONFstart(&fps);

        ov_ctrl_stderr_restore(saved);
        ov_cmdlog_push(log, rc == RETURN_SUCCESS ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                       "FPS \"%s\": CONFstart %s", f->name,
                       rc == RETURN_SUCCESS ? "sent" : "failed");
    }
    fps_disconnect(&fps);
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
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Stream \"%s\" — delete failed (open)", s->name);
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
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Stream \"%s\" — delete failed (unlink)",
                           s->name);
        }
        return;
    }

    /* Trigger immediate scan to refresh model */
    ov_scan_force_update();

    if (log != NULL)
    {
        ov_cmdlog_push(log, OV_CMDLOG_OK, "🗑️ Stream \"%s\" — deleted", s->name);
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
 * ov_ctrl_fps_remove - stop conf/run then remove FPS.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_remove(const OV_FPS *f, OV_CMDLOG *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }

    FPS fps;
    memset(&fps, 0, sizeof(fps));
    if (fps_connect(f->name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\": erase failed (connect)", f->name);
        return;
    }

    int ts = ov_ctrl_fps_guard(f, &fps, "erase", log);
    if (ts >= 0)
    {
        int saved = ov_ctrl_stderr_mute();

        functionparameter_CONFstop(&fps);
        if (ts == 1)
        {
            functionparameter_RUNstop(&fps); /* without a session RUNstop has nothing to talk to */
        }
        functionparameter_FPSremove(&fps);
        ov_ctrl_stderr_restore(saved);

        int gone = !function_parameter_struct_isvalid(&fps); /* shm file unlinked */
        ov_cmdlog_push(log, gone ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "FPS \"%s\": %s", f->name,
                       gone ? "erased" : "erase failed, shm file still present");
    }
    fps_disconnect(&fps);
}
