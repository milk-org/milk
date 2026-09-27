// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_ctrl.c
 * @brief Control-mode actions for FPS and shared memory streams
 */

#include "overview_input_internal.h"
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
#include "fps_tmux.h"

int pid_is_stopped(pid_t pid);

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

    if (f->run_alive)
    {
        int rc = ov_ctrl_fps_action(f->name, functionparameter_RUNstop);
        if (log != NULL)
        {
            ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "FPS \"%s\" — RUNstop %s",
                           f->name, rc == 0 ? "succeeded" : "failed");
        }
    }
    else
    {
        FPS fps;
        memset(&fps, 0, sizeof(fps));

        long crc = fps_connect(f->name, &fps, FPSCONNECT_SIMPLE);
        if (crc == -1)
        {
            if (log != NULL)
            {
                ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\" — RUNstart connect failed",
                               f->name);
            }
            return;
        }

        int saved_stderr = dup(STDERR_FILENO);
        {
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0)
            {
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
        }

        functionparameter_FPS_tmux_ensure(&fps);

        if (saved_stderr >= 0)
        {
            dup2(saved_stderr, STDERR_FILENO);
            close(saved_stderr);
        }

        errno_t arc = functionparameter_RUNstart(&fps);
        fps_disconnect(&fps);

        int rc = (arc == RETURN_SUCCESS) ? 0 : -1;
        if (log != NULL)
        {
            ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "FPS \"%s\" — RUNstart %s",
                           f->name, rc == 0 ? "succeeded" : "failed");
        }
    }
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
        if (log != NULL)
        {
            ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "FPS \"%s\" — CONFstop %s",
                           f->name, rc == 0 ? "succeeded" : "failed");
        }
    }
    else
    {
        FPS fps;
        memset(&fps, 0, sizeof(fps));

        long crc = fps_connect(f->name, &fps, FPSCONNECT_SIMPLE);
        if (crc == -1)
        {
            if (log != NULL)
            {
                ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\" — CONFstart connect failed",
                               f->name);
            }
            return;
        }

        int saved_stderr = dup(STDERR_FILENO);
        {
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0)
            {
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
        }

        functionparameter_FPS_tmux_ensure(&fps);

        if (saved_stderr >= 0)
        {
            dup2(saved_stderr, STDERR_FILENO);
            close(saved_stderr);
        }

        errno_t arc = functionparameter_CONFstart(&fps);
        fps_disconnect(&fps);

        int rc = (arc == RETURN_SUCCESS) ? 0 : -1;
        if (log != NULL)
        {
            ov_cmdlog_push(log, rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                           "FPS \"%s\" — CONFstart %s", f->name, rc == 0 ? "succeeded" : "failed");
        }
    }
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

    char fullpath[512];
    ImageStreamIO_filename(fullpath, sizeof(fullpath), s->name);

    if (unlink(fullpath) != 0)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "🚫 Stream \"%s\" — delete failed (unlink)",
                           s->name);
        }
        return;
    }

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
    int ok = 0;
    if (f->run_alive && f->runpid > 0)
    {
        if (kill(f->runpid, sig) == 0)
        {
            ok = 1;
        }
    }
    if (f->conf_alive && f->confpid > 0)
    {
        if (kill(f->confpid, sig) == 0)
        {
            ok = 1;
        }
    }
    if (log != NULL)
    {
        const char *signame = (sig == SIGTERM)   ? "SIGTERM"
                              : (sig == SIGKILL) ? "SIGKILL"
                                                 : "signal";
        ov_cmdlog_push(log, ok ? OV_CMDLOG_OK : OV_CMDLOG_FAIL, "FPS \"%s\" — %s sent", f->name,
                       signame);
    }
}

/**
 * ov_ctrl_fps_pause_toggle - toggle SIGSTOP/SIGCONT for FPS run and conf processes.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_pause_toggle(const OV_FPS *f, OV_CMDLOG *log)
{
    if (f == NULL)
    {
        return;
    }
    /* Use runpid state to decide direction */
    int stopped = 0;
    if (f->run_alive && f->runpid > 0)
    {
        stopped = pid_is_stopped(f->runpid);
    }
    int sig = stopped ? SIGCONT : SIGSTOP;
    if (f->run_alive && f->runpid > 0)
    {
        kill(f->runpid, sig);
    }
    if (f->conf_alive && f->confpid > 0)
    {
        kill(f->confpid, sig);
    }
    if (log != NULL)
    {
        ov_cmdlog_push(log, OV_CMDLOG_OK, "%s FPS \"%s\" — %s", stopped ? "⏯️" : "⏸️", f->name,
                       stopped ? "resumed" : "paused");
    }
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

    long rc = fps_connect(f->name, &fps, FPSCONNECT_SIMPLE);
    if (rc == -1)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL, "FPS \"%s\" — erase failed (connect)", f->name);
        }
        return;
    }

    /* Suppress stderr from tmux commands to avoid TUI display corruptions */
    int saved_stderr = dup(STDERR_FILENO);
    {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0)
        {
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
    }

    functionparameter_CONFstop(&fps);
    functionparameter_RUNstop(&fps);
    functionparameter_FPSremove(&fps);

    /* Restore stderr */
    if (saved_stderr >= 0)
    {
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
    }

    fps_disconnect(&fps);

    if (log != NULL)
    {
        ov_cmdlog_push(log, OV_CMDLOG_OK, "🗑️ FPS \"%s\" — erased", f->name);
    }
}
