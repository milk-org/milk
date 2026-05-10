/**
 * @file overview_ctrl.c
 * @brief Control-mode actions for milkCTRL
 *
 * Implements write operations available when CONTROL mode is ON:
 *   - FPS: toggle run process (r key), toggle conf process (s key)
 *   - Stream: delete SHM (d key)
 *   - Process: send SIGTERM (k key)
 */

#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>

#include "overview_defs.h"
#include "overview_data.h"
#include "overview_layout.h"

/* fps_types.h for FPS and FPSCMDCODE_* */
#include "fps_types.h"

errno_t functionparameter_CONFstop(FPS *fps);
errno_t functionparameter_RUNstop(FPS *fps);
errno_t functionparameter_FPSremove(FPS *fps);

/* ImageStreamIO for stream open/destroy */
#include "ImageStreamIO/ImageStreamIO.h"

/* Forward-declare FPS connect/disconnect (same pattern as overview_data.c) */
long fps_connect(
    const char               *name,
    FPS *fps,
    int                        fpsconnectmode);
int fps_disconnect(
    FPS *fps);

/* =========================================================
 * Internal FPS SHM signal helper
 * ========================================================= */

/**
 * ov_ctrl_fps_signal - write a command code into the FPS SHM.
 * @fps_name: FPS instance name
 * @cmd:      FPSCMDCODE_* value to OR into md->signal
 *
 * Opens the FPS in simple (read/write) mode, ORs the command,
 * and immediately disconnects.
 *
 * Return: 0 on success, -1 if the SHM could not be opened.
 */
static int ov_ctrl_fps_signal(
    const char *fps_name,
    uint32_t    cmd)
{
    FPS fps;
    memset(&fps, 0, sizeof(fps));

    long rc = fps_connect(
                  fps_name, &fps, FPSCONNECT_SIMPLE);
    if (rc == -1)
    {
        return -1;
    }

    if (fps.md != NULL)
    {
        fps.md->signal |= (uint64_t) cmd;
    }

    fps_disconnect(&fps);
    return 0;
}

/* =========================================================
 * Public control actions
 * ========================================================= */

/**
 * ov_ctrl_fps_run_toggle - start or stop the FPS run process.
 * @f: FPS model entry (read-only snapshot from OV_MODEL)
 *
 * Sends FPSCMDCODE_RUNSTART when run is not alive,
 * FPSCMDCODE_RUNSTOP when it is.
 */
void ov_ctrl_fps_run_toggle(const OV_FPS *f)
{
    if (f == NULL || !f->valid)
    {
        return;
    }

    uint32_t cmd = f->run_alive
                   ? FPSCMDCODE_RUNSTOP
                   : FPSCMDCODE_RUNSTART;

    ov_ctrl_fps_signal(f->name, cmd);
}

/**
 * ov_ctrl_fps_conf_toggle - start or stop the FPS conf process.
 * @f: FPS model entry (read-only snapshot from OV_MODEL)
 *
 * Sends FPSCMDCODE_CONFSTART when conf is not alive,
 * FPSCMDCODE_CONFSTOP when it is.
 */
void ov_ctrl_fps_conf_toggle(const OV_FPS *f)
{
    if (f == NULL || !f->valid)
    {
        return;
    }

    uint32_t cmd = f->conf_alive
                   ? FPSCMDCODE_CONFSTOP
                   : FPSCMDCODE_CONFSTART;

    ov_ctrl_fps_signal(f->name, cmd);
}

/**
 * ov_ctrl_stream_delete - destroy a shared memory stream.
 * @s: stream model entry (read-only snapshot from OV_MODEL)
 *
 * Opens the stream SHM, calls ImageStreamIO_destroyIm() to
 * unmap and unlink the file, then marks the IMAGE as freed.
 */
void ov_ctrl_stream_delete(const OV_STREAM *s)
{
    if (s == NULL || !s->valid)
    {
        return;
    }

    IMAGE im;
    memset(&im, 0, sizeof(im));

    if (ImageStreamIO_openIm(&im, s->name) != 0)
    {
        return;
    }

    ImageStreamIO_destroyIm(&im);
}

/**
 * ov_ctrl_proc_kill - send SIGTERM to a process.
 * @p: process model entry (read-only snapshot from OV_MODEL)
 *
 * Sends SIGTERM to p->PID.  No error is reported if the
 * process has already exited (kill returns ESRCH).
 */
void ov_ctrl_proc_kill(const OV_PROC *p)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }

    kill(p->PID, SIGTERM);
}

/**
 * ov_ctrl_proc_sigkill - send SIGKILL to a process.
 * @p: process model entry
 */
void ov_ctrl_proc_sigkill(const OV_PROC *p)
{
    if (p == NULL || p->PID <= 0) return;
    kill(p->PID, SIGKILL);
}

/**
 * pid_is_stopped - check if a process is in 'T' state.
 * @pid: process PID
 *
 * Reads /proc/[pid]/stat and returns 1 if the state
 * character is 'T' (stopped), 0 otherwise.
 */
static int pid_is_stopped(pid_t pid)
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
    int p;
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
 * ov_ctrl_proc_pause_toggle - toggle SIGSTOP/SIGCONT
 * for a process.
 * @p: process model entry
 */
void ov_ctrl_proc_pause_toggle(const OV_PROC *p)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }
    kill(p->PID, pid_is_stopped(p->PID)
                 ? SIGCONT : SIGSTOP);
}

/**
 * ov_ctrl_fps_signal_pid - send signal to FPS PIDs.
 * @f:   FPS model entry
 * @sig: signal number
 */
void ov_ctrl_fps_signal_pid(const OV_FPS *f, int sig)
{
    if (f == NULL)
    {
        return;
    }
    if (f->run_alive && f->runpid > 0)
    {
        kill(f->runpid, sig);
    }
    if (f->conf_alive && f->confpid > 0)
    {
        kill(f->confpid, sig);
    }
}

/**
 * ov_ctrl_fps_pause_toggle - toggle SIGSTOP/SIGCONT
 * for FPS run and conf processes.
 * @f: FPS model entry
 */
void ov_ctrl_fps_pause_toggle(const OV_FPS *f)
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
}

/**
 * ov_ctrl_fps_remove - stop conf/run then remove the FPS SHM.
 * @f: FPS model entry (read-only snapshot from OV_MODEL)
 *
 * Connects to the FPS, sends CONFstop and RUNstop, then
 * removes the shared-memory file and associated tmux session.
 * Mirrors fpsCTRL's ctrl+e behaviour.
 */
void ov_ctrl_fps_remove(const OV_FPS *f)
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
        return;
    }

    functionparameter_CONFstop(&fps);
    functionparameter_RUNstop(&fps);
    functionparameter_FPSremove(&fps);

    fps_disconnect(&fps);
}
