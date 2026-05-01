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

/* fps_types.h for FUNCTION_PARAMETER_STRUCT and FPSCMDCODE_* */
#include "fps_types.h"

/* ImageStreamIO for stream open/destroy */
#include "ImageStreamIO/ImageStreamIO.h"

/* Forward-declare FPS connect/disconnect (same pattern as overview_data.c) */
long function_parameter_struct_connect(
    const char               *name,
    FUNCTION_PARAMETER_STRUCT *fps,
    int                        fpsconnectmode);
int function_parameter_struct_disconnect(
    FUNCTION_PARAMETER_STRUCT *fps);

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
    FUNCTION_PARAMETER_STRUCT fps;
    memset(&fps, 0, sizeof(fps));

    long rc = function_parameter_struct_connect(
                  fps_name, &fps, FPSCONNECT_SIMPLE);
    if (rc != 0)
    {
        return -1;
    }

    if (fps.md != NULL)
    {
        fps.md->signal |= (uint64_t) cmd;
    }

    function_parameter_struct_disconnect(&fps);
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
