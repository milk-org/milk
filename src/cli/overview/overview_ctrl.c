/**
 * @file overview_ctrl.c
 * @brief Control-mode actions for milk-CTRL
 *
 * Implements write operations available when CONTROL mode is ON:
 *   - FPS: toggle run process (r key), toggle conf process (s key)
 *   - Stream: delete SHM (d key)
 *   - Process: send SIGTERM (k key)
 *
 * Each function posts a status message to the OV_CMDLOG ring
 * buffer so the user gets visual feedback on what happened.
 */

#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "overview_defs.h"
#include "overview_data.h"
#include "overview_layout.h"
#include "overview_ctrl.h"
#include "overview_ansi.h"

/* fps_types.h for FPS and FPSCMDCODE_* */
#include "fps_types.h"

/* EXECUTE_SYSTEM_COMMAND_NOCHECK macro */
#undef STRINGMAXLEN_DIRNAME
#undef STRINGMAXLEN_FULLFILENAME
#undef STRINGMAXLEN_COMMAND
#undef PRINT_ERROR
#include "milkDebugTools.h"

/**
 * @brief Forward declaration: start FPS configuration.
 */
errno_t functionparameter_CONFstart(FPS *fps);
/**
 * @brief Forward declaration: stop FPS configuration.
 */
errno_t functionparameter_CONFstop(FPS *fps);
/**
 * @brief Forward declaration: start FPS run process.
 */
errno_t functionparameter_RUNstart(FPS *fps);
errno_t functionparameter_RUNstop(FPS *fps);
errno_t functionparameter_FPSremove(FPS *fps);
int functionparameter_FPS_tmux_ensure(FPS *fps);

/* ImageStreamIO for stream open/destroy */
#include "ImageStreamIO/ImageStreamIO.h"

/* Forward-declare FPS connect/disconnect */
long fps_connect(
    const char *name,
    FPS        *fps,
    int        fpsconnectmode);
int fps_disconnect(
    FPS *fps);

/* =========================================================
 * Internal FPS action helper
 * ========================================================= */

/**
 * ov_ctrl_fps_action - perform an FPS function parameter action
 * @fps_name: FPS instance name
 * @action:   Function to execute
 *
 * Opens the FPS in simple mode, runs the action, and disconnects.
 * Suppresses stderr to prevent tmux errors from corrupting TUI.
 *
 * Return: 0 on success, -1 if the SHM could not be opened.
 */
static int ov_ctrl_fps_action(
    const char *fps_name,
    errno_t (*action)(FPS *))
{
    FPS fps;
    memset(&fps, 0, sizeof(fps));

    long rc = fps_connect(
                  fps_name, &fps, FPSCONNECT_SIMPLE);
    if (rc == -1)
    {
        return -1;
    }

    /* Suppress stderr from tmux commands to avoid TUI corruption */
    int saved_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0)
    {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }

    action(&fps);

    /* Restore stderr */
    if (saved_stderr >= 0)
    {
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
    }

    fps_disconnect(&fps);
    return 0;
}

/* =========================================================
 * Public control actions
 * ========================================================= */

/**
 * ov_ctrl_fps_run_toggle - start or stop the FPS run.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 *
 * For runstart, we bypass functionparameter_RUNstart()
 * because it gates on CHECKOK and silently no-ops if
 * conf hasn't validated parameters yet. Instead we
 * directly send tmux commands, mirroring fpsCTRL's 'R'
 * key handler.  For runstop, we delegate to
 * functionparameter_RUNstop() which has no such gate.
 */
void ov_ctrl_fps_run_toggle(
    const OV_FPS *f,
    OV_CMDLOG    *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }

    if (f->run_alive)
    {
        /* --- RUN stop (no CHECKOK gate) --- */
        int rc = ov_ctrl_fps_action(
                     f->name, functionparameter_RUNstop);
        if (log != NULL)
        {
            ov_cmdlog_push(log,
                           rc == 0 ? OV_CMDLOG_OK
                                   : OV_CMDLOG_FAIL,
                           "⏹️ FPS \"%s\" — RUN stop",
                           f->name);
        }
        return;
    }

    /* --- RUN start: direct tmux dispatch --- */
    FPS fps;
    memset(&fps, 0, sizeof(fps));

    long rc = fps_connect(
                  f->name, &fps, FPSCONNECT_SIMPLE);
    if (rc == -1)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "▶️ FPS \"%s\" — RUN start"
                           " failed (connect)",
                           f->name);
        }
        return;
    }

    /* Suppress stderr from tmux commands */
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

    /* cd to workdir */
    EXECUTE_SYSTEM_COMMAND_NOCHECK(
        "tmux send-keys -t %s:run \" cd %s\" C-m",
        fps.md->name, fps.md->workdir);

    /* Determine executable */
    char progexec[1024];
    {
        const char *ep = fps.md->execfullpath;
        char *bn = strrchr(ep, '/');
        const char *base = bn ? bn + 1 : ep;
        if (strlen(ep) > 0
            && strcmp(base, "unknown") != 0
            && strcmp(base, "milk") != 0
            && strcmp(base, "cacao") != 0)
        {
            strncpy(progexec, ep, sizeof(progexec) - 1);
            progexec[sizeof(progexec) - 1] = '\0';
        }
        else
        {
            snprintf(progexec, sizeof(progexec),
                     "%s-exec",
                     fps.md->callprogname);
        }
    }

    /* Send run command */
    EXECUTE_SYSTEM_COMMAND_NOCHECK(
        "tmux send-keys -t %s:run \" %s %s:runstart\""
        " C-m",
        fps.md->name, progexec, fps.md->name);

    fps.md->status |=
        FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    fps.md->signal |=
        FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

    /* Restore stderr */
    if (saved_stderr >= 0)
    {
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
    }

    fps_disconnect(&fps);

    if (log != NULL)
    {
        ov_cmdlog_push(log, OV_CMDLOG_OK,
                       "▶️ FPS \"%s\" — RUN start",
                       f->name);
    }
}

/**
 * ov_ctrl_fps_conf_toggle - start or stop the FPS conf.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_conf_toggle(
    const OV_FPS *f,
    OV_CMDLOG    *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }

    errno_t (*action_fn)(FPS *) = f->conf_alive
                                  ? functionparameter_CONFstop
                                  : functionparameter_CONFstart;
    const char *action = f->conf_alive
                         ? "CONF stop" : "CONF start";

    int rc = ov_ctrl_fps_action(f->name, action_fn);
    if (log != NULL)
    {
        ov_cmdlog_push(log,
                       rc == 0 ? OV_CMDLOG_OK
                               : OV_CMDLOG_FAIL,
                       "%s FPS \"%s\" — %s",
                       f->conf_alive ? "⏹️" : "▶️",
                       f->name, action);
    }
}

/**
 * ov_ctrl_stream_delete - destroy a shared memory stream.
 * @s:   stream model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_stream_delete(
    const OV_STREAM *s,
    OV_CMDLOG       *log)
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
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "🚫 Stream \"%s\" — delete"
                           " failed (open)",
                           s->name);
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
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "🚫 Stream \"%s\" — delete"
                           " failed (unlink)",
                           s->name);
        }
        return;
    }

    if (log != NULL)
    {
        ov_cmdlog_push(log, OV_CMDLOG_OK,
                       "🗑️ Stream \"%s\" — deleted",
                       s->name);
    }
}

/**
 * ov_ctrl_proc_kill - send SIGTERM to a process.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_kill(
    const OV_PROC *p,
    OV_CMDLOG     *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }

    int rc = kill(p->PID, SIGTERM);
    if (log != NULL)
    {
        ov_cmdlog_push(log,
                       rc == 0 ? OV_CMDLOG_OK
                               : OV_CMDLOG_FAIL,
                       "💀 Process \"%s\" (PID %d)"
                       " — SIGTERM",
                       p->name, p->PID);
    }
}

/**
 * ov_ctrl_proc_sigkill - send SIGKILL to a process.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_sigkill(
    const OV_PROC *p,
    OV_CMDLOG     *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }
    int rc = kill(p->PID, SIGKILL);
    if (log != NULL)
    {
        ov_cmdlog_push(log,
                       rc == 0 ? OV_CMDLOG_OK
                               : OV_CMDLOG_FAIL,
                       "Process \"%s\" (PID %d)"
                       " — SIGKILL",
                       p->name, p->PID);
    }
}

#include <errno.h>

/**
 * ov_ctrl_proc_set_ctrlval - mutate process CTRLval.
 * @p:   process model entry
 * @val: new value (-1 to toggle between 0 and 1)
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_set_ctrlval(
    const OV_PROC *p,
    int           val,
    OV_CMDLOG     *log)
{
    if (p == NULL || p->PID <= 0 || !p->valid)
    {
        return;
    }

    char fname[1024];
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm",
             ov_get_shmdir(), p->name, (int)p->PID);

    int fd = open(fname, O_RDWR);
    if (fd < 0)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "🚫 Process \"%s\" — ctrl failed (open)", p->name);
        }
        return;
    }

    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size < (off_t)sizeof(PROCESSINFO))
    {
        close(fd);
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "🚫 Process \"%s\" — ctrl failed (stat)", p->name);
        }
        return;
    }

    PROCESSINFO *pinfo = (PROCESSINFO *)mmap(NULL, sizeof(PROCESSINFO),
                                             PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pinfo == MAP_FAILED)
    {
        close(fd);
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "🚫 Process \"%s\" — ctrl failed (mmap)", p->name);
        }
        return;
    }

    int old_val = pinfo->CTRLval;
    int new_val = (val == -1) ? (old_val == 0 ? 1 : 0) : val;
    pinfo->CTRLval = new_val;

    munmap(pinfo, sizeof(PROCESSINFO));
    close(fd);

    if (log != NULL)
    {
        const char *action;
        const char *emoji = "⚡";
        if (new_val == 0) { action = "Resume"; emoji = "⏯️"; }
        else if (new_val == 1) { action = "Pause"; emoji = "⏸️"; }
        else if (new_val == 2) { action = "Step"; emoji = "⏭️"; }
        else if (new_val == 3) { action = "Exit request"; emoji = "⏹️"; }
        else { action = "CTRLval updated"; }

        ov_cmdlog_push(log, OV_CMDLOG_OK,
                       "%s Process \"%s\" — %s", emoji, p->name, action);
    }
}

/**
 * ov_ctrl_proc_zero_counters - reset process loopcnt.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_zero_counters(
    const OV_PROC *p,
    OV_CMDLOG     *log)
{
    if (p == NULL || p->PID <= 0 || !p->valid)
    {
        return;
    }

    char fname[1024];
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm",
             ov_get_shmdir(), p->name, (int)p->PID);

    int fd = open(fname, O_RDWR);
    if (fd < 0)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "🚫 Process \"%s\" — zero failed (open)", p->name);
        }
        return;
    }

    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size < (off_t)sizeof(PROCESSINFO))
    {
        close(fd);
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "🚫 Process \"%s\" — zero failed (stat)", p->name);
        }
        return;
    }

    PROCESSINFO *pinfo = (PROCESSINFO *)mmap(NULL, sizeof(PROCESSINFO),
                                             PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pinfo == MAP_FAILED)
    {
        close(fd);
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "🚫 Process \"%s\" — zero failed (mmap)", p->name);
        }
        return;
    }

    pinfo->loopcnt = 0;

    munmap(pinfo, sizeof(PROCESSINFO));
    close(fd);

    if (log != NULL)
    {
        ov_cmdlog_push(log, OV_CMDLOG_OK,
                       "0️⃣ Process \"%s\" — Counters zeroed", p->name);
    }
}

/**
 * ov_ctrl_proc_remove - remove a single process from shm.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_remove(
    const OV_PROC *p,
    OV_CMDLOG     *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }

    if (kill(p->PID, 0) == 0 || errno == EPERM)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log,
                           OV_CMDLOG_FAIL,
                           "🚫 Process \"%s\" (PID %d) is still alive",
                           p->name, p->PID);
        }
        return;
    }

    char fname[1024];
    snprintf(fname, sizeof(fname), "%s/proc.%s.%06d.shm",
             ov_get_shmdir(), p->name, (int)p->PID);

    int rc = unlink(fname);
    
    if (log != NULL)
    {
        if (rc == 0)
        {
            ov_cmdlog_push(log,
                           OV_CMDLOG_OK,
                           "file %s removed 🗑",
                           fname);
        }
        else
        {
            ov_cmdlog_push(log,
                           OV_CMDLOG_FAIL,
                           "failed to remove file %s",
                           fname);
        }
    }
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
 * ov_ctrl_proc_pause_toggle - toggle SIGSTOP/SIGCONT.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_pause_toggle(
    const OV_PROC *p,
    OV_CMDLOG     *log)
{
    if (p == NULL || p->PID <= 0)
    {
        return;
    }
    int stopped = pid_is_stopped(p->PID);
    int sig = stopped ? SIGCONT : SIGSTOP;
    int rc = kill(p->PID, sig);
    if (log != NULL)
    {
        ov_cmdlog_push(log,
                       rc == 0 ? OV_CMDLOG_OK
                               : OV_CMDLOG_FAIL,
                       "%s Process \"%s\" (PID %d) — %s",
                       stopped ? "⏯️" : "⏸️",
                       p->name, p->PID,
                       stopped ? "resumed"
                               : "paused");
    }
}

/**
 * ov_ctrl_fps_signal_pid - send signal to FPS PIDs.
 * @f:   FPS model entry
 * @sig: signal number
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_signal_pid(
    const OV_FPS *f,
    int          sig,
    OV_CMDLOG    *log)
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
        const char *signame =
            (sig == SIGTERM) ? "SIGTERM"
            : (sig == SIGKILL) ? "SIGKILL"
            : "signal";
        ov_cmdlog_push(log,
                       ok ? OV_CMDLOG_OK
                          : OV_CMDLOG_FAIL,
                       "FPS \"%s\" — %s sent",
                       f->name, signame);
    }
}

/**
 * ov_ctrl_fps_pause_toggle - toggle SIGSTOP/SIGCONT
 * for FPS run and conf processes.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_pause_toggle(
    const OV_FPS *f,
    OV_CMDLOG    *log)
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
        ov_cmdlog_push(log, OV_CMDLOG_OK,
                       "%s FPS \"%s\" — %s",
                       stopped ? "⏯️" : "⏸️",
                       f->name,
                       stopped ? "resumed"
                               : "paused");
    }
}

/**
 * ov_ctrl_fps_remove - stop conf/run then remove FPS.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 *
 * The underlying FPS functions call system("tmux ...")
 * which can print "can't find window" errors to stderr.
 * We temporarily redirect stderr to /dev/null to prevent
 * TUI corruption, then restore it.
 */
void ov_ctrl_fps_remove(
    const OV_FPS *f,
    OV_CMDLOG    *log)
{
    if (f == NULL || !f->valid)
    {
        return;
    }

    FPS fps;
    memset(&fps, 0, sizeof(fps));

    long rc = fps_connect(
                  f->name, &fps, FPSCONNECT_SIMPLE);
    if (rc == -1)
    {
        if (log != NULL)
        {
            ov_cmdlog_push(log, OV_CMDLOG_FAIL,
                           "FPS \"%s\" — erase"
                           " failed (connect)",
                           f->name);
        }
        return;
    }

    /* Suppress stderr from tmux commands to avoid
     * "can't find window/session" messages that
     * corrupt the TUI display. */
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
        ov_cmdlog_push(log, OV_CMDLOG_OK,
                       "🗑️ FPS \"%s\" — erased",
                       f->name);
    }
}

/**
 * ov_ctrl_procs_cleanup - remove crashed/stopped processes
 * @log: command log (may be NULL)
 */
void ov_ctrl_procs_cleanup(
    OV_CMDLOG *log)
{
    /* Silently remove crashed/stopped procinfo entries */
    int rc = system("milk-procinfo-rm -c >/dev/null 2>&1");
    if (log != NULL)
    {
        ov_cmdlog_push(log,
                       rc == 0 ? OV_CMDLOG_OK : OV_CMDLOG_FAIL,
                       "🧹 Process cleanup requested");
    }
}

/**
 * ov_ctrl_inspect_item - spawn an interactive detailed view
 * @panel: the active panel type
 * @item:  pointer to the selected item (OV_STREAM, OV_PROC, or OV_FPS)
 */
void ov_ctrl_inspect_item(
    ov_focus_t panel,
    const void *item)
{
    if (item == NULL)
    {
        return;
    }

    char cmd[512] = {0};

    if (panel == OV_FOCUS_STREAMS)
    {
        const OV_STREAM *s = (const OV_STREAM *)item;
        snprintf(cmd, sizeof(cmd), "milk-stream-info %s | less -R", s->name);
    }
    else if (panel == OV_FOCUS_PROCS)
    {
        const OV_PROC *p = (const OV_PROC *)item;
        snprintf(cmd, sizeof(cmd), "milk-procinfo-info %s | less -R", p->name);
    }
    else if (panel == OV_FOCUS_FPS)
    {
        const OV_FPS *f = (const OV_FPS *)item;
        snprintf(cmd, sizeof(cmd), "milk-fps-info %s | less -R", f->name);
    }
    else
    {
        return;
    }

    /* Suspend TUI */
    ov_raw_mode_exit();
    int rc_clear = system("clear");
    (void)rc_clear;
    
    /* Spawn interactive diagnostic tool */
    int rc_cmd = system(cmd);
    (void)rc_cmd;
    
    /* Resume TUI */
    ov_raw_mode_enter();
    ov_buf_force_clear();
}

