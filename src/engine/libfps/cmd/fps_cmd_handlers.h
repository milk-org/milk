/**
 * @file fps_cmd_handlers.h
 * @brief Shared FPS command handlers for tmux, conf, and run
 * lifecycles.
 *
 * Included by fps_processcmdline_interactive.c and
 * fpsseq_cmdexec.c to avoid duplicating the three identical
 * per-FPS handler functions.
 *
 * Must be included after fps.h, fps_CONFstart.h,
 * fps_RUNstart.h, and the project debug macros.
 */

#ifndef FPS_CMD_HANDLERS_H
#define FPS_CMD_HANDLERS_H

#include "fps.h"
#include "fps_CONFstart.h"
#include "fps_RUNstart.h"

/**
 * FPS_CMD_RESULT - return status for FPS command dispatch handlers
 * @FPS_CMD_NOT_FOUND: command verb not recognized by this handler
 * @FPS_CMD_FAIL:      command recognized; execution failed
 * @FPS_CMD_OK:        command recognized; succeeded (log CMDOK)
 * @FPS_CMD_OK_QUIET:  command recognized; succeeded silently (no log)
 */
typedef enum
{
    FPS_CMD_NOT_FOUND = 0,
    FPS_CMD_FAIL      = -1,
    FPS_CMD_OK        = 1,
    FPS_CMD_OK_QUIET  = 2,
} FPS_CMD_RESULT;

/**
 * fps_cmd_handle_tmux - Handle tmux start/stop commands
 * @FPScommand:  Command verb
 * @nbword:      Word count
 * @fps:         FPS array
 * @fpsindex:    Index of the target FPS entry
 * @taskstatus:  OR-ed with error flags
 *
 * Dispatches: tmuxstart, tmuxstop.
 *
 * Return: FPS_CMD_OK on success, FPS_CMD_FAIL on argument error,
 *         FPS_CMD_NOT_FOUND if verb not recognized.
 */
static FPS_CMD_RESULT fps_cmd_handle_tmux(const char *FPScommand,
                                          int         nbword,
                                          FPS        *fps,
                                          int         fpsindex,
                                          uint64_t   *taskstatus)
{
    // tmuxstart
    if (strcmp(FPScommand, "tmuxstart") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "%s", "COMMAND tmuxstart takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        DEBUG_TRACEPOINT(" ");
        functionparameter_FPS_tmux_init(&fps[fpsindex]);
        functionparameter_outlog("FPSCTRL", "TMUXSTART %s", fps[fpsindex].md->name);
        return FPS_CMD_OK;
    }

    // tmuxstop
    if (strcmp(FPScommand, "tmuxstop") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "%s", "COMMAND tmuxstop takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        DEBUG_TRACEPOINT(" ");
        functionparameter_FPS_tmux_kill(&fps[fpsindex]);
        functionparameter_outlog("FPSCTRL", "TMUXSTOP %s", fps[fpsindex].md->name);
        return FPS_CMD_OK;
    }

    return FPS_CMD_NOT_FOUND;
}

/**
 * fps_cmd_handle_conf - Handle configuration commands
 * @FPScommand:  Command verb
 * @nbword:      Word count
 * @fps:         FPS array
 * @fpsindex:    Index of the target FPS entry
 * @taskstatus:  OR-ed with error flags
 *
 * Dispatches: confstart, confstop, confupdate, confwupdate.
 * confwupdate polls until the FPS acknowledges the update
 * or a timeout is reached.
 *
 * Return: FPS_CMD_OK on success, FPS_CMD_FAIL on argument error,
 *         FPS_CMD_NOT_FOUND if verb not recognized.
 */
static FPS_CMD_RESULT fps_cmd_handle_conf(const char *FPScommand,
                                          int         nbword,
                                          FPS        *fps,
                                          int         fpsindex,
                                          uint64_t   *taskstatus)
{
    // confstart
    if (strcmp(FPScommand, "confstart") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "%s", "COMMAND confstart takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        DEBUG_TRACEPOINT(" ");
        functionparameter_CONFstart(&fps[fpsindex]);
        functionparameter_outlog("FPSCTRL", "CONFSTART %s", fps[fpsindex].md->name);
        return FPS_CMD_OK;
    }

    // confstop
    if (strcmp(FPScommand, "confstop") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND confstop takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        DEBUG_TRACEPOINT(" ");
        functionparameter_CONFstop(&fps[fpsindex]);
        functionparameter_outlog("FPSCTRL", "CONFSTOP %s", fps[fpsindex].md->name);
        return FPS_CMD_OK;
    }

    // confupdate
    if (strcmp(FPScommand, "confupdate") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND confupdate takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        DEBUG_TRACEPOINT(" ");
        fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED;
        fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
        functionparameter_outlog("FPSCTRL", "CONFUPDATE %s", fps[fpsindex].md->name);
        return FPS_CMD_OK;
    }

    // confwupdate
    if (strcmp(FPScommand, "confwupdate") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND confwupdate takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        {
            int          looptry     = 1;
            int          looptrycnt  = 0;
            unsigned int timercnt    = 0;
            useconds_t   dt          = 100;
            unsigned int timercntmax = 10000;

            while (looptry == 1)
            {
                DEBUG_TRACEPOINT(" ");
                fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED;
                fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

                while (((fps[fpsindex].md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED)) &&
                       (timercnt < timercntmax))
                {
                    usleep(dt);
                    timercnt++;
                }
                usleep(dt);
                timercnt++;

                functionparameter_outlog("DEBUG",
                                         "CONFWUPDATE [%d] waited %d us on FPS %d %s. "
                                         "conferrcnt = %d",
                                         looptrycnt, dt * timercnt, fpsindex,
                                         fps[fpsindex].md->name, fps[fpsindex].md->conferrcnt);

                looptrycnt++;

                if (fps[fpsindex].md->conferrcnt == 0)
                {
                    looptry = 0;
                }

                if (timercnt > timercntmax)
                {
                    looptry = 0;
                }
            }
        }
        return FPS_CMD_OK;
    }

    return FPS_CMD_NOT_FOUND;
}
/**
 * fps_cmd_handle_run - Handle run lifecycle commands
 * @FPScommand:  Command verb
 * @nbword:      Word count
 * @fps:         FPS array
 * @fpsindex:    Index of the target FPS entry
 * @taskstatus:  OR-ed with error flags
 *
 * Dispatches: runstart, runwait, runstop.
 * runwait polls FPS status flags until CMDRUN clears
 * or a timeout is reached.
 *
 * Return: FPS_CMD_OK on success, FPS_CMD_FAIL on argument error,
 *         FPS_CMD_NOT_FOUND if verb not recognized.
 */
static FPS_CMD_RESULT fps_cmd_handle_run(const char *FPScommand,
                                         int         nbword,
                                         FPS        *fps,
                                         int         fpsindex,
                                         uint64_t   *taskstatus)
{
    // runstart
    if (strcmp(FPScommand, "runstart") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND runstart takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        DEBUG_TRACEPOINT(" ");
        functionparameter_RUNstart(&fps[fpsindex]);
        functionparameter_outlog("FPSCTRL", "RUNSTART %s", fps[fpsindex].md->name);
        return FPS_CMD_OK;
    }

    // runwait
    if (strcmp(FPScommand, "runwait") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND runwait takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        DEBUG_TRACEPOINT(" ");
        {
            unsigned int timercnt    = 0;
            useconds_t   dt          = 10000;
            unsigned int timercntmax = 100000;

            while (((fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)) &&
                   (timercnt < timercntmax))
            {
                usleep(dt);
                timercnt++;
            }
            functionparameter_outlog("FPSCTRL", "RUNWAIT waited %d us on FPS %s", dt * timercnt,
                                     fps[fpsindex].md->name);
        }
        return FPS_CMD_OK;
    }

    // runstop
    if (strcmp(FPScommand, "runstop") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND runstop takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        DEBUG_TRACEPOINT(" ");
        functionparameter_RUNstop(&fps[fpsindex]);
        functionparameter_outlog("FPSCTRL", "RUNSTOP %s", fps[fpsindex].md->name);
        return FPS_CMD_OK;
    }

    return FPS_CMD_NOT_FOUND;
}

/**
 * fps_cmd_handle_sys_common - Handle system commands common to all dispatchers
 * @FPScommand:    Command verb
 * @nbword:        Word count
 * @FPSarg0:       First argument token
 * @FPSarg1:       Second argument token
 * @fpsCTRLvar:    fpsCTRL process state (exitloop, scan params)
 * @fps:           FPS array (for rescan)
 * @keywnode:      Keyword tree root (for rescan)
 * @queuelist:     Task queue array (for queueprio)
 * @taskstatus:    OR-ed with error flags
 * @testcnt:       Counter incremented by the "cntinc" command
 *
 * Dispatches: exit, rescan, cntinc, logsymlink, queueprio.
 *
 * Return: FPS_CMD_OK_QUIET on success, FPS_CMD_FAIL on argument
 *         error, FPS_CMD_NOT_FOUND if verb not recognized.
 */
static FPS_CMD_RESULT fps_cmd_handle_sys_common(const char           *FPScommand,
                                                int                   nbword,
                                                const char           *FPSarg0,
                                                const char           *FPSarg1,
                                                FPSCTRL_PROCESS_VARS *fpsCTRLvar,
                                                FPS                  *fps,
                                                KEYWORD_TREE_NODE    *keywnode,
                                                FPSCTRL_TASK_QUEUE   *queuelist,
                                                uint64_t             *taskstatus,
                                                int                  *testcnt)
{
    // exit
    if (strcmp(FPScommand, "exit") == 0)
    {
        if (nbword != 1)
        {
            functionparameter_outlog("ERROR", "COMMAND exit takes NBARGS = 0");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        fpsCTRLvar->exitloop = 1;
        functionparameter_outlog("DEBUG", "EXIT");
        return FPS_CMD_OK_QUIET;
    }

    // rescan
    if (strcmp(FPScommand, "rescan") == 0)
    {
        if (nbword != 1)
        {
            functionparameter_outlog("ERROR", "COMMAND rescan takes NBARGS = 0");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        functionparameter_scan_fps(fpsCTRLvar->mode, fpsCTRLvar->fpsnamemask, fps, keywnode,
                                   &fpsCTRLvar->NBkwn, &fpsCTRLvar->NBfps, &fpsCTRLvar->NBindex, 0);
        functionparameter_outlog("DEBUG", "RESCAN");
        return FPS_CMD_OK_QUIET;
    }

    // cntinc
    if (strcmp(FPScommand, "cntinc") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND cntinc takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        (*testcnt)++;
        functionparameter_outlog("DEBUG", "TEST [%d] counter = %d", atoi(FPSarg0), *testcnt);
        return FPS_CMD_OK_QUIET;
    }

    // logsymlink
    if (strcmp(FPScommand, "logsymlink") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND logsymlink takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        {
            char logfname[STRINGMAXLEN_FULLFILENAME];
            getFPSlogfname(logfname);
            functionparameter_outlog("DEBUG", "CREATE SYM LINK %s <- %s", FPSarg0, logfname);
            if (symlink(logfname, FPSarg0) != 0)
            {
                PRINT_ERROR("symlink error %s %s", logfname, FPSarg0);
            }
        }
        return FPS_CMD_OK_QUIET;
    }

    // queueprio
    if (strcmp(FPScommand, "queueprio") == 0)
    {
        if (nbword != 3)
        {
            functionparameter_outlog("ERROR", "COMMAND queueprio takes NBARGS = 2");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        {
            int queue = atoi(FPSarg0);
            int prio  = atoi(FPSarg1);
            if ((queue >= 0) && (queue < NB_FPSCTRL_TASKQUEUE_MAX))
            {
                queuelist[queue].priority = prio;
                functionparameter_outlog("FPSCTRL", "%s", "QUEUE %d PRIO = %d", queue, prio);
            }
        }
        return FPS_CMD_OK_QUIET;
    }

    return FPS_CMD_NOT_FOUND;
}

#endif /* FPS_CMD_HANDLERS_H */
