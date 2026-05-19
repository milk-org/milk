/**
 * @file CLIcore_script_traps.c
 *
 * @brief Signal traps and engine event traps.
 *
 * Implements the `trap` builtin (POSIX signal traps
 * bound to CLI commands), plus the non-blocking engine
 * event trap system (STREAM:, FPS:, PROC: prefixes)
 * that fires commands between CLI cycles.
 *
 * Public API (declared in CLIcore_script.h):
 *   cli_trap_run()
 *   cli_trap_run_exit()
 *   cli_engine_traps_poll()
 *   cli_engine_traps_cleanup()
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "ImageStreamIO/ImageStreamIO.h"

/* processinfo functions — linked via milkprocessinfo */
extern PROCESSINFO *processinfo_shm_link(
    const char *pname, int *fd);
extern errno_t processinfo_procdirname(
    char *procdname);


/* ============================================================
 *  Global engine trap table
 * ============================================================
 */

CLI_ENGINE_TRAP cli_engine_traps[CLI_ENGINE_TRAP_MAX];


/* ============================================================
 *  Signal-based traps
 * ============================================================
 */

/**
 * @brief Execute trap handlers for signal
 *
 * Scans the trap table for entries matching
 * @signum and runs their associated command.
 */
void cli_trap_run(int signum)
{
    for(int i = 0;
        i < CLI_TRAP_MAXSIGS; i++)
    {
        if(cli_traps[i].used
           && cli_traps[i].signum
           == signum)
        {
            strncpy(data.CLIcmdline, cli_traps[i].cmd, STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
            CLI_execute_line();
        }
    }
}

/**
 * @brief Run EXIT traps (signal 0)
 *
 * Called at script termination. Executes deferred
 * cleanup commands, disconnects engine traps, then
 * runs any registered EXIT (signal 0) trap handler.
 */
void cli_trap_run_exit(void)
{
    cli_defer_run();
    cli_engine_traps_cleanup();
    cli_trap_run(0);
}


/* ============================================================
 *  Engine event traps — non-blocking poll
 * ============================================================
 */

/**
 * etrap_connect - lazy-connect a trap handle
 * @et: engine trap entry
 *
 * Opens the SHM stream, FPS, or processinfo
 * handle on first poll. Returns 1 on success,
 * 0 if the resource is not yet available.
 */
static int etrap_connect(CLI_ENGINE_TRAP *et)
{
    if(et->connected)
    {
        return 1;
    }

    switch(et->type)
    {
    case CLI_ETRAP_STREAM:
    {
        memset(&et->img, 0, sizeof(IMAGE));
        if(ImageStreamIO_read_sharedmem_image_toIMAGE(
               et->target, &et->img)
           != IMAGESTREAMIO_SUCCESS)
        {
            return 0;
        }
        et->last_cnt0 = et->img.md->cnt0;
        et->connected = 1;
        return 1;
    }

    case CLI_ETRAP_FPS:
    {
        et->fps.SMfd = -1;
        int rc = fps_connect(et->target, &et->fps, FPSCONNECT_SIMPLE);
        if(rc == -1 || et->fps.md == NULL
           || et->fps.parray == NULL)
        {
            return 0;
        }

        int paramindex = functionparameter_GetParamIndex(&et->fps, et->param);
        if(paramindex < 0)
        {
            char dottedparam[STRINGMAXLEN_FULLFILENAME];
            if(snprintf(dottedparam,
                        sizeof(dottedparam),
                        ".%s", et->param)
               >= (int) sizeof(dottedparam))
            {
                return 0;
            }
            paramindex = functionparameter_GetParamIndex(&et->fps, dottedparam);
        }
        if(paramindex < 0)
        {
            return 0;
        }

        /* Record initial value */
        functionparameter_GetParamValueString(
            &et->fps.parray[paramindex], et->last_fpsval, (int) sizeof(et->last_fpsval));
        et->connected = 1;
        return 1;
    }

    case CLI_ETRAP_PROC:
        /* No persistent handle needed;
         * we scan pinfolist each poll */
        et->connected = 1;
        return 1;

    default: return 0;
    }
}

/**
 * etrap_check_fire - check if trap should fire
 * @et: engine trap entry
 *
 * Tests the current state of the event source
 * against the trap's trigger condition. Returns
 * 1 if the event has occurred since last check,
 * triggering a transition-based edge detection.
 */
static int etrap_check_fire(CLI_ENGINE_TRAP *et)
{
    switch(et->type)
    {
    case CLI_ETRAP_STREAM:
    {
        if(et->img.md == NULL)
        {
            return 0;
        }
        uint64_t cur = et->img.md->cnt0;
        if(cur != et->last_cnt0)
        {
            et->last_cnt0 = cur;
            return 1;
        }
        return 0;
    }

    case CLI_ETRAP_FPS:
    {
        if(et->fps.md == NULL
           || et->fps.parray == NULL)
        {
            return 0;
        }
        for(int pi = 0;
            pi < et->fps.md->NBparamMAX;
            pi++)
        {
            if(!(et->fps.parray[pi].fpflag
                 & FPFLAG_ACTIVE))
            {
                continue;
            }
            if(strcmp(
                   et->fps.parray[pi]
                       .keyword[0],
                   et->param)
               != 0)
            {
                continue;
            }
            char cur[256];
            functionparameter_GetParamValueString(&et->fps.parray[pi], cur, (int) sizeof(cur));

            /* Compare based on operator */
            double dval = strtod(cur, NULL);
            int match = 0;
            switch(et->op)
            {
            case CLI_ETRAP_OP_EQ:
                if(et->has_cmp)
                {
                    match = (dval == et->cmpval);
                }
                else
                {
                    /* Fire on any change */
                    match = (strcmp(cur, et->last_fpsval) != 0);
                }
                break;
            case CLI_ETRAP_OP_NE: match = (dval != et->cmpval);
                break;
            case CLI_ETRAP_OP_GE: match = (dval >= et->cmpval);
                break;
            case CLI_ETRAP_OP_LE: match = (dval <= et->cmpval);
                break;
            }

            /* Only fire on rising-edge
             * transition into match */
            int prev_match = 0;
            {
                double pval = strtod(et->last_fpsval, NULL);
                switch(et->op)
                {
                case CLI_ETRAP_OP_EQ:
                    if(et->has_cmp)
                    {
                        prev_match = (pval == et->cmpval);
                    }
                    else
                    {
                        prev_match = 0;
                    }
                    break;
                case CLI_ETRAP_OP_NE: prev_match = (pval != et->cmpval);
                    break;
                case CLI_ETRAP_OP_GE: prev_match = (pval >= et->cmpval);
                    break;
                case CLI_ETRAP_OP_LE: prev_match = (pval <= et->cmpval);
                    break;
                }
            }

            strncpy(et->last_fpsval, cur, sizeof(et->last_fpsval) - 1);
            et->last_fpsval[sizeof(et->last_fpsval) - 1] = '\0';

            if(match && !prev_match)
            {
                return 1;
            }
            return 0;
        }
        return 0;
    }

    case CLI_ETRAP_PROC:
    {
        if(pinfolist == NULL)
        {
            return 0;
        }
        for(int pi = 0;
            pi < PROCESSINFOLISTSIZE; pi++)
        {
            if(!pinfolist->active[pi])
            {
                continue;
            }
            if(strcmp(
                   pinfolist->pnamearray[pi],
                   et->target) != 0)
            {
                continue;
            }
            pid_t fpid = pinfolist->PIDarray[pi];
            if(fpid <= 0)
            {
                continue;
            }
            char pfn[512];
            char pdname[256];
            processinfo_procdirname(pdname);
            snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
            int pfd = -1;
            PROCESSINFO *pi_shm = processinfo_shm_link(pfn, &pfd);
            if(pi_shm == MAP_FAILED
               || pi_shm == NULL)
            {
                if(pfd >= 0)
                {
                    close(pfd);
                }
                continue;
            }
            int cur_state = pi_shm->loopstat;
            munmap(pi_shm, sizeof(PROCESSINFO));
            close(pfd);

            if(cur_state == et->proc_state)
            {
                return 1;
            }
            return 0;
        }
        return 0;
    }

    default: return 0;
    }
}

/**
 * @brief Check all engine traps, fire if triggered
 *
 * Called at the top of each CLI command cycle.
 * For each registered trap, checks if the event
 * occurred and fires the command if so, respecting
 * throttle (min_interval_ms) and fire-count limits.
 */
void cli_engine_traps_poll(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    for(int i = 0;
        i < CLI_ENGINE_TRAP_MAX; i++)
    {
        CLI_ENGINE_TRAP *et = &cli_engine_traps[i];
        if(!et->used)
        {
            continue;
        }

        /* Lazy connect */
        if(!et->connected)
        {
            if(!etrap_connect(et))
            {
                continue;
            }
        }

        /* Check throttle interval */
        if(et->min_interval_ms > 0)
        {
            long elapsed_ms =
                (now.tv_sec
                 - et->last_fire_ts.tv_sec)
                * 1000L + (now.tv_nsec - et->last_fire_ts .tv_nsec) / 1000000L;
            if(elapsed_ms
               < et->min_interval_ms)
            {
                continue;
            }
        }

        /* Check fire count limit */
        if(et->max_fires > 0
           && et->fire_count
           >= et->max_fires)
        {
            et->used = 0;
            et->connected = 0;
            continue;
        }

        /* Check event and fire */
        if(etrap_check_fire(et))
        {
            et->fire_count++;
            et->last_fire_ts = now;
            CLI_execute_string(et->cmd);
        }
    }
}

/**
 * @brief Disconnect and deactivate all engine traps
 *
 * Called at script exit to release SHM stream
 * and FPS handles acquired during lazy-connect.
 */
void cli_engine_traps_cleanup(void)
{
    for(int i = 0;
        i < CLI_ENGINE_TRAP_MAX; i++)
    {
        CLI_ENGINE_TRAP *et = &cli_engine_traps[i];
        if(!et->used)
        {
            continue;
        }
        if(et->connected)
        {
            if(et->type == CLI_ETRAP_FPS)
            {
                fps_disconnect(&et->fps);
            }
            else if(et->type
                    == CLI_ETRAP_STREAM)
            {
                ImageStreamIO_closeIm(&et->img);
            }
            et->connected = 0;
        }
        et->used = 0;
    }
}
