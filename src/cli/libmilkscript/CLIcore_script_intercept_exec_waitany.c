#include <stddef.h>
extern int cli_find_in_path(const char *cmd, char *outpath, size_t outsize);
extern int processinfo_procdirname(char *procdirname);
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include "CLIcore.h"
#include "CLIcore_script.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>

extern int cli_block_level;
extern int cli_break_flag;
extern int CLI_trap_enable;
extern int cli_cmd_delay_us;

struct wa_event;
/**
 * @brief Evaluate FPS conditions for waitany.
 *
 * Checks if any FPS parameter meets the
 * specified condition.
 */
static int eval_waitany_fps(struct wa_event *ev, const char *vstr);

enum
{
    WA_STREAM,
    WA_FPS_PARAM,
    WA_PROC_STATE
};

enum
{
    CMP_EQ,
    CMP_NE,
    CMP_GE,
    CMP_LE
};

enum
{
    WA_MAX_EVENTS = 16
};

struct wa_event
{
    int  type;
    char name[256];
    /* FPS */
    char param[256];
    char target_val[256];
    int  cmp_op;
    /* Process */
    int target_state;
    /* Runtime handles */
    IMAGE    img;
    uint64_t start_cnt0;
    int      img_open;
    FPS      fps;
    int      fps_pindex;
    int      fps_open;
};

/**
 * @brief Parse waitany command arguments.
 *
 * Extracts stream/FPS names and timeout values
 * from the argument list.
 */
static int parse_waitany_args(const char *p, struct wa_event *events, double *timeout_v)
{
    char argbuf[STRINGMAXLEN_CLICMDLINE];
    strncpy(argbuf, p, STRINGMAXLEN_CLICMDLINE - 1);
    argbuf[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    int    nevents = 0;
    double timeout = 30.0;

    char *sav = NULL;
    char *tok = strtok_r(argbuf, " \t", &sav);
    /* skip "wait_any" */
    tok = strtok_r(NULL, " \t", &sav);

    while (tok != NULL && nevents < WA_MAX_EVENTS)
    {
        /* -t timeout */
        if (strcmp(tok, "-t") == 0)
        {
            tok = strtok_r(NULL, " \t", &sav);
            if (tok == NULL)
            {
                PRINT_ERROR("ERROR: wait_any: missing timeout value after -t");
                PRINT_ERROR("USAGE: wait_any [-t timeout] <events...>");
                cli_last_retval = 255;
                return -1;
            }
            timeout = strtod(tok, NULL);
            tok     = strtok_r(NULL, " \t", &sav);
            continue;
        }

        struct wa_event *ev = &events[nevents];
        memset(ev, 0, sizeof(*ev));

        if (strncmp(tok, "S:", 2) == 0)
        {
            ev->type = WA_STREAM;
            strncpy(ev->name, tok + 2, sizeof(ev->name) - 1);
            nevents++;
        }
        else if (strncmp(tok, "F:", 2) == 0)
        {
            ev->type         = WA_FPS_PARAM;
            const char *body = tok + 2;
            const char *dot  = strchr(body, '.');
            if (dot == NULL)
            {
                PRINT_ERROR("wait_any: bad F: token: %s", tok);
                cli_last_retval = 255;
                return -1;
            }
            int nlen = (int) (dot - body);
            if (nlen >= (int) sizeof(ev->name))
            {
                nlen = (int) sizeof(ev->name) - 1;
            }
            memcpy(ev->name, body, (size_t) nlen);
            ev->name[nlen] = '\0';

            const char *rest   = dot + 1;
            const char *op_pos = NULL;
            int         op_len = 0;
            ev->cmp_op         = CMP_EQ;

            op_pos = strstr(rest, ">=");
            if (op_pos != NULL)
            {
                ev->cmp_op = CMP_GE;
                op_len     = 2;
            }
            if (op_pos == NULL)
            {
                op_pos = strstr(rest, "<=");
                if (op_pos != NULL)
                {
                    ev->cmp_op = CMP_LE;
                    op_len     = 2;
                }
            }
            if (op_pos == NULL)
            {
                op_pos = strstr(rest, "!=");
                if (op_pos != NULL)
                {
                    ev->cmp_op = CMP_NE;
                    op_len     = 2;
                }
            }
            if (op_pos == NULL)
            {
                op_pos = strchr(rest, '=');
                if (op_pos != NULL)
                {
                    ev->cmp_op = CMP_EQ;
                    op_len     = 1;
                }
            }

            if (op_pos == NULL)
            {
                PRINT_ERROR("wait_any: no operator in F: token: %s", tok);
                cli_last_retval = 255;
                return -1;
            }

            int plen = (int) (op_pos - rest);
            if (plen >= (int) sizeof(ev->param))
            {
                plen = (int) sizeof(ev->param) - 1;
            }
            memcpy(ev->param, rest, (size_t) plen);
            ev->param[plen] = '\0';

            strncpy(ev->target_val, op_pos + op_len, sizeof(ev->target_val) - 1);
            nevents++;
        }
        else if (strncmp(tok, "P:", 2) == 0)
        {
            ev->type          = WA_PROC_STATE;
            const char *body  = tok + 2;
            const char *colon = strchr(body, ':');
            if (colon == NULL)
            {
                PRINT_ERROR("wait_any: bad P: token: %s", tok);
                cli_last_retval = 255;
                return -1;
            }
            int nlen = (int) (colon - body);
            if (nlen >= (int) sizeof(ev->name))
            {
                nlen = (int) sizeof(ev->name) - 1;
            }
            memcpy(ev->name, body, (size_t) nlen);
            ev->name[nlen] = '\0';

            const char *st = colon + 1;
            if (strcasecmp(st, "INIT") == 0)
            {
                ev->target_state = PROCESSINFO_LOOPSTAT_INIT;
            }
            else if (strcasecmp(st, "ACTIVE") == 0)
            {
                ev->target_state = PROCESSINFO_LOOPSTAT_ACTIVE;
            }
            else if (strcasecmp(st, "PAUSE") == 0)
            {
                ev->target_state = PROCESSINFO_LOOPSTAT_PAUSE;
            }
            else if (strcasecmp(st, "STOP") == 0)
            {
                ev->target_state = PROCESSINFO_LOOPSTAT_STOP;
            }
            else if (strcasecmp(st, "ERROR") == 0)
            {
                ev->target_state = PROCESSINFO_LOOPSTAT_ERROR;
            }
            else if (strcasecmp(st, "SPIN") == 0)
            {
                ev->target_state = PROCESSINFO_LOOPSTAT_SPIN;
            }
            else if (strcasecmp(st, "CRASHED") == 0)
            {
                ev->target_state = PROCESSINFO_LOOPSTAT_CRASHED;
            }
            else
            {
                ev->target_state = (int) strtol(st, NULL, 0);
            }

            nevents++;
        }
        else
        {
            PRINT_ERROR("wait_any: unknown event prefix: %s", tok);
            cli_last_retval = 255;
            return -1;
        }
        tok = strtok_r(NULL, " \t", &sav);
    }

    if (tok != NULL)
    {
        PRINT_ERROR("ERROR: wait_any: too many events (max %d)", WA_MAX_EVENTS);
        cli_last_retval = 255;
        return -1;
    }

    if (nevents == 0)
    {
        printf("Usage: wait_any [-t timeout] S:stream [F:fps.p=v] [P:proc:STATE]\n");
        cli_last_retval = 255;
        return -1;
    }

    if (timeout_v != NULL)
    {
        *timeout_v = timeout;
    }
    return nevents;
}

/**
 * @brief Open shared memory handles for waitany.
 *
 * Connects to all specified streams/FPS instances
 * for event monitoring.
 */
static int open_waitany_handles(struct wa_event *events, int nevents)
{
    int any_open = 0;
    for (int i = 0; i < nevents; i++)
    {
        struct wa_event *ev = &events[i];
        ev->img_open        = 0;
        ev->fps_open        = 0;

        if (ev->type == WA_STREAM)
        {
            if (ImageStreamIO_read_sharedmem_image_toIMAGE(ev->name, &ev->img) ==
                IMAGESTREAMIO_SUCCESS)
            {
                ev->start_cnt0 = ev->img.md->cnt0;
                ev->img_open   = 1;
                any_open       = 1;
            }
        }
        else if (ev->type == WA_FPS_PARAM)
        {
            if (fps_connect(ev->name, &ev->fps, FPSCONNECT_SIMPLE) != -1 && ev->fps.parray != NULL)
            {
                ev->fps_pindex = functionparameter_GetParamIndex(&ev->fps, ev->param);
                if (ev->fps_pindex < 0)
                {
                    char dname[512];
                    snprintf(dname, sizeof(dname), ".%s", ev->param);
                    ev->fps_pindex = functionparameter_GetParamIndex(&ev->fps, dname);
                }
                if (ev->fps_pindex >= 0)
                {
                    ev->fps_open = 1;
                    any_open     = 1;
                }
                else
                {
                    fps_disconnect(&ev->fps);
                }
            }
        }
        else if (ev->type == WA_PROC_STATE)
        {
            any_open = 1;
        }
    }
    return any_open;
}

static void close_waitany_handles(struct wa_event *events, int nevents)
{
    for (int i = 0; i < nevents; i++)
    {
        if (events[i].img_open)
        {
            ImageStreamIO_closeIm(&events[i].img);
        }
        if (events[i].fps_open)
        {
            fps_disconnect(&events[i].fps);
        }
    }
}

static int poll_waitany_events(struct wa_event *events, int nevents, double timeout)
{
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    cli_last_retval = 254;

    while (!cli_break_flag)
    {
        for (int i = 0; i < nevents; i++)
        {
            struct wa_event *ev    = &events[i];
            int              fired = 0;

            if (ev->type == WA_STREAM && ev->img_open)
            {
                if (ev->img.md->cnt0 != ev->start_cnt0)
                {
                    fired = 1;
                }
            }
            else if (ev->type == WA_FPS_PARAM && ev->fps_open)
            {
                char vstr[512];
                functionparameter_GetParamValueString(&ev->fps.parray[ev->fps_pindex], vstr,
                                                      sizeof(vstr));
                fired = eval_waitany_fps(ev, vstr);
            }
            else if (ev->type == WA_PROC_STATE)
            {
                if (pinfolist != NULL)
                {
                    for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
                    {
                        if (!pinfolist->active[pi])
                        {
                            continue;
                        }
                        if (strcmp(pinfolist->pnamearray[pi], ev->name) != 0)
                        {
                            continue;
                        }
                        pid_t fpid = pinfolist->PIDarray[pi];
                        char  pfn[512];
                        char  pdname[256];
                        processinfo_procdirname(pdname);
                        snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
                        int          pfd = -1;
                        PROCESSINFO *pii = processinfo_shm_link(pfn, &pfd);
                        if (pii != MAP_FAILED && pii != NULL)
                        {
                            if (pii->loopstat == ev->target_state)
                            {
                                fired = 1;
                            }
                            munmap(pii, sizeof(PROCESSINFO));
                            close(pfd);
                        }
                        else if (pfd >= 0)
                        {
                            close(pfd);
                        }
                        break;
                    }
                }
            }

            if (fired)
            {
                cli_last_retval = i;
                return 1;
            }
        }

        if (timeout >= 0.0)
        {
            struct timespec ts_now;
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            double elapsed = (double) (ts_now.tv_sec - ts_start.tv_sec) +
                             1e-9 * (double) (ts_now.tv_nsec - ts_start.tv_nsec);
            if (elapsed >= timeout)
            {
                cli_last_retval = 254;
                return 1;
            }
        }
        usleep(1000);
    }
    return 1;
}

/**
 * @brief Evaluate FPS conditions for waitany.
 *
 * Checks if any FPS parameter meets the
 * specified condition.
 */
static int eval_waitany_fps(struct wa_event *ev, const char *vstr)
{
    int fired = 0;
    switch (ev->cmp_op)
    {
    case CMP_EQ:
    {
        if (strcmp(vstr, ev->target_val) == 0)
        {
            fired = 1;
            break;
        }
        char  *e1 = NULL;
        char  *e2 = NULL;
        double d1 = strtod(vstr, &e1);
        double d2 = strtod(ev->target_val, &e2);
        if (e1 != vstr && *e1 == '\0' && e2 != ev->target_val && *e2 == '\0' && d1 == d2)
        {
            fired = 1;
        }
    }
    break;
    case CMP_NE:
    {
        int eq = 0;
        if (strcmp(vstr, ev->target_val) == 0)
        {
            eq = 1;
        }
        else
        {
            char  *e1 = NULL;
            char  *e2 = NULL;
            double d1 = strtod(vstr, &e1);
            double d2 = strtod(ev->target_val, &e2);
            if (e1 != vstr && *e1 == '\0' && e2 != ev->target_val && *e2 == '\0' && d1 == d2)
            {
                eq = 1;
            }
        }
        if (!eq)
        {
            fired = 1;
        }
    }
    break;
    case CMP_GE:
    case CMP_LE:
    {
        char  *e1 = NULL;
        char  *e2 = NULL;
        double d1 = strtod(vstr, &e1);
        double d2 = strtod(ev->target_val, &e2);
        if (e1 != vstr && *e1 == '\0' && e2 != ev->target_val && *e2 == '\0')
        {
            if (ev->cmp_op == CMP_GE && d1 >= d2)
            {
                fired = 1;
            }
            if (ev->cmp_op == CMP_LE && d1 <= d2)
            {
                fired = 1;
            }
        }
    }
    break;
    }
    return fired;
}

int cli_intercept_cmd_wait_any(const char *p)
{
    if (starts_with(p, "wait_any ") || starts_with(p, "wait_any\t") || strcmp(p, "wait_any") == 0)
    {
        /* --- local types --- */

        struct wa_event events[WA_MAX_EVENTS];
        int             nevents = 0;
        double          timeout = 30.0;

        nevents = parse_waitany_args(p, events, &timeout);
        if (nevents <= 0)
        {
            return 1;
        }

        /* --- open event handles --- */
        if (!open_waitany_handles(events, nevents))
        {
            PRINT_ERROR("wait_any: no events could be opened");
            cli_last_retval = 255;
            /* Still close what was opened if any */
            close_waitany_handles(events, nevents);
            return 1;
        }

        poll_waitany_events(events, nevents, timeout);
        close_waitany_handles(events, nevents);
        return 1;
    }
    return 0;
}
