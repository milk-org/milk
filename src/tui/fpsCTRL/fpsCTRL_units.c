// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fpsCTRL_units.c
 * @brief   fpsCTRL's unit actions, unit states, output view and clipboard (fpsCTRL_units.h)
 *
 * Starts and stops run as milk-fps-<op> tools that the main loop polls (fps_unit_job_*), so the
 * screen keeps refreshing while a unit starts or stops (U3).
 */

#include <stdarg.h>
#include <time.h>

#include "computeunit.h"
#include "fpsCTRL_TUIcompat.h"
#include "fpsCTRL_units.h"

#define UNITS_JOBS 8
#define UNITS_CACHE 128
#define UNITS_MSG_S 20.0     /* how long the message line stays */
#define UNITS_STATUS_S 0.5   /* how old a cached state may be */
#define UNITS_TAIL_MAX 65536 /* bytes of output the view reads */

static FPS_UNIT_JOB jobs[UNITS_JOBS];
static char         then_op[UNITS_JOBS][16]; /* started once the job ended well (restart) */

static char   msg[256];
static int    msg_err;
static double msg_t = -1e9;

static struct
{
    char            name[STRINGMAXLEN_FPS_NAME];
    int             role;
    double          t;
    FPS_UNIT_STATUS st;
} cache[UNITS_CACHE];
static int cache_next;

static char out_name[STRINGMAXLEN_FPS_NAME];

static double units_now(void)
{
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

void fpsctrl_unit_say(int is_error, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg_err = is_error;
    msg_t   = units_now();
}

const char *fpsctrl_units_message(int *is_error)
{
    *is_error = msg_err;
    return (msg[0] != '\0' && units_now() - msg_t < UNITS_MSG_S) ? msg : NULL;
}

/* A state read now for @name, not from the cache: after an action on it */
static void units_forget(const char *name)
{
    for (int i = 0; i < UNITS_CACHE; i++)
    {
        if (strcmp(cache[i].name, name) == 0)
        {
            cache[i].t = -1e9;
        }
    }
}

/* Start the tool, one at a time per unit. Return: its slot, or -1 */
static int units_start(const char *name, const char *op)
{
    int slot = -1;

    for (int i = 0; i < UNITS_JOBS; i++)
    {
        if (jobs[i].pid > 0 && strcmp(jobs[i].name, name) == 0)
        {
            fpsctrl_unit_say(1, "%s: %s is still running", name, jobs[i].op);
            return -1;
        }
        if (jobs[i].pid <= 0 && slot < 0)
        {
            slot = i;
        }
    }
    if (slot < 0)
    {
        fpsctrl_unit_say(1, "%s %s: wait, %d unit actions are running", name, op, UNITS_JOBS);
        return -1;
    }
    if (fps_unit_job_start(&jobs[slot], name, op) != 0)
    {
        fpsctrl_unit_say(1, "%s %s: cannot run milk-fps-%s: %s", name, op, op, strerror(errno));
        return -1;
    }
    then_op[slot][0] = '\0';
    fpsctrl_unit_say(0, "%s %s ...", name, op);
    return slot;
}

int fpsctrl_unit_act(const char *name, const char *op)
{
    return (units_start(name, op) < 0) ? -1 : 0;
}

int fpsctrl_unit_restart(const char *name)
{
    int slot = units_start(name, "runstop");

    if (slot < 0)
    {
        return -1;
    }
    snprintf(then_op[slot], sizeof(then_op[slot]), "runstart");
    return 0;
}

int fpsctrl_units_poll(void)
{
    int ended = 0;

    for (int i = 0; i < UNITS_JOBS; i++)
    {
        if (jobs[i].pid <= 0)
        {
            continue;
        }
        int rc = fps_unit_job_poll(&jobs[i]);
        if (rc < 0)
        {
            continue;
        }
        ended = 1;
        units_forget(jobs[i].name);
        if (jobs[i].msg[0] != '\0')
        {
            fpsctrl_unit_say(rc != 0, "%s", jobs[i].msg);
        }
        else
        {
            fpsctrl_unit_say(rc != 0, "%s %s: %s (status %d)", jobs[i].name, jobs[i].op,
                             (rc == 0) ? "done" : "failed", rc);
        }
        if (rc == 0 && then_op[i][0] != '\0')
        {
            char name[STRINGMAXLEN_FPS_NAME];
            char op[16];
            snprintf(name, sizeof(name), "%s", jobs[i].name);
            snprintf(op, sizeof(op), "%s", then_op[i]);
            then_op[i][0] = '\0';
            units_start(name, op);
        }
    }
    return ended;
}

void fpsctrl_unit_status(FPS *fps, FPS_UNIT_ROLE role, FPS_UNIT_STATUS *st)
{
    double now  = units_now();
    int    slot = -1;

    for (int i = 0; i < UNITS_CACHE && slot < 0; i++)
    {
        if (cache[i].role == (int) role && strcmp(cache[i].name, fps->md->name) == 0)
        {
            slot = i;
        }
    }
    if (slot >= 0 && now - cache[slot].t < UNITS_STATUS_S)
    {
        *st = cache[slot].st;
        return;
    }
    if (slot < 0)
    {
        slot       = cache_next;
        cache_next = (cache_next + 1) % UNITS_CACHE;
    }
    fps_unit_status_fps(fps, role, st, FPS_UNIT_REPAIR);
    snprintf(cache[slot].name, sizeof(cache[slot].name), "%s", fps->md->name);
    cache[slot].role = (int) role;
    cache[slot].t    = now;
    cache[slot].st   = *st;
}

void fpsctrl_output_open(const char *name)
{
    snprintf(out_name, sizeof(out_name), "%s", name);
}

const char *fpsctrl_output_name(void)
{
    return out_name;
}

/* One line of unit output for the screen: no terminal sequences or control characters */
static void units_print_line(const char *l, size_t n)
{
    char   line[1024];
    size_t o   = 0;
    int    esc = 0;

    for (size_t i = 0; i < n && o < sizeof(line) - 1 && (int) o < sc_term_cols; i++)
    {
        unsigned char c = (unsigned char) l[i];
        if (esc == 1)
        {
            esc = (c == '[') ? 2 : 0;
        }
        else if (esc == 2)
        {
            esc = (c >= 0x40 && c <= 0x7e) ? 0 : 2;
        }
        else if (c == 0x1b)
        {
            esc = 1;
        }
        else if (c == '\t')
        {
            line[o++] = ' ';
        }
        else if (c >= 0x20 && c != 0x7f)
        {
            line[o++] = (char) c;
        }
    }
    line[o] = '\0';
    TUI_printfw("%s\n", line);
}

void fpsctrl_output_view(void)
{
    static char            buf[UNITS_TAIL_MAX];
    static double          st_t = -1e9;
    static FPS_UNIT_STATUS st;
    static char            st_name[STRINGMAXLEN_FPS_NAME];
    int                    rows = sc_term_rows - 4; // header, rule, message line, status bar

    if (rows < 1)
    {
        rows = 1;
    }
    if (strcmp(st_name, out_name) != 0 || units_now() - st_t >= UNITS_STATUS_S)
    {
        if (fps_unit_status(out_name, FPS_UNIT_RUN, &st) != 0)
        {
            st.state = FPS_UNIT_STOPPED;
            st.pid   = 0;
        }
        snprintf(st_name, sizeof(st_name), "%s", out_name);
        st_t = units_now();
    }

    screenprint_setbold();
    TUI_printfw(" OUTPUT %s run: %s", out_name, fps_unit_state_name(st.state));
    screenprint_unsetbold();
    if (st.pid > 0)
    {
        TUI_printfw(", pid %d", (int) st.pid);
    }
    if (fps_unit_state_alive(st.state) && st.uptime > 0)
    {
        TUI_printfw(", up %.0f s", st.uptime);
    }
    else if (st.last[0] != '\0')
    {
        TUI_printfw(" (%s)", st.last);
    }
    TUI_printfw("      Ctrl-C runstop   r restart   Esc back\n");
    for (int c = 0; c < sc_term_cols; c++)
    {
        TUI_printfw("-");
    }
    TUI_printfw("\n");

    size_t size = (size_t) rows * 512;
    int    n    = fps_unit_out_tail(out_name, FPS_UNIT_RUN, rows, buf,
                                    (size < sizeof(buf)) ? size : sizeof(buf));
    if (n < 0)
    {
        TUI_printfw("  (no output: the unit has not run)\n");
        return;
    }
    for (char *l = buf; l < buf + n;)
    {
        char  *nl  = memchr(l, '\n', (size_t) (buf + n - l));
        size_t len = (nl != NULL) ? (size_t) (nl - l) : (size_t) (buf + n - l);
        units_print_line(l, len);
        l += len + 1;
    }
}

void fpsctrl_copy(const char *text)
{
    static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char              out[1200];
    size_t            n = strlen(text);
    size_t            o = (size_t) snprintf(out, sizeof(out), "\033]52;c;");

    for (size_t i = 0; i < n && o + 6 < sizeof(out); i += 3)
    {
        unsigned v = (unsigned char) text[i] << 16;
        v |= (i + 1 < n) ? (unsigned char) text[i + 1] << 8 : 0;
        v |= (i + 2 < n) ? (unsigned char) text[i + 2] : 0;
        out[o++] = b64[(v >> 18) & 63];
        out[o++] = b64[(v >> 12) & 63];
        out[o++] = (i + 1 < n) ? b64[(v >> 6) & 63] : '=';
        out[o++] = (i + 2 < n) ? b64[v & 63] : '=';
    }
    out[o++] = '\a';
    if (write(STDOUT_FILENO, out, o) < 0)
    {
    }
}
