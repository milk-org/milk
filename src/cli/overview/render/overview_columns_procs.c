// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_columns_procs.c
 * @brief   Column table of the processes panel.
 */

#include "overview_render_internal.h"
#include <inttypes.h>

#define PROC(rc) (&(rc)->m->procs[(rc)->idx])

/** proc_rel - 1 if the row is related to the selection, *@written set when the process writes it. */
static int proc_rel(const OV_ROW_CTX *rc, int *written)
{
    int has  = (rc->rel != NULL && bget(rc->rel->procs, rc->idx));
    *written = has ? bget(rc->rel->proc_writes, rc->idx) : 0;
    return has;
}

/** cell_anc - lineage relative to the selection. */
static void cell_anc(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    int w;
    int has = proc_rel(rc, &w);
    ov_cell_ancestry(c, rc, PROC(rc)->nb_loops, has ? w : -1);
}

/** cell_name - process name. */
static void cell_name(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    ov_cell_name(c, rc, PROC(rc)->name, OV_FG_PROC);
}

/** cell_pid - process ID. */
static void cell_pid(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    c->fg = ov_pid_color(PROC(rc)->PID);
    snprintf(c->text, sizeof(c->text), "%d", (int) PROC(rc)->PID);
}

/** cell_prio - real-time priority, "-" when not real-time. */
static void cell_prio(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    int prio = PROC(rc)->rt_priority;
    c->fg    = (prio > -1) ? OV_FG_WARN : OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), (prio > -1) ? "%d" : "-", prio);
}

/** cell_stat - loop state (STAL when a running loop stalled). */
static void cell_stat(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    snprintf(c->text, sizeof(c->text), "%s", ov_proc_status(PROC(rc), &c->fg));
}

/** cell_hz - loop rate. */
static void cell_hz(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    double hz = PROC(rc)->loop_hz;
    c->fg = (hz > 0.1) ? ov_rgb_lerp(OV_FG_DIM, OV_FG_ACTIVE, (float) (hz / 5000.0)) : OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), (hz > 0.1) ? "%.1f" : "-", hz);
}

/** cell_uptime - time since the process started. */
static void cell_uptime(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    snprintf(c->text, sizeof(c->text), "-");
    if (PROC(rc)->start_time_sec > 0)
    {
        format_uptime(c->text, (int) sizeof(c->text), PROC(rc)->start_time_sec);
    }
}

/** cell_trg - trigger mode. */
static void cell_trg(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    c->fg = OV_FG_CONN;
    snprintf(c->text, sizeof(c->text), "%s", ov_trigmode_label(PROC(rc)->triggermode, 0));
}

/** cell_trigstrm - trigger stream. */
static void cell_trigstrm(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_PROC *p = PROC(rc);
    int            t = (p->trigstreamname[0] != '\0' && p->triggermode > 0);
    c->fg            = t ? OV_FG_STREAM : OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "%s", t ? p->trigstreamname : "-");
}

/** cell_exec - median execution time in ms; W/R when it writes/reads the selected stream. */
static void cell_exec(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_PROC *p = PROC(rc);
    int            w;
    int            has    = proc_rel(rc, &w);
    const char    *suffix = has ? (w ? " W" : " R") : "";
    if (p->MeasureTiming && p->dtmedian_exec_ns > 0)
    {
        double ms = 1.0e-6 * (double) p->dtmedian_exec_ns;
        c->fg     = (ms < 1.0) ? OV_FG_ACTIVE : (ms < 10.0) ? OV_FG_WARN : OV_FG_ERROR;
        snprintf(c->text, sizeof(c->text), "%.3f%s", ms, suffix);
    }
    else
    {
        c->fg = OV_FG_DIM;
        snprintf(c->text, sizeof(c->text), "-%s", suffix);
    }
}

/** cell_duty - execution time over iteration time. */
static void cell_duty(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_PROC *p = PROC(rc);
    c->fg            = OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "-");
    if (p->MeasureTiming && p->dtmedian_exec_ns > 0 && p->dtmedian_iter_ns > 0)
    {
        double duty = 100.0 * (double) p->dtmedian_exec_ns / (double) p->dtmedian_iter_ns;
        c->fg       = (duty > 90.0) ? OV_FG_ERROR : (duty > 50.0) ? OV_FG_WARN : OV_FG_ACTIVE;
        snprintf(c->text, sizeof(c->text), "%.0f%%", duty);
    }
}

/** cell_cpu - CPU use of the process. */
static void cell_cpu(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    snprintf(c->text, sizeof(c->text), "%.1f%%", PROC(rc)->cpu_used);
}

/** cell_loopcnt - loop counter (active while it moves). */
static void cell_loopcnt(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    c->fg = PROC(rc)->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "%" PRId64, (int64_t) PROC(rc)->loopcnt);
}

/** cell_mem - resident memory. */
static void cell_mem(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    format_mem_kb(c->text, sizeof(c->text), PROC(rc)->mem_rss_kb);
}

/** cell_missed - missed triggers since start. */
static void cell_missed(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    uint64_t n = PROC(rc)->triggermissed_cumul;
    c->fg      = (n > 0) ? OV_FG_WARN : OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "%" PRIu64, n);
}

/** cell_msg - status message of the loop. */
static void cell_msg(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    snprintf(c->text, sizeof(c->text), "%s", PROC(rc)->statusmsg);
}

static const OV_COLUMN proc_cols[] = {
    { "anc", "A", "ancestry", 3, 8, 0, 5, NULL, cell_anc },
    { "name", "NAME", "name", 14, 0, 0, 0, NULL, cell_name },
    { "pid", "PID", NULL, 7, 1, OV_COLF_RIGHT, 1, NULL, cell_pid },
    { "prio", "PRIO", NULL, 4, 9, OV_COLF_RIGHT, 6, NULL, cell_prio },
    { "stat", "STAT", "stat", 4, 2, OV_COLF_RIGHT, 2, NULL, cell_stat },
    { "hz", "Hz", NULL, 6, 3, OV_COLF_RIGHT, 3, NULL, cell_hz },
    { "uptime", "UPTIME", NULL, 6, 10, OV_COLF_RIGHT, 7, NULL, cell_uptime },
    { "trg", "TRG", NULL, 3, 11, OV_COLF_RIGHT | OV_COLF_COMPACT_HIDE, -1, NULL, cell_trg },
    { "trigstrm", "trig-strm", NULL, 10, 12, OV_COLF_COMPACT_HIDE, -1, NULL, cell_trigstrm },
    { "exec", "exec", NULL, 9, 13, OV_COLF_RIGHT | OV_COLF_COMPACT_HIDE, -1, NULL, cell_exec },
    { "duty", "DUTY", NULL, 5, 14, OV_COLF_RIGHT | OV_COLF_COMPACT_HIDE, 10, NULL, cell_duty },
    { "cpu", "CPU%", NULL, 6, 4, OV_COLF_RIGHT, 8, NULL, cell_cpu },
    { "loopcnt", "LOOPCNT", NULL, 10, 6, OV_COLF_RIGHT, 9, NULL, cell_loopcnt },
    { "mem", "MEM", NULL, 5, 5, OV_COLF_RIGHT, 4, NULL, cell_mem },
    { "missed", "MISSED", NULL, 10, 15, OV_COLF_RIGHT | OV_COLF_COMPACT_HIDE, -1, NULL,
      cell_missed },
    { "msg", "MSG", NULL, 200, 16, 0, -1, NULL, cell_msg },
};

const OV_TABLE ov_procs_table = { proc_cols, (int) (sizeof(proc_cols) / sizeof(proc_cols[0])),
                                  OV_FOCUS_PROCS, "processes" };
