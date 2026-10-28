// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_columns_fps.c
 * @brief   Column table of the FPS panel.
 */

#include "overview_render_internal.h"
#include "fps_unit_state.h"

#define FPS(rc) (&(rc)->m->fps[(rc)->idx])

/** ov_cell_ancestry - lineage mark of a row: distance to the selection, its direction, or loop membership. */
void ov_cell_ancestry(OV_COLCELL *c, const OV_ROW_CTX *rc, int nb_loops, int written)
{
    int d = (rc->depth < 0) ? -rc->depth : rc->depth;
    d     = (d > 99) ? 99 : d;
    c->fg = (nb_loops > 1) ? OV_FG_LOOP_SHARED : (nb_loops == 1) ? OV_FG_LOOP : OV_FG_WARN;
    if (rc->depth != 0 && !rc->is_sel && !rc->is_frozen)
    {
        snprintf(c->text, sizeof(c->text), (rc->depth < 0) ? "\xe2\x97\x80%d" : "%d\xe2\x96\xb6",
                 d);
    }
    else if (rc->is_sel || rc->is_frozen)
    {
        snprintf(c->text, sizeof(c->text), "\xe2\x97\x8f"); /* ● */
    }
    else if (rc->eff_focus == OV_FOCUS_STREAMS && rc->is_rel && written >= 0)
    {
        snprintf(c->text, sizeof(c->text), written ? "\xe2\x96\xb6" : "\xe2\x97\x80");
    }
    else if (nb_loops > 1)
    {
        snprintf(c->text, sizeof(c->text), "\xe2\xae\x82"); /* ⮂ */
    }
    else if (nb_loops == 1)
    {
        snprintf(c->text, sizeof(c->text), "\xe2\x86\xba"); /* ↺ */
    }
}

/** ov_cell_name - entry name, the panel filter's match highlighted. */
void ov_cell_name(OV_COLCELL *c, const OV_ROW_CTX *rc, const char *name, ov_rgb_t fg)
{
    regmatch_t pm[1];
    c->fg = fg;
    if (rc->re != NULL && regexec(rc->re, name, 1, pm, 0) == 0 && pm[0].rm_eo > pm[0].rm_so)
    {
        snprintf(c->text, sizeof(c->text), "%.*s\x01%.*s\x02%s", (int) pm[0].rm_so, name,
                 (int) (pm[0].rm_eo - pm[0].rm_so), name + pm[0].rm_so, name + pm[0].rm_eo);
    }
    else
    {
        snprintf(c->text, sizeof(c->text), "%s", name);
    }
}

/** ov_cell_pid - a PID: highlighted when it is the selected process, red when that process crashed. */
void ov_cell_pid(OV_COLCELL *c, const OV_ROW_CTX *rc, pid_t pid)
{
    if (pid <= 0)
    {
        c->fg = OV_FG_DIM;
        snprintf(c->text, sizeof(c->text), "-");
        return;
    }
    int pi      = ov_find_proc_by_pid(rc->m, pid);
    int crashed = (pi >= 0 && rc->m->procs[pi].loopstat == PROCESSINFO_LOOPSTAT_CRASHED);
    int match   = (rc->sel_pid > 0 && pid == rc->sel_pid);
    c->fg       = crashed ? OV_FG_ERROR : ov_pid_color(pid);
    if (match)
    {
        c->has_bg = 1;
        c->bg     = crashed ? OV_FG_ERROR : OV_BG_PID_MATCH;
        c->fg     = ov_rgb_on(OV_BG_PID_MATCH);
        c->bold   = 1;
    }
    snprintf(c->text, sizeof(c->text), "%d", (int) pid);
}

/** cell_mark - batch mark of the FPS (b / a in CONTROL mode). */
static void cell_mark(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    c->fg = OV_FG_ACTIVE;
    snprintf(c->text, sizeof(c->text), "%s",
             ov_msel_has(rc->lay, FPS(rc)->name) ? "\xe2\x97\x86" : "");
}

/** cell_anc - lineage relative to the selection. */
static void cell_anc(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    ov_cell_ancestry(c, rc, FPS(rc)->nb_loops,
                     (rc->rel != NULL) ? bget(rc->rel->fps_writes, rc->idx) : -1);
}

/** cell_name - FPS name. */
static void cell_name(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    ov_cell_name(c, rc, FPS(rc)->name, OV_FG_FPS);
}

/** cell_state - state of the run process (U10): running, starting, stopping, failed, exited, stopped. */
static void cell_state(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    FPS_UNIT_STATE s = (FPS_UNIT_STATE) FPS(rc)->run_state;
    c->fg            = (s == FPS_UNIT_RUNNING)   ? OV_FG_ACTIVE
                       : (s == FPS_UNIT_FAILED)  ? OV_FG_ERROR
                       : fps_unit_state_alive(s) ? OV_FG_WARN
                                                 : OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "%s", fps_unit_state_name(s));
}

/** cell_up - how long the run process has been running. */
static void cell_up(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_FPS *f = FPS(rc);
    c->fg           = OV_FG_DIM;
    if (fps_unit_state_alive((FPS_UNIT_STATE) f->run_state))
    {
        format_uptime(c->text, (int) sizeof(c->text), (int64_t) f->run_uptime);
    }
}

/** cell_last - how the last run ended, once none runs. */
static void cell_last(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_FPS *f = FPS(rc);
    c->fg           = (f->run_state == FPS_UNIT_FAILED) ? OV_FG_ERROR : OV_FG_DIM;
    if (!fps_unit_state_alive((FPS_UNIT_STATE) f->run_state))
    {
        snprintf(c->text, sizeof(c->text), "%s", f->run_last);
    }
}

/** cell_cpid - CONF process. */
static void cell_cpid(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    ov_cell_pid(c, rc, FPS(rc)->confpid);
}

/** cell_rpid - RUN process. */
static void cell_rpid(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    ov_cell_pid(c, rc, FPS(rc)->runpid);
}

/** cell_str - number of stream parameters. */
static void cell_str(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    snprintf(c->text, sizeof(c->text), "%d", FPS(rc)->nb_stream_params);
}

/** cell_mem - resident memory of the FPS processes. */
static void cell_mem(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    format_mem_kb(c->text, sizeof(c->text), FPS(rc)->mem_rss_kb);
}

/** cell_desc - FPS description. */
static void cell_desc(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    c->fg = OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "%s", FPS(rc)->description);
}

/** cell_hist - Hz history as a sparkline. */
static void cell_hist(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_FPS *f = FPS(rc);
    c->fg           = OV_FG_FPS;
    if (f->hz_hist_idx > 0)
    {
        ov_sparkline_str(c->text, sizeof(c->text), f->hz_hist, f->hz_hist_idx, OV_SPARKLINE_LEN, 8);
    }
}

/** desc_width - the description is wider in the F5 view. */
static int desc_width(const OV_LAYOUT *lay)
{
    return (lay->view == OV_VIEW_FPS) ? 30 : 20;
}

static const OV_COLUMN fps_cols[] = {
    { "mark", "", NULL, 1, 9, OV_COLF_NOLABEL | OV_COLF_PIN, -1, NULL, cell_mark },
    { "anc", "A", "ancestry", 3, 8, OV_COLF_PIN, 3, NULL, cell_anc },
    { "name", "NAME", "name", 18, 0, OV_COLF_PIN | OV_COLF_ELASTIC, 0, NULL, cell_name },
    { "state", "STATE", NULL, 8, 3, 0, 5, NULL, cell_state },
    { "up", "UP", NULL, 6, 6, OV_COLF_RIGHT, -1, NULL, cell_up },
    { "cpid", "CPID", NULL, 7, 2, OV_COLF_RIGHT, 1, NULL, cell_cpid },
    { "rpid", "RPID", NULL, 7, 1, OV_COLF_RIGHT, 4, NULL, cell_rpid },
    { "str", "STR", NULL, 3, 7, OV_COLF_RIGHT, 6, NULL, cell_str },
    { "mem", "MEM", NULL, 5, 5, OV_COLF_RIGHT, 2, NULL, cell_mem },
    { "last", "LAST", NULL, 16, 8, OV_COLF_COMPACT_HIDE, -1, NULL, cell_last },
    { "desc", "DESCRIPTION", NULL, 20, 10, OV_COLF_COMPACT_HIDE, -1, desc_width, cell_desc },
    { "hist", "", NULL, 8, 11, OV_COLF_NOLABEL | OV_COLF_COMPACT_HIDE, -1, NULL, cell_hist },
};

const OV_TABLE ov_fps_table = { fps_cols, (int) (sizeof(fps_cols) / sizeof(fps_cols[0])),
                                OV_FOCUS_FPS, "FPS" };
