// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_columns_streams.c
 * @brief   Column table of the streams panel.
 */

#include "overview_render_internal.h"
#include <inttypes.h>

#define STRM(rc) (&(rc)->m->streams[(rc)->idx])

#define RPID_W 23 /* read PIDs shown before the +N overflow marker */

/** cell_anc - distance to the selection; else loop membership or activity, then read/write direction. */
static void cell_anc(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_STREAM *s = STRM(rc);
    int              d = (rc->depth < 0) ? -rc->depth : rc->depth;
    d                  = (d > 99) ? 99 : d;
    c->fg = (s->nb_loops > 1) ? OV_FG_LOOP_SHARED : (s->nb_loops == 1) ? OV_FG_LOOP : OV_FG_WARN;
    if (rc->depth != 0 && !rc->is_sel && !rc->is_frozen)
    {
        snprintf(c->text, sizeof(c->text), (rc->depth < 0) ? "\xe2\x97\x80%d" : "%d\xe2\x96\xb6",
                 d);
        return;
    }
    const char *mark = (s->nb_loops > 1)      ? "\xe2\xae\x82" /* ⮂ */
                       : (s->nb_loops == 1)   ? "\xe2\x86\xba" /* ↺ */
                       : (s->update_hz > 0.1) ? "\xe2\x97\x8f" /* ● */
                                              : " ";
    const char *dir  = " ";
    if ((rc->eff_focus == OV_FOCUS_PROCS || rc->eff_focus == OV_FOCUS_FPS) && rc->is_rel)
    {
        dir = bget(rc->rel->stream_written, rc->idx) ? "\xe2\x96\xb6" : "\xe2\x97\x80";
    }
    snprintf(c->text, sizeof(c->text), "%s %s", mark, dir);
}

/** cell_name - stream name, dim when the stream is not updating. */
static void cell_name(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    ov_cell_name(c, rc, STRM(rc)->name, STRM(rc)->active ? OV_FG_STREAM : OV_FG_DIM);
}

/** cell_typ - data type. */
static void cell_typ(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    c->fg = OV_FG_MUTED;
    snprintf(c->text, sizeof(c->text), "%s", render_dtype(STRM(rc)->datatype));
}

/** cell_size - dimensions. */
static void cell_size(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    snprintf(c->text, sizeof(c->text), "%s", STRM(rc)->size_str);
}

/** cell_hz - update rate. */
static void cell_hz(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    double hz = STRM(rc)->update_hz;
    c->fg     = (hz > 0.1) ? OV_FG_ACTIVE : OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), (hz > 0.1) ? "%.1f" : "-", hz);
}

/** cell_mbps - throughput. */
static void cell_mbps(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_STREAM *s = STRM(rc);
    c->fg              = OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "-");
    if (s->update_hz > 0.1)
    {
        double mb =
            s->update_hz * (double) s->nelement * dtype_bytesize(s->datatype) / (1024.0 * 1024.0);
        c->fg = OV_FG_ACTIVE;
        snprintf(c->text, sizeof(c->text), (mb >= 1000.0) ? "%.1fG" : "%.1fM",
                 (mb >= 1000.0) ? mb / 1024.0 : mb);
    }
}

/** cell_inode - inode of the shm file. */
static void cell_inode(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    c->fg = OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "%" PRIu64, (uint64_t) STRM(rc)->inode);
}

/** cell_owner - creator process. */
static void cell_owner(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    ov_cell_pid(c, rc, STRM(rc)->ownerPID);
}

/** cell_count - update counter, active while it moves. */
static void cell_count(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    c->fg = STRM(rc)->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM;
    snprintf(c->text, sizeof(c->text), "%" PRIu64, (uint64_t) STRM(rc)->cnt0);
}

/** cell_sems - one character per semaphore: its value (+ above 9), '.' for absent ones. */
static void cell_sems(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_STREAM *s = STRM(rc);
    for (int i = 0; i < 10; i++)
    {
        int v = (i < s->nb_sem) ? s->semval[i] : 0;
        if (i >= s->nb_sem)
        {
            ov_cell_span(c, (OV_COLSPAN) { .fg = OV_FG_DIM }, ".");
        }
        else
        {
            ov_cell_span(c, (OV_COLSPAN) { .fg = ov_get_sem_color(v) }, "%c",
                         (v < 0)   ? '-'
                         : (v > 9) ? '+'
                                   : (char) ('0' + v));
        }
    }
}

/** cell_wpid - last writer process. */
static void cell_wpid(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    ov_cell_pid(c, rc, STRM(rc)->write_pid);
}

/** span_of - @c's single-cell style as a run. */
static OV_COLSPAN span_of(const OV_COLCELL *c)
{
    return (OV_COLSPAN) { .fg = c->fg, .has_bg = c->has_bg, .bg = c->bg, .bold = c->bold };
}

/** cell_rpid - reader processes, ':'-separated; +N counts those that do not fit. */
static void cell_rpid(OV_COLCELL *c, const OV_ROW_CTX *rc)
{
    const OV_STREAM *s    = STRM(rc);
    int              used = 0;
    if (s->nb_read_pids <= 0)
    {
        ov_cell_span(c, (OV_COLSPAN) { .fg = OV_FG_DIM }, "-");
        return;
    }
    for (int i = 0; i < s->nb_read_pids; i++)
    {
        OV_COLCELL p;
        memset(&p, 0, sizeof(p));
        ov_cell_pid(&p, rc, s->read_pids[i]);
        int rest = s->nb_read_pids - i - 1;
        int need =
            (i > 0) + (int) strlen(p.text) + (rest > 0 ? 1 + snprintf(NULL, 0, "%d", rest) : 0);
        if (used + need > RPID_W)
        {
            ov_cell_span(c, (OV_COLSPAN) { .fg = OV_FG_WARN }, "+%d", rest + 1);
            return;
        }
        if (i > 0)
        {
            ov_cell_span(c, (OV_COLSPAN) { .fg = OV_FG_DIM }, ":");
        }
        ov_cell_span(c, span_of(&p), "%s", p.text);
        used += (i > 0) + (int) strlen(p.text);
    }
}

static const OV_COLUMN stream_cols[] = {
    { "anc", "A", "ancestry", 3, 7, OV_COLF_PIN, 7, NULL, cell_anc },
    { "name", "NAME", "name", 14, 0, OV_COLF_PIN | OV_COLF_ELASTIC, 0, NULL, cell_name },
    { "typ", "TYP", "typ", 4, 5, OV_COLF_RIGHT, 1, NULL, cell_typ },
    { "size", "SIZE", "size", 11, 3, OV_COLF_RIGHT, 2, NULL, cell_size },
    { "hz", "Hz", NULL, 6, 1, OV_COLF_RIGHT, 3, NULL, cell_hz },
    { "mbps", "MB/s", NULL, 7, 4, OV_COLF_RIGHT, 4, NULL, cell_mbps },
    { "inode", "INODE", "inode", 10, 9, OV_COLF_RIGHT | OV_COLF_COMPACT_HIDE, 5, NULL, cell_inode },
    { "owner", "OWNER", NULL, 7, 6, OV_COLF_RIGHT, -1, NULL, cell_owner },
    { "count", "COUNT", "count", 10, 8, OV_COLF_RIGHT | OV_COLF_COMPACT_HIDE, 6, NULL, cell_count },
    { "sems", "SEMS", NULL, 10, 10, OV_COLF_COMPACT_HIDE, -1, NULL, cell_sems },
    { "wpid", "WPID", NULL, 7, 2, OV_COLF_RIGHT, -1, NULL, cell_wpid },
    { "rpid", "RPID", NULL, RPID_W, 11, 0, -1, NULL, cell_rpid },
};

const OV_TABLE ov_streams_table = { stream_cols,
                                    (int) (sizeof(stream_cols) / sizeof(stream_cols[0])),
                                    OV_FOCUS_STREAMS, "streams" };
