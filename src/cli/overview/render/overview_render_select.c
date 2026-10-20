// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_select.c
 * @brief   One resolver for drawn rows, cursor and action targets in the list panels.
 *
 * The cursor of each list panel is an identity (OV_TARGET). Rows are derived per frame from the
 * rows actually drawn, so render, hover, navigation and actions always agree on the entity.
 */

#include "overview_render_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_tgt_set - capture into @t the identity of entry @idx of @panel (@idx < 0 clears it).
 */
void ov_tgt_set(OV_TARGET *t, const OV_MODEL *m, ov_focus_t panel, int idx)
{
    memset(t, 0, sizeof(*t));
    if (m == NULL || idx < 0)
    {
        return;
    }
    if (panel == OV_FOCUS_STREAMS && idx < m->nb_streams)
    {
        snprintf(t->name, sizeof(t->name), "%s", m->streams[idx].name);
        t->inode = (uint64_t) m->streams[idx].inode;
    }
    else if (panel == OV_FOCUS_PROCS && idx < m->nb_procs)
    {
        snprintf(t->name, sizeof(t->name), "%s", m->procs[idx].name);
        t->pid = (int) m->procs[idx].PID;
    }
    else if (panel == OV_FOCUS_FPS && idx < m->nb_fps)
    {
        snprintf(t->name, sizeof(t->name), "%s", m->fps[idx].name);
    }
}

/**
 * ov_tgt_find - model index of the entry with identity @t, or -1.
 */
int ov_tgt_find(const OV_MODEL *m, ov_focus_t panel, const OV_TARGET *t)
{
    if (m == NULL || t == NULL || t->name[0] == '\0')
    {
        return -1;
    }
    if (panel == OV_FOCUS_STREAMS)
    {
        for (int i = 0; i < m->nb_streams; i++)
        {
            if (strcmp(m->streams[i].name, t->name) == 0 &&
                (t->inode == 0 || (uint64_t) m->streams[i].inode == t->inode))
            {
                return i;
            }
        }
    }
    else if (panel == OV_FOCUS_PROCS)
    {
        for (int i = 0; i < m->nb_procs; i++)
        {
            if (strcmp(m->procs[i].name, t->name) == 0 && (int) m->procs[i].PID == t->pid)
            {
                return i;
            }
        }
    }
    else if (panel == OV_FOCUS_FPS)
    {
        for (int i = 0; i < m->nb_fps; i++)
        {
            if (strcmp(m->fps[i].name, t->name) == 0)
            {
                return i;
            }
        }
    }
    return -1;
}

/**
 * ov_rows - model indices of the rows @panel draws: regex, loop isolation, then the relation of
 * the frozen subject (never of the hovered item). Return: number of rows.
 */
int ov_rows(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, int *fidx, int max)
{
    static OV_RELATED rel;
    const OV_RELATED *rp = NULL;

    if (lay->freeze && lay->freeze_focus < OV_FOCUS_GRAPH && lay->freeze_focus != panel)
    {
        ov_compute_related_frozen(lay, m, &rel);
        rp = &rel;
    }
    switch (panel)
    {
    case OV_FOCUS_STREAMS:
        return ov_filter_streams(lay, m, rp, fidx, max);
    case OV_FOCUS_PROCS:
        return ov_filter_procs(lay, m, rp, fidx, max);
    case OV_FOCUS_FPS:
        return ov_filter_fps(lay, m, rp, fidx, max);
    default:
        return 0;
    }
}

/**
 * ov_row_to_idx - model index drawn at @row of @panel, or -1.
 */
int ov_row_to_idx(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, int row)
{
    int fidx[OV_MAX_NODES];
    int n = ov_rows(lay, m, panel, fidx, OV_MAX_NODES);
    return (row >= 0 && row < n) ? fidx[row] : -1;
}

/** ov_sel_rowp - the cursor row field of @panel. */
static int *ov_sel_rowp(OV_LAYOUT *lay, ov_focus_t panel)
{
    return (panel == OV_FOCUS_STREAMS) ? &lay->sel_stream
           : (panel == OV_FOCUS_PROCS) ? &lay->sel_proc
                                       : &lay->sel_fps;
}

/**
 * ov_sel_row - put the cursor of @panel on drawn @row and capture its identity.
 * Return: model index, or -1 when @row is not drawn (cursor unchanged).
 */
int ov_sel_row(OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, int row)
{
    int idx = ov_row_to_idx(lay, m, panel, row);
    if (idx < 0 || panel >= OV_FOCUS_GRAPH)
    {
        return -1;
    }
    *ov_sel_rowp(lay, panel) = row;
    ov_tgt_set(&lay->sel_tgt[panel], m, panel, idx);
    return idx;
}

/**
 * ov_sel_model - put the cursor of @panel on model entry @idx (graph or search jumps).
 * Return: drawn row, or -1 with a command-log warning when a panel filter hides the entry.
 */
int ov_sel_model(OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, int idx)
{
    int fidx[OV_MAX_NODES];
    int n = ov_rows(lay, m, panel, fidx, OV_MAX_NODES);
    for (int r = 0; r < n; r++)
    {
        if (fidx[r] == idx)
        {
            *ov_sel_rowp(lay, panel) = r;
            ov_tgt_set(&lay->sel_tgt[panel], m, panel, idx);
            return r;
        }
    }
    OV_TARGET t;
    ov_tgt_set(&t, m, panel, idx);
    if (t.name[0] != '\0')
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN, "\"%s\" is hidden by the panel filter",
                       t.name);
    }
    return -1;
}

/**
 * ov_sel_idx - model index of the cursor of @panel (not the frozen item), or -1.
 */
int ov_sel_idx(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel)
{
    return (panel < OV_FOCUS_GRAPH) ? ov_tgt_find(m, panel, &lay->sel_tgt[panel]) : -1;
}

/**
 * ov_get_selected_stream_idx - model index of the selected stream (frozen one when frozen), or -1.
 */
int ov_get_selected_stream_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_TARGET *t =
        lay->freeze ? &lay->freeze_tgt[OV_FOCUS_STREAMS] : &lay->sel_tgt[OV_FOCUS_STREAMS];
    return ov_tgt_find(m, OV_FOCUS_STREAMS, t);
}

/**
 * ov_get_selected_proc_idx - model index of the selected process (frozen one when frozen), or -1.
 */
int ov_get_selected_proc_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_TARGET *t =
        lay->freeze ? &lay->freeze_tgt[OV_FOCUS_PROCS] : &lay->sel_tgt[OV_FOCUS_PROCS];
    return ov_tgt_find(m, OV_FOCUS_PROCS, t);
}

/**
 * ov_get_selected_fps_idx - model index of the selected FPS (frozen one when frozen), or -1.
 */
int ov_get_selected_fps_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_TARGET *t = lay->freeze ? &lay->freeze_tgt[OV_FOCUS_FPS] : &lay->sel_tgt[OV_FOCUS_FPS];
    return ov_tgt_find(m, OV_FOCUS_FPS, t);
}

/** ov_sel_clamp_scroll - keep @row visible in a panel of @height lines (3 lines of chrome). */
static void ov_sel_clamp_scroll(int row, int height, int *scroll)
{
    int page_h = height - 3;
    if (page_h <= 0)
    {
        return;
    }
    if (row < *scroll)
    {
        *scroll = row;
    }
    if (row >= *scroll + page_h)
    {
        *scroll = row - page_h + 1;
    }
}

/**
 * ov_sel_sync - re-derive rows from identities after a new model, a sort or a filter change.
 *
 * A cursor whose entity left the model moves to the neighbouring row (never back to row 0) and
 * the command log says so. A freeze whose subject left the model is released.
 */
void ov_sel_sync(OV_LAYOUT *lay, const OV_MODEL *m)
{
    static int fidx[OV_MAX_NODES];
    int *scrolls[OV_NB_LIST_PANELS] = { &lay->scroll_stream, &lay->scroll_proc, &lay->scroll_fps };
    const int heights[OV_NB_LIST_PANELS] = { lay->r_streams.height, lay->r_procs.height,
                                             lay->r_fps.height };
    int      *frows[OV_NB_LIST_PANELS]   = { &lay->freeze_sel_stream, &lay->freeze_sel_proc,
                                             &lay->freeze_sel_fps };

    if (lay->freeze && lay->freeze_focus < OV_FOCUS_GRAPH &&
        ov_tgt_find(m, lay->freeze_focus, &lay->freeze_tgt[lay->freeze_focus]) < 0)
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Frozen item \"%s\" is gone: freeze released",
                       lay->freeze_tgt[lay->freeze_focus].name);
        lay->freeze = 0;
    }

    for (int p = 0; p < OV_NB_LIST_PANELS; p++)
    {
        ov_focus_t panel = (ov_focus_t) p;
        int       *row   = ov_sel_rowp(lay, panel);
        int        n     = ov_rows(lay, m, panel, fidx, OV_MAX_NODES);
        if (n == 0)
        {
            *row = 0;
            memset(&lay->sel_tgt[p], 0, sizeof(lay->sel_tgt[p]));
        }
        else
        {
            int idx = ov_tgt_find(m, panel, &lay->sel_tgt[p]);
            int pos = -1;
            for (int r = 0; r < n && idx >= 0; r++)
            {
                if (fidx[r] == idx)
                {
                    pos = r;
                    break;
                }
            }
            if (pos < 0)
            {
                pos = (*row < 0) ? 0 : ((*row >= n) ? n - 1 : *row);
                if (lay->sel_tgt[p].name[0] != '\0' && idx < 0)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "\"%s\" is gone; cursor on \"%s\"",
                                   lay->sel_tgt[p].name,
                                   (panel == OV_FOCUS_STREAMS) ? m->streams[fidx[pos]].name
                                   : (panel == OV_FOCUS_PROCS) ? m->procs[fidx[pos]].name
                                                               : m->fps[fidx[pos]].name);
                }
                ov_tgt_set(&lay->sel_tgt[p], m, panel, fidx[pos]);
            }
            *row = pos;
            ov_sel_clamp_scroll(pos, heights[p], scrolls[p]);
        }

        *frows[p] = -1;
        if (lay->freeze)
        {
            int fi = ov_tgt_find(m, panel, &lay->freeze_tgt[p]);
            for (int r = 0; r < n && fi >= 0; r++)
            {
                if (fidx[r] == fi)
                {
                    *frows[p] = r;
                    break;
                }
            }
        }
    }
}
