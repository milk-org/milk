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
        t->pid   = (int) m->procs[idx].PID;
        t->start = m->procs[idx].start_ticks;
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
            if (strcmp(m->procs[i].name, t->name) == 0 && (int) m->procs[i].PID == t->pid &&
                (t->start == 0 || m->procs[i].start_ticks == 0 ||
                 m->procs[i].start_ticks == t->start))
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
    return (panel == OV_FOCUS_STREAMS) ? &lay->panel[OV_FOCUS_STREAMS].sel
           : (panel == OV_FOCUS_PROCS) ? &lay->panel[OV_FOCUS_PROCS].sel
                                       : &lay->panel[OV_FOCUS_FPS].sel;
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
    ov_tgt_set(&lay->panel[panel].tgt, m, panel, idx);
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
            ov_tgt_set(&lay->panel[panel].tgt, m, panel, idx);
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
    return (panel < OV_FOCUS_GRAPH) ? ov_tgt_find(m, panel, &lay->panel[panel].tgt) : -1;
}

/**
 * ov_get_selected_stream_idx - model index of the selected stream (frozen one when frozen), or -1.
 */
int ov_get_selected_stream_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_TARGET *t =
        lay->freeze ? &lay->panel[OV_FOCUS_STREAMS].freeze_tgt : &lay->panel[OV_FOCUS_STREAMS].tgt;
    return ov_tgt_find(m, OV_FOCUS_STREAMS, t);
}

/**
 * ov_get_selected_proc_idx - model index of the selected process (frozen one when frozen), or -1.
 */
int ov_get_selected_proc_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_TARGET *t =
        lay->freeze ? &lay->panel[OV_FOCUS_PROCS].freeze_tgt : &lay->panel[OV_FOCUS_PROCS].tgt;
    return ov_tgt_find(m, OV_FOCUS_PROCS, t);
}

/**
 * ov_get_selected_fps_idx - model index of the selected FPS (frozen one when frozen), or -1.
 */
int ov_get_selected_fps_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_TARGET *t =
        lay->freeze ? &lay->panel[OV_FOCUS_FPS].freeze_tgt : &lay->panel[OV_FOCUS_FPS].tgt;
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
    int       *scrolls[OV_NB_LIST_PANELS] = { &lay->panel[OV_FOCUS_STREAMS].scroll,
                                              &lay->panel[OV_FOCUS_PROCS].scroll,
                                              &lay->panel[OV_FOCUS_FPS].scroll };
    const int  heights[OV_NB_LIST_PANELS] = { lay->r_streams.height, lay->r_procs.height,
                                              lay->r_fps.height };
    int       *frows[OV_NB_LIST_PANELS]   = { &lay->panel[OV_FOCUS_STREAMS].freeze_row,
                                              &lay->panel[OV_FOCUS_PROCS].freeze_row,
                                              &lay->panel[OV_FOCUS_FPS].freeze_row };

    if (lay->freeze && lay->freeze_focus < OV_FOCUS_GRAPH &&
        ov_tgt_find(m, lay->freeze_focus, &lay->panel[lay->freeze_focus].freeze_tgt) < 0)
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Frozen item \"%s\" is gone: freeze released",
                       lay->panel[lay->freeze_focus].freeze_tgt.name);
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
            memset(&lay->panel[p].tgt, 0, sizeof(lay->panel[p].tgt));
        }
        else
        {
            int idx = ov_tgt_find(m, panel, &lay->panel[p].tgt);
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
                if (lay->panel[p].tgt.name[0] != '\0' && idx < 0)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "\"%s\" is gone; cursor on \"%s\"",
                                   lay->panel[p].tgt.name,
                                   (panel == OV_FOCUS_STREAMS) ? m->streams[fidx[pos]].name
                                   : (panel == OV_FOCUS_PROCS) ? m->procs[fidx[pos]].name
                                                               : m->fps[fidx[pos]].name);
                }
                ov_tgt_set(&lay->panel[p].tgt, m, panel, fidx[pos]);
            }
            *row = pos;
            ov_sel_clamp_scroll(pos, heights[p], scrolls[p]);
        }

        *frows[p] = -1;
        if (lay->freeze)
        {
            int fi = ov_tgt_find(m, panel, &lay->panel[p].freeze_tgt);
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

/** ov_msel_find - slot of the FPS batch mark @name, or -1. */
static int ov_msel_find(const OV_LAYOUT *lay, const char *name)
{
    for (int k = 0; k < lay->multi_sel_count; k++)
    {
        if (strcmp(lay->multi_sel_fps[k].name, name) == 0)
        {
            return k;
        }
    }
    return -1;
}

/** ov_msel_has - 1 if FPS @name carries a batch mark. */
int ov_msel_has(const OV_LAYOUT *lay, const char *name)
{
    return ov_msel_find(lay, name) >= 0;
}

/** ov_msel_set - add (@on) or remove the batch mark of FPS @name; set semantics. */
void ov_msel_set(OV_LAYOUT *lay, const char *name, int on)
{
    int k = ov_msel_find(lay, name);
    if (on && k < 0 && lay->multi_sel_count < OV_MSEL_MAX)
    {
        OV_TARGET *t = &lay->multi_sel_fps[lay->multi_sel_count++];
        memset(t, 0, sizeof(*t));
        snprintf(t->name, sizeof(t->name), "%s", name);
    }
    else if (!on && k >= 0)
    {
        lay->multi_sel_fps[k] = lay->multi_sel_fps[--lay->multi_sel_count];
    }
}

/** ov_msel_clear - drop every FPS batch mark. */
void ov_msel_clear(OV_LAYOUT *lay)
{
    lay->multi_sel_count = 0;
}

/**
 * ov_msel_title - FPS panel title showing the batch scope: "FPS", "FPS ◆N" or "FPS ◆N (K hidden)",
 * K counting marks not drawn (filtered out or gone). Return: @buf.
 */
const char *ov_msel_title(const OV_LAYOUT *lay, const OV_MODEL *m, char *buf, size_t n)
{
    static int fidx[OV_MAX_NODES];
    if (lay->multi_sel_count == 0)
    {
        snprintf(buf, n, "FPS");
        return buf;
    }
    int rows   = ov_rows(lay, m, OV_FOCUS_FPS, fidx, OV_MAX_NODES);
    int hidden = 0;
    for (int k = 0; k < lay->multi_sel_count; k++)
    {
        int fi    = ov_tgt_find(m, OV_FOCUS_FPS, &lay->multi_sel_fps[k]);
        int shown = 0;
        for (int r = 0; r < rows && fi >= 0 && !shown; r++)
        {
            shown = (fidx[r] == fi);
        }
        hidden += !shown;
    }
    if (hidden > 0)
    {
        snprintf(buf, n, "FPS \xe2\x97\x86%d (%d hidden)", lay->multi_sel_count, hidden);
    }
    else
    {
        snprintf(buf, n, "FPS \xe2\x97\x86%d", lay->multi_sel_count);
    }
    return buf;
}
