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

/** ov_tgt_match - 1 if model entry @i of @panel has identity @t. */
static int ov_tgt_match(const OV_MODEL *m, ov_focus_t panel, int i, const OV_TARGET *t)
{
    if (panel == OV_FOCUS_STREAMS)
    {
        return strcmp(m->streams[i].name, t->name) == 0 &&
               (t->inode == 0 || (uint64_t) m->streams[i].inode == t->inode);
    }
    if (panel == OV_FOCUS_PROCS)
    {
        return strcmp(m->procs[i].name, t->name) == 0 && (int) m->procs[i].PID == t->pid &&
               (t->start == 0 || m->procs[i].start_ticks == 0 ||
                m->procs[i].start_ticks == t->start);
    }
    return strcmp(m->fps[i].name, t->name) == 0;
}

#define OV_ID_SLOTS 32768 /* power of two, > 2 * OV_MAX_STREAMS */

/* Name index of one panel of one published model: slot = model index + 1, 0 = empty */
typedef struct
{
    const OV_MODEL *m;
    uint64_t        seq;
    int16_t         slot[OV_ID_SLOTS];
} OV_ID_INDEX;

static OV_ID_INDEX s_id[OV_NB_LIST_PANELS];

/** ov_id_hash - FNV-1a of @name, folded to a slot. */
static uint32_t ov_id_hash(const char *name)
{
    uint32_t h = 2166136261u;
    for (; *name != '\0'; name++)
    {
        h = (h ^ (uint8_t) *name) * 16777619u;
    }
    return h & (OV_ID_SLOTS - 1);
}

/** ov_id_index - the name index of @panel for @m, rebuilt when @m is a new publication. */
static const OV_ID_INDEX *ov_id_index(const OV_MODEL *m, ov_focus_t panel)
{
    OV_ID_INDEX *x = &s_id[panel];
    if (x->m != m || x->seq != m->seq)
    {
        int n = (panel == OV_FOCUS_STREAMS) ? m->nb_streams
                : (panel == OV_FOCUS_PROCS) ? m->nb_procs
                                            : m->nb_fps;
        memset(x->slot, 0, sizeof(x->slot));
        for (int i = 0; i < n; i++)
        {
            const char *name = (panel == OV_FOCUS_STREAMS) ? m->streams[i].name
                               : (panel == OV_FOCUS_PROCS) ? m->procs[i].name
                                                           : m->fps[i].name;
            uint32_t    h    = ov_id_hash(name);
            while (x->slot[h] != 0)
            {
                h = (h + 1) & (OV_ID_SLOTS - 1);
            }
            x->slot[h] = (int16_t) (i + 1);
        }
        x->m   = m;
        x->seq = m->seq;
    }
    return x;
}

/**
 * ov_tgt_find - model index of the entry with identity @t, or -1. Published models (seq != 0) use a
 * name index built once per model; others are scanned.
 */
int ov_tgt_find(const OV_MODEL *m, ov_focus_t panel, const OV_TARGET *t)
{
    if (m == NULL || t == NULL || t->name[0] == '\0' || panel >= OV_NB_LIST_PANELS)
    {
        return -1;
    }
    if (m->seq != 0)
    {
        const OV_ID_INDEX *x = ov_id_index(m, panel);
        for (uint32_t h = ov_id_hash(t->name); x->slot[h] != 0; h = (h + 1) & (OV_ID_SLOTS - 1))
        {
            if (ov_tgt_match(m, panel, x->slot[h] - 1, t))
            {
                return x->slot[h] - 1;
            }
        }
        return -1;
    }
    int n = (panel == OV_FOCUS_STREAMS) ? m->nb_streams
            : (panel == OV_FOCUS_PROCS) ? m->nb_procs
                                        : m->nb_fps;
    for (int i = 0; i < n; i++)
    {
        if (ov_tgt_match(m, panel, i, t))
        {
            return i;
        }
    }
    return -1;
}

/**
 * ov_rows - copy into @fidx (at most @max) the model indices @panel draws, in drawn order: display
 * order, regex, loop isolation, then the frozen subject's relation. Return: number of rows.
 */
int ov_rows(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, int *fidx, int max)
{
    int        n;
    const int *rows = ov_view_rows(lay, m, panel, &n);
    n               = (n < max) ? n : max;
    memcpy(fidx, rows, (size_t) n * sizeof(int));
    return n;
}

/**
 * ov_row_to_idx - model index drawn at @row of @panel, or -1.
 */
int ov_row_to_idx(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, int row)
{
    int        n;
    const int *rows = ov_view_rows(lay, m, panel, &n);
    return (row >= 0 && row < n) ? rows[row] : -1;
}

/** ov_sel_rowp - the cursor row field of @panel. */
static int *ov_sel_rowp(OV_LAYOUT *lay, ov_focus_t panel)
{
    return &lay->panel[panel].sel;
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
    int r = ov_view_row_of(lay, m, panel, idx);
    if (r >= 0)
    {
        *ov_sel_rowp(lay, panel) = r;
        ov_tgt_set(&lay->panel[panel].tgt, m, panel, idx);
        return r;
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

/** ov_selected_idx - model index of @panel's selection (its frozen entry when frozen), or -1. */
static int ov_selected_idx(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel)
{
    const OV_PANEL_STATE *ps = &lay->panel[panel];
    return ov_tgt_find(m, panel, lay->freeze ? &ps->freeze_tgt : &ps->tgt);
}

/** ov_get_selected_stream_idx - model index of the selected stream (frozen one when frozen), or -1. */
int ov_get_selected_stream_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_selected_idx(lay, m, OV_FOCUS_STREAMS);
}

/** ov_get_selected_proc_idx - model index of the selected process (frozen one when frozen), or -1. */
int ov_get_selected_proc_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_selected_idx(lay, m, OV_FOCUS_PROCS);
}

/** ov_get_selected_fps_idx - model index of the selected FPS (frozen one when frozen), or -1. */
int ov_get_selected_fps_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_selected_idx(lay, m, OV_FOCUS_FPS);
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
    const int heights[OV_NB_LIST_PANELS] = { lay->r_streams.height, lay->r_procs.height,
                                             lay->r_fps.height };

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
        int        n;
        const int *fidx = ov_view_rows(lay, m, panel, &n);
        if (n == 0)
        {
            *row = 0;
            memset(&lay->panel[p].tgt, 0, sizeof(lay->panel[p].tgt));
        }
        else
        {
            int idx = ov_tgt_find(m, panel, &lay->panel[p].tgt);
            int pos = (idx >= 0) ? ov_view_row_of(lay, m, panel, idx) : -1;
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
            ov_sel_clamp_scroll(pos, heights[p], &lay->panel[p].scroll);
        }

        lay->panel[p].freeze_row = -1;
        if (lay->freeze)
        {
            int fi                   = ov_tgt_find(m, panel, &lay->panel[p].freeze_tgt);
            lay->panel[p].freeze_row = (fi >= 0) ? ov_view_row_of(lay, m, panel, fi) : -1;
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
