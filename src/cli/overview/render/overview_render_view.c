// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_view.c
 * @brief   Per-panel view cache: the rows a list panel draws, rebuilt only when their inputs change.
 *
 * The rows depend on the model, its display order, the panel's applied filter, loop isolation and
 * the frozen subject of another panel. Those inputs form a memcmp-able key; any change rebuilds the
 * panel's rows and their inverse (model index -> row). An identity index makes ov_tgt_find O(1).
 */

#include "overview_render_internal.h"
#include <string.h>

/* Inputs of one panel's rows: memset 0, then filled, so memcmp compares them */
typedef struct
{
    const OV_MODEL *m;
    uint64_t        seq;
    uint32_t        order_gen;
    int32_t         loop;         /* isolated loop index, -1 = none */
    int32_t         freeze_focus; /* panel whose frozen subject filters this one, -1 = none */
    OV_TARGET       freeze_tgt;
    char            filter[64]; /* applied regex, "" = none */
} OV_VIEW_KEY;

typedef struct
{
    OV_VIEW_KEY key;
    int         valid;
    int         n;
    int         rows[OV_MAX_NODES];   /* model indices in drawn order */
    int16_t     row_of[OV_MAX_NODES]; /* model index -> drawn row, -1 = not drawn */
} OV_PANEL_VIEW;

static OV_PANEL_VIEW s_view[OV_NB_LIST_PANELS];

/** ov_view_key - the inputs of @panel's rows (call after ov_order_get, which may rebuild the order). */
static void ov_view_key(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, OV_VIEW_KEY *k)
{
    memset(k, 0, sizeof(*k));
    k->m            = m;
    k->seq          = m->seq;
    k->order_gen    = ov_order_gen(panel);
    k->loop         = (lay->loop_filter_active && lay->sub[OV_SUB_LOOPS].sel >= 0 &&
                       lay->sub[OV_SUB_LOOPS].sel < m->nb_loops)
                          ? lay->sub[OV_SUB_LOOPS].sel
                          : -1;
    k->freeze_focus = -1;
    if (lay->freeze && lay->freeze_focus < OV_FOCUS_GRAPH && lay->freeze_focus != panel)
    {
        k->freeze_focus = lay->freeze_focus;
        k->freeze_tgt   = lay->panel[lay->freeze_focus].freeze_tgt;
    }
    snprintf(k->filter, sizeof(k->filter), "%s", ov_get_active_filter_for(lay, panel));
}

/** ov_view_build - filter @panel (model order), then lay the kept entries out in display order. */
static void ov_view_build(OV_PANEL_VIEW   *v,
                          const OV_LAYOUT *lay,
                          const OV_MODEL  *m,
                          ov_focus_t       panel,
                          const int       *order,
                          int              no)
{
    static OV_RELATED rel;
    static int        keep[OV_MAX_NODES];
    static uint8_t    mark[OV_MAX_NODES];
    const OV_RELATED *rp = NULL;
    int               n  = 0;

    if (v->key.freeze_focus >= 0)
    {
        ov_compute_related_frozen(lay, m, &rel);
        rp = &rel;
    }
    if (panel == OV_FOCUS_STREAMS)
    {
        n = ov_filter_streams(lay, m, rp, keep, OV_MAX_NODES);
    }
    else if (panel == OV_FOCUS_PROCS)
    {
        n = ov_filter_procs(lay, m, rp, keep, OV_MAX_NODES);
    }
    else
    {
        n = ov_filter_fps(lay, m, rp, keep, OV_MAX_NODES);
    }
    memset(mark, 0, (size_t) no);
    for (int i = 0; i < n; i++)
    {
        mark[keep[i]] = 1;
    }
    memset(v->row_of, 0xff, sizeof(v->row_of));
    v->n = 0;
    for (int r = 0; r < no; r++)
    {
        if (mark[order[r]])
        {
            v->row_of[order[r]] = (int16_t) v->n;
            v->rows[v->n++]     = order[r];
        }
    }
    v->valid = 1;
}

/**
 * ov_view_rows - model indices drawn by list panel @panel, in order (rebuilt only when an input
 * changed). Return: the rows (valid until the next model or state change), *@n of them.
 */
const int *ov_view_rows(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, int *n)
{
    *n = 0;
    if (m == NULL || panel >= OV_NB_LIST_PANELS)
    {
        return s_view[0].rows;
    }
    OV_PANEL_VIEW *v = &s_view[panel];
    int            no;
    const int     *order = ov_order_get(m, panel, &no);
    OV_VIEW_KEY    k;
    ov_view_key(lay, m, panel, &k);
    if (!v->valid || m->seq == 0 || memcmp(&k, &v->key, sizeof(k)) != 0)
    {
        v->key = k;
        ov_view_build(v, lay, m, panel, order, no);
    }
    *n = v->n;
    return v->rows;
}

/** ov_view_row_of - drawn row of model entry @idx in @panel, or -1. */
int ov_view_row_of(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t panel, int idx)
{
    int n;
    ov_view_rows(lay, m, panel, &n);
    return (panel < OV_NB_LIST_PANELS && idx >= 0 && idx < OV_MAX_NODES) ? s_view[panel].row_of[idx]
                                                                         : -1;
}
