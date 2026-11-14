// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_hittest_cols.c
 * @brief Header click detection, column sort hit-tests, and global hover resolution
 */

#include "overview_input_internal.h"

/**
 * ov_input__table_header_click - sort list panel @panel by the column clicked at screen column @mc
 * (the same column again flips the direction). Return: 1 when a sortable column was hit.
 */
int ov_input__table_header_click(OV_LAYOUT *lay, ov_focus_t panel, int mc)
{
    const OV_TABLE *t = ov_table_for(panel);
    int             c = (t != NULL) ? ov_table_col_at(panel, mc) : -1;
    if (c < 0 || t->cols[c].sort_key < 0)
    {
        return 0;
    }
    OV_PANEL_STATE *ps = &lay->panel[panel];
    ps->sort_dir       = (ps->sort_key == t->cols[c].sort_key) ? !ps->sort_dir : 0;
    ps->sort_key       = t->cols[c].sort_key;
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort %s by %s %s", t->what, t->cols[c].label,
                   ps->sort_dir ? "\xe2\x96\xbc" : "\xe2\x96\xb2");
    lay->sort_pending = 1;
    ov_scan_force_update();
    return 1;
}

/**
 * ov_hittest_resolve_globals - Resolve hover index into global indices for cross-panel lineage
 * @lay: Pointer to layout structure
 * @m:   Pointer to data model snapshot
 */
void ov_hittest_resolve_globals(OV_LAYOUT *lay, const OV_MODEL *m)
{
    for (int p = 0; p < OV_NB_LIST_PANELS; p++)
    {
        lay->panel[p].hover_idx = -1;
    }

    if (!lay->mouse_hover || lay->hover_idx < 0)
    {
        return;
    }

    /* hover_idx is a drawn row: resolve it through the same rows the panel draws */
    if (lay->hover_view == OV_FOCUS_STREAMS)
    {
        lay->panel[OV_FOCUS_STREAMS].hover_idx =
            ov_row_to_idx(lay, m, OV_FOCUS_STREAMS, lay->hover_idx);
    }
    else if (lay->hover_view == OV_FOCUS_PROCS)
    {
        lay->panel[OV_FOCUS_PROCS].hover_idx =
            ov_row_to_idx(lay, m, OV_FOCUS_PROCS, lay->hover_idx);
    }
    else if (lay->hover_view == OV_FOCUS_FPS)
    {
        lay->panel[OV_FOCUS_FPS].hover_idx = ov_row_to_idx(lay, m, OV_FOCUS_FPS, lay->hover_idx);
    }
    else if (lay->hover_view == OV_FOCUS_GRAPH)
    {
        int                 n;
        const SG_TREE_NODE *rows = ov_graph_rows(lay, m, &n);
        if (lay->hover_idx >= 0 && lay->hover_idx < n)
        {
            lay->panel[OV_FOCUS_STREAMS].hover_idx = rows[lay->hover_idx].stream_idx;
        }
    }
}
