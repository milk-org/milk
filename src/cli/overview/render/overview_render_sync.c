// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_sync.c
 * @brief   Selection bounds clamping and node synchronization for milk-CTRL.
 */

#include "overview_render_internal.h"
#include "overview_render_fps_params.h"
#include "overview_render_loops.h"
#include "overview_data_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_render__sync_selection - apply a pending sort, then re-derive cursor rows from identities.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 */
void ov_render__sync_selection(OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay->focus != OV_FOCUS_GRAPH)
    {
        lay->graph_root_focus = lay->focus; /* the graph cursor never re-roots the tree */
    }

    /* One-shot sort when the user asks (S, s, header click...): the order then stays until the
     * next request, also across new models; the selection follows by identity */
    if (lay->sort_pending)
    {
        int8_t depths[OV_MAX_NODES];
        for (int i = 0; i < OV_MAX_NODES; i++)
        {
            depths[i] = 127;
        }

        /* Ancestry depths relative to the selected (or frozen) entry */
        ov_focus_t focus    = lay->freeze ? lay->freeze_focus : lay->focus;
        int        sel_node = -1;
        int        idx      = -1;
        if (focus == OV_FOCUS_STREAMS && (idx = ov_get_selected_stream_idx(lay, m)) >= 0)
        {
            sel_node = m->streams[idx].node_idx;
        }
        else if (focus == OV_FOCUS_PROCS && (idx = ov_get_selected_proc_idx(lay, m)) >= 0)
        {
            sel_node = m->procs[idx].node_idx;
        }
        else if (focus == OV_FOCUS_FPS && (idx = ov_get_selected_fps_idx(lay, m)) >= 0)
        {
            sel_node = m->fps[idx].node_idx;
        }
        if (sel_node >= 0)
        {
            sg_mode_t smode = (focus == OV_FOCUS_FPS) ? SG_MODE_FPS : SG_MODE_FULL;
            sg_compute_node_depths(m, sel_node, smode, depths);
        }
        ov_sort_set_depths(depths);

        for (int p = 0; p < OV_NB_LIST_PANELS; p++)
        {
            ov_order_sort(m, p, lay->panel[p].sort_key, lay->panel[p].sort_dir);
        }
        lay->sort_pending = 0;
    }

    /* Rows follow the display order (ov_order_get); re-derive the cursor rows from identities */
    ov_sel_sync(lay, m);

    /* F5 parameter tree follows the FPS cursor: keep a valid parameter row */
    int cur_fidx = ov_sel_idx(lay, m, OV_FOCUS_FPS);
    if (lay->view == OV_VIEW_FPS && cur_fidx >= 0)
    {
        static fps_tree_item_t items[1024];
        int nitems = ov_get_fps_tree_items(&m->fps[cur_fidx], lay->fps_param_path, items, 1024);
        if (nitems > 0)
        {
            if (lay->fps_param_sel < 0)
            {
                lay->fps_param_sel = 0;
            }
            else if (lay->fps_param_sel >= nitems)
            {
                lay->fps_param_sel = nitems - 1;
            }
        }
    }
}
