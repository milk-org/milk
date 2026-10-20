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

static const OV_MODEL *g_last_model = NULL;

/**
 * ov_render__sync_selection - clamp selections, handle sort freezing, and sync node indices.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 */
void ov_render__sync_selection(OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_MODEL *mm = (OV_MODEL *) (uintptr_t) m;

    if (lay->focus != OV_FOCUS_GRAPH)
    {
        lay->graph_root_focus = lay->focus; /* the graph cursor never re-roots the tree */
    }

    /* One-shot sort: only runs once when the user presses S or s. Order stays frozen until
     * the user explicitly sorts again; the selection follows by identity. */
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
            sel_node = mm->streams[idx].node_idx;
        }
        else if (focus == OV_FOCUS_PROCS && (idx = ov_get_selected_proc_idx(lay, m)) >= 0)
        {
            sel_node = mm->procs[idx].node_idx;
        }
        else if (focus == OV_FOCUS_FPS && (idx = ov_get_selected_fps_idx(lay, m)) >= 0)
        {
            sel_node = mm->fps[idx].node_idx;
        }
        if (sel_node >= 0)
        {
            sg_mode_t smode = (focus == OV_FOCUS_FPS) ? SG_MODE_FPS : SG_MODE_FULL;
            sg_compute_node_depths(mm, sel_node, smode, depths);
        }
        ov_sort_set_depths(depths);

        ov_sort_streams(mm, lay->panel[OV_FOCUS_STREAMS].sort_key,
                        lay->panel[OV_FOCUS_STREAMS].sort_dir);
        ov_sort_procs(mm, lay->panel[OV_FOCUS_PROCS].sort_key, lay->panel[OV_FOCUS_PROCS].sort_dir);
        ov_sort_fps(mm, lay->panel[OV_FOCUS_FPS].sort_key, lay->panel[OV_FOCUS_FPS].sort_dir);
        ov_sort_freeze_snapshot(mm);

        lay->sort_pending = 0;
        g_last_model      = m;
    }
    else if (m != g_last_model)
    {
        /* A new scan model arrived: re-apply the saved order once, so items don't shuffle */
        ov_sort_apply_ranks(mm);
        g_last_model = m;
    }

    /* Patch graph node .index fields to the current array positions (the frozen relation and
     * F7 read them), then re-derive the cursor rows from the selection identities */
    for (int i = 0; i < mm->nb_streams; i++)
    {
        int ni = mm->streams[i].node_idx;
        if (ni >= 0 && ni < mm->nb_nodes)
        {
            mm->nodes[ni].index = i;
        }
    }
    for (int i = 0; i < mm->nb_fps; i++)
    {
        int ni = mm->fps[i].node_idx;
        if (ni >= 0 && ni < mm->nb_nodes)
        {
            mm->nodes[ni].index = i;
        }
    }
    for (int i = 0; i < mm->nb_procs; i++)
    {
        int ni = mm->procs[i].node_idx;
        if (ni >= 0 && ni < mm->nb_nodes)
        {
            mm->nodes[ni].index = i;
        }
    }
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
