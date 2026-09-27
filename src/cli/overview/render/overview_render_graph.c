// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_graph.c
 * @brief Stream-process connection graph tree rendering.
 */

#include "overview_render_internal.h"
#include "overview_data_loops.h"
#include "stream_graph.h"

/**
 * get_graph_start_node - resolve source graph node index for connection tree root.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: Node index in m->nodes, or -1 if no matching selection.
 */
static int get_graph_start_node(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_focus_t eff_focus   = lay->freeze ? lay->freeze_focus : lay->focus;
    int        target_type = -1;
    int        target_idx  = -1;

    if (eff_focus == OV_FOCUS_STREAMS || eff_focus == OV_FOCUS_GRAPH)
    {
        target_type = OV_NODE_STREAM;
        target_idx  = ov_get_selected_stream_idx(lay, m);
    }
    else if (eff_focus == OV_FOCUS_PROCS)
    {
        target_type = OV_NODE_PROC;
        target_idx  = ov_get_selected_proc_idx(lay, m);
    }
    else if (eff_focus == OV_FOCUS_FPS)
    {
        target_type = OV_NODE_FPS;
        target_idx  = ov_get_selected_fps_idx(lay, m);
    }

    if (target_type != -1 && target_idx != -1)
    {
        for (int i = 0; i < m->nb_nodes; i++)
        {
            if (m->nodes[i].type == target_type && m->nodes[i].index == target_idx)
            {
                return i;
            }
        }
    }
    return -1;
}

/**
 * ov_render_graph_panel - render the stream-process graph panel.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 */
void ov_render_graph_panel(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT     r      = lay->r_graph;
    const char *tabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
    ov_draw_panel_tabs(r.row, r.col, r.height, r.width, tabs, 4, lay->graph_tab_mode, OV_FG_CONN,
                       lay->focus == OV_FOCUS_GRAPH);

    int max_rows = r.height - 3;
    int row      = r.row + 1;

    /* Render Header */
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);
    char htext[256];
    int  hlen =
        snprintf(htext, sizeof(htext), " %-12s      %s", "MODE", sg_mode_label(lay->lineage_mode));
    ov_buf_printf("%s", htext);
    render_pad_spaces(hlen, r.width);
    row++;

    int          start_node = get_graph_start_node(lay, m);
    SG_TREE_NODE rnodes[OV_MAX_NODES];
    int          nb_rnodes = 0;

    if (start_node >= 0)
    {
        nb_rnodes = sg_compute_render_tree(m, start_node, lay->lineage_mode, rnodes);
    }

    if (nb_rnodes == 0)
    {
        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("  ");
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("No graph available");
        render_pad_spaces(2 + 18, r.width);
        row++;
        for (int i = 1; i < max_rows; i++, row++)
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return;
    }

    int rendered_rows = 0;
    int scroll        = lay->scroll_graph;

    for (int ri = scroll; ri < nb_rnodes && rendered_rows < max_rows; ri++)
    {
        const SG_TREE_NODE *rn = &rnodes[ri];

        ov_buf_pos(row, r.col + 1);

        int      is_sel   = (ri == lay->sel_graph && lay->focus == OV_FOCUS_GRAPH);
        int      is_hover = (lay->mouse_hover && lay->hover_view == OV_FOCUS_GRAPH &&
                             lay->graph_tab_mode == 0 && ri == lay->hover_idx);
        ov_rgb_t row_bg   = is_hover ? OV_BG_HOVER : OV_BG_PANEL;
        int      use_ul   = 0;
        ov_rgb_t ul_color = { 0, 0, 0 };

        if (is_sel)
        {
            use_ul   = 1;
            ul_color = OV_FG_BRIGHT;
        }

        ov_theme_bg(row_bg);
        if (use_ul)
        {
            ov_theme_ul(ul_color);
            ov_buf_underline();
        }
        ov_buf_printf(" ");

        int printed = 1;
        int avail   = r.width - 2;

#define GRAPH_FIELD(color, fmt, ...)                                \
    do                                                              \
    {                                                               \
        char _fb[128];                                              \
        int  _fl  = snprintf(_fb, sizeof(_fb), fmt, ##__VA_ARGS__); \
        int  _vis = _fl;                                            \
        int  _max = avail - printed;                                \
        if (_vis > _max)                                            \
            _vis = _max;                                            \
        if (_vis > 0)                                               \
        {                                                           \
            ov_theme_fg(color);                                     \
            ov_buf_printf("%.*s", _vis, _fb);                       \
            printed += _vis;                                        \
        }                                                           \
    } while (0)

        /* Draw tree prefix */
        GRAPH_FIELD(OV_FG_DIM, "%s", rn->tree_prefix);

        /* Selection marker / Target marker */
        if (rn->is_target)
        {
            GRAPH_FIELD(OV_FG_WARN, "\xe2\x96\xb6 "); /* Arrow */
        }

        /* Draw node name */
        ov_rgb_t name_color = rn->is_target ? OV_FG_WARN : OV_FG_STREAM;
        int      hl_stream  = (lay->mouse_hover && lay->hover_global_stream >= 0 &&
                               rn->stream_idx == lay->hover_global_stream);
        if (hl_stream)
        {
            ov_theme_bg(OV_BG_HOVER);
        }
        GRAPH_FIELD(name_color, "%s", rn->name);
        if (hl_stream)
        {
            ov_theme_bg(row_bg);
        }

        /* Loop badge & cycle closure indicator */
        if (rn->is_loop)
        {
            if (rn->stream_idx >= 0 && rn->stream_idx < m->nb_streams &&
                m->streams[rn->stream_idx].primary_loop_id > 0)
            {
                int         lid   = m->streams[rn->stream_idx].primary_loop_id;
                const char *lname = ov_get_loop_name(m, lid);
                GRAPH_FIELD(OV_FG_LOOP, " \xe2\x86\xba [L%02d: %s]", lid, lname);
                if (m->streams[rn->stream_idx].nb_loops > 1)
                {
                    GRAPH_FIELD(OV_FG_LOOP_SHARED, " (\xe2\xae\x82 %d loops)",
                                m->streams[rn->stream_idx].nb_loops);
                }
            }
            else
            {
                GRAPH_FIELD(OV_FG_LOOP, " \xe2\x86\xba (loop)");
            }
        }
        else if (rn->stream_idx >= 0 && rn->stream_idx < m->nb_streams &&
                 m->streams[rn->stream_idx].nb_loops > 0)
        {
            int lid = m->streams[rn->stream_idx].primary_loop_id;
            if (m->streams[rn->stream_idx].nb_loops > 1)
            {
                GRAPH_FIELD(OV_FG_LOOP_SHARED, " [\xe2\xae\x82 L%02d+]", lid);
            }
            else
            {
                GRAPH_FIELD(OV_FG_LOOP, " [L%02d]", lid);
            }
        }

        /* Draw reader proc */
        if (rn->reader_name[0] != '\0')
        {
            GRAPH_FIELD(OV_FG_DIM, " [");
            if (rn->is_target_proc)
            {
                GRAPH_FIELD(OV_FG_WARN, "\xe2\x96\xb6 ");
            }
            ov_rgb_t proc_color = rn->is_target_proc ? OV_FG_WARN : OV_FG_PROC;

            int proc_idx = -1;
            for (int i = 0; i < m->nb_procs; i++)
            {
                if (strcmp(m->procs[i].name, rn->reader_name) == 0)
                {
                    proc_idx = i;
                    break;
                }
            }
            int hl_proc = (lay->mouse_hover && lay->hover_global_proc >= 0 &&
                           proc_idx == lay->hover_global_proc);

            if (hl_proc)
            {
                ov_theme_bg(OV_BG_HOVER);
            }
            GRAPH_FIELD(proc_color, "%s", rn->reader_name);
            if (proc_idx >= 0 && proc_idx < m->nb_procs && m->procs[proc_idx].nb_loops > 0)
            {
                int plid = m->procs[proc_idx].primary_loop_id;
                if (m->procs[proc_idx].nb_loops > 1)
                {
                    GRAPH_FIELD(OV_FG_LOOP_SHARED, " \xe2\xae\x82L%02d+", plid);
                }
                else
                {
                    GRAPH_FIELD(OV_FG_LOOP, " L%02d", plid);
                }
            }
            if (hl_proc)
            {
                ov_theme_bg(row_bg);
            }

            GRAPH_FIELD(OV_FG_DIM, "]");
        }

#undef GRAPH_FIELD

        render_pad_spaces(printed, r.width);
        ov_buf_reset_attr();
        row++;
        rendered_rows++;
    }

    for (; rendered_rows < max_rows; rendered_rows++, row++)
    {
        clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
    }

    ov_buf_reset_attr();
}
