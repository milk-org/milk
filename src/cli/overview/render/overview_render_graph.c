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
 * ov_graph_start_node - graph node of the selected (or frozen) entry: the connection tree root.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: Node index in m->nodes, or -1 if no matching selection.
 */
int ov_graph_start_node(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_focus_t eff_focus   = lay->freeze ? lay->freeze_focus : lay->focus;
    int        target_type = -1;
    int        target_idx  = -1;

    if (eff_focus == OV_FOCUS_GRAPH)
    {
        eff_focus = lay->graph_root_focus;
    }
    if (eff_focus == OV_FOCUS_STREAMS)
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

    if (target_idx < 0)
    {
        return -1;
    }
    return (target_type == OV_NODE_STREAM) ? m->streams[target_idx].node_idx
           : (target_type == OV_NODE_PROC) ? m->procs[target_idx].node_idx
           : (target_type == OV_NODE_FPS)  ? m->fps[target_idx].node_idx
                                           : -1;
}

/**
 * ov_graph_rows - the CONNECTIONS rows as drawn, for render and input alike; @n gets their count.
 * Static buffer: UI thread only; recomputed when the model, the root or the lineage mode changes.
 */
const SG_TREE_NODE *ov_graph_rows(const OV_LAYOUT *lay, const OV_MODEL *m, int *n)
{
    static SG_TREE_NODE    rows[OV_MAX_NODES];
    static int             nb;
    static const OV_MODEL *key_m; /* rows are for this model, root and lineage mode */
    static uint64_t        key_seq;
    static int             key_start = -2, key_mode = -1;
    int                    start = ov_graph_start_node(lay, m);
    if (m->seq == 0 || m != key_m || m->seq != key_seq || start != key_start ||
        lay->lineage_mode != key_mode)
    {
        nb        = (start >= 0) ? sg_compute_render_tree(m, start, lay->lineage_mode, rows) : 0;
        key_m     = m;
        key_seq   = m->seq;
        key_start = start;
        key_mode  = lay->lineage_mode;
    }
    *n = nb;
    return rows;
}

/* Health columns on the right of each CONNECTIONS row, in screen order */
#define OV_HCOLS 5
#define OV_GRAPH_TREE_MIN 24 /* the tree keeps at least this many columns */
static const struct
{
    const char *title;
    int         w;
    int         prio; /* lower stays longer when the panel is narrow */
} hcol[OV_HCOLS] = { { "Hz", 7, 1 },
                     { "STATE", 5, 0 },
                     { "EXEC", 7, 2 },
                     { "MISS", 5, 3 },
                     { "FPS", 14, 4 } };

/** health_cols - the health columns that fit next to the tree in @avail columns, in @shown. Return: width. */
static int health_cols(int avail, int *shown, int *n)
{
    int w = 0, keep[OV_HCOLS] = { 0 };
    for (int p = 0; p < OV_HCOLS; p++)
    {
        for (int c = 0; c < OV_HCOLS; c++)
        {
            if (hcol[c].prio == p && avail - (w + hcol[c].w + 1) >= OV_GRAPH_TREE_MIN)
            {
                keep[c] = 1;
                w += hcol[c].w + 1;
            }
        }
    }
    *n = 0;
    for (int c = 0; c < OV_HCOLS; c++)
    {
        if (keep[c])
        {
            shown[(*n)++] = c;
        }
    }
    return w;
}

/**
 * ov_render_graph_panel - render the stream-process graph panel.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 */
void ov_render_graph_panel(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r = lay->r_graph;
    ov_subtab_bar(lay, OV_FG_CONN);

    int max_rows = r.height - 3;
    int row      = r.row + 1;

    /* Render Header */
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);
    int  avail = r.width - 2;
    int  shown[OV_HCOLS], nshown;
    int  hw        = health_cols(avail, shown, &nshown);
    int  tree_cols = avail - hw;
    char htext[256];
    int  hlen =
        snprintf(htext, sizeof(htext), " %-12s      %s", "MODE", sg_mode_label(lay->lineage_mode));
    ov_buf_printf("%.*s", tree_cols, htext);
    hlen = (hlen < tree_cols) ? hlen : tree_cols;
    render_pad_spaces(hlen, tree_cols + 1);
    for (int c = 0; c < nshown; c++)
    {
        ov_buf_printf(" %*s", hcol[shown[c]].w, hcol[shown[c]].title);
    }
    render_pad_spaces(avail, r.width);
    row++;

    int                 nb_rnodes;
    const SG_TREE_NODE *rnodes = ov_graph_rows(lay, m, &nb_rnodes);

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

    int           rendered_rows = 0;
    int           scroll        = lay->sub[OV_SUB_CONN].scroll;
    static int8_t cause[OV_MAX_NODES], dim[OV_MAX_NODES];
    ov_graph_stalls(m, rnodes, nb_rnodes, cause, dim);

    for (int ri = scroll; ri < nb_rnodes && rendered_rows < max_rows; ri++)
    {
        const SG_TREE_NODE *rn = &rnodes[ri];

        ov_buf_pos(row, r.col + 1);

        int      is_sel   = (ri == lay->sub[OV_SUB_CONN].sel && lay->focus == OV_FOCUS_GRAPH);
        int      is_hover = (lay->mouse_hover && lay->hover_view == OV_FOCUS_GRAPH &&
                             lay->subtab == OV_SUB_CONN && ri == lay->hover_idx);
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

/* Print one field clipped to the columns left; counts columns, never cuts a UTF-8 character */
#define GRAPH_FIELD(color, fmt, ...)                                                           \
    do                                                                                         \
    {                                                                                          \
        char _fb[128];                                                                         \
        int  _fl   = snprintf(_fb, sizeof(_fb), fmt, ##__VA_ARGS__);                           \
        int  _cols = 0;                                                                        \
        int  _nb   = ov_utf8_fit(_fb, (_fl < (int) sizeof(_fb)) ? _fl : (int) sizeof(_fb) - 1, \
                                 tree_cols - printed, &_cols);                                 \
        if (_nb > 0)                                                                           \
        {                                                                                      \
            ov_theme_fg(dim[ri] ? OV_FG_DIM : (color));                                        \
            ov_buf_printf("%.*s", _nb, _fb);                                                   \
            printed += _cols;                                                                  \
        }                                                                                      \
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
        int      hl_stream  = (lay->mouse_hover && lay->panel[OV_FOCUS_STREAMS].hover_idx >= 0 &&
                               rn->stream_idx == lay->panel[OV_FOCUS_STREAMS].hover_idx);
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

            int proc_idx = sg_tree_reader_proc(m, rn);
            int hl_proc  = (lay->mouse_hover && lay->panel[OV_FOCUS_PROCS].hover_idx >= 0 &&
                            proc_idx == lay->panel[OV_FOCUS_PROCS].hover_idx);

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

        /* Health on the right: below a stall the row is dimmed, the stall itself says so in red */
        OV_ROW_HEALTH h;
        ov_graph_row_health(m, rn, &h);
        render_pad_spaces(printed, tree_cols + 1);
        char hz[16];
        snprintf(hz, sizeof(hz), "%s%s", h.hz,
                 h.mismatch ? "!" : ""); /* ! : the reader is slower */
        const char *txt[OV_HCOLS] = { hz, cause[ri] ? "STALL" : h.state, h.exec, h.miss, h.fps };
        ov_rgb_t    col[OV_HCOLS] = { h.mismatch ? OV_FG_WARN : OV_FG_TEXT,
                                      cause[ri] ? OV_FG_ERROR : h.state_fg, OV_FG_TEXT, OV_FG_TEXT,
                                      OV_FG_FPS };
        for (int c = 0; c < nshown; c++)
        {
            ov_theme_fg(dim[ri] ? OV_FG_DIM : col[shown[c]]);
            ov_buf_printf(" %*.*s", hcol[shown[c]].w, hcol[shown[c]].w, txt[shown[c]]);
        }
        render_pad_spaces(avail, r.width);
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
