// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_graph.c
 */

#include "overview_render_internal.h"
#include "stream_graph.h"

/**
 * Renders on row 2, between header and panels.
 * Shows untruncated fields for the focused panel's
 * selected item.
 */
void ov_render_preview_line(OV_LAYOUT *lay, const OV_MODEL *m)
{
    int W = lay->term_cols;

    /* Reset button tracking */
    lay->nb_preview_btns = 0;

    ov_buf_pos(2, 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_hline(' ', W);
    ov_buf_pos(2, 1);

    /* Use frozen selection when freeze is active */
    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int        ssel  = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    int        psel  = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
    int        fsel  = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;

    char     line[512];
    int      len         = 0;
    ov_rgb_t label_color = OV_FG_DIM;

    switch (focus)
    {
    case OV_FOCUS_STREAMS:
    {
        label_color = OV_FG_STREAM;
        if (ssel < 0 || ssel >= m->nb_streams)
        {
            break;
        }
        const OV_STREAM *s = &m->streams[ssel];
        char             szb[32];
        if (s->naxis == 1)
        {
            snprintf(szb, sizeof(szb), "%u", (unsigned) s->size[0]);
        }
        else if (s->naxis == 2)
        {
            snprintf(szb, sizeof(szb), "%ux%u", (unsigned) s->size[0], (unsigned) s->size[1]);
        }
        else
        {
            snprintf(szb, sizeof(szb), "%ux%ux%u", (unsigned) s->size[0], (unsigned) s->size[1],
                     (unsigned) s->size[2]);
        }
        len = snprintf(line, sizeof(line),
                       " STM  %s  %s %s"
                       "  Hz:%.1f  ino:%" PRIu64 ""
                       "  own:%d  cnt:%" PRIu64 ""
                       "  wpid:%d  sem:%d",
                       s->name, render_dtype(s->datatype), szb, s->update_hz, (uint64_t) s->inode,
                       (int) s->ownerPID, (uint64_t) s->cnt0, (int) s->write_pid, s->nb_sem);
        break;
    }
    case OV_FOCUS_PROCS:
    {
        label_color = OV_FG_PROC;
        if (psel < 0 || psel >= m->nb_procs)
        {
            break;
        }
        const OV_PROC *p = &m->procs[psel];
        const char    *sl;
        switch (p->loopstat)
        {
        case 0:
            sl = "IDLE";
            break;
        case 1:
            sl = "RUN";
            break;
        case 2:
            sl = "PAUS";
            break;
        case 3:
            sl = "TERM";
            break;
        case 4:
            sl = "ERR";
            break;
        default:
            sl = "??";
            break;
        }
        len = snprintf(line, sizeof(line),
                       " PRC  %s  PID:%d  %s"
                       "  Hz:%.1f  trig:%s"
                       "  sem:%d  loop:%" PRId64 ""
                       "  miss:%d  prio:%d",
                       p->name, (int) p->PID, sl, p->loop_hz,
                       p->trigstreamname[0] ? p->trigstreamname : "-", p->triggersem,
                       (int64_t) p->loopcnt, p->triggermissed, p->rt_priority);
        break;
    }
    case OV_FOCUS_FPS:
    {
        label_color = OV_FG_FPS;
        if (fsel < 0 || fsel >= m->nb_fps)
        {
            break;
        }
        const OV_FPS *f = &m->fps[fsel];
        len = snprintf(line, sizeof(line),
                       " FPS  %s  C:%s R:%s"
                       "  st:%08X  cpid:%d  rpid:%d"
                       "  %s",
                       f->name, f->conf_alive ? "Y" : "-", f->run_alive ? "Y" : "-", f->md_status,
                       (int) f->confpid, (int) f->runpid, f->description);
        break;
    }
    default:
        break;
    }

    if (len > 0)
    {
        /* Label badge */
        ov_theme_bg(label_color);
        ov_buf_fg(0, 0, 0);
        ov_buf_bold();
        ov_buf_printf("%.*s", 5, line);
        ov_buf_reset_attr();

        /* Remaining content */
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_TEXT);
        int rem = len - 5;
        if (rem > W - 5)
        {
            rem = W - 5;
        }
        if (rem > 0)
        {
            ov_buf_printf("%.*s", rem, line + 5);
        }
    }

    /* ---- Action buttons (right-aligned) ---- */
    {
        /* Button definitions: label, bg color, id,
         * always_active (rendered even w/o ctrl_mode) */
        struct
        {
            const char *label;
            ov_rgb_t    bg;
            int         id;
            int         always_active;
        } btns[6];
        int nb = 0;

        if (focus == OV_FOCUS_PROCS && psel >= 0 && psel < m->nb_procs)
        {
            const OV_PROC *p = &m->procs[psel];
            /* Pause / Resume */
            if (p->loopstat == 2)
            {
                btns[nb].label         = " [p] \xe2\x96\xb6 Resume ";
                btns[nb].bg            = (ov_rgb_t) { 30, 120, 60 };
                btns[nb].id            = OV_BTN_PROC_PAUSE;
                btns[nb].always_active = 0;
                nb++;

                /* Step */
                btns[nb].label         = " [s] \xe2\x8f\xad Step ";
                btns[nb].bg            = (ov_rgb_t) { 120, 100, 30 };
                btns[nb].id            = OV_BTN_PROC_STEP;
                btns[nb].always_active = 0;
                nb++;
            }
            else
            {
                btns[nb].label         = " [p] \xe2\x8f\xb8 Pause ";
                btns[nb].bg            = (ov_rgb_t) { 50, 90, 160 };
                btns[nb].id            = OV_BTN_PROC_PAUSE;
                btns[nb].always_active = 0;
                nb++;
            }
            /* Exit (clean stop) */
            btns[nb].label         = " [e] \xe2\x8f\xbb Exit ";
            btns[nb].bg            = (ov_rgb_t) { 160, 120, 30 };
            btns[nb].id            = OV_BTN_PROC_EXIT;
            btns[nb].always_active = 0;
            nb++;
            /* Kill (SIGTERM) */
            btns[nb].label         = " [k] \xe2\x98\xa0 Kill ";
            btns[nb].bg            = (ov_rgb_t) { 180, 40, 40 };
            btns[nb].id            = OV_BTN_PROC_KILL;
            btns[nb].always_active = 0;
            nb++;
            /* Inspect */
            btns[nb].label         = " [i] Inspect ";
            btns[nb].bg            = (ov_rgb_t) { 50, 90, 160 };
            btns[nb].id            = OV_BTN_INSPECT;
            btns[nb].always_active = 1;
            nb++;
        }
        else if (focus == OV_FOCUS_FPS && fsel >= 0 && fsel < m->nb_fps)
        {
            const OV_FPS *f = &m->fps[fsel];
            /* Conf toggle */
            btns[nb].label = f->conf_alive ? " [s] \xe2\x96\xa0 Conf " : " [s] \xe2\x96\xb6 Conf ";
            btns[nb].bg = f->conf_alive ? (ov_rgb_t) { 160, 120, 30 } : (ov_rgb_t) { 30, 120, 60 };
            btns[nb].id = OV_BTN_FPS_CONF;
            btns[nb].always_active = 0;
            nb++;
            /* Run toggle */
            btns[nb].label = f->run_alive ? " [r] \xe2\x96\xa0 Run " : " [r] \xe2\x96\xb6 Run ";
            btns[nb].bg = f->run_alive ? (ov_rgb_t) { 160, 120, 30 } : (ov_rgb_t) { 30, 120, 60 };
            btns[nb].id = OV_BTN_FPS_RUN;
            btns[nb].always_active = 0;
            nb++;
            /* Kill */
            btns[nb].label         = " [k] \xe2\x98\xa0 Kill ";
            btns[nb].bg            = (ov_rgb_t) { 180, 40, 40 };
            btns[nb].id            = OV_BTN_FPS_KILL;
            btns[nb].always_active = 0;
            nb++;
            /* Inspect */
            btns[nb].label         = " [i] Inspect ";
            btns[nb].bg            = (ov_rgb_t) { 50, 90, 160 };
            btns[nb].id            = OV_BTN_INSPECT;
            btns[nb].always_active = 1;
            nb++;
        }
        else if (focus == OV_FOCUS_STREAMS && ssel >= 0 && ssel < m->nb_streams)
        {
            /* Delete stream */
            btns[nb].label         = " [DEL] \xe2\x9c\x95 Delete ";
            btns[nb].bg            = (ov_rgb_t) { 180, 40, 40 };
            btns[nb].id            = OV_BTN_STREAM_DEL;
            btns[nb].always_active = 0;
            nb++;
            /* Inspect */
            btns[nb].label         = " [i] Inspect ";
            btns[nb].bg            = (ov_rgb_t) { 50, 90, 160 };
            btns[nb].id            = OV_BTN_INSPECT;
            btns[nb].always_active = 1;
            nb++;
        }

        if (nb > 0)
        {
            /* Compute total width of all buttons
             * (1 space gap between each) */
            int btn_widths[6];
            int total_w = 0;
            for (int bi = 0; bi < nb; bi++)
            {
                /* strlen gives byte count; UTF-8
                 * symbols use 3 bytes per glyph,
                 * so subtract 2 per symbol for
                 * display width */
                int blen = (int) strlen(btns[bi].label);
                /* Count UTF-8 leading bytes
                 * (0xE2) to find symbol count */
                int syms = 0;
                for (int k = 0; k < blen; k++)
                {
                    if ((unsigned char) btns[bi].label[k] == 0xE2)
                    {
                        syms++;
                    }
                }
                btn_widths[bi] = blen - syms * 2;
                total_w += btn_widths[bi];
            }
            total_w += (nb - 1); /* gaps */

            /* Reserve space for SELECTED badge */
            int reserved  = lay->freeze ? 12 : 1;
            int start_col = W - total_w - reserved + 1;
            if (start_col < 1)
            {
                start_col = 1;
            }

            int col = start_col;
            for (int bi = 0; bi < nb; bi++)
            {
                ov_buf_pos(2, col);
                if (btns[bi].always_active)
                {
                    /* Always-active buttons (Inspect)
                     * render in full color */
                    ov_buf_bg(btns[bi].bg.r, btns[bi].bg.g, btns[bi].bg.b);
                    ov_buf_fg(255, 255, 255);
                }
                else if (!lay->ctrl_mode)
                {
                    ov_buf_bg(60, 60, 60);
                    ov_buf_fg(180, 180, 180);
                }
                else
                {
                    ov_buf_bg(btns[bi].bg.r, btns[bi].bg.g, btns[bi].bg.b);
                    ov_buf_fg(255, 255, 255);
                }
                ov_buf_bold();
                ov_buf_printf("%s", btns[bi].label);
                ov_buf_reset_attr();

                /* Record button position */
                if (lay->nb_preview_btns < 6)
                {
                    int idx                      = lay->nb_preview_btns;
                    lay->preview_btns[idx].col   = col;
                    lay->preview_btns[idx].width = btn_widths[bi];
                    lay->preview_btns[idx].id    = btns[bi].id;
                    lay->nb_preview_btns++;
                }

                col += btn_widths[bi] + 1;
            }
        }
    }

    /* [SELECTED] badge on right edge when frozen */
    if (lay->freeze)
    {
        const char *badge = " SELECTED ";
        int         bw    = 10;
        int         col   = W - bw + 1;
        if (col > 1)
        {
            ov_buf_pos(2, col);
            ov_buf_bg(60, 130, 200);
            ov_buf_fg(255, 255, 255);
            ov_buf_bold();
            ov_buf_printf("%s", badge);
        }
    }

    ov_buf_reset_attr();
}

static int get_graph_start_node(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_focus_t eff_focus   = lay->freeze ? lay->freeze_focus : lay->focus;
    int        target_type = -1;
    int        target_idx  = -1;

    if (eff_focus == OV_FOCUS_STREAMS || eff_focus == OV_FOCUS_GRAPH)
    {
        target_type = OV_NODE_STREAM;
        target_idx  = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    }
    else if (eff_focus == OV_FOCUS_PROCS)
    {
        target_type = OV_NODE_PROC;
        target_idx  = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
    }
    else if (eff_focus == OV_FOCUS_FPS)
    {
        target_type = OV_NODE_FPS;
        target_idx  = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;
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
 * @brief Render the stream-process graph panel.
 */
void ov_render_graph_panel(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT     r      = lay->r_graph;
    const char *tabs[] = { "CONNECTIONS", "DETAILS", "RESOURCES" };
    ov_draw_panel_tabs(r.row, r.col, r.height, r.width, tabs, 3, lay->graph_tab_mode, OV_FG_CONN,
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
