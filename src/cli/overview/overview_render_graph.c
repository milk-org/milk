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
void ov_render_preview_line(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    int W = lay->term_cols;

    ov_buf_pos(2, 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_hline(' ', W);
    ov_buf_pos(2, 1);

    /* Use frozen selection when freeze is active */
    ov_focus_t focus = lay->freeze
                       ? lay->freeze_focus
                       : lay->focus;
    int ssel = lay->freeze
               ? lay->freeze_sel_stream
               : lay->sel_stream;
    int psel = lay->freeze
               ? lay->freeze_sel_proc
               : lay->sel_proc;
    int fsel = lay->freeze
               ? lay->freeze_sel_fps
               : lay->sel_fps;

    char line[512];
    int  len = 0;
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
        char szb[32];
        if (s->naxis == 1)
        {
            snprintf(szb, sizeof(szb), "%u",
                (unsigned) s->size[0]);
        }
        else if (s->naxis == 2)
        {
            snprintf(szb, sizeof(szb), "%ux%u",
                (unsigned) s->size[0],
                (unsigned) s->size[1]);
        }
        else
        {
            snprintf(szb, sizeof(szb),
                "%ux%ux%u",
                (unsigned) s->size[0],
                (unsigned) s->size[1],
                (unsigned) s->size[2]);
        }
        len = snprintf(line, sizeof(line),
            " STM  %s  %s %s"
            "  Hz:%.1f  ino:%lu"
            "  own:%d  cnt:%lu"
            "  wpid:%d  sem:%d",
            s->name,
            render_dtype(s->datatype),
            szb,
            s->update_hz,
            (unsigned long) s->inode,
            (int) s->ownerPID,
            (unsigned long) s->cnt0,
            (int) s->write_pid,
            s->nb_sem);
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
        const char *sl;
        switch (p->loopstat)
        {
        case 0:  sl = "IDLE"; break;
        case 1:  sl = "RUN";  break;
        case 2:  sl = "PAUS"; break;
        case 3:  sl = "TERM"; break;
        case 4:  sl = "ERR";  break;
        default: sl = "??";   break;
        }
        len = snprintf(line, sizeof(line),
            " PRC  %s  PID:%d  %s"
            "  Hz:%.1f  trig:%s"
            "  sem:%d  loop:%ld"
            "  miss:%d  prio:%d",
            p->name, (int) p->PID, sl,
            p->loop_hz,
            p->trigstreamname[0]
                ? p->trigstreamname : "-",
            p->triggersem,
            (long) p->loopcnt,
            p->triggermissed,
            p->rt_priority);
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
            f->name,
            f->conf_alive ? "Y" : "-",
            f->run_alive  ? "Y" : "-",
            f->md_status,
            (int) f->confpid,
            (int) f->runpid,
            f->description);
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

    /* [FREEZE] badge on right edge when frozen */
    if (lay->freeze)
    {
        const char *badge = " FREEZE ";
        int bw = 8;
        int col = W - bw + 1;
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


void ov_render_graph_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    OV_RECT r = lay->r_graph;
    ov_draw_panel_border(r.row, r.col, r.height, r.width, "CONNECTIONS", OV_FG_CONN, lay->focus == OV_FOCUS_GRAPH);

    int max_rows = r.height - 3;
    int row = r.row + 1;

    /* Render Header */
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);
    char htext[256];
    int hlen = snprintf(htext, sizeof(htext), " %-12s %-4s %-12s %-6s %-16s %-4s %-5s %-4s", 
                        "PROCESS", "PID", "STREAM", "STATUS", "LOOPCNT", "MISS", "MEM", "PR");
    ov_buf_printf("%s", htext);
    render_pad_spaces(hlen, r.width);
    row++;

    if (m->nb_edges == 0)
    {
        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("  ");
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("No connections detected");
        render_pad_spaces(2 + 23, r.width);
        row++;
        for (int i = 1; i < max_rows; i++, row++) {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return;
    }

    int rendered_rows = 0;
    for (int ei = 0; ei < m->nb_edges && rendered_rows < max_rows; ei++)
    {
        const OV_EDGE *e = &m->edges[ei];
        if (e->src_node < 0 || e->src_node >= m->nb_nodes || e->tgt_node < 0 || e->tgt_node >= m->nb_nodes) continue;

        const OV_NODE *src = &m->nodes[e->src_node];
        const OV_NODE *tgt = &m->nodes[e->tgt_node];

        ov_buf_pos(row, r.col + 1);
        
        int is_sel = (ei == lay->sel_graph && lay->focus == OV_FOCUS_GRAPH);
        ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : OV_BG_PANEL;
        ov_theme_bg(row_bg);
        ov_buf_printf(" ");

        /* Detailed rendering with color */
        ov_rgb_t sc;
        const char *styp = "UNK";
        switch (src->type) {
        case OV_NODE_STREAM: sc = OV_FG_STREAM; styp = "STRM"; break;
        case OV_NODE_FPS:    sc = OV_FG_FPS;    styp = "FPS "; break;
        case OV_NODE_PROC:   sc = OV_FG_PROC;   styp = "PROC"; break;
        }

        ov_rgb_t tc;
        const char *ttyp = "UNK";
        switch (tgt->type) {
        case OV_NODE_STREAM: tc = OV_FG_STREAM; ttyp = "STRM"; break;
        case OV_NODE_FPS:    tc = OV_FG_FPS;    ttyp = "FPS "; break;
        case OV_NODE_PROC:   tc = OV_FG_PROC;   ttyp = "PROC"; break;
        }

        const char *etype = "UNKNOWN";
        switch (e->type) {
        case OV_EDGE_PROC_WRITES_STREAM:   etype = "PROC_WRITES_STRM"; break;
        case OV_EDGE_STREAM_TRIGGERS_PROC: etype = "STRM_TRIGS_PROC "; break;
        case OV_EDGE_FPS_RUNS_PROC:        etype = "FPS_RUNS_PROC   "; break;
        case OV_EDGE_FPS_INPUT_STREAM:     etype = "FPS_INPUT_STRM  "; break;
        case OV_EDGE_FPS_OUTPUT_STREAM:    etype = "FPS_OUTPUT_STRM "; break;
        case OV_EDGE_PROC_TRIGGER_STREAM:  etype = "PROC_TRIGS_STRM "; break;
        }

        int printed = 1;
        int avail = r.width - 2;

        #define GRAPH_FIELD(color, fmt, ...)         \
        do {                                         \
            char _fb[128];                           \
            int _fl = snprintf(                      \
                _fb, sizeof(_fb), fmt,               \
                ##__VA_ARGS__);                      \
            int _vis = _fl;                          \
            int _max = avail - printed;              \
            if (_vis > _max) _vis = _max;            \
            if (_vis > 0) {                          \
                ov_theme_fg(color);                  \
                ov_buf_printf("%.*s", _vis, _fb);    \
                printed += _vis;                     \
            }                                        \
        } while(0)

        GRAPH_FIELD(sc, "%-12.12s", src->name);
        
        /* Box drawing character length workaround */
        if (avail - printed >= 4) {
            ov_theme_fg(OV_FG_CONN);
            ov_buf_printf(" %s%s ", OV_BOX_H, OV_TRI_R);
            printed += 4;
        }

        GRAPH_FIELD(tc, "%-12.12s", tgt->name);
        GRAPH_FIELD(OV_FG_DIM, " [%.6s]", e->label);
        GRAPH_FIELD(OV_FG_TEXT, " %-16.16s ", etype);
        GRAPH_FIELD(sc, "%-4s ", styp);
        GRAPH_FIELD(tc, "%-4s", ttyp);

        #undef GRAPH_FIELD

        render_pad_spaces(printed, r.width);
        row++;
        rendered_rows++;
    }
    
    for (; rendered_rows < max_rows; rendered_rows++, row++)
    {
        clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    
    ov_buf_reset_attr();
}

