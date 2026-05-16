/**
 * @file    overview_render_streams.c
 * @brief   STREAMS panel rendering for milk-CTRL
 *
 * Split from overview_render.c for navigability.
 */

#include "overview_render_internal.h"

void ov_render_streams_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    const OV_RELATED *rel)
{
    OV_RECT r = lay->r_streams;

    /* Build filtered index array */
    const char *names[OV_MAX_STREAMS];
    for (int i = 0; i < m->nb_streams; i++)
    {
        names[i] = m->streams[i].name;
    }
    int filt_idx[OV_MAX_STREAMS];
    int filt_n = ov_filter_build(
        lay->filter_stream, names,
        m->nb_streams, filt_idx, OV_MAX_STREAMS);

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_STREAMS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->streams, filt_idx[i]))
            {
                filt_idx[new_filt_n++] = filt_idx[i];
            }
        }
        filt_n = new_filt_n;
    }

    int has_re = 0;
    regex_t re;
    if (lay->filter_stream[0] != '\0')
    {
        if (regcomp(&re, lay->filter_stream, REG_EXTENDED | REG_ICASE) == 0)
        {
            has_re = 1;
        }
    }

    /* Panel title with filter indicator */
    char title[80];
    if (lay->filter_stream[0] != '\0')
    {
        snprintf(title, sizeof(title),
                 "STREAMS /%s/", lay->filter_stream);
    }
    else
    {
        snprintf(title, sizeof(title), "STREAMS");
    }
    ov_draw_panel_border(
        r.row, r.col, r.height, r.width,
        title, OV_FG_STREAM,
        lay->focus == OV_FOCUS_STREAMS, 0);

    int hrow = r.row + 1;
    int hs   = lay->hscroll_stream;

    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf("    ");
    ov_theme_fg(OV_FG_STREAM_HDR);
    
    char htext[300];
    int hlen;
    {
        int sk = lay->sort_key_stream;
        int sd = lay->sort_dir_stream;
        char c_name[20], c_typ[10], c_size[16];
        char c_hz[10], c_mbps[12], c_ino[16], c_cnt[16];
        int w_name = sort_col_label(c_name, sizeof(c_name),
                       (sk == 7) ? "ANCESTRY" : "NAME",
                       (sk == 7) ? 7 : 0, sk, sd, 14);
        int w_typ = sort_col_label(c_typ, sizeof(c_typ),
                       "TYP", 1, sk, sd, 4);
        int w_size = sort_col_label(c_size, sizeof(c_size),
                       "SIZE", 2, sk, sd, 11);
        int w_hz = sort_col_label(c_hz, sizeof(c_hz),
                       "Hz", 3, sk, sd, 6);
        int w_mbps = sort_col_label(c_mbps, sizeof(c_mbps),
                       "MB/s", 4, sk, sd, 7);
        int w_ino = sort_col_label(c_ino, sizeof(c_ino),
                       "INODE", 5, sk, sd, 10);
        int w_cnt = sort_col_label(c_cnt, sizeof(c_cnt),
                       "COUNT", 6, sk, sd, 10);
        hlen = snprintf(
            htext, sizeof(htext),
            "%-*s %*s %*s %*s"
            " %*s %*s %7s %*s %10s"
            " %7s %s",
            w_name, c_name, w_typ, c_typ,
            w_size, c_size, w_hz, c_hz,
            w_mbps, c_mbps, w_ino, c_ino,
            "OWNER", w_cnt, c_cnt, "SEMS",
            "WPID", "RPID");
    }
    
    {
        int vis_width = r.width - 4;
        if (vis_width < 0) vis_width = 0;
        int printed = ov_render_header_text(
            htext, hs, vis_width, OV_FG_STREAM_HDR);
        render_pad_spaces(4 + printed, r.width);
    }

    /* Separator between header and data rows */
    render_separator(
        hrow + 1, r.col + 1,
        r.width - 2, OV_FG_STREAM_HDR);

    int max_rows = r.height - 4;
    int start = lay->scroll_stream;

    /* Compute lineage depths when a stream
     * is selected.  sel_stream is a position in
     * the filtered list; convert via filt_idx[]
     * to obtain the model-level stream index. */
    int8_t local_depth[OV_MAX_STREAMS];
    memset(local_depth, 0, sizeof(local_depth));
    {
        int eff_sel = -1;
        if (lay->freeze
            && lay->freeze_focus
               == OV_FOCUS_STREAMS
            && lay->freeze_sel_stream >= 0
            && lay->freeze_sel_stream < filt_n)
        {
            eff_sel = lay->freeze_sel_stream;
        }
        else if (lay->focus == OV_FOCUS_STREAMS
                 && lay->sel_stream >= 0
                 && lay->sel_stream < filt_n)
        {
            eff_sel = lay->sel_stream;
        }
        if (eff_sel >= 0)
        {
            int root_si = filt_idx[eff_sel];
            SG_LINEAGE lin;
            sg_compute_lineage(
                m, root_si,
                SG_MODE_FULL, &lin);

            for (int a = 0;
                 a < lin.nb_ancestors; a++)
            {
                int si =
                    lin.ancestors[a].stream_idx;
                if (si >= 0
                    && si < m->nb_streams)
                {
                    int d =
                        lin.ancestors[a].depth;
                    if (d > 127) { d = 127; }
                    local_depth[si] =
                        (int8_t)(-d);
                }
            }
            for (int di = 0;
                 di < lin.nb_descendants; di++)
            {
                int si =
                    lin.descendants[di]
                        .stream_idx;
                if (si >= 0
                    && si < m->nb_streams)
                {
                    int dp =
                        lin.descendants[di]
                            .depth;
                    if (dp > 127) { dp = 127; }
                    local_depth[si] =
                        (int8_t) dp;
                }
            }
            /* DEBUG: dump to file once */
            {
                static int dbg_done = 0;
                if (!dbg_done)
                {
                    dbg_done = 1;
                    FILE *df = fopen(
                        "/tmp/lineage_debug.txt",
                        "w");
                    if (df)
                    {
                        fprintf(df,
                            "root_si=%d name=%s "
                            "node=%d\n",
                            root_si,
                            m->streams[root_si]
                                .name,
                            m->streams[root_si]
                                .node_idx);
                        fprintf(df,
                            "nb_edges=%d "
                            "nb_nodes=%d\n",
                            m->nb_edges,
                            m->nb_nodes);
                        const char *etnames[] = {
                            "PROC_WRITES_STREAM",
                            "STREAM_TRIGGERS_PROC",
                            "FPS_RUNS_PROC",
                            "FPS_INPUT_STREAM",
                            "FPS_OUTPUT_STREAM",
                            "PROC_TRIGGER_STREAM",
                            "STREAM_READ_BY_PROC"
                        };
                        for (int ei = 0;
                             ei < m->nb_edges;
                             ei++)
                        {
                            const OV_EDGE *e =
                                &m->edges[ei];
                            const char *sn =
                                (e->src_node >= 0
                                 && e->src_node
                                    < m->nb_nodes)
                                ? m->nodes[
                                    e->src_node]
                                    .name
                                : "??";
                            const char *tn =
                                (e->tgt_node >= 0
                                 && e->tgt_node
                                    < m->nb_nodes)
                                ? m->nodes[
                                    e->tgt_node]
                                    .name
                                : "??";
                            const char *et =
                                (e->type <= 6)
                                ? etnames[e->type]
                                : "??";
                            fprintf(df,
                                "EDGE[%d] "
                                "%-20s -> "
                                "%-20s %s\n",
                                ei, sn, tn, et);
                        }
                        fprintf(df,
                            "\nAncestors: %d\n",
                            lin.nb_ancestors);
                        for (int a = 0;
                             a < lin.nb_ancestors;
                             a++)
                        {
                            fprintf(df,
                                "  anc[%d] "
                                "depth=%d "
                                "si=%d name=%s "
                                "via=%s\n",
                                a,
                                lin.ancestors[a]
                                    .depth,
                                lin.ancestors[a]
                                    .stream_idx,
                                m->streams[
                                    lin.ancestors[a]
                                    .stream_idx]
                                    .name,
                                lin.ancestors[a]
                                    .via_name);
                        }
                        fprintf(df,
                            "\nDescendants: %d\n",
                            lin.nb_descendants);
                        for (int di = 0;
                             di < lin
                                 .nb_descendants;
                             di++)
                        {
                            fprintf(df,
                                "  desc[%d] "
                                "depth=%d "
                                "si=%d name=%s "
                                "via=%s\n",
                                di,
                                lin.descendants[di]
                                    .depth,
                                lin.descendants[di]
                                    .stream_idx,
                                m->streams[
                                    lin.descendants
                                    [di]
                                    .stream_idx]
                                    .name,
                                lin.descendants[di]
                                    .via_name);
                        }
                        fclose(df);
                    }
                }
            }
        }
    }

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 2 + i;
        int fi = start + i;
        if (fi < filt_n)
        {
            int si = filt_idx[fi];
            const OV_STREAM *s = &m->streams[si];
            int is_sel =
                (fi == lay->sel_stream
                 && (lay->focus == OV_FOCUS_STREAMS
                     || lay->focus == OV_FOCUS_GRAPH));
            int is_frozen = (lay->freeze
                && lay->freeze_focus
                   == OV_FOCUS_STREAMS
                && fi == lay->freeze_sel_stream);
            ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int is_rel =
                (!is_sel && !is_frozen
                 && eff_focus != OV_FOCUS_STREAMS
                 && rel != NULL
                 && bget(rel->streams, si));
            ov_rgb_t row_bg = OV_BG_PANEL;
            if (is_sel) {
                row_bg = OV_BG_SELECTED;
            } else if (is_frozen) {
                row_bg = OV_BG_FROZEN;
            } else if (is_rel) {
                row_bg = OV_BG_RELATED;
            } else if (s->is_new > 0) {
                row_bg = OV_BG_NEW_ITEM;
            }
            row_bg = zebra_bg(row_bg, i);

            int hs_rem = hs;
            int printed = 4;
            int avail = r.width - 2;

            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);

            /* Focus ring accent strip (#10) */
            int panel_focused =
                (lay->focus == OV_FOCUS_STREAMS);
            render_focus_strip(
                row, r.col + 1,
                panel_focused,
                OV_FG_STREAM, row_bg);

            #define STRM_FIELD(color, fmt, ...)        \
            do {                                       \
                char _fb[80];                          \
                int _fl = snprintf(                    \
                    _fb, sizeof(_fb), fmt,              \
                    ##__VA_ARGS__);                     \
                int _skip = 0;                         \
                if (hs_rem > 0) {                      \
                    _skip = (hs_rem < _fl) ? hs_rem : _fl; \
                    hs_rem -= _skip;                   \
                }                                      \
                int _vis = _fl - _skip;                \
                int _max = avail - printed;            \
                if (_vis > _max) _vis = _max;          \
                if (_vis > 0) {                        \
                    ov_theme_fg(color);                \
                    ov_buf_printf("%.*s", _vis, _fb + _skip); \
                    printed += _vis;                   \
                }                                      \
            } while(0)

            /* PID field with inverted highlight when it
             * matches the selected process PID:
             * green background + bold black text */
            #define STRM_PID_FIELD(pid_val, fmt, ...)  \
            do {                                       \
                int _match = (_spid > 0                \
                    && (pid_t)(pid_val) == _spid);     \
                if (_match) {                          \
                    ov_theme_bg(OV_BG_PID_MATCH);      \
                    ov_buf_bold();                      \
                }                                      \
                STRM_FIELD(                            \
                    _match                             \
                    ? ((ov_rgb_t){0,0,0})              \
                    : ov_pid_color((pid_val)),          \
                    fmt, ##__VA_ARGS__);                \
                if (_match) {                          \
                    ov_buf_reset_attr();                \
                    if (is_sel || is_frozen || is_rel) {   \
                        ov_theme_bg(row_bg);            \
                    }                                   \
                }                                      \
            } while(0)

            pid_t _spid = (rel != NULL)
                        ? rel->sel_pid : 0;

            /* Lineage depth badge: ←N or N→ */
            int8_t sdepth = local_depth[si];
            if (sdepth != 0 && !is_sel && !is_frozen)
            {
                int abs_d = sdepth < 0 ? -sdepth : sdepth;
                if (abs_d > 99) abs_d = 99;
                ov_theme_fg(OV_FG_WARN);
                if (sdepth < 0) {
                    if (abs_d < 10) ov_buf_printf("\xe2\x97\x80%d  ", abs_d);
                    else ov_buf_printf("\xe2\x97\x80%d ", abs_d);
                } else {
                    if (abs_d < 10) ov_buf_printf("%d\xe2\x96\xb6  ", abs_d);
                    else ov_buf_printf("%d\xe2\x96\xb6 ", abs_d);
                }
            }
            else
            {
                if (s->update_hz > 0.1) {
                    ov_theme_fg(OV_FG_ACTIVE);
                    ov_buf_printf("\xe2\x97\x8f ");
                } else {
                    ov_buf_printf("  ");
                }

                if ((eff_focus == OV_FOCUS_PROCS || eff_focus == OV_FOCUS_FPS) && is_rel && rel != NULL) {
                    int is_written = bget(rel->stream_written, si);
                    if (is_written) {
                        ov_theme_fg(OV_FG_ERROR);
                        ov_buf_printf("\xe2\x96\xb6 ");
                    } else {
                        ov_theme_fg(OV_FG_ACTIVE);
                        ov_buf_printf("\xe2\x97\x80 ");
                    }
                } else {
                    ov_buf_printf("  ");
                }
            }

            ov_rgb_t base_color = s->active
                ? OV_FG_STREAM : OV_FG_DIM;
            
            STRM_FIELD(base_color,
                "%-14.14s ", s->name);
            STRM_FIELD(OV_FG_MUTED, "%4s ", render_dtype(s->datatype));
            
            STRM_FIELD(OV_FG_TEXT, "%11s ", s->size_str);
            
            if (s->update_hz > 0.1) {
                STRM_FIELD(OV_FG_ACTIVE, "%6.1f ", s->update_hz);
            } else {
                STRM_FIELD(OV_FG_DIM, "     - ");
            }

            /* MB/s throughput */
            if (s->update_hz > 0.1) {
                double mbps = s->update_hz
                    * (double) s->nelement
                    * dtype_bytesize(s->datatype)
                    / (1024.0 * 1024.0);
                if (mbps >= 1000.0) {
                    STRM_FIELD(OV_FG_ACTIVE,
                        "%6.1fG ", mbps / 1024.0);
                } else {
                    STRM_FIELD(OV_FG_ACTIVE,
                        "%6.1fM ", mbps);
                }
            } else {
                STRM_FIELD(OV_FG_DIM, "      - ");
            }

            if (!lay->compact_mode)
            {
                STRM_FIELD(OV_FG_DIM, "%10" PRIu64 " ", (uint64_t) s->inode);
            }
            
            STRM_PID_FIELD(
                s->ownerPID,
                "%7d ", (int) s->ownerPID);
            if (!lay->compact_mode)
            {
                STRM_FIELD(s->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM,
                    "%10" PRIu64 " ", (uint64_t) s->cnt0);

                for (int sm = 0; sm < 10; sm++) {
                    if (sm < s->nb_sem) {
                        int val = s->semval[sm];
                        char c;
                        if (val < 0) c = '-';
                        else if (val > 9) c = '+';
                        else c = '0' + val;
                        STRM_FIELD(ov_get_sem_color(val), "%c", c);
                    } else {
                        STRM_FIELD(OV_FG_DIM, ".");
                    }
                }
                STRM_FIELD(OV_FG_DIM, " ");
            }

            /* Write PID */
            if (s->write_pid > 0) {
                STRM_PID_FIELD(
                    s->write_pid,
                    "%7d ",
                    (int) s->write_pid);
            } else {
                STRM_FIELD(OV_FG_DIM, "      - ");
            }

            /* Read PIDs (compact list) */
            if (s->nb_read_pids > 0) {
                for (int rp = 0;
                     rp < s->nb_read_pids; rp++)
                {
                    if (rp > 0) {
                        STRM_FIELD(OV_FG_DIM, ":");
                    }
                    STRM_PID_FIELD(
                        s->read_pids[rp],
                        "%d",
                        (int) s->read_pids[rp]);
                }
                STRM_FIELD(OV_FG_DIM, " ");
            } else {
                STRM_FIELD(OV_FG_DIM, "- ");
            }

            #undef STRM_PID_FIELD
            #undef STRM_FIELD
            
            // The active dot is now printed at the start of the line
            
            render_pad_spaces(printed, r.width);
            ov_buf_reset_attr();
        }
        else
        {
            clear_row(
                row, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(
        r, lay->scroll_stream, max_rows,
        filt_n, OV_FG_STREAM);

    /* Total MB/s footer on bottom border */
    {
        /* Totals over ALL streams */
        double total_all_bps = 0.0;
        for (int i = 0; i < m->nb_streams; i++)
        {
            const OV_STREAM *s = &m->streams[i];
            if (s->update_hz > 0.1)
            {
                total_all_bps += s->update_hz
                    * (double) s->nelement
                    * dtype_bytesize(s->datatype);
            }
        }

        /* Totals over filtered subset */
        double total_flt_bps = 0.0;
        for (int i = 0; i < filt_n; i++)
        {
            int si = filt_idx[i];
            const OV_STREAM *s = &m->streams[si];
            if (s->update_hz > 0.1)
            {
                total_flt_bps += s->update_hz
                    * (double) s->nelement
                    * dtype_bytesize(s->datatype);
            }
        }

        int  brow = r.row + r.height - 1;
        int  is_subset =
            (filt_n < m->nb_streams);

        /* Right side: total (always) */
        double total_all_mb =
            total_all_bps / (1024.0 * 1024.0);
        char rbuf[40];
        if (total_all_mb >= 1000.0)
        {
            snprintf(rbuf, sizeof(rbuf),
                " %.1f GB/s ",
                total_all_mb / 1024.0);
        }
        else
        {
            snprintf(rbuf, sizeof(rbuf),
                " %.1f MB/s ",
                total_all_mb);
        }
        int rlen = (int) strlen(rbuf);
        int below = filt_n - lay->scroll_stream - max_rows;
        int dw = 0;
        if (below > 0)
        {
            dw = 3;
            int tmp = below;
            while (tmp > 0) { dw++; tmp /= 10; }
        }
        int rcol = r.col + r.width - rlen - dw - 4;
        if (rcol > r.col + 1)
        {
            ov_buf_pos(brow, rcol);
            ov_theme_fg(OV_FG_ACTIVE);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", rbuf);
        }

        /* Left side: filtered (only when
         * filter is active) */
        if (is_subset)
        {
            double flt_mb =
                total_flt_bps
                / (1024.0 * 1024.0);
            char lbuf[40];
            if (flt_mb >= 1000.0)
            {
                snprintf(lbuf, sizeof(lbuf),
                    " %.1f GB/s ",
                    flt_mb / 1024.0);
            }
            else
            {
                snprintf(lbuf, sizeof(lbuf),
                    " %.1f MB/s ",
                    flt_mb);
            }
            int llen = (int) strlen(lbuf);
            int lcol = r.col + 2;
            if (lcol + llen < rcol)
            {
                ov_buf_pos(brow, lcol);
                ov_theme_fg(OV_FG_WARN);
                ov_theme_bg(OV_BG_PANEL);
                ov_buf_printf("%s", lbuf);
            }
        }
    }

    ov_buf_reset_attr();

    if (has_re)
    {
        regfree(&re);
    }
}
