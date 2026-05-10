/**
 * @file    overview_render_streams.c
 * @brief   STREAMS panel rendering for milkCTRL
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
        lay->focus == OV_FOCUS_STREAMS);

    int hrow = r.row + 1;
    int hs   = lay->hscroll_stream;

    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");
    ov_theme_fg(OV_FG_DIM);
    
    char htext[300];
    int hlen;
    {
        int sk = lay->sort_key_stream;
        int sd = lay->sort_dir_stream;
        char c_name[20], c_typ[10], c_size[16];
        char c_hz[10], c_mbps[12], c_ino[16], c_cnt[16];
        int w_name = sort_col_label(c_name, sizeof(c_name),
                       "NAME", 0, sk, sd, 14);
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
        int vis = hlen - hs;
        if (vis < 0) vis = 0;
        const char *start_str = htext + hs;
        if (hs >= hlen) { start_str = ""; vis = 0; }
        ov_buf_printf("%.*s", vis, start_str);
        render_pad_spaces(1 + vis, r.width);
    }

    int max_rows = r.height - 3;
    int start = lay->scroll_stream;

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 1 + i;
        int fi = start + i;
        if (fi < filt_n)
        {
            int si = filt_idx[fi];
            const OV_STREAM *s = &m->streams[si];
            int is_sel =
                (fi == lay->sel_stream
                 && lay->focus == OV_FOCUS_STREAMS);
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
            ov_rgb_t row_bg = is_sel
                ? OV_BG_SELECTED
                : is_frozen ? OV_BG_FROZEN
                : is_rel ? OV_BG_RELATED
                         : OV_BG_PANEL;

            int hs_rem = hs;
            int printed = 1;
            int avail = r.width - 2;

            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);
            if (s->update_hz > 0.1) {
                ov_theme_fg(OV_FG_ACTIVE);
                ov_buf_printf("●");
            } else {
                ov_buf_printf(" ");
            }

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
                    ov_theme_bg(row_bg);                \
                }                                      \
            } while(0)

            pid_t _spid = (rel != NULL)
                        ? rel->sel_pid : 0;

            ov_rgb_t base_color = s->active ? OV_FG_STREAM : OV_FG_DIM;
            
            STRM_FIELD(base_color, "%-14.14s ", s->name);
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

            STRM_FIELD(OV_FG_DIM, "%10lu ", (unsigned long) s->inode);
            
            STRM_PID_FIELD(
                s->ownerPID,
                "%7d ", (int) s->ownerPID);
            STRM_FIELD(OV_FG_CONN, "%10lu ", (unsigned long) s->cnt0);
            
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
        double total_bps = 0.0;
        for (int i = 0; i < filt_n; i++)
        {
            int si = filt_idx[i];
            const OV_STREAM *s = &m->streams[si];
            if (s->update_hz > 0.1)
            {
                total_bps += s->update_hz
                    * (double) s->nelement
                    * dtype_bytesize(s->datatype);
            }
        }
        double total_mb = total_bps
            / (1024.0 * 1024.0);
        char tbuf[40];
        if (total_mb >= 1000.0)
        {
            snprintf(tbuf, sizeof(tbuf),
                " %.1f GB/s ",
                total_mb / 1024.0);
        }
        else
        {
            snprintf(tbuf, sizeof(tbuf),
                " %.1f MB/s ",
                total_mb);
        }
        int tlen = (int) strlen(tbuf);
        int brow = r.row + r.height - 1;
        int bcol = r.col + r.width
            - tlen - 2;
        if (bcol > r.col + 1)
        {
            ov_buf_pos(brow, bcol);
            ov_theme_fg(OV_FG_ACTIVE);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", tbuf);
        }
    }

    ov_buf_reset_attr();

    if (has_re)
    {
        regfree(&re);
    }
}
