/**
 * @file    overview_render_fps.c
 * @brief   FPS panel + detail panel rendering for milk-CTRL
 *
 * Split from overview_render.c for navigability.
 */

#include "overview_render_internal.h"

void ov_render_fps_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    const OV_RELATED *rel)
{
    OV_RECT r = lay->r_fps;

    /* Build filtered index array */
    const char *names[OV_MAX_FPS];
    for (int i = 0; i < m->nb_fps; i++)
    {
        names[i] = m->fps[i].name;
    }
    int fidx[OV_MAX_FPS];
    int filt_n = ov_filter_build(
        lay->filter_fps, names,
        m->nb_fps, fidx, OV_MAX_FPS);

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_FPS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->fps, fidx[i]))
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    int has_re = 0;
    regex_t re;
    if (lay->filter_fps[0] != '\0')
    {
        if (regcomp(&re, lay->filter_fps, REG_EXTENDED | REG_ICASE) == 0)
        {
            has_re = 1;
        }
    }

    char title[80];
    if (lay->filter_fps[0] != '\0')
    {
        snprintf(title, sizeof(title),
                 "FPS /%s/", lay->filter_fps);
    }
    else
    {
        snprintf(title, sizeof(title), "FPS");
    }
    ov_draw_panel_border(
        r.row, r.col, r.height, r.width,
        title, OV_FG_FPS,
        lay->focus == OV_FOCUS_FPS, 0);

    int hrow = r.row + 1;
    int hs   = lay->hscroll_fps;

    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf("    ");
    ov_theme_fg(OV_FG_DIM);
    char htext[256];
    int hlen;
    {
        int sk = lay->sort_key_fps;
        int sd = lay->sort_dir_fps;
        char c_name[24], c_c[8], c_r[8], c_mem[10];
        int w_name = sort_col_label(c_name, sizeof(c_name),
                       (sk == 3) ? "ANCESTRY" : "NAME",
                       (sk == 3) ? 3 : 0, sk, sd, 18);
        int w_c = sort_col_label(c_c, sizeof(c_c),
                       "CPID", 1, sk, sd, 7);
        int w_r = sort_col_label(c_r, sizeof(c_r),
                       "RPID", 4, sk, sd, 7);
        int w_mem = sort_col_label(c_mem, sizeof(c_mem),
                       "MEM", 2, sk, sd, 5);
        int desc_w =
            (lay->view == OV_VIEW_FPS)
            ? 30 : 20;
        hlen = snprintf(
            htext, sizeof(htext),
            "%-*s %3s %*s %*s %3s %*s"
            " %-*s",
            w_name, c_name, "TMX", w_c, c_c, w_r, c_r, "STR", w_mem, c_mem,
            desc_w, "DESCRIPTION");
    }
    
    {
        int vis_width = r.width - 4;
        if (vis_width < 0) vis_width = 0;
        int printed = ov_render_header_text(htext, hs, vis_width);
        render_pad_spaces(4 + printed, r.width);
    }

    int8_t local_depth[OV_MAX_FPS];
    memset(local_depth, 0, sizeof(local_depth));
    {
        int eff_sel = -1;
        if (lay->freeze && lay->freeze_focus == OV_FOCUS_FPS
            && lay->freeze_sel_fps >= 0 && lay->freeze_sel_fps < filt_n) {
            eff_sel = lay->freeze_sel_fps;
        } else if (lay->focus == OV_FOCUS_FPS
                   && lay->sel_fps >= 0 && lay->sel_fps < filt_n) {
            eff_sel = lay->sel_fps;
        }
        if (eff_sel >= 0) {
            int root_fi = fidx[eff_sel];
            int root_node = m->fps[root_fi].node_idx;
            if (root_node >= 0) {
                int8_t node_depths[OV_MAX_NODES];
                sg_compute_node_depths(
                    m, root_node,
                    SG_MODE_FPS, node_depths);
                for (int fi = 0; fi < m->nb_fps; fi++) {
                    int n = m->fps[fi].node_idx;
                    if (n >= 0 && node_depths[n] != 127)
                        local_depth[fi] = node_depths[n];
                }
            }
        }
    }

    int max_rows = r.height - 3;
    int start = lay->scroll_fps;

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 1 + i;
        int ffi = start + i;
        if (ffi < filt_n)
        {
            int fi = fidx[ffi];
            const OV_FPS *f = &m->fps[fi];
            int is_sel = (ffi == lay->sel_fps
                          && (lay->focus == OV_FOCUS_FPS
                              || lay->focus == OV_FOCUS_GRAPH));
            int is_frozen = (lay->freeze
                && lay->freeze_focus
                   == OV_FOCUS_FPS
                && ffi == lay->freeze_sel_fps);
            ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int has_rel = (rel != NULL && bget(rel->fps, fi));
            int is_rel = (!is_sel && !is_frozen
                && eff_focus != OV_FOCUS_FPS
                && has_rel);
            ov_rgb_t row_bg = OV_BG_PANEL;

            if (is_sel) {
                row_bg = OV_BG_SELECTED;
            } else if (is_frozen) {
                row_bg = OV_BG_FROZEN;
            } else if (is_rel) {
                row_bg = OV_BG_RELATED;
            }

            int hs_rem = hs;
            int printed = 4;
            int avail = r.width - 2;

            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);

            /* Lineage depth badge: ◀N or N▶ */
            int8_t sdepth = local_depth[fi];
            if (sdepth != 0 && !is_sel && !is_frozen) {
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
            } else {
                if (is_sel || is_frozen) {
                    ov_theme_fg(OV_FG_ACTIVE);
                    ov_buf_printf("\xe2\x97\x8f   ");
                } else if (eff_focus == OV_FOCUS_STREAMS && is_rel && rel != NULL) {
                    int is_written = bget(rel->fps_writes, fi);
                    if (is_written) {
                        ov_theme_fg(OV_FG_ERROR);
                        ov_buf_printf("\xe2\x96\xb6   ");
                    } else {
                        ov_theme_fg(OV_FG_ACTIVE);
                        ov_buf_printf("\xe2\x97\x80   ");
                    }
                } else {
                    ov_buf_printf("    ");
                }
            }

            #define FPS_FIELD(color, fmt, ...)         \
            do {                                       \
                char _fb[128];                         \
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
            #define FPS_PID_FIELD(pid_val, fmt, ...)    \
            do {                                       \
                int _match = (_spid > 0                \
                    && (pid_t)(pid_val) == _spid);     \
                if (_match) {                          \
                    ov_theme_bg(OV_BG_PID_MATCH);      \
                    ov_buf_bold();                      \
                }                                      \
                FPS_FIELD(                             \
                    _match                             \
                    ? ((ov_rgb_t){0,0,0})              \
                    : ov_pid_color((pid_val)),          \
                    fmt, ##__VA_ARGS__);                \
                if (_match) {                          \
                    ov_buf_reset_attr();                \
                    if (is_sel || is_frozen || is_rel) { \
                        ov_theme_bg(row_bg);            \
                    }                                   \
                }                                      \
            } while(0)

            pid_t _spid = (rel != NULL)
                        ? rel->sel_pid : 0;

            FPS_FIELD(OV_FG_FPS, "%-18.18s ", f->name);

            char tmx_str[4] = {
                (f->tmux_flags & OV_TMUX_CTRL) ? 'c' : '-',
                (f->tmux_flags & OV_TMUX_CONF) ? 'C' : '-',
                (f->tmux_flags & OV_TMUX_RUN)  ? 'r' : '-',
                '\0'
            };
            FPS_FIELD(OV_FG_DIM, "%3s ", tmx_str);

            if (f->confpid > 0) FPS_PID_FIELD(f->confpid, "%7d ", (int) f->confpid);
            else FPS_FIELD(OV_FG_DIM, "%7s ", "-");
            if (f->runpid > 0) FPS_PID_FIELD(f->runpid, "%7d ", (int) f->runpid);
            else FPS_FIELD(OV_FG_DIM, "%7s ", "-");
            FPS_FIELD(OV_FG_TEXT, "%3d ", f->nb_stream_params);
            
            /* MEM */
            char memstr[16];
            format_mem_kb(memstr, sizeof(memstr), f->mem_rss_kb);
            FPS_FIELD(OV_FG_TEXT, "%5s ", memstr);

            /* Detailed columns */
            if (lay->view == OV_VIEW_FPS)
            {
                FPS_FIELD(OV_FG_DIM,
                    "%-30.30s ",
                    f->description);
            }
            else
            {
                FPS_FIELD(OV_FG_DIM,
                    "%-20.20s ",
                    f->description);
            }
            
            #undef FPS_PID_FIELD
            #undef FPS_FIELD

            /* When cross-highlighted by a stream selection, iterate all
             * stream params of this FPS that match the selected stream */
            int n5 = 0;
            if (has_rel && eff_focus == OV_FOCUS_STREAMS)
            {
                uint32_t mask = rel->fps_param_mask[fi];
                for (int sp = 0; mask != 0 && sp < f->nb_stream_params; sp++, mask >>= 1)
                {
                    if (!(mask & 1)) continue;

                    const char *kname = f->stream_param_name[sp];

                    if (strcmp(kname, "procinfo.triggersname") == 0)
                    {
                        ov_buf_bg(120, 80, 10);
                        ov_buf_fg(255, 210, 80);
                        ov_buf_bold();
                        int w = snprintf(NULL, 0, " [TRIG]");
                        ov_buf_printf(" [TRIG]");
                        ov_buf_reset_attr();
                        if (is_sel || is_frozen || is_rel) {
                            ov_theme_bg(row_bg);
                        }
                        n5 += w;
                    }
                    else
                    {
                        ov_theme_fg(OV_FG_CONN);
                        int w = snprintf(NULL, 0, " :%s", kname);
                        ov_buf_printf(" :%s", kname);
                        n5 += w;
                    }
                }
            }

            render_pad_spaces(printed + n5, r.width);
            ov_buf_reset_attr();
        }
        else
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(
        r, lay->scroll_fps, max_rows,
        filt_n, OV_FG_FPS);

    /* ---- Footer stats on bottom border ---- */
    {
        /* Totals over ALL FPS */
        int  tot_conf = 0;
        int  tot_run  = 0;
        int64_t tot_mem  = 0;
        for (int j = 0; j < m->nb_fps; j++)
        {
            const OV_FPS *f = &m->fps[j];
            if (f->conf_alive) { tot_conf++; }
            if (f->run_alive)  { tot_run++;  }
            tot_mem += f->mem_rss_kb;
        }

        /* Totals over filtered subset */
        int  flt_conf = 0;
        int  flt_run  = 0;
        int64_t flt_mem  = 0;
        for (int j = 0; j < filt_n; j++)
        {
            const OV_FPS *f = &m->fps[fidx[j]];
            if (f->conf_alive) { flt_conf++; }
            if (f->run_alive)  { flt_run++;  }
            flt_mem += f->mem_rss_kb;
        }

        int brow = r.row + r.height - 1;
        int is_subset =
            (filt_n < m->nb_fps);

        /* Right side: total stats (always) */
        char tmem[16];
        format_mem_kb(tmem, sizeof(tmem), tot_mem);
        char rbuf[80];
        snprintf(rbuf, sizeof(rbuf),
            " %d conf \u2502 %d run \u2502 %s ",
            tot_conf, tot_run, tmem);
        int rlen = (int) strlen(rbuf);
        int below = filt_n - lay->scroll_fps - max_rows;
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
            ov_theme_fg(
                tot_run > 0
                ? OV_FG_ACTIVE : OV_FG_DIM);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", rbuf);
        }

        /* Left side: filtered stats */
        if (is_subset)
        {
            char fmem[16];
            format_mem_kb(
                fmem, sizeof(fmem), flt_mem);
            char lbuf[80];
            snprintf(lbuf, sizeof(lbuf),
                " %d conf \u2502 %d run \u2502 %s ",
                flt_conf, flt_run, fmem);
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

/* =========================================================
 * Detail pane — replaces CONNECTIONS when item selected
 * ========================================================= */

/**
 * ov_render_detail_panel - show detail for selected item.
 * @lay: layout (holds selection + graph rect)
 * @m:   data model
 *
 * Renders into the graph panel rectangle. Returns 1 if
 * detail was drawn, 0 if nothing to show (caller should
 * fall back to graph panel).
 */
