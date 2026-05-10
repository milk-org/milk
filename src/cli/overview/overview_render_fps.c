/**
 * @file    overview_render_fps.c
 * @brief   FPS panel + detail panel rendering for milkCTRL
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
        lay->focus == OV_FOCUS_FPS);

    int hrow = r.row + 1;
    int hs   = lay->hscroll_fps;

    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");
    ov_theme_fg(OV_FG_DIM);
    char htext[256];
    int hlen;
    {
        int sk = lay->sort_key_fps;
        int sd = lay->sort_dir_fps;
        char c_name[24], c_c[8], c_mem[10];
        int w_name = sort_col_label(c_name, sizeof(c_name),
                       "NAME", 0, sk, sd, 18);
        int w_c = sort_col_label(c_c, sizeof(c_c),
                       "C", 1, sk, sd, 2);
        int w_mem = sort_col_label(c_mem, sizeof(c_mem),
                       "MEM", 2, sk, sd, 5);
        int desc_w =
            (lay->view == OV_VIEW_FPS)
            ? 30 : 20;
        hlen = snprintf(
            htext, sizeof(htext),
            "%-*s %*s %1s %3s %*s"
            " %-*s %8s %7s %7s",
            w_name, c_name, w_c, c_c, "R", "STR", w_mem, c_mem,
            desc_w, "DESCRIPTION",
            "STATUS", "CPID", "RPID");
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
                          && lay->focus
                             == OV_FOCUS_FPS);
            int is_frozen = (lay->freeze
                && lay->freeze_focus
                   == OV_FOCUS_FPS
                && ffi == lay->freeze_sel_fps);
            ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int has_rel = (rel != NULL && bget(rel->fps, fi));
            int is_rel = (!is_sel && !is_frozen
                && eff_focus != OV_FOCUS_FPS
                && has_rel);
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
            ov_buf_printf(" ");

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
                    ov_theme_bg(row_bg);                \
                }                                      \
            } while(0)

            pid_t _spid = (rel != NULL)
                        ? rel->sel_pid : 0;

            FPS_FIELD(OV_FG_FPS, "%-18.18s ", f->name);
            FPS_PID_FIELD(f->confpid, "%2s ", f->conf_alive ? "C" : "-");
            FPS_PID_FIELD(f->runpid, "%s ", f->run_alive ? "R" : "-");
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
            FPS_FIELD(OV_FG_MUTED, "%08X ", f->md_status);
            FPS_PID_FIELD(f->confpid, "%7d ", (int) f->confpid);
            FPS_PID_FIELD(f->runpid, "%7d", (int) f->runpid);
            
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
                        ov_theme_bg(is_rel ? OV_BG_RELATED : OV_BG_PANEL);
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
        }
        else
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(
        r, lay->scroll_fps, max_rows,
        filt_n, OV_FG_FPS);
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
