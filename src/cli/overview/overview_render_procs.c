/**
 * @file    overview_render_procs.c
 * @brief   PROCS panel rendering for milk-CTRL
 *
 * Split from overview_render.c for navigability.
 */

#include "overview_render_internal.h"

/* render_trigmode_label and format_mem_kb are
 * now static inline in overview_render_internal.h */


static int ov_procs__filter(const OV_LAYOUT  *lay,
                            const OV_MODEL   *m,
                            const OV_RELATED *rel,
                            int              *filt_idx,
                            int              *has_re,
                            regex_t          *re)
{
    const char *names[OV_MAX_PROCS];
    for (int i = 0; i < m->nb_procs; i++)
    {
        names[i] = m->procs[i].name;
    }
    int filt_n = ov_filter_build(lay->filter_proc, names, m->nb_procs, filt_idx, OV_MAX_PROCS);

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_PROCS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->procs, filt_idx[i]))
            {
                filt_idx[new_filt_n++] = filt_idx[i];
            }
        }
        filt_n = new_filt_n;
    }

    *has_re = 0;
    if (lay->filter_proc[0] != '\0')
    {
        if (regcomp(re, lay->filter_proc, REG_EXTENDED | REG_ICASE) == 0)
        {
            *has_re = 1;
        }
    }
    return filt_n;
}

/**
 * @brief Render the process panel column headers.
 */
static void ov_procs__render_header(const OV_LAYOUT *lay, int hrow, int hs, OV_RECT r)
{
    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf("    ");
    ov_theme_fg(OV_FG_PROC_HDR);

    char htext[256];
    int  hlen;
    {
        int  sk = lay->sort_key_proc;
        int  sd = lay->sort_dir_proc;
        char c_anc[8], c_name[20], c_pid[12];
        char c_stat[10], c_hz[10], c_mem[10];
        int  w_anc  = sort_col_label(c_anc, sizeof(c_anc), "A", 5, sk, sd, 3);
        int  w_name = sort_col_label(c_name, sizeof(c_name), "NAME", 0, sk, sd, 14);
        int  w_pid  = sort_col_label(c_pid, sizeof(c_pid), "PID", 1, sk, sd, 7);
        int  w_stat = sort_col_label(c_stat, sizeof(c_stat), "STAT", 2, sk, sd, 5);
        int  w_hz   = sort_col_label(c_hz, sizeof(c_hz), "Hz", 3, sk, sd, 6);
        int  w_mem  = sort_col_label(c_mem, sizeof(c_mem), "MEM", 4, sk, sd, 5);
        hlen        = snprintf(htext, sizeof(htext),
                               "%-*s %-*s %*s %*s %*s"
                                      " %6s"
                                      " %3s %-10s %7s %5s"
                                      "  CPU%%  %10s %*s %10s %4s",
                               w_anc, c_anc, w_name, c_name, w_pid, c_pid, w_stat, c_stat, w_hz, c_hz,
                               "UPTIME", "TRG", "trig-strm", "exec", "DUTY", "LOOPCNT", w_mem, c_mem,
                               "MISSED", "PRIO");
    }
    int vis_width = r.width - 4;
    if (vis_width < 0)
    {
        vis_width = 0;
    }
    int printed = ov_render_header_text(htext, hs, vis_width, OV_FG_PROC_HDR);
    render_pad_spaces(4 + printed, r.width);

    /* Separator between header and data rows */
    render_separator(hrow + 1, r.col + 1, r.width - 2, OV_FG_PROC_HDR);
}

/**
 * @brief Render process rows in the overview.
 */
static void ov_procs__render_rows(const OV_LAYOUT  *lay,
                                  const OV_MODEL   *m,
                                  const OV_RELATED *rel,
                                  int               hrow,
                                  int               hs,
                                  OV_RECT           r,
                                  const int        *filt_idx,
                                  int               filt_n,
                                  int               has_re,
                                  const regex_t    *re)
{
    int8_t local_depth[OV_MAX_PROCS];
    memset(local_depth, 0, sizeof(local_depth));
    {
        int eff_sel = -1;
        if (lay->freeze && lay->freeze_focus == OV_FOCUS_PROCS && lay->freeze_sel_proc >= 0 &&
            lay->freeze_sel_proc < filt_n)
        {
            eff_sel = lay->freeze_sel_proc;
        }
        else if (lay->focus == OV_FOCUS_PROCS && lay->sel_proc >= 0 && lay->sel_proc < filt_n)
        {
            eff_sel = lay->sel_proc;
        }
        if (eff_sel >= 0)
        {
            int root_pi   = filt_idx[eff_sel];
            int root_node = m->procs[root_pi].node_idx;
            if (root_node >= 0)
            {
                int8_t node_depths[OV_MAX_NODES];
                sg_compute_node_depths(m, root_node, SG_MODE_FULL, node_depths);
                for (int pi = 0; pi < m->nb_procs; pi++)
                {
                    int n = m->procs[pi].node_idx;
                    if (n >= 0 && node_depths[n] != 127)
                    {
                        local_depth[pi] = node_depths[n];
                    }
                }
            }
        }
    }

    int max_rows = r.height - 4;
    int start    = lay->scroll_proc;

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 2 + i;
        int fi  = start + i;
        if (fi < filt_n)
        {
            int            pi     = filt_idx[fi];
            const OV_PROC *p      = &m->procs[pi];
            int            is_sel = (fi == lay->sel_proc &&
                          (lay->focus == OV_FOCUS_PROCS || lay->focus == OV_FOCUS_GRAPH));
            int            is_frozen =
                (lay->freeze && lay->freeze_focus == OV_FOCUS_PROCS && fi == lay->freeze_sel_proc);
            ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int        has_rel   = (rel != NULL && bget(rel->procs, pi));
            int        is_rel   = (!is_sel && !is_frozen && eff_focus != OV_FOCUS_PROCS && has_rel);
            int        is_write = (has_rel && rel != NULL && bget(rel->proc_writes, pi));
            ov_rgb_t   row_bg   = OV_BG_PANEL;
            if (is_sel)
            {
                row_bg = OV_BG_SELECTED;
            }
            else if (is_frozen)
            {
                row_bg = OV_BG_FROZEN;
            }
            else if (is_rel)
            {
                row_bg = OV_BG_RELATED;
            }
            else if (p->stale_count >= 3)
            {
                row_bg = OV_BG_STALE;
            }
            else if (p->is_new > 0)
            {
                row_bg = OV_BG_NEW_ITEM;
            }
            else if (lay->mouse_hover && lay->hover_global_proc == pi)
            {
                row_bg = OV_BG_HOVER;
            }
            row_bg = zebra_bg(row_bg, i);

            /* Build the full row text into a local
             * buffer so we can apply hscroll */
            char rbuf[256];
            int  rlen = 0;

            /* Name */
            rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%-14.14s ", p->name);

            /* PID */
            rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%7d ", (int) p->PID);

            /* Status label */
            const char *sl;
            switch (p->loopstat)
            {
            case PROCESSINFO_LOOPSTAT_INIT:
                sl = "INIT";
                break;
            case PROCESSINFO_LOOPSTAT_ACTIVE:
                sl = " RUN";
                break;
            case PROCESSINFO_LOOPSTAT_PAUSE:
                sl = "PAUS";
                break;
            case PROCESSINFO_LOOPSTAT_STOP:
                sl = "STOP";
                break;
            case PROCESSINFO_LOOPSTAT_ERROR:
                sl = "ERR!";
                break;
            case PROCESSINFO_LOOPSTAT_SPIN:
                sl = "SPIN";
                break;
            case PROCESSINFO_LOOPSTAT_CRASHED:
                sl = "CRSH";
                break;
            default:
                sl = " ?? ";
                break;
            }
            rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%4s ", sl);

            /* Hz */
            if (p->loop_hz > 0.1)
            {
                rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%6.1f ", p->loop_hz);
            }
            else
            {
                rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "     - ");
            }

            /* Uptime (#7) */
            {
                char uptstr[12] = "-";
                if (p->start_time_sec > 0)
                {
                    format_uptime(uptstr, sizeof(uptstr), p->start_time_sec);
                }
                rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%6s ", uptstr);
            }

            /* Trigger, exec time, missed (hidden
             * in compact mode #13) */
            if (!lay->compact_mode)
            {
                /* Trigger mode label */
                rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%3s ",
                                 render_trigmode_label(p->triggermode));

                /* Trigger stream name (truncated) */
                if (p->trigstreamname[0] != '\0' && p->triggermode > 0)
                {
                    rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%-10.10s ",
                                     p->trigstreamname);
                }
                else
                {
                    rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%-10s ", "-");
                }

                /* Exec time (ms) */
                if (p->MeasureTiming && p->dtmedian_exec_ns > 0)
                {
                    double exec_ms = 1.0e-6 * (double) p->dtmedian_exec_ns;
                    rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%7.3f", exec_ms);
                }
                else
                {
                    rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "      -");
                }

                /* Direction arrow */
                if (has_rel)
                {
                    const char *arr = is_write ? " W" : " R";
                    rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, "%s", arr);
                }

                /* Missed frame badge */
                if (p->triggermissed > 0)
                {
                    rlen += snprintf(rbuf + rlen, sizeof(rbuf) - (size_t) rlen, " M:%d",
                                     p->triggermissed);
                }
            }

            /* Now render with hscroll and color */
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);
            ov_buf_printf(" ");

            /* Apply hscroll: show rbuf[hs..] */
            int vis = rlen - hs;
            if (vis < 0)
            {
                vis = 0;
            }

            /* Colorize fields individually.
             * For simplicity, render the scrolled
             * text as a single colored block.
             * Advanced per-field coloring can be
             * added later. */
            {
                /* Choose name color */
                ov_theme_fg(OV_FG_PROC);
                /* For the full-row approach, we print
                 * each segment manually to colorize */
            }

            /* Simpler approach: print full row, then
             * pad — use per-field color for key cols */
            /* Reset and reposition */
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);

            /* Focus ring accent strip (#10) */
            int panel_focused = (lay->focus == OV_FOCUS_PROCS);
            render_focus_strip(row, r.col + 1, panel_focused, OV_FG_PROC, row_bg);

            ov_buf_printf(" ");

            /* ---- per-field colored output ---- */
            int printed = 1;
            int avail   = r.width - 2;

            /* Helper macro: skip hs chars, then
             * print at most avail-printed chars */
#define PROC_FIELD(color, fmt, ...)                                  \
    do                                                               \
    {                                                                \
        char _fb[80];                                                \
        int  _fl   = snprintf(_fb, sizeof(_fb), fmt, ##__VA_ARGS__); \
        int  _skip = 0;                                              \
        if (hs_rem > 0)                                              \
        {                                                            \
            _skip = (hs_rem < _fl) ? hs_rem : _fl;                   \
            hs_rem -= _skip;                                         \
        }                                                            \
        int _vis = _fl - _skip;                                      \
        int _max = avail - printed;                                  \
        if (_vis > _max)                                             \
            _vis = _max;                                             \
        if (_vis > 0)                                                \
        {                                                            \
            ov_theme_fg(color);                                      \
            ov_buf_printf("%.*s", _vis, _fb + _skip);                \
            printed += _vis;                                         \
        }                                                            \
    } while (0)

            /* We need a mutable copy of hs for
             * the macro-based field skipping */
            int hs_rem = lay->hscroll_proc;

            /* Re-do per-field with colors */
            ov_buf_pos(row, r.col + 1);
            if (is_sel || is_frozen || is_rel)
            {
                ov_theme_bg(row_bg);
            }

            /* Ancestry column — rendered raw (UTF-8
             * arrows are 3 bytes / 1 display char).
             * Fixed width: 4 display chars. */
            int8_t sdepth = local_depth[pi];
            if (sdepth != 0 && !is_sel && !is_frozen)
            {
                int abs_d = sdepth < 0 ? -sdepth : sdepth;
                if (abs_d > 99)
                {
                    abs_d = 99;
                }
                ov_theme_fg(OV_FG_WARN);
                if (sdepth < 0)
                {
                    if (abs_d < 10)
                    {
                        ov_buf_printf("\xe2\x97\x80%d  ", abs_d);
                    }
                    else
                    {
                        ov_buf_printf("\xe2\x97\x80%d ", abs_d);
                    }
                }
                else
                {
                    if (abs_d < 10)
                    {
                        ov_buf_printf("%d\xe2\x96\xb6  ", abs_d);
                    }
                    else
                    {
                        ov_buf_printf("%d\xe2\x96\xb6 ", abs_d);
                    }
                }
            }
            else if (eff_focus == OV_FOCUS_STREAMS && has_rel)
            {
                if (is_write)
                {
                    ov_theme_fg(OV_FG_ERROR);
                    ov_buf_printf("\xe2\x96\xb6   ");
                }
                else
                {
                    ov_theme_fg(OV_FG_ACTIVE);
                    ov_buf_printf("\xe2\x97\x80   ");
                }
            }
            else if (is_sel || is_frozen)
            {
                ov_theme_fg(OV_FG_ACTIVE);
                ov_buf_printf("\xe2\x97\x8f   ");
            }
            else
            {
                ov_buf_printf("    ");
            }
            printed += 4;

            /* Name */
            {
                char fb[80];
                int  fl   = snprintf(fb, sizeof(fb), "%-14.14s ", p->name);
                int  skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    regmatch_t pm[1];
                    if (has_re && regexec(re, p->name, 1, pm, 0) == 0)
                    {
                        int b_len = pm[0].rm_so;
                        if (b_len > 14)
                        {
                            b_len = 14;
                        }
                        int m_len = pm[0].rm_eo - pm[0].rm_so;
                        if (b_len + m_len > 14)
                        {
                            m_len = 14 - b_len;
                        }

                        int seg_start[3] = { 0, b_len, b_len + m_len };
                        int seg_len[3]   = { b_len, m_len, fl - (b_len + m_len) };
                        int is_match[3]  = { 0, 1, 0 };

                        for (int s = 0; s < 3; s++)
                        {
                            int s_start = seg_start[s];
                            int s_len   = seg_len[s];
                            if (s_len <= 0)
                            {
                                continue;
                            }

                            int print_start = s_start;
                            if (print_start < skip)
                            {
                                print_start = skip;
                            }
                            int print_end = s_start + s_len;
                            if (print_end > skip + vv)
                            {
                                print_end = skip + vv;
                            }

                            if (print_end > print_start)
                            {
                                int len_to_print = print_end - print_start;
                                if (is_match[s])
                                {
                                    ov_buf_bold();
                                    ov_buf_fg(255, 255, 255);
                                }
                                else
                                {
                                    ov_theme_fg(OV_FG_PROC);
                                }
                                ov_buf_printf("%.*s", len_to_print, fb + print_start);
                                if (is_match[s])
                                {
                                    ov_buf_reset_attr();
                                    if (is_sel || is_frozen || is_rel)
                                    {
                                        ov_theme_bg(row_bg);
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        ov_theme_fg(OV_FG_PROC);
                        ov_buf_printf("%.*s", vv, fb + skip);
                    }
                    printed += vv;
                }
            }
            /* Calculate Status Color First */
            ov_rgb_t sc;
            switch (p->loopstat)
            {
            case PROCESSINFO_LOOPSTAT_INIT:
                sc = OV_FG_DIM;
                break;
            case PROCESSINFO_LOOPSTAT_ACTIVE:
                sc = OV_FG_ACTIVE;
                break;
            case PROCESSINFO_LOOPSTAT_PAUSE:
                sc = OV_FG_WARN;
                break;
            case PROCESSINFO_LOOPSTAT_STOP:
                sc = OV_FG_ZOMBIE;
                break;
            case PROCESSINFO_LOOPSTAT_ERROR:
                sc = OV_FG_ERROR;
                break;
            case PROCESSINFO_LOOPSTAT_SPIN:
                sc = OV_FG_WARN;
                break;
            case PROCESSINFO_LOOPSTAT_CRASHED:
                sc = OV_FG_ERROR;
                break;
            default:
                sc = OV_FG_DIM;
                break;
            }
            /* PID */
            {
                char fb[80];
                int  fl   = snprintf(fb, sizeof(fb), "%7d ", (int) p->PID);
                int  skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(ov_pid_color(p->PID));
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Status */
            {
                char fb[80];
                int  fl   = snprintf(fb, sizeof(fb), "%5s ", sl);
                int  skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(sc);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Hz */
            {
                char     fb[80];
                int      fl;
                ov_rgb_t hzc;
                if (p->loop_hz > 0.1)
                {
                    hzc = ov_rgb_lerp(OV_FG_DIM, OV_FG_ACTIVE, (float) (p->loop_hz / 5000.0));
                    fl  = snprintf(fb, sizeof(fb), "%6.1f ", p->loop_hz);
                }
                else
                {
                    hzc = OV_FG_DIM;
                    fl  = snprintf(fb, sizeof(fb), "     - ");
                }
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(hzc);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Trigger mode */
            {
                char fb[80];
                int  fl   = snprintf(fb, sizeof(fb), "%3s ", render_trigmode_label(p->triggermode));
                int  skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_CONN);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Trigger stream */
            {
                char fb[80];
                int  fl;
                if (p->trigstreamname[0] != '\0' && p->triggermode > 0)
                {
                    fl = snprintf(fb, sizeof(fb), "%-10.10s ", p->trigstreamname);
                }
                else
                {
                    fl = snprintf(fb, sizeof(fb), "%-10s ", "-");
                }
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_STREAM);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Exec time */
            {
                char     fb[80];
                int      fl;
                ov_rgb_t ec = OV_FG_DIM;
                if (p->MeasureTiming && p->dtmedian_exec_ns > 0)
                {
                    double exec_ms = 1.0e-6 * (double) p->dtmedian_exec_ns;
                    fl             = snprintf(fb, sizeof(fb), "%7.3f", exec_ms);
                    /* Color: green < 1ms, yellow < 10ms,
                     * red > 10ms */
                    if (exec_ms < 1.0)
                    {
                        ec = OV_FG_ACTIVE;
                    }
                    else if (exec_ms < 10.0)
                    {
                        ec = OV_FG_WARN;
                    }
                    else
                    {
                        ec = OV_FG_ERROR;
                    }
                }
                else
                {
                    fl = snprintf(fb, sizeof(fb), "      -");
                }
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(ec);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* DUTY% (exec / iter) */
            {
                char     fb[80];
                int      fl;
                ov_rgb_t dc = OV_FG_DIM;
                if (p->MeasureTiming && p->dtmedian_exec_ns > 0 && p->dtmedian_iter_ns > 0)
                {
                    double duty =
                        100.0 * (double) p->dtmedian_exec_ns / (double) p->dtmedian_iter_ns;
                    fl = snprintf(fb, sizeof(fb), " %4.0f%%", duty);
                    if (duty > 90.0)
                    {
                        dc = OV_FG_ERROR;
                    }
                    else if (duty > 50.0)
                    {
                        dc = OV_FG_WARN;
                    }
                    else
                    {
                        dc = OV_FG_ACTIVE;
                    }
                }
                else
                {
                    fl = snprintf(fb, sizeof(fb), "    -");
                }
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(dc);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* CPU% */
            {
                char fb[80];
                int  fl   = snprintf(fb, sizeof(fb), " %5.1f%% ", p->cpu_used);
                int  skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_TEXT);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* LOOPCNT */
            {
                char fb[64];
                int  fl   = snprintf(fb, sizeof(fb), "%10" PRId64 " ", (int64_t) p->loopcnt);
                int  skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(p->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* MEM */
            {
                char memstr[16];
                format_mem_kb(memstr, sizeof(memstr), p->mem_rss_kb);
                char fb[32];
                int  fl   = snprintf(fb, sizeof(fb), "%5s ", memstr);
                int  skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_TEXT);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* MISSED */
            {
                char fb[64];
                int  fl =
                    snprintf(fb, sizeof(fb), "%10" PRIu64 " ", (uint64_t) p->triggermissed_cumul);
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(p->triggermissed_cumul > 0 ? OV_FG_WARN : OV_FG_DIM);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* PRIO */
            {
                char fb[80];
                int  fl   = snprintf(fb, sizeof(fb), "%4d", p->rt_priority);
                int  skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl) ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx)
                {
                    vv = mx;
                }
                if (vv > 0)
                {
                    ov_theme_fg(p->rt_priority > 0 ? OV_FG_ACTIVE : OV_FG_DIM);
                    ov_buf_printf("%.*s", vv, fb + skip);
                    printed += vv;
                }
            }

#undef PROC_FIELD

            if (lay->mouse_hover && lay->hover_view == OV_FOCUS_PROCS && lay->hover_idx == fi)
            {
                char loc_upt[12] = "-";
                if (p->start_time_sec > 0)
                {
                    format_uptime(loc_upt, sizeof(loc_upt), p->start_time_sec);
                }
                snprintf((char *) lay->hover_tooltip, sizeof(lay->hover_tooltip),
                         "PID: %d | Up: %s | CPU: %4.1f%% | Exec: %zu ns", (int) p->PID, loc_upt,
                         p->cpu_used, (size_t) p->dtmedian_exec_ns);

                int btn_w = 8; /* " [Kill] " */
                int rem   = avail - printed;
                if (rem >= btn_w)
                {
                    ov_buf_hline(' ', rem - btn_w);
                    ov_theme_bg(OV_FG_ERROR);
                    ov_theme_fg(OV_FG_TEXT);
                    ov_buf_printf(" [Kill] ");
                    printed += rem;
                }
            }

            /* Pad remainder */
            {
                int rem = avail - printed;
                if (rem > 0)
                {
                    ov_buf_hline(' ', rem);
                }
            }
            ov_buf_reset_attr();
        }
        else
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(r, lay->scroll_proc, max_rows, filt_n, OV_FG_PROC);

    /* ---- Footer stats on bottom border ---- */
    {
        /* Compute totals over ALL procs */
        int     tot_run = 0;
        double  tot_cpu = 0.0;
        int64_t tot_mem = 0;
        for (int j = 0; j < m->nb_procs; j++)
        {
            const OV_PROC *p = &m->procs[j];
            if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE)
            {
                tot_run++;
            }
            if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE ||
                p->loopstat == PROCESSINFO_LOOPSTAT_SPIN ||
                p->loopstat == PROCESSINFO_LOOPSTAT_INIT)
            {
                tot_cpu += p->cpu_used;
            }
            tot_mem += p->mem_rss_kb;
        }

        /* Compute totals over filtered subset */
        int     flt_run = 0;
        double  flt_cpu = 0.0;
        int64_t flt_mem = 0;
        for (int j = 0; j < filt_n; j++)
        {
            const OV_PROC *p = &m->procs[filt_idx[j]];
            if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE)
            {
                flt_run++;
            }
            if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE ||
                p->loopstat == PROCESSINFO_LOOPSTAT_SPIN ||
                p->loopstat == PROCESSINFO_LOOPSTAT_INIT)
            {
                flt_cpu += p->cpu_used;
            }
            flt_mem += p->mem_rss_kb;
        }

        int brow      = r.row + r.height - 1;
        int is_subset = (filt_n < m->nb_procs);

        /* Right side: total stats (always) */
        char tmem[16];
        format_mem_kb(tmem, sizeof(tmem), tot_mem);
        char rbuf[80];
        snprintf(rbuf, sizeof(rbuf), " %d RUN \u2502 %.0f%% CPU \u2502 %s ", tot_run,
                 (double) tot_cpu, tmem);
        int rlen  = (int) strlen(rbuf);
        int below = filt_n - lay->scroll_proc - (r.height - 3);
        int dw    = 0;
        if (below > 0)
        {
            dw      = 3;
            int tmp = below;
            while (tmp > 0)
            {
                dw++;
                tmp /= 10;
            }
        }
        int rcol = r.col + r.width - rlen - dw - 4;
        if (rcol > r.col + 1)
        {
            ov_buf_pos(brow, rcol);
            ov_theme_fg(OV_FG_ACTIVE);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", rbuf);
        }

        /* Left side: filtered stats (only if
         * filter is active) */
        if (is_subset)
        {
            char fmem[16];
            format_mem_kb(fmem, sizeof(fmem), flt_mem);
            char lbuf[80];
            snprintf(lbuf, sizeof(lbuf), " %d RUN \u2502 %.0f%% CPU \u2502 %s ", flt_run,
                     (double) flt_cpu, fmem);
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
}

/**
 * @brief Render the processes panel in the overview.
 */
void ov_render_procs_panel(const OV_LAYOUT *lay, const OV_MODEL *m, const OV_RELATED *rel)
{
    OV_RECT r = lay->r_procs;

    int     filt_idx[OV_MAX_PROCS];
    int     has_re;
    regex_t re;
    int     filt_n = ov_procs__filter(lay, m, rel, filt_idx, &has_re, &re);

    char title[80];
    if (lay->filter_proc[0] != '\0')
    {
        snprintf(title, sizeof(title), "PROCESSINFO /%s/", lay->filter_proc);
    }
    else
    {
        snprintf(title, sizeof(title), "PROCESSINFO");
    }
    ov_draw_panel_border(r.row, r.col, r.height, r.width, title, OV_FG_PROC,
                         lay->focus == OV_FOCUS_PROCS, 0);

    int hrow = r.row + 1;
    int hs   = lay->hscroll_proc;

    ov_procs__render_header(lay, hrow, hs, r);
    ov_procs__render_rows(lay, m, rel, hrow, hs, r, filt_idx, filt_n, has_re, &re);

    if (has_re)
    {
        regfree(&re);
    }
}
