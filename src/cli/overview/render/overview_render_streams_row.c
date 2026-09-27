// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_streams_row.c
 * @brief   Single stream row rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/**
 * ov_streams_render_single_row - render a single stream row in the streams panel.
 * @lay:    Pointer to layout structure.
 * @m:      Pointer to data model snapshot.
 * @rel:    Pointer to relationship lookup tables.
 * @row:    Terminal screen row index.
 * @i:      Row index within visible panel list.
 * @fi:     Index in filtered stream array.
 * @si:     Stream index in data model.
 * @sdepth: Lineage depth for ancestry indicators.
 * @has_re: Non-zero if regex filter is active and compiled.
 * @re:     Pointer to compiled regex.
 * @r:      Bounding rectangle of streams panel.
 */
void ov_streams_render_single_row(const OV_LAYOUT  *lay,
                                  const OV_MODEL   *m,
                                  const OV_RELATED *rel,
                                  int               row,
                                  int               i,
                                  int               fi,
                                  int               si,
                                  int8_t            sdepth,
                                  int               has_re,
                                  const regex_t    *re,
                                  OV_RECT           r)
{
    const OV_STREAM *s = &m->streams[si];
    int              is_sel =
        (fi == lay->sel_stream && (lay->focus == OV_FOCUS_STREAMS || lay->focus == OV_FOCUS_GRAPH));
    int is_frozen =
        (lay->freeze && lay->freeze_focus == OV_FOCUS_STREAMS && fi == lay->freeze_sel_stream);
    ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int        is_rel    = (!is_sel && !is_frozen && eff_focus != OV_FOCUS_STREAMS && rel != NULL &&
                            bget(rel->streams, si));
    int        is_loop_member = 0;
    if ((lay->graph_tab_mode == 1 || lay->view == OV_VIEW_LOOPS) && lay->sel_loop >= 0 &&
        lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        if (s->loop_mask & active_mask)
        {
            is_loop_member = 1;
        }
    }

    ov_rgb_t row_bg = OV_BG_PANEL;
    if (is_sel)
    {
        row_bg = OV_BG_SELECTED;
    }
    else if (is_frozen)
    {
        row_bg = OV_BG_FROZEN;
    }
    else if (is_loop_member)
    {
        row_bg = (s->nb_loops > 1) ? OV_BG_LOOP_SHARED : OV_BG_LOOP;
    }
    else if (is_rel)
    {
        row_bg = OV_BG_RELATED;
    }
    else if (s->is_new > 0)
    {
        row_bg = OV_BG_NEW_ITEM;
    }
    else if (lay->mouse_hover && lay->hover_global_stream == si)
    {
        row_bg = OV_BG_HOVER;
    }
    row_bg = zebra_bg(row_bg, i);

    int hs_rem  = lay->hscroll_stream;
    int printed = 1;
    int avail   = r.width - 2;

    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(row_bg);

    /* Focus ring accent strip (#10) */
    int panel_focused = (lay->focus == OV_FOCUS_STREAMS);
    render_focus_strip(row, r.col + 1, panel_focused, OV_FG_STREAM, row_bg);

#define STRM_FIELD_WITH_COL(vcol_idx, logical_idx, color, bg_color, fmt, ...)                     \
    do                                                                                            \
    {                                                                                             \
        char _fb[128];                                                                            \
        int  _fl = snprintf(_fb, sizeof(_fb), fmt, ##__VA_ARGS__);                                \
        (void) _fl;                                                                               \
        ov_render_cell(logical_idx, vcol_idx, (color), (bg_color), _fb, &hs_rem, &printed, avail, \
                       lay->highlight_col_stream, lay->col_collapsed_stream);                     \
    } while (0)

#define STRM_FIELD(color, fmt, ...)                                                   \
    do                                                                                \
    {                                                                                 \
        int logical_idx = ov_get_logical_col_stream(vcol, lay->compact_mode);         \
        STRM_FIELD_WITH_COL(vcol, logical_idx, (color), cell_bg, fmt, ##__VA_ARGS__); \
        vcol++;                                                                       \
    } while (0)

#define STRM_PID_FIELD(pid_val, fmt, ...)                                            \
    do                                                                               \
    {                                                                                \
        int      _match  = (_spid > 0 && (pid_t) (pid_val) == _spid);                \
        ov_rgb_t prev_bg = cell_bg;                                                  \
        if (_match)                                                                  \
        {                                                                            \
            cell_bg = OV_BG_PID_MATCH;                                               \
            ov_buf_bold();                                                           \
        }                                                                            \
        STRM_FIELD(_match ? ((ov_rgb_t) { 0, 0, 0 }) : ov_pid_color((pid_val)), fmt, \
                   ##__VA_ARGS__);                                                   \
        if (_match)                                                                  \
        {                                                                            \
            ov_buf_reset_attr();                                                     \
            cell_bg = prev_bg;                                                       \
        }                                                                            \
    } while (0)

    pid_t    _spid   = (rel != NULL) ? rel->sel_pid : 0;
    int      vcol    = 1;
    ov_rgb_t cell_bg = row_bg;

    /* Ancestry column — rendered raw */
    char anc_str[64] = "";
    if (sdepth != 0 && !is_sel && !is_frozen)
    {
        int abs_d = sdepth < 0 ? -sdepth : sdepth;
        if (abs_d > 99)
        {
            abs_d = 99;
        }
        if (sdepth < 0)
        {
            snprintf(anc_str, sizeof(anc_str), abs_d < 10 ? "\xe2\x97\x80%d  " : "\xe2\x97\x80%d ",
                     abs_d);
        }
        else
        {
            snprintf(anc_str, sizeof(anc_str), abs_d < 10 ? "%d\xe2\x96\xb6  " : "%d\xe2\x96\xb6 ",
                     abs_d);
        }
    }
    else
    {
        /* Activity & Loop indicator */
        if (s->nb_loops > 1)
        {
            snprintf(anc_str, sizeof(anc_str), "\xe2\xae\x82 "); /* ⮂ */
        }
        else if (s->nb_loops == 1)
        {
            snprintf(anc_str, sizeof(anc_str), "\xe2\x86\xba "); /* ↺ */
        }
        else if (s->update_hz > 0.1)
        {
            snprintf(anc_str, sizeof(anc_str), "\xe2\x97\x8f ");
        }
        else
        {
            snprintf(anc_str, sizeof(anc_str), "  ");
        }

        /* R/W direction arrow */
        if ((eff_focus == OV_FOCUS_PROCS || eff_focus == OV_FOCUS_FPS) && is_rel && rel != NULL)
        {
            int is_written = bget(rel->stream_written, si);
            strncat(anc_str, is_written ? "\xe2\x96\xb6 " : "\xe2\x97\x80 ",
                    sizeof(anc_str) - strlen(anc_str) - 1);
        }
        else
        {
            strncat(anc_str, "  ", sizeof(anc_str) - strlen(anc_str) - 1);
        }
    }

    ov_rgb_t anc_color =
        (s->nb_loops > 1) ? OV_FG_LOOP_SHARED : ((s->nb_loops == 1) ? OV_FG_LOOP : OV_FG_WARN);
    ov_render_cell(0, 0, anc_color, row_bg, anc_str, &hs_rem, &printed, avail,
                   lay->highlight_col_stream, lay->col_collapsed_stream);

    ov_rgb_t base_color = s->active ? OV_FG_STREAM : OV_FG_DIM;

    /* Stream Name with regex match highlighting */
    {
        char       name_cell[128];
        regmatch_t pm[1];
        if (has_re && re != NULL && regexec(re, s->name, 1, pm, 0) == 0)
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
            int tail_len = 14 - (b_len + m_len);
            if (tail_len < 0)
            {
                tail_len = 0;
            }
            snprintf(name_cell, sizeof(name_cell), "%.*s\x01%.*s\x02%.*s ", b_len, s->name, m_len,
                     s->name + b_len, tail_len, s->name + b_len + m_len);
        }
        else
        {
            snprintf(name_cell, sizeof(name_cell), "%-14.14s ", s->name);
        }
        STRM_FIELD(base_color, "%s", name_cell);
    }
    STRM_FIELD(OV_FG_MUTED, "%4s ", render_dtype(s->datatype));

    STRM_FIELD(OV_FG_TEXT, "%11s ", s->size_str);

    if (s->update_hz > 0.1)
    {
        STRM_FIELD(OV_FG_ACTIVE, "%6.1f ", s->update_hz);
    }
    else
    {
        STRM_FIELD(OV_FG_DIM, "     - ");
    }

    /* MB/s throughput */
    if (s->update_hz > 0.1)
    {
        double mbps =
            s->update_hz * (double) s->nelement * dtype_bytesize(s->datatype) / (1024.0 * 1024.0);
        if (mbps >= 1000.0)
        {
            STRM_FIELD(OV_FG_ACTIVE, "%6.1fG ", mbps / 1024.0);
        }
        else
        {
            STRM_FIELD(OV_FG_ACTIVE, "%6.1fM ", mbps);
        }
    }
    else
    {
        STRM_FIELD(OV_FG_DIM, "      - ");
    }

    if (!lay->compact_mode)
    {
        STRM_FIELD(OV_FG_DIM, "%10" PRIu64 " ", (uint64_t) s->inode);
    }

    STRM_PID_FIELD(s->ownerPID, "%7d ", (int) s->ownerPID);
    if (!lay->compact_mode)
    {
        STRM_FIELD(s->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM, "%10" PRIu64 " ", (uint64_t) s->cnt0);

        int sem_logical   = ov_get_logical_col_stream(vcol, lay->compact_mode);
        int sem_collapsed = (lay->col_collapsed_stream & (1U << sem_logical)) != 0;
        if (sem_collapsed)
        {
            STRM_FIELD_WITH_COL(vcol, sem_logical, OV_FG_DIM, cell_bg, ".");
        }
        else
        {
            for (int sm = 0; sm < 10; sm++)
            {
                if (sm < s->nb_sem)
                {
                    int  val = s->semval[sm];
                    char c;
                    if (val < 0)
                    {
                        c = '-';
                    }
                    else if (val > 9)
                    {
                        c = '+';
                    }
                    else
                    {
                        c = '0' + val;
                    }
                    STRM_FIELD_WITH_COL(vcol, sem_logical, ov_get_sem_color(val), cell_bg, "%c", c);
                }
                else
                {
                    STRM_FIELD_WITH_COL(vcol, sem_logical, OV_FG_DIM, cell_bg, ".");
                }
            }
            STRM_FIELD_WITH_COL(vcol, sem_logical, OV_FG_DIM, cell_bg, " ");
        }
        vcol++;
    }

    /* Write PID */
    if (s->write_pid > 0)
    {
        STRM_PID_FIELD(s->write_pid, "%7d ", (int) s->write_pid);
    }
    else
    {
        STRM_FIELD(OV_FG_DIM, "      - ");
    }

    /* Read PIDs (compact list) */
    int rpid_logical   = ov_get_logical_col_stream(vcol, lay->compact_mode);
    int rpid_collapsed = (lay->col_collapsed_stream & (1U << rpid_logical)) != 0;
    if (rpid_collapsed)
    {
        STRM_FIELD_WITH_COL(vcol, rpid_logical, OV_FG_DIM, cell_bg, "-");
    }
    else
    {
        if (s->nb_read_pids > 0)
        {
            for (int rp = 0; rp < s->nb_read_pids; rp++)
            {
                if (rp > 0)
                {
                    STRM_FIELD_WITH_COL(vcol, rpid_logical, OV_FG_DIM, cell_bg, ":");
                }
                int      _match  = (_spid > 0 && (pid_t) s->read_pids[rp] == _spid);
                ov_rgb_t prev_bg = cell_bg;
                if (_match)
                {
                    cell_bg = OV_BG_PID_MATCH;
                    ov_buf_bold();
                }
                STRM_FIELD_WITH_COL(vcol, rpid_logical,
                                    _match ? ((ov_rgb_t) { 0, 0, 0 })
                                           : ov_pid_color((s->read_pids[rp])),
                                    cell_bg, "%d", (int) s->read_pids[rp]);
                if (_match)
                {
                    ov_buf_reset_attr();
                    cell_bg = prev_bg;
                }
            }
            STRM_FIELD_WITH_COL(vcol, rpid_logical, OV_FG_DIM, cell_bg, " ");
        }
        else
        {
            STRM_FIELD_WITH_COL(vcol, rpid_logical, OV_FG_DIM, cell_bg, "- ");
        }
    }

#undef STRM_PID_FIELD
#undef STRM_FIELD
#undef STRM_FIELD_WITH_COL

    if (lay->mouse_hover && lay->hover_view == OV_FOCUS_STREAMS && lay->hover_idx == si)
    {
        snprintf((char *) lay->hover_tooltip, sizeof(lay->hover_tooltip),
                 "Stream: %s | Dimensions: %dD (%s) | Semaphores: %d | inode: %" PRIu64, s->name,
                 s->naxis, s->size_str, s->nb_sem, (uint64_t) s->inode);

        int btn_w = 10; /* " [Delete] " */
        int rem   = r.width - printed;
        if (rem >= btn_w)
        {
            render_pad_spaces(printed, r.width - btn_w);
            ov_theme_bg(OV_FG_ERROR);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf(" [Delete] ");
            printed = r.width;
        }
    }

    render_pad_spaces(printed, r.width);
    ov_buf_reset_attr();
}
