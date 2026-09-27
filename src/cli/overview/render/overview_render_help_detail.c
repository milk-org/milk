// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_help_detail.c
 * @brief   Contextual help detail pane rendering for milk-CTRL.
 */

#include "overview_render_help_internal.h"
#include <stdio.h>
/**
 * ov_help_print_wrapped - word-wrap and print text within a rectangular area.
 * @text:      String to wrap and print.
 * @row:       Starting terminal row.
 * @col:       Starting terminal column.
 * @max_w:     Maximum width in columns.
 * @max_lines: Maximum number of lines.
 * @fg:        Foreground color.
 * @bg:        Background color.
 */
void ov_help_print_wrapped(const char *text,
                           int         row,
                           int         col,
                           int         max_w,
                           int         max_lines,
                           ov_rgb_t    fg,
                           ov_rgb_t    bg)
{
    if (text == NULL || max_w <= 0 || max_lines <= 0)
    {
        return;
    }

    const char *p             = text;
    int         lines_printed = 0;

    while (*p != '\0' && lines_printed < max_lines)
    {
        /* Skip leading whitespace on new line */
        while (*p == ' ')
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }

        int len        = 0;
        int last_space = -1;
        while (p[len] != '\0' && p[len] != '\n' && len < max_w)
        {
            if (p[len] == ' ')
            {
                last_space = len;
            }
            len++;
        }

        int line_len = len;
        if (p[len] == '\n')
        {
            line_len = len;
            len++;
        }
        else if (p[len] != '\0' && last_space > 0)
        {
            line_len = last_space;
            len      = last_space + 1;
        }

        ov_buf_pos(row + lines_printed, col);
        ov_theme_bg(bg);
        ov_theme_fg(fg);
        ov_buf_printf("%.*s", line_len, p);

        int pad = max_w - line_len;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }

        p += len;
        lines_printed++;
    }

    /* Clear any remaining allocated lines */
    for (int r = lines_printed; r < max_lines; r++)
    {
        ov_buf_pos(row + r, col);
        ov_theme_bg(bg);
        ov_buf_hline(' ', max_w);
    }
}

/**
 * ov_help_render_detail - render in-depth help and target info for selected item.
 * @lay:      layout state
 * @m:        system model
 * @entry:    currently highlighted help entry
 * @split_r:  row of separator line
 * @pc:       left column of help box
 * @pw:       width of help box
 * @detail_h: height of detail pane
 */
void ov_help_render_detail(const OV_LAYOUT    *lay,
                           const OV_MODEL     *m,
                           const help_entry_t *entry,
                           int                 split_r,
                           int                 pc,
                           int                 pw,
                           int                 detail_h)
{
    int inner_w = pw - 4;
    int col     = pc + 2;

    /* Row 1: Header (Key / Command title + Status badges) */
    ov_buf_pos(split_r + 1, col);
    ov_theme_bg(OV_BG_PANEL);

    if (entry->flags & HF_SECTION)
    {
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        if (entry->section == HS_INTRO)
        {
            ov_buf_printf("■ INTRODUCTION: %s", entry->label);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            const char *hint = "[Press 1, 'i', or ENTER to open full interactive guide]";
            int         rem  = inner_w - (16 + (int) strlen(entry->label) + (int) strlen(hint));
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("%s", hint);
        }
        else
        {
            ov_buf_printf("■ PANEL OVERVIEW: %s", entry->label);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            const char *hint = help_is_expanded(lay, entry->section)
                                   ? "[Press ← / ENTER to collapse]"
                                   : "[Press → / ENTER to expand]";
            int         rem  = inner_w - (18 + (int) strlen(entry->label) + (int) strlen(hint));
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%s", hint);
        }
    }
    else if (entry->flags & HF_ENTRY)
    {
        ov_buf_bold();
        if (entry->flags & HF_CTRL_MODE)
        {
            if (lay->ctrl_mode)
            {
                ov_buf_fg(255, 95, 75);
                ov_buf_printf("Key: [ %s ]", entry->key);
                ov_buf_fg(255, 80, 80);
                ov_buf_printf("  ⚡ CONTROL MODE: ON (Ready)");
            }
            else
            {
                ov_buf_fg(205, 140, 50);
                ov_buf_printf("Key: [ %s ]", entry->key);
                ov_buf_fg(205, 140, 50);
                ov_buf_printf("  🔒 REQUIRES CONTROL MODE (Press 'c')");
            }
        }
        else
        {
            ov_buf_fg(130, 205, 255);
            ov_buf_printf("Key: [ %s ]", entry->key);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("  (Standard Command)");
        }
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);

        /* Section scope badge on right */
        const char *sname   = ov_help_section_name(entry->section);
        int         key_len = (int) strlen(entry->key) + 9;
        int         badge_l = (int) strlen(sname) + 10;
        int         rem     = inner_w - (key_len + 35 + badge_l);
        if (rem > 0)
        {
            ov_buf_hline(' ', rem);
        }
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("[Panel: %s]", sname);
    }
    else
    {
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("■ %s", entry->label);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_hline(' ', inner_w - (2 + (int) strlen(entry->label)));
    }

    /* Rows 2..N: Word-wrapped descriptive detail */
    int text_lines = detail_h - 3;
    if (text_lines > 0)
    {
        ov_help_print_wrapped(entry->detail, split_r + 2, col, inner_w, text_lines, OV_FG_TEXT,
                              OV_BG_PANEL);
    }

    /* Last row: Live Target info for the currently selected GUI item */
    int target_row = split_r + detail_h - 1;
    ov_buf_pos(target_row, col);
    ov_theme_bg(OV_BG_PANEL);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_bold();
    ov_buf_printf("Target: ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    int  target_avail = inner_w - 8;
    char tbuf[160];
    tbuf[0]            = '\0';
    ov_rgb_t target_fg = OV_FG_DIM;

    if (entry->section == HS_INTRO)
    {
        target_fg = OV_FG_TITLE;
        snprintf(tbuf, sizeof(tbuf),
                 "Intro Guide: Press '1', 'i', or ENTER to open the interactive introduction");
    }
    else if (entry->section == HS_STREAMS)
    {
        int si = ov_get_selected_stream_idx(lay, m);
        if (si >= 0 && si < m->nb_streams)
        {
            const OV_STREAM *s = &m->streams[si];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t) { 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will delete stream '%s' (%s %s) from SHM", s->name,
                         s->size_str, render_dtype(s->datatype));
            }
            else
            {
                target_fg = OV_FG_STREAM;
                snprintf(tbuf, sizeof(tbuf), "Selected stream '%s' (%s %s, %.1f Hz)", s->name,
                         s->size_str, render_dtype(s->datatype), s->update_hz);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No stream selected (Streams panel is empty)");
        }
    }
    else if (entry->section == HS_PROCS)
    {
        int pi = ov_get_selected_proc_idx(lay, m);
        if (pi >= 0 && pi < m->nb_procs)
        {
            const OV_PROC *p = &m->procs[pi];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t) { 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will signal process '%s' (PID %d, %s)", p->name,
                         (int) p->PID, p->statusmsg);
            }
            else
            {
                target_fg = OV_FG_PROC;
                snprintf(tbuf, sizeof(tbuf), "Selected process '%s' (PID %d, CPU %.1f%%, %s)",
                         p->name, (int) p->PID, p->cpu_used, p->statusmsg);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No process selected (Processes panel is empty)");
        }
    }
    else if (entry->section == HS_FPS)
    {
        int fi = ov_get_selected_fps_idx(lay, m);
        if (fi >= 0 && fi < m->nb_fps)
        {
            const OV_FPS *f = &m->fps[fi];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t) { 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will control FPS module '%s' (run=%s, conf=%s)",
                         f->name, f->run_alive ? "ON" : "OFF", f->conf_alive ? "ON" : "OFF");
            }
            else
            {
                target_fg = OV_FG_FPS;
                snprintf(tbuf, sizeof(tbuf),
                         "Selected FPS module '%s' (run=%s, conf=%s, RSS: %ld KB)", f->name,
                         f->run_alive ? "ON" : "OFF", f->conf_alive ? "ON" : "OFF",
                         (long) f->mem_rss_kb);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No FPS module selected");
        }
    }
    else if (entry->section == HS_GRAPH)
    {
        if (m != NULL && lay->sel_graph >= 0 && lay->sel_graph < m->nb_nodes)
        {
            const OV_NODE *n = &m->nodes[lay->sel_graph];
            target_fg        = OV_FG_CONN;
            snprintf(tbuf, sizeof(tbuf), "Selected node '%s' (type: %s)", n->name,
                     (n->type == OV_NODE_STREAM) ? "Stream"
                                                 : ((n->type == OV_NODE_PROC) ? "Process" : "FPS"));
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "Node graph cursor active");
        }
    }
    else
    {
        /* Global & Display sections: show active GUI selection */
        int sel_s = ov_get_selected_stream_idx(lay, m);
        int sel_p = ov_get_selected_proc_idx(lay, m);
        int sel_f = ov_get_selected_fps_idx(lay, m);
        if (lay->focus == OV_FOCUS_STREAMS && m && sel_s >= 0 && sel_s < m->nb_streams)
        {
            target_fg = OV_FG_STREAM;
            snprintf(tbuf, sizeof(tbuf), "Active selection: Stream '%s' (Panel: Streams)",
                     m->streams[sel_s].name);
        }
        else if (lay->focus == OV_FOCUS_PROCS && m && sel_p >= 0 && sel_p < m->nb_procs)
        {
            target_fg = OV_FG_PROC;
            snprintf(tbuf, sizeof(tbuf), "Active selection: Process '%s' (PID %d) (Panel: Procs)",
                     m->procs[sel_p].name, (int) m->procs[sel_p].PID);
        }
        else if (lay->focus == OV_FOCUS_FPS && m && sel_f >= 0 && sel_f < m->nb_fps)
        {
            target_fg = OV_FG_FPS;
            snprintf(tbuf, sizeof(tbuf), "Active selection: FPS '%s' (Panel: FPS)",
                     m->fps[sel_f].name);
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "Global action — applies across all dashboard views");
        }
    }

    ov_theme_fg(target_fg);
    ov_buf_printf("%s", tbuf);
    int chars_used = (int) strlen(tbuf);
    int target_pad = target_avail - chars_used;
    if (target_pad > 0)
    {
        ov_buf_hline(' ', target_pad);
    }
}
