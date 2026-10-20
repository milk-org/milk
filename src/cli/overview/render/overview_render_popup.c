// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_popup.c
 * @brief   Theme selector popup and y/N confirmation box for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>

void ov_render_theme_popup(OV_LAYOUT *lay)
{
    if (!lay->theme_popup.active)
    {
        return;
    }

    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts);
    double elapsed = (now_ts.tv_sec - lay->theme_popup.ts.tv_sec) +
                     (now_ts.tv_nsec - lay->theme_popup.ts.tv_nsec) * 1e-9;
    if (elapsed >= 1.0)
    {
        lay->theme_popup.active = 0;
        return;
    }

    int nthemes = ov_theme_count();
    int pw      = 46;
    int ph      = nthemes + 2;

    int pr = lay->term_rows - ph;
    int pc = lay->term_cols - pw - 2;

    if (pr < 2)
    {
        pr = 2;
    }
    if (pc < 1)
    {
        pc = 1;
    }
    if (pr + ph > lay->term_rows)
    {
        ph = lay->term_rows - pr;
    }
    if (pc + pw > lay->term_cols)
    {
        pw = lay->term_cols - pc;
    }

    lay->theme_popup.rect.row    = pr;
    lay->theme_popup.rect.col    = pc;
    lay->theme_popup.rect.height = ph;
    lay->theme_popup.rect.width  = pw;

    /* Top border */
    ov_buf_pos(pr, pc);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_WARN);
    ov_buf_bold();
    ov_buf_printf("%s%s", OV_BOX_TL, OV_BOX_H);
    ov_theme_fg(OV_FG_TITLE);
    ov_buf_printf(" THEME SELECTOR (↑/↓ • ESC) ");
    ov_theme_fg(OV_FG_WARN);
    int top_rem = (pc + pw - 1) - ov__cursor_col;
    if (top_rem > 0)
    {
        ov_buf_hline_utf8(OV_BOX_H, top_rem);
    }
    ov_buf_printf("%s", OV_BOX_TR);
    ov_buf_reset_attr();

    /* Render theme items */
    for (int i = 0; i < nthemes && (pr + 1 + i) < (pr + ph - 1); i++)
    {
        int               row       = pr + 1 + i;
        int               is_sel    = (i == lay->theme_popup.sel);
        int               is_active = (i == ov_theme_get_active_index());
        const ov_theme_t *th        = ov_theme_get(i);

        ov_buf_pos(row, pc);

        /* Left border */
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("%s", OV_BOX_V);

        /* Row content */
        ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : OV_BG_PANEL;
        ov_theme_bg(row_bg);

        if (is_sel)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf(" ▶ ");
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("   ");
        }

        /* Swatch: 4 color preview blocks */
        ov_buf_bg(th->bg_terminal.r, th->bg_terminal.g, th->bg_terminal.b);
        ov_buf_fg(th->fg_title.r, th->fg_title.g, th->fg_title.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_stream.r, th->fg_stream.g, th->fg_stream.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_active.r, th->fg_active.g, th->fg_active.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_warn.r, th->fg_warn.g, th->fg_warn.b);
        ov_buf_printf("■ ");

        /* Restore row background */
        ov_theme_bg(row_bg);

        /* Theme name */
        if (is_sel)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_BRIGHT);
        }
        else
        {
            ov_theme_fg(OV_FG_TEXT);
        }
        ov_buf_printf("%-18.18s ", th->name);

        /* Active tag */
        if (is_active)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf("● active");
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("        ");
        }

        /* Pad row to right edge */
        render_pad_to_col(pc + pw - 1);

        /* Right border */
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("%s", OV_BOX_V);
        ov_buf_reset_attr();
    }

    /* Bottom border with auto-close countdown */
    int bot_row = pr + ph - 1;
    ov_buf_pos(bot_row, pc);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_WARN);
    ov_buf_bold();
    ov_buf_printf("%s%s", OV_BOX_BL, OV_BOX_H);

    double remain = 1.0 - elapsed;
    if (remain < 0.0)
    {
        remain = 0.0;
    }
    char hint[48];
    snprintf(hint, sizeof(hint), " auto-closes in %.1fs ", remain);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("%s", hint);

    ov_theme_fg(OV_FG_WARN);
    int bot_rem = (pc + pw - 1) - ov__cursor_col;
    if (bot_rem > 0)
    {
        ov_buf_hline_utf8(OV_BOX_H, bot_rem);
    }
    ov_buf_printf("%s", OV_BOX_BR);
    ov_buf_reset_attr();
}

/** confirm_line - one interior line of the confirm box at @row, text clipped to the box. */
static void confirm_line(int row, int col, int w, ov_rgb_t fg, const char *text)
{
    clear_row(row, col + 1, w - 2, OV_BG_PANEL);
    ov_buf_pos(row, col + 2);
    ov_theme_bg(OV_BG_PANEL);
    ov_theme_fg(fg);
    ov_buf_printf("%-*.*s", w - 4, w - 4, text);
}

/** confirm_batch_lines - count, first names, missing and skipped marks of a batch verb. */
static int confirm_batch_lines(const OV_LAYOUT *lay, const OV_MODEL *m, char lines[][160], int max)
{
    const OV_CONFIRM *c = &lay->confirm;
    int               n = 0, missing = 0, skipped = 0;
    char              names[160] = "";
    for (int k = 0; k < lay->multi_sel_count; k++)
    {
        int fi = ov_tgt_find(m, OV_FOCUS_FPS, &lay->multi_sel_fps[k]);
        if (fi < 0)
        {
            missing++;
            continue;
        }
        const OV_FPS *f = &m->fps[fi];
        if (c->verb == OV_ACT_BATCH_SIGNAL
                ? (!f->run_alive && !f->conf_alive)
                : ((c->verb == OV_ACT_BATCH_CONF ? !!f->conf_alive : !!f->run_alive) == c->arg))
        {
            skipped++;
        }
        if (k < 5)
        {
            size_t l = strlen(names);
            snprintf(names + l, sizeof(names) - l, "%s%s", l ? ", " : "", f->name);
        }
    }
    snprintf(lines[n++], 160, "%s", names);
    if (lay->multi_sel_count > 5 && n < max)
    {
        snprintf(lines[n++], 160, "+%d more", lay->multi_sel_count - 5);
    }
    if (n < max)
    {
        snprintf(lines[n++], 160, "%d already in that state (skipped), %d no longer present",
                 skipped, missing);
    }
    return n;
}

/**
 * ov_render_confirm - draw the y/N box of a pending destructive or batch verb: verb, target, impact.
 */
void ov_render_confirm(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_CONFIRM *c = &lay->confirm;
    if (c->verb == OV_ACT_NONE)
    {
        return;
    }
    char lines[6][160];
    int  nl = 0;
    int idx = (c->verb < OV_ACT_BATCH_SIGNAL) ? ov_tgt_find(m, (ov_focus_t) c->panel, &c->tgt) : -1;
    const char *verb = "";
    switch (c->verb)
    {
    case OV_ACT_PROC_SIGKILL:
        verb = "SIGKILL process (no cleanup, its loop stops)";
        break;
    case OV_ACT_PROC_REMOVE:
        verb = "Remove process entry (proc file and registry slot)";
        break;
    case OV_ACT_PROC_CLEANUP:
        verb = "Clean up dead process entries (milk-procinfo-rm -c)";
        break;
    case OV_ACT_FPS_SIGKILL:
        verb = "SIGKILL FPS conf/run processes";
        break;
    case OV_ACT_FPS_ERASE:
        verb = "Erase FPS: stops CONF+RUN, closes tmux, deletes shm+fpslog";
        break;
    case OV_ACT_STREAM_DELETE:
        verb = "Delete stream (removes its shm file)";
        break;
    case OV_ACT_BATCH_SIGNAL:
        verb = (c->arg == SIGKILL) ? "Batch SIGKILL" : "Batch SIGTERM";
        break;
    case OV_ACT_BATCH_RUN:
        verb = c->arg ? "Batch RUN start" : "Batch RUN stop";
        break;
    default:
        verb = c->arg ? "Batch CONF start" : "Batch CONF stop";
        break;
    }
    snprintf(lines[nl++], 160, "%s", verb);

    if (c->verb >= OV_ACT_BATCH_SIGNAL)
    {
        snprintf(lines[nl++], 160, "%d marked FPS:", lay->multi_sel_count);
        nl += confirm_batch_lines(lay, m, &lines[nl], 6 - nl);
    }
    else if (c->verb == OV_ACT_PROC_CLEANUP)
    {
        int dead = 0;
        for (int i = 0; i < m->nb_procs; i++)
        {
            dead += (m->procs[i].loopstat == PROCESSINFO_LOOPSTAT_STOP ||
                     m->procs[i].loopstat == PROCESSINFO_LOOPSTAT_CRASHED);
        }
        snprintf(lines[nl++], 160, "%d STOP/CRSH entries listed", dead);
    }
    else if (idx < 0)
    {
        snprintf(lines[nl++], 160, "%s  (no longer present)", c->tgt.name);
    }
    else if (c->panel == OV_FOCUS_PROCS)
    {
        snprintf(lines[nl++], 160, "%s  PID %d  %s", m->procs[idx].name, (int) m->procs[idx].PID,
                 ov_state_info(ov_proc_state(&m->procs[idx])).lname);
    }
    else if (c->panel == OV_FOCUS_FPS)
    {
        snprintf(lines[nl++], 160, "%s  conf PID %d  run PID %d", m->fps[idx].name,
                 (int) m->fps[idx].confpid, (int) m->fps[idx].runpid);
    }
    else
    {
        const OV_STREAM *s = &m->streams[idx];
        snprintf(lines[nl++], 160, "%s  inode %lu  writer PID %d  in %d loop(s)", s->name,
                 (unsigned long) s->inode, (int) s->write_pid, s->nb_loops);
    }

    int w   = (lay->term_cols - 4 < 72) ? lay->term_cols - 4 : 72;
    int h   = nl + 4;
    int row = (lay->term_rows - h) / 2;
    int col = (lay->term_cols - w) / 2;
    if (w < 20 || row < 1)
    {
        return;
    }
    ov_draw_panel_border(row, col, h, w, " CONFIRM ", OV_FG_ERROR, 1, 1);
    for (int i = 0; i < nl; i++)
    {
        confirm_line(row + 1 + i, col, w, (i == 0) ? OV_FG_WARN : OV_FG_TEXT, lines[i]);
    }
    confirm_line(row + 1 + nl, col, w, OV_FG_DIM, "");
    confirm_line(row + 2 + nl, col, w, OV_FG_BRIGHT, "[y] confirm   any other key: cancel");
    ov_buf_reset_attr();
}
