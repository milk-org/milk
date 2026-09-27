// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_preview.c
 * @brief Preview banner and quick-action buttons rendered on top of panels.
 */

#include "overview_render_internal.h"
#include "overview_data_loops.h"
#include "stream_graph.h"

/**
 * ov_render_preview_line - render on row 3 between tabs and panels.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Shows untruncated fields for the focused panel's selected item
 * and action buttons on the right.
 */
void ov_render_preview_line(OV_LAYOUT *lay, const OV_MODEL *m)
{
    int W = lay->term_cols;

    /* Reset button tracking */
    lay->nb_preview_btns = 0;

    ov_buf_pos(3, 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_hline(' ', W);
    ov_buf_pos(3, 1);

    /* Use resolved selection from model */
    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int        ssel  = ov_get_selected_stream_idx(lay, m);
    int        psel  = ov_get_selected_proc_idx(lay, m);
    int        fsel  = ov_get_selected_fps_idx(lay, m);

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
        char loopinfo[48];
        loopinfo[0] = '\0';
        if (s->nb_loops > 1)
        {
            snprintf(loopinfo, sizeof(loopinfo), "  \xe2\xae\x82 %d loops", s->nb_loops);
        }
        else if (s->nb_loops == 1)
        {
            snprintf(loopinfo, sizeof(loopinfo), "  \xe2\x86\xba L%02d", s->primary_loop_id);
        }
        len = snprintf(line, sizeof(line),
                       " STM  %s  %s %s"
                       "  Hz:%.1f  ino:%" PRIu64 ""
                       "  own:%d  cnt:%" PRIu64 ""
                       "  wpid:%d  sem:%d%s",
                       s->name, render_dtype(s->datatype), szb, s->update_hz, (uint64_t) s->inode,
                       (int) s->ownerPID, (uint64_t) s->cnt0, (int) s->write_pid, s->nb_sem,
                       loopinfo);
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
        char loopinfo[48];
        loopinfo[0] = '\0';
        if (p->nb_loops > 1)
        {
            snprintf(loopinfo, sizeof(loopinfo), "  \xe2\xae\x82 %d loops", p->nb_loops);
        }
        else if (p->nb_loops == 1)
        {
            snprintf(loopinfo, sizeof(loopinfo), "  \xe2\x86\xba L%02d", p->primary_loop_id);
        }
        len = snprintf(line, sizeof(line),
                       " PRC  %s  PID:%d  %s"
                       "  Hz:%.1f  trig:%s"
                       "  sem:%d  loop:%" PRId64 ""
                       "  miss:%d  prio:%d%s",
                       p->name, (int) p->PID, sl, p->loop_hz,
                       p->trigstreamname[0] ? p->trigstreamname : "-", p->triggersem,
                       (int64_t) p->loopcnt, p->triggermissed, p->rt_priority, loopinfo);
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
        char          loopinfo[48];
        loopinfo[0] = '\0';
        if (f->nb_loops > 1)
        {
            snprintf(loopinfo, sizeof(loopinfo), "  \xe2\xae\x82 %d loops", f->nb_loops);
        }
        else if (f->nb_loops == 1)
        {
            snprintf(loopinfo, sizeof(loopinfo), "  \xe2\x86\xba L%02d", f->primary_loop_id);
        }
        len = snprintf(line, sizeof(line),
                       " FPS  %s  C:%s R:%s"
                       "  st:%08X  cpid:%d  rpid:%d"
                       "  %s%s",
                       f->name, f->conf_alive ? "Y" : "-", f->run_alive ? "Y" : "-", f->md_status,
                       (int) f->confpid, (int) f->runpid, f->description, loopinfo);
        break;
    }
    case OV_FOCUS_GRAPH:
    {
        label_color = OV_FG_LOOP;
        if (lay->graph_tab_mode == 1 && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
        {
            const OV_LOOP *lp = &m->loops[lay->sel_loop];
            len               = snprintf(
                line, sizeof(line), " LOOP L%02d  %-20.20s  Nodes:%d (%ds, %dp)  Hz:%.1f  %s  %s",
                lp->loop_id, lp->name, lp->nb_nodes, lp->nb_streams, lp->nb_procs + lp->nb_fps,
                lp->min_hz, lp->is_running ? "RUN" : (lp->is_paused ? "PAUS" : "IDLE"),
                (lp->overlap_mask != 0) ? "OVERLAPPING" : "EXCLUSIVE");
        }
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

    /* Action buttons (right-aligned) */
    {
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
            int btn_widths[6];
            int total_w = 0;
            for (int bi = 0; bi < nb; bi++)
            {
                int blen = (int) strlen(btns[bi].label);
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

            int reserved  = lay->freeze ? 12 : 1;
            int start_col = W - total_w - reserved + 1;
            if (start_col < 1)
            {
                start_col = 1;
            }

            int col = start_col;
            for (int bi = 0; bi < nb; bi++)
            {
                ov_buf_pos(3, col);
                if (btns[bi].always_active)
                {
                    ov_buf_bg(btns[bi].bg.r, btns[bi].bg.g, btns[bi].bg.b);
                    ov_buf_fg(255, 255, 255);
                }
                else if (!lay->ctrl_mode)
                {
                    ov_theme_bg(OV_BG_PANEL_ALT);
                    ov_theme_fg(OV_FG_MUTED);
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
            ov_buf_pos(3, col);
            ov_buf_bg(60, 130, 200);
            ov_buf_fg(255, 255, 255);
            ov_buf_bold();
            ov_buf_printf("%s", badge);
        }
    }

    ov_buf_reset_attr();
}
