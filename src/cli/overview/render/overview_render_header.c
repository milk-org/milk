// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_header.c
 * @brief   Top status header and metrics banner rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include "overview_data_internal.h"
#include "milk_config.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

extern float ov_scan_get_interval(void);

/**
 * ov_render_header_text - render formatted header text with highlight markers and UTF-8 handling.
 * @text:          Input string with optional \x01 (highlight on) and \x02 (highlight off).
 * @hs:            Horizontal scroll offset.
 * @max_vis_width: Maximum visible columns allowed to be printed.
 * @base_fg:       Base foreground color when not highlighted.
 *
 * Return: Number of visible character cells printed.
 */
int ov_render_header_text(
    const char *text,
    int         hs,
    int         max_vis_width,
    ov_rgb_t    base_fg)
{
    int vis_col = 0;
    int printed = 0;
    int i       = 0;

    while (text[i] != '\0' && printed < max_vis_width)
    {
        if (text[i] == '\x01')
        {
            if (vis_col >= hs)
            {
                ov_theme_fg(OV_FG_BRIGHT);
                ov_buf_bold();
                ov_buf_underline();
            }
            i++;
        }
        else if (text[i] == '\x02')
        {
            if (vis_col >= hs)
            {
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_HEADER);
                ov_theme_fg(base_fg);
            }
            i++;
        }
        else
        {
            int clen = 1;
            if ((text[i] & 0xE0) == 0xC0)
            {
                clen = 2;
            }
            else if ((text[i] & 0xF0) == 0xE0)
            {
                clen = 3;
            }
            else if ((text[i] & 0xF8) == 0xF0)
            {
                clen = 4;
            }

            if (vis_col >= hs)
            {
                ov_buf_printf("%.*s", clen, text + i);
                printed++;
            }
            vis_col++;
            i += clen;
        }
    }
    return printed;
}

void render_pad_spaces(int chars_written, int panel_width);

/**
 * view_label - get short 4-letter view mode tag string.
 * @v: View mode enum.
 *
 * Return: Static label string ("DASH", "STRM", "PROC", "FPS", "CONN", "LOOPS").
 */
static const char *view_label(ov_view_t v)
{
    switch (v)
    {
    case OV_VIEW_DASHBOARD:
        return "DASH";
    case OV_VIEW_STREAMS:
        return "STRM";
    case OV_VIEW_PROCS:
        return "PROC";
    case OV_VIEW_FPS:
        return "FPS";
    case OV_VIEW_GRAPH:
        return "CONN";
    case OV_VIEW_LOOPS:
        return "LOOPS";
    default:
        return "";
    }
}

/**
 * ov_render_header - render the top status header bar of milk-CTRL.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 */
void ov_render_header(
    OV_LAYOUT      *lay,
    const OV_MODEL *m)
{
    /* Advance blink counter each frame */
    lay->ctrl_blink++;

    OV_RECT r = lay->r_header;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);

    /* ── Heartbeat: fast-pulsing indicator ── */
    {
        int beat = lay->ctrl_blink % 2;
        if (beat == 0)
        {
            /* Bright beat — vivid red */
            ov_buf_fg(255, 50, 50);
            ov_buf_bold();
            ov_buf_printf("\xe2\x99\xa5"); /* ♥ */
            ov_buf_reset_attr();
        }
        else
        {
            /* Dim beat — dark red */
            ov_buf_fg(100, 30, 30);
            ov_buf_printf("\xe2\x99\xa5"); /* ♥ */
        }
        ov_theme_bg(OV_BG_HEADER);
    }

    /* LCARS-style rounded end cap */
    ov_theme_fg(OV_GRAD_LO);
    ov_buf_printf("%s", OV_LCARS_LEFT);

    /* Gradient header text */
    ov_buf_bold();
    ov_buf_printf_gradient(OV_GRAD_LO, OV_GRAD_HI, " %s milk-CTRL ", OV_BULLET);
    ov_buf_reset_attr();

    /* LCARS-style rounded end cap (matching the gradient end) */
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_GRAD_HI);
    ov_buf_printf("%s ", OV_LCARS_RIGHT);

    /* Version / commit tracking */
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("[%s] ", MILK_GIT_COMMIT);

    /* Shared memory directory */
    const char *shmdir = ov_get_shmdir();
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("[shm: %s] ", shmdir);

    /* Blinking badge — visible when ctrl_mode is ON, READ ONLY when OFF */
    int ctrl_w = 0;
    if (lay->ctrl_mode)
    {
        /* Software blinking badge for "CONTROL" (fast 2.5Hz blink) */
        if ((lay->ctrl_blink % 4) < 2)
        {
            ov_buf_bg(OV_ANIM_PULSE_BG_MAX.r, OV_ANIM_PULSE_BG_MAX.g, OV_ANIM_PULSE_BG_MAX.b);
            ov_buf_fg(OV_ANIM_PULSE_FG_MAX.r, OV_ANIM_PULSE_FG_MAX.g, OV_ANIM_PULSE_FG_MAX.b);
        }
        else
        {
            ov_buf_bg(220, 40, 40);   /* vibrant red */
            ov_buf_fg(255, 255, 255); /* white text */
        }
        ov_buf_bold();
        ov_buf_printf(" [c] CONTROL ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ctrl_w = 13; /* visual width of " [c] CONTROL " */
    }
    else
    {
        /* READ ONLY badge (green) */
        ov_buf_bg(20, 180, 20);   /* deep green background */
        ov_buf_fg(220, 255, 220); /* light text */
        ov_buf_bold();
        ov_buf_printf(" [c] READ ONLY ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ctrl_w = 15; /* visual width of " [c] READ ONLY " */
    }

    ov_buf_printf(" ");
    int hover_w = 0;
    if (lay->mouse_hover)
    {
        /* Mouse hover active badge */
        ov_buf_bg(180, 180, 20); /* deep yellow background */
        ov_buf_fg(20, 20, 20);   /* dark text */
        ov_buf_bold();
        ov_buf_printf(" [m] HOVER: ON ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        hover_w = 15; /* visual width of " [m] HOVER: ON " */
    }
    else
    {
        /* Mouse hover inactive badge */
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_bold();
        ov_buf_printf(" [m] HOVER: OFF ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        hover_w = 16; /* visual width of " [m] HOVER: OFF " */
    }

    int filter_w        = 0;
    lay->r_filter_count = 0;
    int b_start = 17 + (int) strlen(MILK_GIT_COMMIT) + 3 + (int) strlen(shmdir) + 8 + 1 + ctrl_w +
                  1 + hover_w + 1;

    ov_focus_t fpanel = ov_get_effective_filter_panel(lay);

    if (lay->view == OV_VIEW_DASHBOARD)
    {
        ov_focus_t  d_panels[3] = { OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS };
        const char *d_pnames[3] = { "STRM", "PROC", "FPS" };

        for (int p = 0; p < 3; p++)
        {
            ov_focus_t  curr_p  = d_panels[p];
            const char *curr_nm = d_pnames[p];
            const char *fpat    = ov_get_panel_filter_pattern(lay, curr_p);
            int         is_act  = ov_is_panel_filter_active(lay, curr_p);
            int         is_foc  = (fpanel == curr_p);

            ov_buf_printf(" ");
            filter_w++;

            char fbadge[64];
            if (is_act)
            {
                if (is_foc)
                {
                    if ((lay->ctrl_blink % 4) < 2)
                    {
                        ov_buf_bg(255, 190, 0); /* bright amber/gold */
                        ov_buf_fg(20, 20, 20);  /* dark text */
                    }
                    else
                    {
                        ov_buf_bg(230, 80, 20);   /* vibrant red-orange */
                        ov_buf_fg(255, 255, 255); /* white text */
                    }
                    snprintf(fbadge, sizeof(fbadge), " [f] %s: /%.8s/ ", curr_nm, fpat);
                }
                else
                {
                    /* Unselected panel: solid vivid amber pill */
                    ov_buf_bg(220, 130, 20);
                    ov_buf_fg(255, 255, 255);
                    snprintf(fbadge, sizeof(fbadge), " %s: /%.8s/ ", curr_nm, fpat);
                }
                ov_buf_bold();
                ov_buf_printf("%s", fbadge);
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_HEADER);
            }
            else if (fpat[0] != '\0')
            {
                ov_theme_bg(OV_BG_PANEL_ALT);
                ov_theme_fg(OV_FG_WARN);
                ov_buf_bold();
                if (is_foc)
                {
                    snprintf(fbadge, sizeof(fbadge), " [f] %s: OFF (/%.6s/) ", curr_nm, fpat);
                }
                else
                {
                    snprintf(fbadge, sizeof(fbadge), " %s: OFF ", curr_nm);
                }
                ov_buf_printf("%s", fbadge);
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_HEADER);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_DIM);
                ov_buf_bold();
                if (is_foc)
                {
                    snprintf(fbadge, sizeof(fbadge), " [/] %s: ALL ", curr_nm);
                }
                else
                {
                    snprintf(fbadge, sizeof(fbadge), " %s: ALL ", curr_nm);
                }
                ov_buf_printf("%s", fbadge);
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_HEADER);
            }

            int bw = (int) strlen(fbadge);
            if (lay->r_filter_count < 4)
            {
                lay->r_filter_start[lay->r_filter_count] = b_start;
                lay->r_filter_width[lay->r_filter_count] = bw;
                lay->r_filter_panel[lay->r_filter_count] = curr_p;
                lay->r_filter_count++;
            }
            b_start += bw + 1;
            filter_w += bw;
        }
    }
    else
    {
        /* Dedicated or graph view: show focused panel filter badge */
        const char *pname  = (fpanel == OV_FOCUS_STREAMS) ? "STRM"
                             : (fpanel == OV_FOCUS_PROCS) ? "PROC"
                             : (fpanel == OV_FOCUS_FPS)   ? "FPS"
                                                          : "FILTER";
        const char *fpat   = (fpanel != OV_FOCUS_GRAPH) ? ov_get_panel_filter_pattern(lay, fpanel)
                                                        : ov_get_filter_pattern(lay);
        int         is_act = (fpanel != OV_FOCUS_GRAPH) ? ov_is_panel_filter_active(lay, fpanel)
                                                        : ov_is_filter_active(lay);

        ov_buf_printf(" ");
        filter_w++;

        char fbadge[64];
        if (is_act)
        {
            if ((lay->ctrl_blink % 4) < 2)
            {
                ov_buf_bg(255, 190, 0);
                ov_buf_fg(20, 20, 20);
            }
            else
            {
                ov_buf_bg(230, 80, 20);
                ov_buf_fg(255, 255, 255);
            }
            ov_buf_bold();
            snprintf(fbadge, sizeof(fbadge), " [f] %s: /%.10s/ ", pname, fpat);
            ov_buf_printf("%s", fbadge);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
        }
        else if (fpat[0] != '\0')
        {
            ov_theme_bg(OV_BG_PANEL_ALT);
            ov_theme_fg(OV_FG_WARN);
            ov_buf_bold();
            snprintf(fbadge, sizeof(fbadge), " [f] %s: OFF (/%.8s/) ", pname, fpat);
            ov_buf_printf("%s", fbadge);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
        }
        else
        {
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_bold();
            snprintf(fbadge, sizeof(fbadge), " [/] %s: ALL ", pname);
            ov_buf_printf("%s", fbadge);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
        }

        int bw                 = (int) strlen(fbadge);
        lay->r_filter_start[0] = b_start;
        lay->r_filter_width[0] = bw;
        lay->r_filter_panel[0] = fpanel;
        lay->r_filter_count    = 1;
        b_start += bw + 1;
        filter_w += bw;

        /* Also show alert pills for any other panels with active filters */
        ov_focus_t  bg_panels[3] = { OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS };
        const char *bg_names[3]  = { "STRM", "PROC", "FPS" };
        for (int p = 0; p < 3; p++)
        {
            if (bg_panels[p] != fpanel && ov_is_panel_filter_active(lay, bg_panels[p]))
            {
                const char *bg_pat = ov_get_panel_filter_pattern(lay, bg_panels[p]);
                char        bg_badge[64];
                snprintf(bg_badge, sizeof(bg_badge), " %s: /%.8s/ ", bg_names[p], bg_pat);
                ov_buf_printf(" ");
                filter_w++;
                ov_buf_bold();
                ov_buf_bg(220, 130, 20);
                ov_buf_fg(255, 255, 255);
                ov_buf_printf("%s", bg_badge);
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_HEADER);

                int bg_w = (int) strlen(bg_badge);
                if (lay->r_filter_count < 4)
                {
                    lay->r_filter_start[lay->r_filter_count] = b_start;
                    lay->r_filter_width[lay->r_filter_count] = bg_w;
                    lay->r_filter_panel[lay->r_filter_count] = bg_panels[p];
                    lay->r_filter_count++;
                }
                b_start += bg_w + 1;
                filter_w += bg_w;
            }
        }
    }

    int commit_w = (int) strlen(MILK_GIT_COMMIT) + 3;
    int shmdir_w = (int) strlen(shmdir) + 8;
    int chars_left =
        17 + commit_w + shmdir_w + 1 + ctrl_w + 1 + hover_w + 1 + filter_w; /* +1 for heartbeat */

    ov_theme_fg(OV_FG_STREAM);
    chars_left += snprintf(NULL, 0, " %d stm", m->nb_streams);
    ov_buf_printf(" %d stm", m->nb_streams);

    ov_theme_fg(OV_FG_PROC);
    chars_left += snprintf(NULL, 0, " %d prc", m->nb_procs);
    ov_buf_printf(" %d prc", m->nb_procs);

    ov_theme_fg(OV_FG_FPS);
    chars_left += snprintf(NULL, 0, " %d fps", m->nb_fps);
    ov_buf_printf(" %d fps", m->nb_fps);

    ov_theme_fg(OV_FG_CONN);
    chars_left += snprintf(NULL, 0, " %d edg", m->nb_edges);
    ov_buf_printf(" %d edg", m->nb_edges);

    ov_theme_fg(OV_FG_DIM);
    {
        double cpu_pct = ov_sys_get_cpu_usage();
        chars_left += snprintf(NULL, 0, "  CPU: %4.1f%%", cpu_pct);
        ov_buf_printf("  CPU: %4.1f%%", cpu_pct);
    }

    {
        double bw_kbs = ov_sys_get_bandwidth_usage();
        chars_left += snprintf(NULL, 0, "  BW: %4.1f kB/s", bw_kbs);
        ov_buf_printf("  BW: %4.1f kB/s", bw_kbs);
    }

    int pad = r.width - chars_left;
    if (pad > 0)
    {
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_hline(' ', pad);
    }

    ov_theme_bg(OV_BG_HEADER);
}

