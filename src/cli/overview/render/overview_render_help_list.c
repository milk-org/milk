// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_help_list.c
 * @brief   Help topics list row rendering for milk-CTRL.
 */

#include "overview_render_help_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_help_render_list_row - render a single visible item or section row in help list.
 * @lay:      Pointer to layout structure.
 * @vr:       Visible row index relative to list top.
 * @scroll:   Vertical scroll offset.
 * @sel:      Currently selected row index.
 * @map:      Array of visible item indices into g_help_entries.
 * @nvis:     Total number of visible rows.
 * @list_top: Screen row of top of list area.
 * @pc:       Left column of help panel.
 * @pw:       Width of help panel.
 * @inner_w:  Inner content width for text padding.
 */
void ov_help_render_list_row(
    const OV_LAYOUT *lay,
    int              vr,
    int              scroll,
    int              sel,
    const int       *map,
    int              nvis,
    int              list_top,
    int              pc,
    int              pw,
    int              inner_w)
{
    (void) nvis;
    (void) pw;
        int                 idx    = map[vr + scroll];
        const help_entry_t *h      = &g_help_entries[idx];
        int                 row    = list_top + vr;
        int                 is_sel = ((vr + scroll) == sel);

        ov_buf_pos(row, pc + 2);
        if (is_sel)
        {
            ov_theme_bg(OV_BG_SELECTED);
        }
        else
        {
            ov_theme_bg(OV_BG_PANEL);
        }

        if (lay->help_search[0] != '\0')
        {
            /* Search match row with section badge */
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf(" ▶ ");
            }
            else
            {
                ov_buf_printf("   ");
            }

            /* Section badge */
            ov_buf_bold();
            ov_theme_fg(ov_help_section_color(h->section));
            ov_buf_printf("[%-4s] ", ov_help_section_tag(h->section));
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_theme_bg(OV_BG_SELECTED);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
            }

            /* Keystroke or category indicator */
            ov_buf_bold();
            if (h->flags & HF_SECTION)
            {
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_printf("%-13s", "Topic");
                ov_buf_printf("   ");
            }
            else if (h->flags & HF_CTRL_MODE)
            {
                if (lay->ctrl_mode)
                {
                    ov_buf_fg(255, 95, 75);
                    ov_buf_printf("%-13s", h->key ? h->key : "");
                    ov_buf_fg(255, 80, 80);
                    ov_buf_printf(" ⚡ ");
                }
                else
                {
                    ov_buf_fg(200, 140, 50);
                    ov_buf_printf("%-13s", h->key ? h->key : "");
                    ov_buf_fg(160, 115, 45);
                    ov_buf_printf(" 🔒 ");
                }
            }
            else
            {
                ov_buf_fg(130, 205, 255);
                ov_buf_printf("%-13s", h->key ? h->key : "");
                ov_buf_printf("   ");
            }

            /* Summary label */
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_theme_bg(OV_BG_SELECTED);
                ov_theme_fg(OV_FG_BRIGHT);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_printf("%s", h->label);

            int used = 3 + 7 + 13 + 3 + (int) strlen(h->label);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }
        else if (h->flags & HF_SECTION)
        {
            int         expanded = help_is_expanded(lay, h->section);
            const char *chev     = expanded ? "▾" : "▸";

            if (is_sel)
            {
                ov_theme_bg(OV_BG_SELECTED);
                ov_buf_bold();
                ov_theme_fg(OV_FG_BRIGHT);
                ov_buf_printf("▶ %s %s", chev, h->label);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL_ALT);
                ov_buf_bold();
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_printf("  %s %s", chev, h->label);
            }

            /* Count child entries in section */
            int nchildren = 0;
            for (int k = 0; k < g_help_total; k++)
            {
                if (g_help_entries[k].section == h->section &&
                    !(g_help_entries[k].flags & HF_SECTION))
                {
                    nchildren++;
                }
            }

            char tag[32];
            snprintf(tag, sizeof(tag), "(%d keys)", nchildren);
            int used = 4 + (int) strlen(h->label) + 1 + (int) strlen(tag);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" %s", tag);
        }
        else if (h->flags & HF_COLORS)
        {
            ov_theme_fg(OV_FG_DIM);
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf("    ▶ ");
            }
            else
            {
                ov_buf_printf("      ");
            }

            ov_theme_fg(OV_FG_STREAM);
            ov_buf_printf("● stream ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_printf("● proc ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_printf("● fps ");
            ov_theme_fg(OV_FG_CONN);
            ov_buf_printf("● conn ");
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf("● active ");
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("● warn ");
            ov_theme_fg(OV_FG_ERROR);
            ov_buf_printf("● error");

            int used = 6 + 53;
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }
        else
        {
            /* Standard keystroke entry with tab offset */
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf("    ▶ ");
            }
            else
            {
                ov_buf_printf("      ");
            }

            /* Keystroke column in standard bold font */
            ov_buf_bold();
            if (h->flags & HF_CTRL_MODE)
            {
                if (lay->ctrl_mode)
                {
                    ov_buf_fg(255, 95, 75);
                    ov_buf_printf("%-13s", h->key);
                    ov_buf_fg(255, 80, 80);
                    ov_buf_printf(" ⚡ ");
                }
                else
                {
                    ov_buf_fg(200, 140, 50);
                    ov_buf_printf("%-13s", h->key);
                    ov_buf_fg(160, 115, 45);
                    ov_buf_printf(" 🔒 ");
                }
            }
            else
            {
                ov_buf_fg(130, 205, 255);
                ov_buf_printf("%-13s", h->key);
                ov_buf_printf("    ");
            }

            /* Summary label */
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_theme_bg(OV_BG_SELECTED);
                ov_theme_fg(OV_FG_BRIGHT);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_printf("%s", h->label);

            int used = 6 + 13 + 4 + (int) strlen(h->label);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }

        ov_buf_reset_attr();
}
