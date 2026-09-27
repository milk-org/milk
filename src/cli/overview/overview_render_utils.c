// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_render_internal.h"
/**
 * @brief Render a name with search-match highlighting.
 *
 * Colors matching substring in the display.
 */
void render_highlighted_name(const char *name,
                             int         max_len,
                             regex_t    *re,
                             int         has_re,
                             ov_rgb_t    normal_fg,
                             ov_rgb_t    row_bg)
{
    int len = (int) strlen(name);
    if (len > max_len)
    {
        len = max_len;
    }

    regmatch_t pm[1];
    if (has_re && regexec(re, name, 1, pm, 0) == 0)
    {
        int b_len = pm[0].rm_so;
        if (b_len > max_len)
        {
            b_len = max_len;
        }

        int m_len = pm[0].rm_eo - pm[0].rm_so;
        if (b_len + m_len > max_len)
        {
            m_len = max_len - b_len;
        }

        int a_len = len - (b_len + m_len);

        if (b_len > 0)
        {
            ov_buf_printf("%.*s", b_len, name);
        }
        if (m_len > 0)
        {
            ov_buf_bold();
            ov_buf_fg(255, 255, 255);
            ov_buf_printf("%.*s", m_len, name + b_len);
            ov_buf_reset_attr();
            ov_theme_bg(row_bg);
            ov_theme_fg(normal_fg);
        }
        if (a_len > 0)
        {
            ov_buf_printf("%.*s", a_len, name + b_len + m_len);
        }
    }
    else
    {
        ov_buf_printf("%.*s", len, name);
    }

    int pad = max_len - len;
    if (pad > 0)
    {
        ov_buf_hline(' ', pad);
    }
}


/**
 * @brief Render a data type badge with color coding.
 */
const char *render_dtype(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "UI8";
    case _DATATYPE_INT8:
        return "SI8";
    case _DATATYPE_UINT16:
        return "U16";
    case _DATATYPE_INT16:
        return "S16";
    case _DATATYPE_UINT32:
        return "U32";
    case _DATATYPE_INT32:
        return "S32";
    case _DATATYPE_UINT64:
        return "U64";
    case _DATATYPE_INT64:
        return "S64";
    case _DATATYPE_FLOAT:
        return "F32";
    case _DATATYPE_DOUBLE:
        return "F64";
    default:
        return "???";
    }
}

/**
 * dtype_bytesize - bytes per element for a datatype.
 */
int dtype_bytesize(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
    case _DATATYPE_INT8:
        return 1;
    case _DATATYPE_UINT16:
    case _DATATYPE_INT16:
        return 2;
    case _DATATYPE_UINT32:
    case _DATATYPE_INT32:
    case _DATATYPE_FLOAT:
        return 4;
    case _DATATYPE_UINT64:
    case _DATATYPE_INT64:
    case _DATATYPE_DOUBLE:
        return 8;
    default:
        return 1;
    }
}

/**
 * @brief Clear a screen row to blank.
 */
void clear_row(int row, int col, int width, ov_rgb_t bg)
{
    ov_buf_reset_attr();
    ov_buf_pos(row, col);
    ov_theme_bg(bg);
    ov_buf_hline(' ', width);
    ov_buf_reset_attr();
}

/* Pad the remainder of a panel's interior row */
void render_pad_spaces(int chars_written, int panel_width)
{
    int remain = (panel_width - 2) - chars_written;
    if (remain > 0)
    {
        ov_buf_hline(' ', remain);
    }
}

/* Pad remainder of row up to target absolute screen column */
void render_pad_to_col(int end_col)
{
    if (ov__cursor_col < end_col)
    {
        ov_buf_hline(' ', end_col - ov__cursor_col);
    }
}

/**
 * render_scroll_indicators - draw ▲N / ▼N on panel borders.
 * @r:         panel rect
 * @scroll:    current scroll offset (first visible index)
 * @max_rows:  visible rows in the panel body
 * @total:     total item count
 * @accent:    panel accent color for the arrow
 *
 * Draws indicators on the top and bottom border lines of
 * the panel to show how many items are hidden above / below.
 */
void render_scroll_indicators(OV_RECT r, int scroll, int max_rows, int total, ov_rgb_t accent)
{
    int above = scroll;
    int below = total - scroll - max_rows;
    if (below < 0)
    {
        below = 0;
    }

    /* Top border: "▲ N more" right-aligned inside border */
    if (above > 0)
    {
        char buf[32];
        int  n = snprintf(buf, sizeof(buf), " ▲%d ", above);
        /* display width: space + ▲(1col) + digits + space */
        int dw = 3;
        {
            int tmp = above;
            while (tmp > 0)
            {
                dw++;
                tmp /= 10;
            }
        }
        int col = r.col + r.width - dw - 2;
        if (col > r.col + 2)
        {
            ov_buf_pos(r.row, col);
            ov_theme_fg(accent);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", buf);
            (void) n;
        }
    }

    /* Bottom border: "▼ N more" right-aligned inside border */
    if (below > 0)
    {
        char buf[32];
        int  n  = snprintf(buf, sizeof(buf), " ▼%d ", below);
        int  dw = 3;
        {
            int tmp = below;
            while (tmp > 0)
            {
                dw++;
                tmp /= 10;
            }
        }
        int brow = r.row + r.height - 1;
        int col  = r.col + r.width - dw - 2;
        if (col > r.col + 2)
        {
            ov_buf_pos(brow, col);
            ov_theme_fg(accent);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", buf);
            (void) n;
        }
    }
}

/**
 * @brief Render a single table cell with highlighting and collapse support.
 */
void ov_render_cell(int         logical_col,
                    int         vis_col,
                    ov_rgb_t    fg,
                    ov_rgb_t    bg,
                    const char *str,
                    int        *hs_rem,
                    int        *printed,
                    int         avail,
                    int         highlighted_col,
                    uint32_t    collapsed_mask)
{
    int is_high = (vis_col == highlighted_col);
    int is_coll = (collapsed_mask & (1U << logical_col)) != 0;

    /* Determine background color */
    ov_rgb_t cell_bg = bg;
    if (is_high)
    {
        cell_bg = ov_theme_highlight_bg(bg);
    }
    ov_theme_bg(cell_bg);

    /* Format the string depending on collapsed state */
    char cell_str[256];
    if (is_coll)
    {
        /* Collapsed to 1 character. Show double-chevron (») indicating it is hidden. */
        strcpy(cell_str, "\xc2\xbb");
    }
    else
    {
        strncpy(cell_str, str, sizeof(cell_str) - 1);
        cell_str[sizeof(cell_str) - 1] = '\0';
    }

    int skip = 0;
    if (hs_rem && *hs_rem > 0)
    {
        skip = *hs_rem;
    }

    /* Print character by character, handling \x01 and \x02 markup */
    int vis_col_ctr = 0;
    int i           = 0;
    ov_theme_fg(fg);

    while (cell_str[i] != '\0' && (!printed || *printed < avail))
    {
        if (cell_str[i] == '\x01')
        {
            if (vis_col_ctr >= skip)
            {
                ov_buf_bold();
                ov_buf_underline();
                ov_theme_fg(OV_FG_BRIGHT);
            }
            i++;
        }
        else if (cell_str[i] == '\x02')
        {
            if (vis_col_ctr >= skip)
            {
                ov_buf_reset_attr();
                ov_theme_bg(cell_bg);
                ov_theme_fg(fg);
            }
            i++;
        }
        else
        {
            int clen = utf8_char_length((unsigned char) cell_str[i]);
            if (vis_col_ctr >= skip)
            {
                ov_buf_append_char(&cell_str[i], clen);
                if (printed)
                {
                    (*printed)++;
                }
            }
            vis_col_ctr++;
            i += clen;
        }
    }

    if (hs_rem && *hs_rem > 0)
    {
        *hs_rem = (vis_col_ctr < *hs_rem) ? 0 : (*hs_rem - vis_col_ctr);
    }
}
