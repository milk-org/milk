// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_render_internal.h"
/**
 * render_highlighted_name - Render a name with regex search-match highlighting
 * @name:      String to render
 * @max_len:   Maximum characters to display
 * @re:        Pointer to compiled regex or NULL
 * @has_re:    Flag indicating whether regex is compiled
 * @normal_fg: Default foreground color
 * @row_bg:    Current row background color
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
            ov_theme_fg(OV_FG_BRIGHT);
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
 * render_dtype - Get 3-character mnemonic string for datatype
 * @dt: Datatype code
 *
 * Return: Short string representation (e.g. "F32", "U16").
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
 * dtype_bytesize - Get number of bytes per element for a datatype
 * @dt: Datatype code
 *
 * Return: Size in bytes (1, 2, 4, or 8).
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
 * clear_row - Clear a screen row range with background color
 * @row:   Terminal row coordinate
 * @col:   Starting column coordinate
 * @width: Number of columns to clear
 * @bg:    Background color
 */
void clear_row(int row, int col, int width, ov_rgb_t bg)
{
    ov_buf_reset_attr();
    ov_buf_pos(row, col);
    ov_theme_bg(bg);
    ov_buf_hline(' ', width);
    ov_buf_reset_attr();
}

/** render_empty_rows - message @msg on the first of @max_rows panel rows from @row, the others cleared. */
void render_empty_rows(OV_RECT r, int row, int max_rows, ov_rgb_t fg, const char *msg)
{
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_theme_fg(fg);
    int cols  = 0;
    int bytes = ov_utf8_fit(msg, (int) strlen(msg), r.width - 2, &cols);
    ov_buf_printf("%.*s", bytes, msg);
    render_pad_spaces(cols, r.width);
    for (int i = 1; i < max_rows; i++)
    {
        clear_row(row + i, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
}

/**
 * render_pad_spaces - Pad the remainder of a panel interior row with spaces
 * @chars_written: Number of characters already written
 * @panel_width:   Total panel width
 */
void render_pad_spaces(int chars_written, int panel_width)
{
    int remain = (panel_width - 2) - chars_written;
    if (remain > 0)
    {
        ov_buf_hline(' ', remain);
    }
}

/**
 * render_pad_to_col - Pad current terminal row with spaces up to end column
 * @end_col: Target 1-based column position
 */
void render_pad_to_col(int end_col)
{
    if (ov__cursor_col < end_col)
    {
        ov_buf_hline(' ', end_col - ov__cursor_col);
    }
}

/**
 * render_scroll_indicators - Draw scroll indicators on panel border
 * @r:        Panel bounding rectangle
 * @scroll:   Current scroll offset (first visible index)
 * @max_rows: Visible rows in panel body
 * @total:    Total item count
 * @accent:   Accent color for the arrows
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
 * ov_render_cell - draw table cell @str of table column @col (\x01..\x02 marks highlighted text) on @bg, lit
 * when @col is @highlighted_col and shown as » when collapsed; stops at @avail columns of *@printed.
 */
void ov_render_cell(int         col,
                    ov_rgb_t    fg,
                    ov_rgb_t    bg,
                    const char *str,
                    int        *printed,
                    int         avail,
                    int         highlighted_col,
                    uint32_t    collapsed_mask)
{
    ov_rgb_t cell_bg = (col == highlighted_col) ? ov_theme_highlight_bg(bg) : bg;
    if (collapsed_mask & (1U << col))
    {
        str = "\xc2\xbb";
    }
    ov_theme_bg(cell_bg);
    ov_theme_fg(fg);
    for (int i = 0; str[i] != '\0' && *printed < avail;)
    {
        if (str[i] == '\x01')
        {
            ov_buf_bold();
            ov_buf_underline();
            ov_theme_fg(OV_FG_BRIGHT);
            i++;
        }
        else if (str[i] == '\x02')
        {
            ov_buf_reset_attr();
            ov_theme_bg(cell_bg);
            ov_theme_fg(fg);
            i++;
        }
        else
        {
            int clen = utf8_char_length((unsigned char) str[i]);
            ov_buf_append_char(&str[i], clen);
            (*printed)++;
            i += clen;
        }
    }
}

/** ov_state_info - names and theme colour of state @st. */
OV_STATE_INFO ov_state_info(ov_state_t st)
{
    ov_rgb_t fg;
    switch (st)
    {
    case OV_ST_RUN:
        fg = OV_FG_ACTIVE;
        break;
    case OV_ST_PAUS:
    case OV_ST_SPIN:
    case OV_ST_STAL:
        fg = OV_FG_WARN;
        break;
    case OV_ST_STOP:
        fg = OV_FG_ZOMBIE;
        break;
    case OV_ST_ERR:
    case OV_ST_CRSH:
        fg = OV_FG_ERROR;
        break;
    default:
        fg = OV_FG_DIM;
        break;
    }
    return (OV_STATE_INFO) { ov_state_name(st, 0), ov_state_name(st, 1), fg };
}

/** ov_proc_state - state of process @p: its loop state, or STAL when an active loop's counter stopped. */
ov_state_t ov_proc_state(const OV_PROC *p)
{
    if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE && p->stale_count >= OV_STALE_SCANS)
    {
        return OV_ST_STAL;
    }
    return ov_state_of_loopstat(p->loopstat);
}

/** ov_loop_state - state of loop @lp, worst first: ERR > STAL > PAUS > RUN > IDLE. */
ov_state_t ov_loop_state(const OV_LOOP *lp)
{
    return lp->is_error     ? OV_ST_ERR
           : lp->is_stale   ? OV_ST_STAL
           : lp->is_paused  ? OV_ST_PAUS
           : lp->is_running ? OV_ST_RUN
                            : OV_ST_IDLE;
}

/** ov_proc_status - short state label of process @p, and its colour in @fg when not NULL. */
const char *ov_proc_status(const OV_PROC *p, ov_rgb_t *fg)
{
    OV_STATE_INFO si = ov_state_info(ov_proc_state(p));
    if (fg != NULL)
    {
        *fg = si.fg;
    }
    return si.sname;
}

/** ov_loop_status - short state label of loop @lp, and its colour in @fg when not NULL. */
const char *ov_loop_status(const OV_LOOP *lp, ov_rgb_t *fg)
{
    OV_STATE_INFO si = ov_state_info(ov_loop_state(lp));
    if (fg != NULL)
    {
        *fg = si.fg;
    }
    return si.sname;
}
