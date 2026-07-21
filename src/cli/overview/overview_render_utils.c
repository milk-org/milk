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

static const char *get_stream_col_desc(int vis_col, int compact)
{
    if (compact)
    {
        switch (vis_col)
        {
        case 0:
            return "A: Ancestry lineage/activity indicators relative to selected stream";
        case 1:
            return "NAME: Shared memory stream name";
        case 2:
            return "TYP: Pixel data type (e.g. FLT=Float, U16=Unsigned 16-bit)";
        case 3:
            return "SIZE: Stream dimensions and axis sizes (width x height)";
        case 4:
            return "Hz: Frame update frequency (iterations/second)";
        case 5:
            return "MB/s: Instantaneous data throughput bandwidth";
        case 6:
            return "OWNER: Process ID (PID) of the stream creator";
        case 7:
            return "WPID: Process ID (PID) of the active writer";
        case 8:
            return "RPID: Process ID(s) of connected readers";
        default:
            return "";
        }
    }
    else
    {
        switch (vis_col)
        {
        case 0:
            return "A: Ancestry lineage/activity indicators relative to selected stream";
        case 1:
            return "NAME: Shared memory stream name";
        case 2:
            return "TYP: Pixel data type (e.g. FLT=Float, U16=Unsigned 16-bit)";
        case 3:
            return "SIZE: Stream dimensions and axis sizes (width x height)";
        case 4:
            return "Hz: Frame update frequency (iterations/second)";
        case 5:
            return "MB/s: Instantaneous data throughput bandwidth";
        case 6:
            return "INODE: Shared memory temporary file system inode number";
        case 7:
            return "OWNER: Process ID (PID) of the stream creator";
        case 8:
            return "COUNT: Cumulative frame write counter";
        case 9:
            return "SEMS: Semaphore state indicators for client synchronization";
        case 10:
            return "WPID: Process ID (PID) of the active writer";
        case 11:
            return "RPID: Process ID(s) of connected readers";
        default:
            return "";
        }
    }
}

static const char *get_proc_col_desc(int vis_col, int compact)
{
    if (compact)
    {
        switch (vis_col)
        {
        case 0:
            return "A: Ancestry relation indicators relative to selected process";
        case 1:
            return "NAME: Executable or command line keyword name";
        case 2:
            return "PID: Linux Process Identifier";
        case 3:
            return "PRIO: Real-time scheduler priority level";
        case 4:
            return "STAT: Process execution status (RUN, PAUS, STOP, CRSH)";
        case 5:
            return "Hz: Process loop cycle execution frequency";
        case 6:
            return "UPTIME: Uptime duration since process start";
        case 7:
            return "CPU%: CPU core consumption percentage";
        case 8:
            return "LOOPCNT: Cumulative iteration step count";
        case 9:
            return "MEM: Resident Set Size (RSS) memory consumption";
        case 10:
            return "MSG: Last status message logged by the process";
        default:
            return "";
        }
    }
    else
    {
        switch (vis_col)
        {
        case 0:
            return "A: Ancestry relation indicators relative to selected process";
        case 1:
            return "NAME: Executable or command line keyword name";
        case 2:
            return "PID: Linux Process Identifier";
        case 3:
            return "PRIO: Real-time scheduler priority level";
        case 4:
            return "STAT: Process execution status (RUN, PAUS, STOP, CRSH)";
        case 5:
            return "Hz: Process loop cycle execution frequency";
        case 6:
            return "UPTIME: Uptime duration since process start";
        case 7:
            return "TRG: Loop execution trigger mode (SEMAPHORE, TIME, FREE)";
        case 8:
            return "trig-strm: Name of trigger stream that fires execution";
        case 9:
            return "exec: Median loop step execution time in milliseconds";
        case 10:
            return "DUTY: Step execution duty cycle percentage";
        case 11:
            return "CPU%: CPU core consumption percentage";
        case 12:
            return "LOOPCNT: Cumulative iteration step count";
        case 13:
            return "MEM: Resident Set Size (RSS) memory consumption";
        case 14:
            return "MISSED: Cumulative loop trigger frames missed";
        case 15:
            return "MSG: Last status message logged by the process";
        default:
            return "";
        }
    }
}

static const char *get_fps_col_desc(int vis_col, int compact)
{
    if (compact)
    {
        switch (vis_col)
        {
        case 0:
            return "A: Ancestry relation marker relative to selected FPS";
        case 1:
            return "NAME: Registered Function Processing System (FPS) name";
        case 2:
            return "TMX: Tmux session container window name/index";
        case 3:
            return "CPID: Process ID (PID) of config/control loop";
        case 4:
            return "RPID: Process ID (PID) of main compute run loop";
        case 5:
            return "STR: Bound primary input stream name";
        case 6:
            return "MEM: Parameter registry shared memory footprint size";
        default:
            return "";
        }
    }
    else
    {
        switch (vis_col)
        {
        case 0:
            return "A: Ancestry relation marker relative to selected FPS";
        case 1:
            return "NAME: Registered Function Processing System (FPS) name";
        case 2:
            return "TMX: Tmux session container window name/index";
        case 3:
            return "CPID: Process ID (PID) of config/control loop";
        case 4:
            return "RPID: Process ID (PID) of main compute run loop";
        case 5:
            return "STR: Bound primary input stream name";
        case 6:
            return "MEM: Parameter registry shared memory footprint size";
        case 7:
            return "DESCRIPTION: Purpose and summary of the FPS module";
        default:
            return "";
        }
    }
}

void ov_render_highlighted_column_description(const OV_LAYOUT *lay)
{
    int W   = lay->term_cols;
    int row = lay->r_streams.row - 1;
    if (row < 2)
    {
        return;
    }

    ov_buf_pos(row, 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_hline(' ', W);

    const char *desc        = "";
    ov_rgb_t    badge_color = OV_FG_TEXT;
    const char *badge_text  = "INFO";

    if (lay->focus == OV_FOCUS_STREAMS)
    {
        desc        = get_stream_col_desc(lay->highlight_col_stream, lay->compact_mode);
        badge_color = OV_FG_STREAM;
        badge_text  = "STREAM";
    }
    else if (lay->focus == OV_FOCUS_PROCS)
    {
        desc        = get_proc_col_desc(lay->highlight_col_proc, lay->compact_mode);
        badge_color = OV_FG_PROC;
        badge_text  = "PROC";
    }
    else if (lay->focus == OV_FOCUS_FPS)
    {
        desc        = get_fps_col_desc(lay->highlight_col_fps, lay->compact_mode);
        badge_color = OV_FG_FPS;
        badge_text  = "FPS";
    }
    else if (lay->focus == OV_FOCUS_GRAPH)
    {
        desc        = "Graph Panel: Visualizes connection topology and data dependencies.";
        badge_color = OV_FG_CONN;
        badge_text  = "GRAPH";
    }

    if (desc && desc[0] != '\0')
    {
        /* Render Badge */
        ov_buf_pos(row, 2);
        ov_theme_bg(badge_color);
        ov_buf_fg(0, 0, 0);
        ov_buf_bold();
        ov_buf_printf(" %s ", badge_text);
        ov_buf_reset_attr();

        /* Render Description */
        ov_buf_pos(row, 2 + strlen(badge_text) + 3);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_TEXT);
        ov_buf_printf("%s", desc);
        ov_buf_reset_attr();
    }
}
