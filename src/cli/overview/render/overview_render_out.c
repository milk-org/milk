// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_out.c
 * @brief   OUTPUT sub-tab: the run state of the DETAILS FPS and the end of its unit's run.out.
 */

#include "overview_render_internal.h"
#include "fps_unit_state.h"

#define OUT_BYTES 65536 /* most of run.out read per frame */

/** out_clean - one line of unit output for the screen: no terminal sequences or control characters. */
static int out_clean(char *dst, size_t size, const char *l, size_t n)
{
    size_t o   = 0;
    int    esc = 0;

    for (size_t i = 0; i < n && o + 1 < size; i++)
    {
        unsigned char c = (unsigned char) l[i];
        if (esc == 1)
        {
            esc = (c == '[') ? 2 : 0;
        }
        else if (esc == 2)
        {
            esc = (c >= 0x40 && c <= 0x7e) ? 0 : 2;
        }
        else if (c == 0x1b)
        {
            esc = 1;
        }
        else if (c == '\t')
        {
            dst[o++] = ' ';
        }
        else if (c >= 0x20 && c != 0x7f)
        {
            dst[o++] = (char) c;
        }
    }
    dst[o] = '\0';
    return (int) o;
}

/** out_row - @text on panel row @row, cut to the panel. */
static void out_row(OV_RECT r, int row, ov_rgb_t fg, const char *text)
{
    int cols  = 0;
    int bytes = ov_utf8_fit(text, (int) strlen(text), r.width - 2, &cols);

    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_theme_fg(fg);
    ov_buf_printf("%.*s", bytes, text);
    render_pad_spaces(cols, r.width);
}

/**
 * ov_render_out_panel - OUTPUT of the DETAILS entry when it is an FPS: its run state, then the last
 * lines its unit wrote (milk-fps-log NAME shows more).
 * Return: 1 (drawn).
 */
int ov_render_out_panel(OV_LAYOUT *lay, const OV_MODEL *m)
{
    static char buf[OUT_BYTES];
    char        line[1024];
    OV_RECT     r        = lay->r_graph;
    int         max_rows = r.height - 2;
    int         row      = r.row + 1;
    ov_focus_t  panel    = OV_FOCUS_STREAMS;
    int         idx      = ov_detail_target(lay, m, &panel);

    ov_subtab_bar(lay, OV_FG_FPS);
    if (max_rows < 1)
    {
        return 1;
    }
    if (idx < 0 || panel != OV_FOCUS_FPS)
    {
        render_empty_rows(r, row, max_rows, OV_FG_DIM,
                          " Select an FPS: OUTPUT shows what its unit's run process wrote");
        return 1;
    }

    const OV_FPS  *f     = &m->fps[idx];
    FPS_UNIT_STATE state = (FPS_UNIT_STATE) f->run_state;
    snprintf(line, sizeof(line), " %s run: %s%s%s%s", f->name, fps_unit_state_name(state),
             (!fps_unit_state_alive(state) && f->run_last[0] != '\0') ? " (" : "",
             fps_unit_state_alive(state) ? "" : f->run_last,
             (!fps_unit_state_alive(state) && f->run_last[0] != '\0') ? ")" : "");
    out_row(r, row, OV_FG_TITLE, line);

    int    lines = max_rows - 1;
    size_t want  = (size_t) lines * 512;
    int    n     = (lines > 0) ? fps_unit_out_tail(f->name, FPS_UNIT_RUN, lines, buf,
                                                   (want < sizeof(buf)) ? want : sizeof(buf))
                               : 0;
    int    ri    = 1;
    if (n < 0)
    {
        out_row(r, row + ri++, OV_FG_DIM, " (no output: the unit has not run)");
    }
    for (char *l = buf; n > 0 && l < buf + n && ri < max_rows;)
    {
        char  *nl  = memchr(l, '\n', (size_t) (buf + n - l));
        size_t len = (nl != NULL) ? (size_t) (nl - l) : (size_t) (buf + n - l);
        out_clean(line + 1, sizeof(line) - 1, l, len);
        line[0] = ' ';
        out_row(r, row + ri++, (strncmp(l, "=== ", 4) == 0) ? OV_FG_DIM : OV_FG_TEXT, line);
        l += len + 1;
    }
    for (; ri < max_rows; ri++)
    {
        clear_row(row + ri, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    return 1;
}
