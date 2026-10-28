// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_status.c
 */

#include "overview_render_internal.h"
#include "../input/overview_input_keys.h"
#include <stdarg.h>

/** ov_status_btn - record a @w-column button starting at the cursor; width 0 when it overflows @cols. */
static void ov_status_btn(OV_RECT *b, int row, int w, int cols)
{
    b->row    = row;
    b->col    = ov__cursor_col;
    b->height = 1;
    b->width  = (ov__cursor_col + w - 1 <= cols) ? w : 0;
}

/* One left-side segment of the status line: text in fg/bg, then an optional tail in tail_fg */
#define SSEG_MAX 24
typedef struct
{
    int      prio;
    int      bold;
    ov_rgb_t fg, bg, tail_fg;
    char     text[96];
    char     tail[24];
} SSEG;

/* Append a segment (printf-style text); empty texts are skipped */
static SSEG *
sseg(SSEG *sg, int *n, int prio, ov_rgb_t fg, ov_rgb_t bg, int bold, const char *fmt, ...)
    __attribute__((format(printf, 7, 8)));

/**
 * sseg - Format and append a prioritized segment to the status-bar segment array
 * @sg:   Destination array of `SSEG` status segments (capacity `SSEG_MAX`)
 * @n:    In/out count of populated segments in @sg
 * @prio: Layout priority (lower numbers are higher priority when width is tight)
 * @fg:   Foreground RGB color for the segment body text
 * @bg:   Background RGB color for the segment
 * @bold: Non-zero to render the segment with bold attribute
 * @fmt:  printf-style format string for the segment text
 *
 * Formats the segment text into `sg[*n]` and increments `*n` only when the
 * resulting string is non-empty. Returns a dummy sink segment if @sg is full.
 *
 * Return: Pointer to the initialized `SSEG` entry (so callers can populate `.tail`).
 */
static SSEG *
sseg(SSEG *sg, int *n, int prio, ov_rgb_t fg, ov_rgb_t bg, int bold, const char *fmt, ...)
{
    static SSEG sink;
    if (*n >= SSEG_MAX)
    {
        return &sink;
    }
    SSEG   *s = &sg[*n];
    va_list ap;
    *s = (SSEG) { .prio = prio, .bold = bold, .fg = fg, .bg = bg };
    va_start(ap, fmt);
    vsnprintf(s->text, sizeof(s->text), fmt, ap);
    va_end(ap);
    if (s->text[0] != '\0')
    {
        (*n)++;
    }
    return s;
}

/**
 * sseg_draw - Emit a single status-bar segment and its optional colored tail to the frame buffer
 * @s: Pointer to the `SSEG` segment to render
 *
 * Applies the segment's background, foreground, and bold attributes, prints `.text`
 * and optional `.tail` (using `.tail_fg`), and resets terminal attributes.
 *
 * Return: void
 */
static void sseg_draw(const SSEG *s)
{
    ov_theme_bg(s->bg);
    ov_theme_fg(s->fg);
    if (s->bold)
    {
        ov_buf_bold();
    }
    ov_buf_printf("%s", s->text);
    if (s->tail[0] != '\0')
    {
        ov_theme_fg(s->tail_fg);
        ov_buf_printf("%s", s->tail);
    }
    ov_buf_reset_attr();
}

/* Breadcrumb "  VIEW › Panel", the panel in its colour */
static void status_crumb(SSEG *sg, int *n, const OV_LAYOUT *lay)
{
    const char *panel_name = "Graph";
    ov_rgb_t    panel_fg   = OV_FG_DIM;
    if (lay->view == OV_VIEW_LOOPS || (lay->view == OV_VIEW_DASHBOARD &&
                                       lay->subtab == OV_SUB_LOOPS && lay->focus == OV_FOCUS_GRAPH))
    {
        panel_name = "Loops";
        panel_fg   = OV_FG_LOOP;
    }
    else if (lay->focus == OV_FOCUS_STREAMS)
    {
        panel_name = "Streams";
        panel_fg   = OV_FG_STREAM;
    }
    else if (lay->focus == OV_FOCUS_FPS)
    {
        panel_name = "FPS";
        panel_fg   = OV_FG_FPS;
    }
    else if (lay->focus == OV_FOCUS_PROCS)
    {
        panel_name = "Procs";
        panel_fg   = OV_FG_PROC;
    }
    SSEG *s =
        sseg(sg, n, 3, OV_FG_DIM, OV_BG_HEADER, 0, "  %s \xe2\x80\xba ", ov_view_label(lay->view));
    snprintf(s->tail, sizeof(s->tail), "%s", panel_name);
    s->tail_fg = panel_fg;
}

/* Filter badges: the focused panel's filter (on: blinking, off: dim), then other panels' active filters */
static void status_filters(SSEG *sg, int *n, OV_LAYOUT *lay)
{
    ov_focus_t  fpanel = ov_get_effective_filter_panel(lay);
    const char *pname  = (fpanel == OV_FOCUS_STREAMS) ? "STRM "
                         : (fpanel == OV_FOCUS_PROCS) ? "PROC "
                         : (fpanel == OV_FOCUS_FPS)   ? "FPS "
                                                      : "";
    int         graph  = (fpanel == OV_FOCUS_GRAPH);
    const char *fpat =
        !graph ? ov_get_panel_filter_pattern(lay, fpanel) : ov_get_filter_pattern(lay);
    int is_act = !graph ? ov_is_panel_filter_active(lay, fpanel) : ov_is_filter_active(lay);

    if (is_act)
    {
        ov_rgb_t bg = ((lay->ctrl_blink % 4) < 2) ? OV_BADGE_WARN : OV_BADGE_DANGER;
        sseg(sg, n, 1, ov_rgb_on(bg), bg, 1, " [%sFILTER ON: /%.*s/ ('f' toggle, ESC clear)] ",
             pname, graph ? 12 : 10, fpat);
    }
    else if (fpat[0] != '\0')
    {
        sseg(sg, n, 2, OV_FG_WARN, OV_BG_PANEL, 1, " [%sFilter OFF: /%.*s/ ('f' enable)] ", pname,
             graph ? 12 : 10, fpat);
    }

    static const ov_focus_t bg_panels[3] = { OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS };
    static const char      *bg_names[3]  = { "STRM", "PROC", "FPS" };
    for (int p = 0; p < 3; p++)
    {
        if (bg_panels[p] != fpanel && ov_is_panel_filter_active(lay, bg_panels[p]))
        {
            sseg(sg, n, 2, ov_rgb_on(OV_BADGE_WARN), OV_BADGE_WARN, 1, " [%s: /%.8s/] ",
                 bg_names[p], ov_get_panel_filter_pattern(lay, bg_panels[p]));
        }
    }
}

/**
 * ov_render_status - Render the bottom status bar and key hints
 * @lay: Layout configuration
 * @m:   Data model snapshot
 */
void ov_render_status(OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r = lay->r_status;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);

    double interval = (double) ov_scan_get_interval();
    double rate_hz  = (interval > 0.0) ? (1.0 / interval) : 0.0;

    /* Sort key label */
    const char *sort_label = "";
    char        sort_buf[48];
    if (lay->focus < OV_NB_LIST_PANELS)
    {
        snprintf(sort_buf, sizeof(sort_buf), " [sort:%s]",
                 ov_table_status_label(ov_table_for(lay->focus), lay->panel[lay->focus].sort_key));
        sort_label = sort_buf;
    }

    /* The sub-tab shown in DASH and F6 */
    char detail_label[16] = "";
    if (lay->view == OV_VIEW_DASHBOARD || lay->view == OV_VIEW_GRAPH)
    {
        snprintf(detail_label, sizeof(detail_label), " [%s]", ov_subtab_status(lay->subtab));
    }
    else if (lay->view == OV_VIEW_STREAMS && lay->r_strm_insp.width > 0)
    {
        snprintf(detail_label, sizeof(detail_label), " [%s]", ov_insp_tab_name(lay->strm_insp_tab));
    }

    /* Filter editing prompt overrides normal status */
    if (lay->filter_editing)
    {
        const char *fstr  = lay->filter;
        const char *pname = (lay->filter_panel == OV_FOCUS_STREAMS) ? "STREAMS"
                            : (lay->filter_panel == OV_FOCUS_PROCS) ? "PROCESSINFO"
                            : (lay->filter_panel == OV_FOCUS_FPS)   ? "FPS"
                                                                    : "FILTER";
        ov_theme_fg(OV_FG_TEXT);
        char pfx = lay->filter_jump ? '?' : '/';
        int  np  = snprintf(NULL, 0, " [%s] %c%s", pname, pfx, fstr);
        ov_buf_printf(" [%s] %c%s", pname, pfx, fstr);

        /* Blinking cursor */
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("█");
        np += 1;

        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("  (ENTER=accept ESC=cancel)");
        np += 26;

        int pad = r.width - np - 1;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }
        ov_buf_reset_attr();
        return;
    }

    /* Left side: segments by importance, cut to what leaves room for the theme/exit buttons and the clock */
    SSEG sg[SSEG_MAX];
    int  n = 0;
    if (lay->paused)
    {
        sseg(sg, &n, 0, OV_FG_ERROR, OV_BG_HEADER, 0, " [PAUSED] ");
    }
    else
    {
        /* the rate scans ran at (configured until two scans); hovering lists the stages */
        sseg(sg, &n, 0, OV_FG_DIM, OV_BG_HEADER, 0, " scan:%.0fms %.1fHz", m->scan_time_ms,
             (m->scan_rate_hz > 0.0) ? m->scan_rate_hz : rate_hz);
    }
    status_crumb(sg, &n, lay);
    status_filters(sg, &n, lay);

    const char *exit_label = " [x] exit ";
    if (lay->help.show)
    {
        exit_label = (lay->help.search[0] != '\0' || lay->help.search_active) ? " [ESC] clear "
                                                                              : " [ESC] close ";
    }
    int n_exit = (int) strlen(exit_label);

    if (lay->help.show)
    {
        const char *help_hints;
        if (lay->help.search_active)
        {
            help_hints = " Type to search   ENTER Browse results   ESC Back";
        }
        else if (lay->help.search[0] != '\0')
        {
            help_hints = " ↑↓ Navigate results   [/] Edit search   ESC Clear search";
        }
        else if (lay->help.mode == 1)
        {
            help_hints = " ↑↓/PgUp/PgDn Scroll   2 Controls Reference   ESC Close";
        }
        else
        {
            help_hints = " 1/i Intro   ↑↓ Nav   →/← Expand   [/] Search   ESC Close";
        }
        sseg(sg, &n, 1, OV_FG_TITLE, OV_BG_HEADER, 1, "%s", help_hints);
    }
    else
    {
        /* Key hints of the context, the most specific first: "k term  p pause  e exit" */
        const OV_KEYHINT *hint[12];
        int               nh = ov_key_hints(ov_key_ctx(lay, m), hint, 12);
        sseg(sg, &n, 4, OV_FG_DIM, OV_BG_HEADER, 0, "%s", sort_label);
        sseg(sg, &n, 6, OV_FG_DIM, OV_BG_HEADER, 0, "%s", detail_label);
        for (int i = 0; i < nh; i++)
        {
            int prio =
                (hint[i]->need != 0) ? 2 : 7 + i; /* the context's own keys before general ones */
            SSEG *s = sseg(sg, &n, prio, OV_FG_TEXT, OV_BG_HEADER, 0, "  %s", hint[i]->key);
            snprintf(s->tail, sizeof(s->tail), " %s", hint[i]->meaning);
            s->tail_fg = OV_FG_DIM;
        }
    }

    time_t     now    = time(NULL);
    struct tm *tm_ptr = localtime(&now);
    char       tstr[16];
    int        n2 = strftime(tstr, sizeof(tstr), "%H:%M:%S", tm_ptr);

    char th_label[32];
    snprintf(th_label, sizeof(th_label), " [%s] ", ov_active_theme->id);
    int n_th = (int) strlen(th_label);

    int         right = n_th + n_exit + n2 + 1;
    OV_BAR_ITEM it[SSEG_MAX];
    for (int i = 0; i < n; i++)
    {
        it[i] = (OV_BAR_ITEM) { ov_str_display_width(sg[i].text) + ov_str_display_width(sg[i].tail),
                                0, sg[i].prio, 0 };
    }
    int used         = ov_bar_layout(it, n, r.width - right);
    lay->r_scan_info = (OV_RECT) { 0, 0, 0, 0 };
    for (int i = 0; i < n; i++)
    {
        if (it[i].shown)
        {
            if (i == 0)
            {
                lay->r_scan_info = (OV_RECT) { r.row, ov__cursor_col, 1, it[0].w };
            }
            sseg_draw(&sg[i]);
        }
    }

    /* Clock on the right edge */
    int pad = r.width - used - right;
    ov_theme_bg(OV_BG_HEADER);
    if (pad > 0)
    {
        ov_buf_hline(' ', pad);
    }

    /* Buttons are hit-tested where they are drawn; one cut off by the terminal edge has no target */
    ov_status_btn(&lay->r_theme_btn, r.row, n_th, lay->term_cols);
    ov_status_btn(&lay->r_exit_btn, r.row, n_exit, lay->term_cols);
    lay->r_exit_btn.col += n_th; /* drawn right after the theme button */
    if (lay->r_exit_btn.col + n_exit - 1 > lay->term_cols)
    {
        lay->r_exit_btn.width = 0;
    }

    /* Render theme button with hover highlight */
    int col_th_start = ov__cursor_col;
    if (lay->mouse_hover && (ov_mouse_row == r.row) && (ov_mouse_col >= col_th_start) &&
        (ov_mouse_col < col_th_start + n_th))
    {
        ov_theme_bg(OV_BG_SELECTED);
        ov_theme_fg(OV_FG_BRIGHT);
    }
    else
    {
        ov_theme_fg(OV_FG_TITLE);
        ov_theme_bg(OV_BG_HEADER);
    }
    ov_buf_printf("%s", th_label);

    /* Render [x] exit with distinct color or hover highlight */
    int col_start = ov__cursor_col;
    if (lay->mouse_hover && (ov_mouse_row == r.row) && (ov_mouse_col >= col_start) &&
        (ov_mouse_col < col_start + n_exit))
    {
        ov_theme_bg(OV_BADGE_DANGER);
        ov_theme_fg(ov_rgb_on(OV_BADGE_DANGER));
    }
    else
    {
        ov_theme_fg(OV_FG_ERROR);
        ov_theme_bg(OV_BG_HEADER);
    }
    ov_buf_printf("%s", exit_label);

    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_TEXT);
    ov_buf_printf("%s ", tstr);
    ov_buf_reset_attr();
}
