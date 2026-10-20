// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_header.c
 * @brief   Top status header and metrics banner rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include "overview_data_internal.h"
#include "overview_ansi.h"
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
int ov_render_header_text(const char *text, int hs, int max_vis_width, ov_rgb_t base_fg)
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

/* One header segment: drawn after @lead spaces of header background */
enum
{
    HS_HEART,
    HS_LOGO,
    HS_TEXT,
    HS_BADGE
};
enum
{
    HIT_NONE,
    HIT_CTRL,
    HIT_HOVER,
    HIT_FILTER
};
typedef struct
{
    int        style, lead, bold, hit, prio;
    ov_rgb_t   fg, bg;
    ov_focus_t panel; /* HIT_FILTER */
    char       text[96];
    char       text_short[48]; /* "" = no short form */
} HSEG;

#define HSEG_MAX 16

/** hseg - append a segment of priority @prio (0 = shown first) to @sg. Return: it. */
static HSEG *hseg(HSEG *sg, int *n, int style, int lead, ov_rgb_t fg, ov_rgb_t bg, int prio)
{
    HSEG *h = &sg[(*n)++];
    memset(h, 0, sizeof(*h));
    h->style = style;
    h->lead  = lead;
    h->fg    = fg;
    h->bg    = bg;
    h->prio  = prio;
    return h;
}

/** filter_badge - the filter badge of panel @p named @nm (@foc: the focused panel), with its colours. */
static void filter_badge(const OV_LAYOUT *lay,
                         HSEG            *h,
                         ov_focus_t       p,
                         const char      *nm,
                         int              foc,
                         int              patlen)
{
    const char *pat =
        (p != OV_FOCUS_GRAPH) ? ov_get_panel_filter_pattern(lay, p) : ov_get_filter_pattern(lay);
    int act = (p != OV_FOCUS_GRAPH) ? ov_is_panel_filter_active(lay, p) : ov_is_filter_active(lay);
    int blink = ((lay->ctrl_blink % 4) < 2);
    h->bold   = 1;
    h->hit    = HIT_FILTER;
    h->panel  = p;
    if (act)
    {
        h->bg = (foc && !blink) ? OV_BADGE_DANGER : OV_BADGE_WARN;
        h->fg = ov_rgb_on(h->bg);
        snprintf(h->text, sizeof(h->text), foc ? " [f] %s: /%.*s/ " : " %s: /%.*s/ ", nm, patlen,
                 pat);
    }
    else if (pat[0] != '\0')
    {
        h->bg = OV_BG_PANEL_ALT;
        h->fg = OV_FG_WARN;
        if (foc)
        {
            snprintf(h->text, sizeof(h->text), " [f] %s: OFF (/%.*s/) ", nm, patlen - 2, pat);
        }
        else
        {
            snprintf(h->text, sizeof(h->text), " %s: OFF ", nm);
        }
    }
    else
    {
        h->bg = OV_BG_PANEL;
        h->fg = OV_FG_DIM;
        snprintf(h->text, sizeof(h->text), foc ? " [/] %s: ALL " : " %s: ALL ", nm);
    }
}

/** basename_of - last component of path @p. */
static const char *basename_of(const char *p)
{
    const char *b = strrchr(p, '/');
    return (b != NULL && b[1] != '\0') ? b + 1 : p;
}

/**
 * ov_render_header - render the top status header bar of milk-CTRL.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Segments are laid out by importance (ov_bar_layout): the mode badge first, then the counts, filters,
 * hover badge, own CPU, logo, shm directory (its last component when short of room) and commit. The
 * badges record where they are drawn, for clicks and tooltips.
 */
void ov_render_header(OV_LAYOUT *lay, const OV_MODEL *m)
{
    static const char *const nm[OV_NB_LIST_PANELS] = { "STRM", "PROC", "FPS" };
    HSEG                     sg[HSEG_MAX];
    int                      n = 0;
    HSEG                    *h;

    lay->ctrl_blink++;
    h = hseg(sg, &n, HS_HEART, 0, OV_FG_ERROR, OV_BG_HEADER, 1);
    snprintf(h->text, sizeof(h->text), "\xe2\x97\x8f"); /* ● heartbeat */
    h = hseg(sg, &n, HS_LOGO, 0, OV_GRAD_LO, OV_BG_HEADER, 5);
    snprintf(h->text, sizeof(h->text), "%s %s milk-CTRL %s", OV_LCARS_LEFT, OV_BULLET,
             OV_LCARS_RIGHT);
    h = hseg(sg, &n, HS_TEXT, 1, OV_FG_DIM, OV_BG_HEADER, 7);
    snprintf(h->text, sizeof(h->text), "[%s]", MILK_GIT_COMMIT);
    h = hseg(sg, &n, HS_TEXT, 1, OV_FG_DIM, OV_BG_HEADER, 6);
    snprintf(h->text, sizeof(h->text), "[shm: %.80s]", ov_get_shmdir());
    snprintf(h->text_short, sizeof(h->text_short), "[shm: %.36s]", basename_of(ov_get_shmdir()));

    int blink = ((lay->ctrl_blink % 4) < 2);
    h         = hseg(sg, &n, HS_BADGE, 1, OV_FG_TEXT, OV_BADGE_OK, 0);
    h->bold   = 1;
    h->hit    = HIT_CTRL;
    if (lay->ctrl_mode)
    {
        h->bg = blink ? OV_ANIM_PULSE_BG_MAX : OV_BADGE_DANGER;
        h->fg = blink ? OV_ANIM_PULSE_FG_MAX : ov_rgb_on(OV_BADGE_DANGER);
    }
    else
    {
        h->fg = ov_rgb_on(OV_BADGE_OK);
    }
    snprintf(h->text, sizeof(h->text), lay->ctrl_mode ? " [c] CONTROL " : " [c] READ ONLY ");
    snprintf(h->text_short, sizeof(h->text_short), lay->ctrl_mode ? " CTRL " : " R/O ");

    h       = hseg(sg, &n, HS_BADGE, 1, OV_FG_DIM, OV_BG_PANEL, 3);
    h->bold = 1;
    h->hit  = HIT_HOVER;
    if (lay->mouse_hover)
    {
        h->bg = OV_BADGE_WARN;
        h->fg = ov_rgb_on(OV_BADGE_WARN);
    }
    snprintf(h->text, sizeof(h->text), lay->mouse_hover ? " [m] HOVER: ON " : " [m] HOVER: OFF ");

    /* Filters: DASH shows its three lists; other views the focused one, plus any other active filter */
    ov_focus_t fpanel = ov_get_effective_filter_panel(lay);
    int        dash   = (lay->view == OV_VIEW_DASHBOARD);
    if (!dash)
    {
        const char *pn = (fpanel < OV_NB_LIST_PANELS) ? nm[fpanel] : "FILTER";
        filter_badge(lay, hseg(sg, &n, HS_BADGE, 1, OV_FG_DIM, OV_BG_PANEL, 2), fpanel, pn, 1, 10);
    }
    for (int p = 0; p < OV_NB_LIST_PANELS; p++)
    {
        if (dash || ((ov_focus_t) p != fpanel && ov_is_panel_filter_active(lay, (ov_focus_t) p)))
        {
            filter_badge(lay, hseg(sg, &n, HS_BADGE, 1, OV_FG_DIM, OV_BG_PANEL, 2), (ov_focus_t) p,
                         nm[p], dash && fpanel == (ov_focus_t) p, 8);
        }
    }

    /* Counts, and milk-CTRL's own cost: its CPU (RUSAGE_SELF) and output rate to the terminal */
    static const char *const unit[4] = { "stm", "prc", "fps", "edg" };
    const ov_rgb_t           ucol[4] = { OV_FG_STREAM, OV_FG_PROC, OV_FG_FPS, OV_FG_CONN };
    const int                cnt[4]  = { m->nb_streams, m->nb_procs, m->nb_fps, m->nb_edges };
    for (int u = 0; u < 4; u++)
    {
        h = hseg(sg, &n, HS_TEXT, 1, ucol[u], OV_BG_HEADER, 1);
        snprintf(h->text, sizeof(h->text), "%d %s", cnt[u], unit[u]);
    }
    h = hseg(sg, &n, HS_TEXT, 2, OV_FG_DIM, OV_BG_HEADER, 4);
    snprintf(h->text, sizeof(h->text), "self %4.1f%% %4.1fkB/s", ov_sys_get_cpu_usage(),
             ov_sys_get_bandwidth_usage());

    OV_BAR_ITEM it[HSEG_MAX];
    for (int i = 0; i < n; i++)
    {
        it[i] = (OV_BAR_ITEM) { sg[i].lead + ov_str_display_width(sg[i].text),
                                sg[i].text_short[0]
                                    ? sg[i].lead + ov_str_display_width(sg[i].text_short)
                                    : 0,
                                sg[i].prio, 0 };
    }
    OV_RECT r = lay->r_header;
    ov_bar_layout(it, n, r.width);

    /* Draw in screen order, recording where the badges land */
    lay->r_ctrl_badge   = (OV_RECT) { 0 };
    lay->r_hover_badge  = (OV_RECT) { 0 };
    lay->r_filter_count = 0;
    int col             = r.col;
    ov_buf_pos(r.row, r.col);
    for (int i = 0; i < n; i++)
    {
        if (!it[i].shown)
        {
            continue;
        }
        const HSEG *g    = &sg[i];
        const char *text = (it[i].shown == 2) ? g->text_short : g->text;
        int         tw   = (it[i].shown == 2 ? it[i].w_short : it[i].w) - g->lead;
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_hline(' ', g->lead);
        col += g->lead;
        OV_RECT hit = { r.row, col, 1, tw };
        if (g->style == HS_HEART)
        {
            ov_theme_fg((lay->ctrl_blink % 2) ? OV_ANIM_PULSE_BG_MIN : OV_FG_ERROR);
            ov_buf_printf("%s", text);
        }
        else if (g->style == HS_LOGO)
        {
            ov_buf_bold();
            ov_buf_printf_gradient(OV_GRAD_LO, OV_GRAD_HI, "%s", text);
            ov_buf_reset_attr();
        }
        else
        {
            ov_theme_bg(g->bg);
            ov_theme_fg(g->fg);
            if (g->bold)
            {
                ov_buf_bold();
            }
            ov_buf_printf("%s", text);
            ov_buf_reset_attr();
        }
        col += tw;
        if (g->hit == HIT_CTRL)
        {
            lay->r_ctrl_badge = hit;
        }
        else if (g->hit == HIT_HOVER)
        {
            lay->r_hover_badge = hit;
        }
        else if (g->hit == HIT_FILTER && lay->r_filter_count < 4)
        {
            lay->r_filter_start[lay->r_filter_count] = hit.col;
            lay->r_filter_width[lay->r_filter_count] = hit.width;
            lay->r_filter_panel[lay->r_filter_count] = g->panel;
            lay->r_filter_count++;
        }
    }
    ov_theme_bg(OV_BG_HEADER);
    render_pad_spaces(col - r.col, r.width);
}
