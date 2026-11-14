// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_view.h
 * @brief View descriptors (tab label, F-key, CLI name, panels shown) and sub-tab tab-bar helpers.
 *
 * Purpose:     a view is one row of ov_views[]: its tab, F-key, --view name, panels, initial
 *              focus and -h line all come from it.
 * Does:        declares OV_VIEW_DESC and ov_views[], the view lookups (ov_view_cycle,
 *              ov_view_by_fkey, ov_view_by_name, ov_view_label), and the hit tests of the tab bar
 *              (ov_tabbar_hit, ov_tabbar_level: render/overview_render_tabs.c) and of the sub-tab
 *              labels (ov_subtab_hit: render/overview_render_subtabs.c).
 * Used by:     overview_layout.h includes it after ov_view_t and ov_focus_t, hence render/ and
 *              input/; tools/milkCTRL.c, tools/milkCTRL_opts.c (-h) and
 *              tools/milkCTRL_opts_parse.c (--view).
 * See also:    overview_view.c; render/overview_render.c (view_render[]: what each view draws);
 *              src/cli/overview/README.md, "Extending milk-CTRL"; docs/guide/milk-CTRL.md, "Views".
 */

#ifndef OVERVIEW_VIEW_H
#define OVERVIEW_VIEW_H

#define OV_PANEL_BIT(f) (1U << (f))

typedef struct
{
    const char *label;         /* tab text, status line, help context: "DASH" */
    const char *cli_name;      /* --view argument: "dash" */
    int         fkey;          /* function key number: 2 = F2 */
    unsigned    panels;        /* OV_PANEL_BIT() of each ov_focus_t panel shown */
    int         default_focus; /* ov_focus_t given on entry, -1 = keep the current focus */
    const char *desc;          /* one line for -h */
} OV_VIEW_DESC;

extern const OV_VIEW_DESC ov_views[OV_VIEW_COUNT];

#define OV_TABBAR_HELP (-2)

int ov_view_cycle(int v, int step);
#define OV_TABBAR_HELP_W 11 /* " [h: HELP] " */
int ov_tabbar_hit(int mc);  /* render/overview_render_tabs.c */
int ov_tabbar_level(int width);
int ov_subtab_hit(int mc); /* render/overview_render_subtabs.c */
int ov_view_by_fkey(int fkey);
int ov_view_by_name(const char *name);

/** ov_view_label - tab label of view @v ("" when out of range). */
static inline const char *ov_view_label(ov_view_t v)
{
    return ((int) v >= 0 && v < OV_VIEW_COUNT) ? ov_views[v].label : "";
}

#endif /* OVERVIEW_VIEW_H */
