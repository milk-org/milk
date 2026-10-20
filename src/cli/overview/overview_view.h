// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_view.h
 * @brief View descriptors (tab label, F-key, CLI name, panels shown) and sub-tab tab-bar helpers
 *
 * Included by overview_layout.h after ov_view_t and ov_focus_t.
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
} OV_VIEW_DESC;

extern const OV_VIEW_DESC ov_views[OV_VIEW_COUNT];

#define OV_TABBAR_HELP (-2)

int ov_view_cycle(int v, int step);
int ov_tabbar_hit(int mc); /* render/overview_render_tabs.c */
int ov_subtab_hit(int mc); /* render/overview_render_subtabs.c */
int ov_view_by_fkey(int fkey);
int ov_view_by_name(const char *name);

/** ov_view_label - tab label of view @v ("" when out of range). */
static inline const char *ov_view_label(ov_view_t v)
{
    return ((int) v >= 0 && v < OV_VIEW_COUNT) ? ov_views[v].label : "";
}

#endif /* OVERVIEW_VIEW_H */
