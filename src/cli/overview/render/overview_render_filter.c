// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_filter.c
 * @brief   Pattern filtering and panel filter state management for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <string.h>

/**
 * ov_get_effective_filter_panel - determine active panel target for filtering.
 * @lay: Pointer to layout structure.
 *
 * Return: Target panel focus enum.
 */
ov_focus_t ov_get_effective_filter_panel(const OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return OV_FOCUS_STREAMS;
    }
    if (lay->view == OV_VIEW_STREAMS)
    {
        return OV_FOCUS_STREAMS;
    }
    if (lay->view == OV_VIEW_PROCS)
    {
        return OV_FOCUS_PROCS;
    }
    if (lay->view == OV_VIEW_FPS)
    {
        return OV_FOCUS_FPS;
    }
    if (lay->focus == OV_FOCUS_STREAMS || lay->focus == OV_FOCUS_PROCS ||
        lay->focus == OV_FOCUS_FPS)
    {
        return lay->focus;
    }
    return OV_FOCUS_GRAPH;
}

/** ov_pf - the panel state for list panel @panel, or NULL (graph or out of range). */
static const OV_PANEL_STATE *ov_pf(const OV_LAYOUT *lay, ov_focus_t panel)
{
    return (lay != NULL && panel >= 0 && panel < OV_NB_LIST_PANELS) ? &lay->panel[panel] : NULL;
}

/**
 * ov_has_panel_filter - 1 if list panel @panel has a filter pattern (applied or not).
 */
int ov_has_panel_filter(const OV_LAYOUT *lay, ov_focus_t panel)
{
    const OV_PANEL_STATE *ps = ov_pf(lay, panel);
    return ps != NULL && ps->filter[0] != '\0';
}

/**
 * ov_is_panel_filter_active - 1 if list panel @panel has a pattern and it is applied.
 */
int ov_is_panel_filter_active(const OV_LAYOUT *lay, ov_focus_t panel)
{
    const OV_PANEL_STATE *ps = ov_pf(lay, panel);
    return ps != NULL && ps->filter_on && ps->filter[0] != '\0';
}

/**
 * ov_get_panel_filter_pattern - pattern of list panel @panel, applied or not ("" when none).
 */
const char *ov_get_panel_filter_pattern(const OV_LAYOUT *lay, ov_focus_t panel)
{
    const OV_PANEL_STATE *ps = ov_pf(lay, panel);
    return (ps != NULL) ? ps->filter : "";
}

/**
 * ov_get_active_filter_for - pattern of list panel @panel if applied, else "".
 */
const char *ov_get_active_filter_for(const OV_LAYOUT *lay, ov_focus_t panel)
{
    return ov_is_panel_filter_active(lay, panel) ? ov_get_panel_filter_pattern(lay, panel) : "";
}

/** ov_filter_any_on - 1 if any list panel has its filter applied. */
int ov_filter_any_on(const OV_LAYOUT *lay)
{
    for (int p = 0; p < OV_NB_LIST_PANELS; p++)
    {
        if (lay->panel[p].filter_on)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * ov_clear_panel_filter - drop the pattern of list panel @panel.
 */
void ov_clear_panel_filter(OV_LAYOUT *lay, ov_focus_t panel)
{
    if (ov_pf(lay, panel) == NULL)
    {
        return;
    }
    lay->panel[panel].filter[0] = '\0';
    lay->panel[panel].filter_on = 0;
    lay->filter_active          = ov_filter_any_on(lay);
}

/**
 * ov_has_filter - 1 if any list panel has a pattern (applied or not).
 */
int ov_has_filter(const OV_LAYOUT *lay)
{
    for (int p = 0; lay != NULL && p < OV_NB_LIST_PANELS; p++)
    {
        if (lay->panel[p].filter[0] != '\0')
        {
            return 1;
        }
    }
    return 0;
}

/**
 * ov_is_filter_active - 1 if the filtered panel (or, from the graph, any panel) has an applied filter.
 */
int ov_is_filter_active(const OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return 0;
    }
    ov_focus_t panel = ov_get_effective_filter_panel(lay);
    if (panel < OV_NB_LIST_PANELS)
    {
        return ov_is_panel_filter_active(lay, panel);
    }
    for (int p = 0; p < OV_NB_LIST_PANELS; p++)
    {
        if (ov_is_panel_filter_active(lay, (ov_focus_t) p))
        {
            return 1;
        }
    }
    return 0;
}

/**
 * ov_get_filter_pattern - pattern of the filtered panel, else the first panel that has one, else "".
 */
const char *ov_get_filter_pattern(const OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return "";
    }
    ov_focus_t panel = ov_get_effective_filter_panel(lay);
    if (ov_has_panel_filter(lay, panel))
    {
        return lay->panel[panel].filter;
    }
    for (int p = 0; p < OV_NB_LIST_PANELS; p++)
    {
        if (lay->panel[p].filter[0] != '\0')
        {
            return lay->panel[p].filter;
        }
    }
    return "";
}

/**
 * ov_get_active_filter - applied pattern of the filtered panel, or "".
 */
const char *ov_get_active_filter(const OV_LAYOUT *lay)
{
    return (lay != NULL) ? ov_get_active_filter_for(lay, ov_get_effective_filter_panel(lay)) : "";
}

/**
 * ov_clear_all_filters - clear all filter strings and reset filter state.
 * @lay: Pointer to layout structure.
 */
void ov_clear_all_filters(OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return;
    }
    for (int p = 0; p < OV_NB_LIST_PANELS; p++)
    {
        lay->panel[p].filter[0] = '\0';
        lay->panel[p].filter_on = 0;
    }
    lay->filter[0]      = '\0';
    lay->filter_active  = 0;
    lay->filter_editing = 0;
    lay->filter_cursor  = 0;
    lay->filter_jump    = 0;
}
