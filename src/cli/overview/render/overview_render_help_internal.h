// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_RENDER_HELP_INTERNAL_H
#define OVERVIEW_RENDER_HELP_INTERNAL_H

/**
 * @file overview_render_help_internal.h
 * @brief Internal declarations and helpers for help overlay subsystems.
 */

#include "overview_render_internal.h"
#include "overview_help_data.h"

/**
 * help_is_expanded - check if a section is expanded.
 * @lay: layout state
 * @sec: section index (HS_NAV .. HS_COLORS)
 *
 * Return: 1 if expanded, 0 if collapsed.
 */
static inline int help_is_expanded(
    const OV_LAYOUT *lay,
    int              sec)
{
    return (lay->help_expand >> sec) & 1;
}

int help_visible_rows(
    const OV_LAYOUT *lay,
    int             *map);

void ov_help_get_rect(
    const OV_LAYOUT *lay,
    int             *pr,
    int             *pc,
    int             *ph,
    int             *pw);

void ov_help_print_wrapped(
    const char *text,
    int         row,
    int         col,
    int         max_w,
    int         max_lines,
    ov_rgb_t    fg,
    ov_rgb_t    bg);

void ov_help_render_detail(
    const OV_LAYOUT    *lay,
    const OV_MODEL     *m,
    const help_entry_t *entry,
    int                 split_r,
    int                 pc,
    int                 pw,
    int                 detail_h);

void ov_help_render_intro(
    const OV_LAYOUT *lay,
    int              pr,
    int              pc,
    int              ph,
    int              pw);

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
    int              inner_w);

#endif /* OVERVIEW_RENDER_HELP_INTERNAL_H */
