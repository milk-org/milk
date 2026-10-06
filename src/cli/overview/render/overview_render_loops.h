// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_loops.h
 * @brief GUI rendering for LOOPS tab and fullscreen view in milk-CTRL
 */

#ifndef OVERVIEW_RENDER_LOOPS_H
#define OVERVIEW_RENDER_LOOPS_H

#include "overview_data.h"
#include "overview_layout.h"

/**
 * ov_render_loops_panel - Render the LOOPS tab inside the dashboard graph panel.
 * @lay: Layout state
 * @m:   System model
 */
void ov_render_loops_panel(const OV_LAYOUT *lay, const OV_MODEL *m);

/**
 * ov_render_loops_view - Render the dedicated fullscreen F7 LOOPS view.
 * @lay: Layout state
 * @m:   System model
 */
void ov_render_loops_view(const OV_LAYOUT *lay, const OV_MODEL *m);

/**
 * ov_format_loop_path - Build a compact path string for loop cycle.
 * @m:   Model
 * @lp:  Loop
 * @buf: Output string buffer
 * @sz:  Buffer capacity
 */
void ov_format_loop_path(const OV_MODEL *m, const OV_LOOP *lp, char *buf, size_t sz);

#endif /* OVERVIEW_RENDER_LOOPS_H */
