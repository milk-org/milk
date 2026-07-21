// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_fps_params.h
 * @brief   FPS parameter tree panel for milk-CTRL F5 view
 */

#ifndef OVERVIEW_RENDER_FPS_PARAMS_H
#define OVERVIEW_RENDER_FPS_PARAMS_H

#include "overview_layout.h"
#include "overview_data.h"

/**
 * ov_render_fps_params_panel - draw FPS parameter tree panel.
 * @lay: layout state (fps_param_focus/sel/scroll consumed)
 * @m:   data model
 *
 * Renders into lay->r_fps_params. Call only when
 * view == OV_VIEW_FPS.
 */
void ov_render_fps_params_panel(OV_LAYOUT *lay, const OV_MODEL *m);

/**
 * ov_render_fps_param_info - draw FPS parameter metadata header on rows 2 and 3.
 * @lay: layout state
 * @m:   data model
 */
void ov_render_fps_param_info(const OV_LAYOUT *lay, const OV_MODEL *m);

#endif /* OVERVIEW_RENDER_FPS_PARAMS_H */
