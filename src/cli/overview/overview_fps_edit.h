// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_fps_edit.h
 * @brief Inline FPS parameter editing for milk-CTRL
 */

#ifndef OVERVIEW_FPS_EDIT_H
#define OVERVIEW_FPS_EDIT_H

#include "overview_layout.h"

/**
 * ov_fps_inline_edit - edit an FPS parameter inline.
 *
 * @lay:       layout state
 * @fps_name:  name of the FPS to edit
 * @disp_idx:  display parameter index
 *
 * Return: 0 on success/abort, -1 on error
 */
int ov_fps_inline_edit(OV_LAYOUT *lay, const char *fps_name, int disp_idx);

#endif /* OVERVIEW_FPS_EDIT_H */
