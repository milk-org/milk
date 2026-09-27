// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_resources_internal.h
 * @brief Internal declarations for resources panel rendering.
 */

#ifndef OVERVIEW_RENDER_RESOURCES_INTERNAL_H
#define OVERVIEW_RENDER_RESOURCES_INTERNAL_H

#include "overview_render_internal.h"
#include "overview_data_internal.h"

void ov_render_resources_perf_section(const OV_LAYOUT *lay,
                                      const OV_MODEL  *m,
                                      pid_t            target_pid,
                                      int              row,
                                      int             *ri,
                                      int             *line_idx,
                                      int              max_rows);

#endif /* OVERVIEW_RENDER_RESOURCES_INTERNAL_H */
