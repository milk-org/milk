// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_data_loops.h
 * @brief Directed cycle detection and loop analysis API for milk-CTRL
 *
 * Provides global directed cycle detection across the stream-process graph,
 * canonical cycle signature computation, overlap classification (shared vs
 * exclusive streams and processes), telemetry aggregation, and loop naming
 * with persistent configuration support.
 */

#ifndef OVERVIEW_DATA_LOOPS_H
#define OVERVIEW_DATA_LOOPS_H

#include "overview_data.h"
#include "stream_graph.h"

/* =========================================================
 * Loop Detection & Analysis API
 * ========================================================= */

/**
 * ov_detect_loops - Detect all elementary directed cycles in the graph.
 * @model: System model containing streams, processes, FPS, nodes, and edges
 * @mode:  Traversal mode (trigger, input, full)
 */
void ov_detect_loops(
    OV_MODEL *model,
    sg_mode_t mode);

/**
 * ov_loop_names_load - Load custom loop names from config file.
 * @model: System model containing detected loops
 */
void ov_loop_names_load(
    OV_MODEL *model);

/**
 * ov_loop_names_save - Save custom loop names to config file.
 * @model: System model containing detected loops
 */
void ov_loop_names_save(
    const OV_MODEL *model);

/**
 * ov_loop_rename - Set a custom name for a loop and persist it.
 * @model:    System model
 * @loop_idx: Index in model->loops[] (0..nb_loops-1)
 * @new_name: New human-readable name string
 *
 * Return: 0 on success, non-zero on error.
 */
int ov_loop_rename(
    OV_MODEL   *model,
    int         loop_idx,
    const char *new_name);

/**
 * ov_get_loop_name - Retrieve display name for a loop ID.
 * @model:   System model
 * @loop_id: 1-based loop ID
 *
 * Return: Display name string or fallback label.
 */
const char *ov_get_loop_name(
    const OV_MODEL *model,
    int             loop_id);

/**
 * ov_find_loop_by_id - Find loop array index from 1-based loop ID.
 * @model:   System model
 * @loop_id: 1-based loop ID
 *
 * Return: Array index in model->loops[], or -1 if not found.
 */
int ov_find_loop_by_id(
    const OV_MODEL *model,
    int             loop_id);

#endif /* OVERVIEW_DATA_LOOPS_H */
