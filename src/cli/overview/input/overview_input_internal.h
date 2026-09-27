// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_internal.h
 * @brief Internal declarations for milk-CTRL input processing modules.
 */

#ifndef OVERVIEW_INPUT_INTERNAL_H
#define OVERVIEW_INPUT_INTERNAL_H

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "overview_defs.h"
#include "overview_ansi.h"
#include "milk_config.h"
#include "overview_data.h"
#include "overview_layout.h"
#include "overview_ctrl.h"
#include "overview_fps_edit.h"
#include "stream_graph.h"
#include "overview_data_internal.h"
#include "overview_render_internal.h"
#include "overview_data_loops.h"

/* libfps headers after overview headers to avoid macro redefinition warnings */
#undef STRINGMAXLEN_DIRNAME
#undef STRINGMAXLEN_FULLFILENAME
#undef STRINGMAXLEN_COMMAND
#undef PRINT_ERROR
#include "fps_types.h"
#include "fps_WriteParameterToDisk.h"
#include "fps_save2disk.h"

#define INSIDE(R, MR, MC) \
    ((MR) >= (R).row && (MR) < (R).row + (R).height && \
     (MC) >= (R).col && (MC) < (R).col + (R).width)

/* External scan and help API */
float ov_scan_get_interval(void);
void  ov_scan_set_interval(float s);
void  ov_help_open(OV_LAYOUT *lay);
int   ov_help_visible_count(const OV_LAYOUT *lay);
int   ov_help_nb_sections(void);
int   ov_help_toggle_at(OV_LAYOUT *lay, int vis_row);
int   ov_help_expand_at(OV_LAYOUT *lay, int vis_row, int expand);
int   ov_help_handle_click(OV_LAYOUT *lay, int mr, int mc);

/* Shared selection and navigation helpers */
int              ov_input_get_graph_start_node(const OV_LAYOUT *lay, const OV_MODEL *m);
int              ov_input_hit_panel_tab(int mc, int panel_col, const char **tabs, int num_tabs);
const OV_STREAM *ov_input_get_sel_stream(const OV_LAYOUT *lay, const OV_MODEL *m);
const OV_PROC   *ov_input_get_sel_proc(const OV_LAYOUT *lay, const OV_MODEL *m);
const OV_FPS    *ov_input_get_sel_fps(const OV_LAYOUT *lay, const OV_MODEL *m);
int              ov_input_get_filtered_count(int focus, const OV_LAYOUT *lay, const OV_MODEL *m);

/* Modal & text input handling */
int ov_input__handle_filter_mode(int key, OV_LAYOUT *lay, const OV_MODEL *m);
int ov_input__handle_loop_rename(int key, OV_LAYOUT *lay, OV_MODEL *m);
int ov_input__handle_loop_actions(int key, OV_LAYOUT *lay, OV_MODEL *m);

/* Hit-test & mouse resolution */
void ov_hittest(OV_LAYOUT *lay, const OV_MODEL *m, int mr, int mc);
void ov_hittest_resolve_globals(OV_LAYOUT *lay, const OV_MODEL *m);
void ov_input__streams_header_click(OV_LAYOUT *lay, int mc);
void ov_input__procs_header_click(OV_LAYOUT *lay, int mc);
void ov_input__fps_header_click(OV_LAYOUT *lay, int mc);
void ov_input__exec_preview_btn(int btn_id, OV_LAYOUT *lay, const OV_MODEL *m);

/* Mouse dispatch & helpers */
int ov_input__handle_mouse(int key, OV_LAYOUT *lay, const OV_MODEL *m);
int ov_input_mouse_drag(OV_LAYOUT *lay, const OV_MODEL *m, int mr, int mc);
int ov_input_mouse_wheel(int key, OV_LAYOUT *lay, const OV_MODEL *m);
int ov_input_mouse_panel_click(OV_LAYOUT *lay, const OV_MODEL *m, int mr, int mc, int is_dbl);

/* View switching, sorting, highlights, and toggles */
int ov_input__handle_view_switch(int key, OV_LAYOUT *lay);
int ov_input__handle_misc_toggles(int key, OV_LAYOUT *lay, const OV_MODEL *m);
int ov_input__handle_column_highlights(int key, OV_LAYOUT *lay, const OV_MODEL *m);
int ov_input__handle_sorting(int key, OV_LAYOUT *lay);

/* Actions: stream delete, proc kill/pause/step, FPS tmux */
int ov_input__handle_actions(int key, OV_LAYOUT *lay, const OV_MODEL *m);

/* Navigation: cursor movement, paging, scrolling */
int ov_input__handle_ancestry_nav(int key, OV_LAYOUT *lay, const OV_MODEL *m);
int ov_input__handle_navigation(int key, OV_LAYOUT *lay, const OV_MODEL *m);
int ov_input_nav_fps(int key, OV_LAYOUT *lay, const OV_MODEL *m);

/* Help overlay key handler */
int ov_input_handle_help_key(int key, OV_LAYOUT *lay);

/* Directory & FPS history serialization */
void ov_input_save_dir_history(OV_LAYOUT *lay, const char *fps_name, const char *path);
void ov_input_load_dir_history(OV_LAYOUT *lay, const char *fps_name, const char *path);
void ov_input_save_fps_history(OV_LAYOUT *lay, const char *fps_name);
void ov_input_load_fps_history(OV_LAYOUT *lay, const char *fps_name);

#endif // OVERVIEW_INPUT_INTERNAL_H
