// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_theme_palettes_extra.c
 * @brief Extended theme palette definitions (dracula, solarized, monokai, matrix).
 */

#include "overview_theme_internal.h"

/* 5. Dracula (Gothic dark slate with vibrant neon accents) */
const ov_theme_t ov_theme_dracula = {
    .id   = "dracula",
    .name = "Dracula",
    .desc = "Gothic dark slate with vibrant purple, pink, & cyan",

    .bg_terminal    = { 40, 42, 54 },
    .bg_panel       = { 52, 55, 70 },
    .bg_panel_alt   = { 46, 48, 62 },
    .bg_header      = { 68, 71, 90 },
    .bg_selected    = { 85, 75, 120 },
    .bg_related     = { 45, 68, 68 },
    .bg_frozen      = { 98, 114, 164 },
    .bg_hover       = { 75, 75, 102 },
    .bg_pid_match   = { 50, 105, 75 },
    .bg_stale       = { 82, 72, 42 },
    .bg_new_item    = { 50, 95, 82 },
    .bg_loop        = { 90, 52, 92 },
    .bg_loop_shared = { 102, 72, 52 },

    .fg_title      = { 189, 147, 249 },
    .fg_dim        = { 125, 135, 160 },
    .fg_text       = { 248, 248, 242 },
    .fg_bright     = { 255, 255, 255 },
    .fg_muted      = { 98, 114, 164 },
    .fg_stream     = { 139, 233, 253 },
    .fg_fps        = { 189, 147, 249 },
    .fg_proc       = { 255, 121, 198 },
    .fg_stream_hdr = { 90, 165, 185 },
    .fg_fps_hdr    = { 135, 110, 185 },
    .fg_proc_hdr   = { 185, 90, 145 },
    .fg_active     = { 80, 250, 123 },
    .fg_idle       = { 130, 140, 165 },

    .anim_pulse_bg_min = { 95, 25, 45 },
    .anim_pulse_bg_max = { 185, 45, 75 },
    .anim_pulse_fg_min = { 205, 125, 145 },
    .anim_pulse_fg_max = { 255, 225, 235 },
    .fg_warn           = { 241, 250, 140 },
    .fg_error          = { 255, 85, 85 },
    .fg_zombie         = { 255, 184, 108 },

    .fg_conn        = { 125, 145, 195 },
    .fg_edge_active = { 139, 233, 253 },
    .fg_loop        = { 255, 121, 198 },
    .fg_loop_shared = { 255, 184, 108 },

    .grad_lo     = { 98, 114, 164 },
    .grad_hi     = { 139, 233, 253 },
    .grad_cpu_lo = { 80, 250, 123 },
    .grad_cpu_hi = { 255, 85, 85 },
};

/* 6. Solarized Dark (Precision cyan & amber dark palette) */
const ov_theme_t ov_theme_solarized_dark = {
    .id   = "solarized-dark",
    .name = "Solarized Dark",
    .desc = "Precision cyan & amber dark palette (Ethan Schoonover)",

    .bg_terminal    = { 0, 43, 54 },
    .bg_panel       = { 7, 54, 66 },
    .bg_panel_alt   = { 4, 48, 60 },
    .bg_header      = { 14, 68, 82 },
    .bg_selected    = { 20, 80, 95 },
    .bg_related     = { 10, 65, 60 },
    .bg_frozen      = { 38, 139, 210 },
    .bg_hover       = { 15, 75, 90 },
    .bg_pid_match   = { 85, 135, 10 },
    .bg_stale       = { 80, 60, 10 },
    .bg_new_item    = { 20, 95, 90 },
    .bg_loop        = { 60, 45, 80 },
    .bg_loop_shared = { 80, 50, 20 },

    .fg_title      = { 38, 139, 210 },
    .fg_dim        = { 101, 123, 131 },
    .fg_text       = { 131, 148, 150 },
    .fg_bright     = { 253, 246, 227 },
    .fg_muted      = { 88, 110, 117 },
    .fg_stream     = { 42, 161, 152 },
    .fg_fps        = { 38, 139, 210 },
    .fg_proc       = { 108, 113, 196 },
    .fg_stream_hdr = { 30, 120, 115 },
    .fg_fps_hdr    = { 28, 105, 160 },
    .fg_proc_hdr   = { 80, 85, 150 },
    .fg_active     = { 133, 153, 0 },
    .fg_idle       = { 101, 123, 131 },

    .anim_pulse_bg_min = { 80, 30, 10 },
    .anim_pulse_bg_max = { 160, 50, 20 },
    .anim_pulse_fg_min = { 180, 100, 60 },
    .anim_pulse_fg_max = { 253, 246, 227 },
    .fg_warn           = { 181, 137, 0 },
    .fg_error          = { 220, 50, 47 },
    .fg_zombie         = { 203, 75, 22 },

    .fg_conn        = { 42, 161, 152 },
    .fg_edge_active = { 38, 139, 210 },
    .fg_loop        = { 211, 54, 130 },
    .fg_loop_shared = { 203, 75, 22 },

    .grad_lo     = { 38, 139, 210 },
    .grad_hi     = { 42, 161, 152 },
    .grad_cpu_lo = { 133, 153, 0 },
    .grad_cpu_hi = { 220, 50, 47 },
};

/* 7. Solarized Light (Warm cream & cyan light palette) */
const ov_theme_t ov_theme_solarized_light = {
    .id   = "solarized-light",
    .name = "Solarized Light",
    .desc = "Warm cream & cyan light palette (Ethan Schoonover)",

    .bg_terminal    = { 253, 246, 227 },
    .bg_panel       = { 238, 232, 213 },
    .bg_panel_alt   = { 245, 239, 220 },
    .bg_header      = { 225, 218, 198 },
    .bg_selected    = { 205, 220, 230 },
    .bg_related     = { 215, 230, 210 },
    .bg_frozen      = { 180, 210, 235 },
    .bg_hover       = { 220, 226, 215 },
    .bg_pid_match   = { 200, 230, 180 },
    .bg_stale       = { 240, 225, 180 },
    .bg_new_item    = { 190, 235, 225 },
    .bg_loop        = { 235, 215, 230 },
    .bg_loop_shared = { 245, 220, 195 },

    .fg_title      = { 38, 139, 210 },
    .fg_dim        = { 147, 161, 161 },
    .fg_text       = { 101, 123, 131 },
    .fg_bright     = { 7, 54, 66 },
    .fg_muted      = { 131, 148, 150 },
    .fg_stream     = { 42, 161, 152 },
    .fg_fps        = { 38, 139, 210 },
    .fg_proc       = { 108, 113, 196 },
    .fg_stream_hdr = { 30, 125, 120 },
    .fg_fps_hdr    = { 28, 110, 170 },
    .fg_proc_hdr   = { 85, 90, 160 },
    .fg_active     = { 133, 153, 0 },
    .fg_idle       = { 147, 161, 161 },

    .anim_pulse_bg_min = { 240, 210, 200 },
    .anim_pulse_bg_max = { 220, 80, 70 },
    .anim_pulse_fg_min = { 160, 60, 50 },
    .anim_pulse_fg_max = { 253, 246, 227 },
    .fg_warn           = { 181, 137, 0 },
    .fg_error          = { 220, 50, 47 },
    .fg_zombie         = { 203, 75, 22 },

    .fg_conn        = { 42, 161, 152 },
    .fg_edge_active = { 38, 139, 210 },
    .fg_loop        = { 211, 54, 130 },
    .fg_loop_shared = { 203, 75, 22 },

    .grad_lo     = { 38, 139, 210 },
    .grad_hi     = { 42, 161, 152 },
    .grad_cpu_lo = { 133, 153, 0 },
    .grad_cpu_hi = { 220, 50, 47 },
};

/* 8. Monokai Pro (Warm charcoal with radiant neon accents) */
const ov_theme_t ov_theme_monokai = {
    .id   = "monokai",
    .name = "Monokai Pro",
    .desc = "Warm charcoal with radiant neon accents",

    .bg_terminal    = { 45, 42, 46 },
    .bg_panel       = { 58, 53, 59 },
    .bg_panel_alt   = { 50, 46, 51 },
    .bg_header      = { 72, 67, 74 },
    .bg_selected    = { 85, 75, 95 },
    .bg_related     = { 55, 70, 60 },
    .bg_frozen      = { 100, 90, 130 },
    .bg_hover       = { 78, 72, 82 },
    .bg_pid_match   = { 60, 110, 75 },
    .bg_stale       = { 90, 75, 35 },
    .bg_new_item    = { 50, 95, 85 },
    .bg_loop        = { 85, 45, 75 },
    .bg_loop_shared = { 95, 65, 40 },

    .fg_title      = { 120, 220, 232 },
    .fg_dim        = { 147, 146, 147 },
    .fg_text       = { 246, 246, 244 },
    .fg_bright     = { 255, 255, 255 },
    .fg_muted      = { 114, 112, 114 },
    .fg_stream     = { 120, 220, 232 },
    .fg_fps        = { 171, 157, 242 },
    .fg_proc       = { 255, 97, 136 },
    .fg_stream_hdr = { 80, 160, 170 },
    .fg_fps_hdr    = { 125, 115, 180 },
    .fg_proc_hdr   = { 190, 70, 100 },
    .fg_active     = { 169, 220, 103 },
    .fg_idle       = { 147, 146, 147 },

    .anim_pulse_bg_min = { 90, 25, 35 },
    .anim_pulse_bg_max = { 190, 45, 70 },
    .anim_pulse_fg_min = { 210, 120, 140 },
    .anim_pulse_fg_max = { 255, 220, 230 },
    .fg_warn           = { 255, 216, 102 },
    .fg_error          = { 255, 97, 136 },
    .fg_zombie         = { 252, 152, 103 },

    .fg_conn        = { 120, 220, 232 },
    .fg_edge_active = { 171, 157, 242 },
    .fg_loop        = { 255, 97, 136 },
    .fg_loop_shared = { 252, 152, 103 },

    .grad_lo     = { 120, 220, 232 },
    .grad_hi     = { 171, 157, 242 },
    .grad_cpu_lo = { 169, 220, 103 },
    .grad_cpu_hi = { 255, 97, 136 },
};

/* 9. Matrix Phosphor (Phosphor green on pure black) */
const ov_theme_t ov_theme_matrix = {
    .id   = "matrix",
    .name = "Matrix Phosphor",
    .desc = "High-contrast phosphor green on pure black",

    .bg_terminal    = { 2, 8, 4 },
    .bg_panel       = { 8, 20, 12 },
    .bg_panel_alt   = { 5, 14, 8 },
    .bg_header      = { 14, 34, 20 },
    .bg_selected    = { 20, 60, 32 },
    .bg_related     = { 12, 40, 22 },
    .bg_frozen      = { 25, 75, 42 },
    .bg_hover       = { 18, 50, 28 },
    .bg_pid_match   = { 30, 95, 50 },
    .bg_stale       = { 50, 45, 15 },
    .bg_new_item    = { 20, 70, 45 },
    .bg_loop        = { 35, 30, 50 },
    .bg_loop_shared = { 55, 45, 20 },

    .fg_title      = { 0, 255, 102 },
    .fg_dim        = { 50, 130, 70 },
    .fg_text       = { 140, 235, 170 },
    .fg_bright     = { 210, 255, 225 },
    .fg_muted      = { 40, 100, 55 },
    .fg_stream     = { 0, 240, 160 },
    .fg_fps        = { 50, 255, 120 },
    .fg_proc       = { 160, 255, 100 },
    .fg_stream_hdr = { 0, 160, 100 },
    .fg_fps_hdr    = { 30, 170, 80 },
    .fg_proc_hdr   = { 110, 170, 70 },
    .fg_active     = { 0, 255, 65 },
    .fg_idle       = { 50, 120, 70 },

    .anim_pulse_bg_min = { 40, 10, 10 },
    .anim_pulse_bg_max = { 120, 20, 20 },
    .anim_pulse_fg_min = { 180, 80, 80 },
    .anim_pulse_fg_max = { 255, 200, 200 },
    .fg_warn           = { 220, 255, 80 },
    .fg_error          = { 255, 70, 70 },
    .fg_zombie         = { 240, 180, 50 },

    .fg_conn        = { 0, 200, 130 },
    .fg_edge_active = { 0, 255, 102 },
    .fg_loop        = { 180, 255, 100 },
    .fg_loop_shared = { 240, 220, 60 },

    .grad_lo     = { 0, 140, 70 },
    .grad_hi     = { 0, 255, 102 },
    .grad_cpu_lo = { 0, 255, 65 },
    .grad_cpu_hi = { 255, 70, 70 },
};
