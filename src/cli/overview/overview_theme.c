// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_theme.c
 * @brief Theme definitions and palette management for milk-CTRL
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "overview_theme.h"
#include "overview_ansi.h"

/* =========================================================
 * Palette definitions
 * ========================================================= */

static const ov_theme_t ov_themes[] = {
    /* -----------------------------------------------------
     * 0. Default Dark (Slate dark palette)
     * ----------------------------------------------------- */
    {
        .id   = "dark",
        .name = "Default Dark",
        .desc = "Default slate dark palette",

        .bg_terminal    = { 20, 22, 28 },
        .bg_panel       = { 30, 32, 40 },
        .bg_panel_alt   = { 25, 27, 35 },
        .bg_header      = { 40, 44, 58 },
        .bg_selected    = { 50, 60, 90 },
        .bg_related     = { 38, 50, 42 },
        .bg_frozen      = { 40, 90, 140 },
        .bg_hover       = { 45, 52, 72 },
        .bg_pid_match   = { 50, 180, 50 },
        .bg_stale       = { 55, 45, 20 },
        .bg_new_item    = { 40, 60, 50 },
        .bg_loop        = { 45, 30, 55 },
        .bg_loop_shared = { 55, 42, 25 },

        .fg_title      = { 130, 170, 255 },
        .fg_dim        = { 100, 105, 120 },
        .fg_text       = { 200, 205, 215 },
        .fg_bright     = { 240, 245, 255 },
        .fg_muted      = { 70, 75, 85 },
        .fg_stream     = { 80, 200, 220 },
        .fg_fps        = { 130, 170, 255 },
        .fg_proc       = { 180, 140, 255 },
        .fg_stream_hdr = { 55, 140, 155 },
        .fg_fps_hdr    = { 90, 120, 180 },
        .fg_proc_hdr   = { 125, 100, 180 },
        .fg_active     = { 80, 220, 80 },
        .fg_idle       = { 130, 140, 160 },

        .anim_pulse_bg_min = { 80, 10, 10 },
        .anim_pulse_bg_max = { 180, 20, 20 },
        .anim_pulse_fg_min = { 160, 80, 80 },
        .anim_pulse_fg_max = { 255, 220, 220 },
        .fg_warn           = { 255, 180, 0 },
        .fg_error          = { 240, 60, 60 },
        .fg_zombie         = { 180, 120, 40 },

        .fg_conn        = { 100, 130, 180 },
        .fg_edge_active = { 140, 200, 255 },
        .fg_loop        = { 220, 120, 255 },
        .fg_loop_shared = { 255, 175, 40 },

        .grad_lo     = { 60, 90, 140 },
        .grad_hi     = { 100, 200, 255 },
        .grad_cpu_lo = { 60, 180, 60 },
        .grad_cpu_hi = { 240, 60, 60 },
    },

    /* -----------------------------------------------------
     * 1. Observatory Red (Monochrome red for dark adaptation)
     * ----------------------------------------------------- */
    {
        .id   = "night",
        .name = "Observatory Red",
        .desc = "Dark-adapted monochrome red palette",

        .bg_terminal    = { 10, 0, 0 },
        .bg_panel       = { 22, 4, 4 },
        .bg_panel_alt   = { 16, 2, 2 },
        .bg_header      = { 36, 8, 8 },
        .bg_selected    = { 70, 15, 15 },
        .bg_related     = { 45, 10, 10 },
        .bg_frozen      = { 85, 20, 20 },
        .bg_hover       = { 50, 12, 12 },
        .bg_pid_match   = { 90, 25, 10 },
        .bg_stale       = { 40, 18, 5 },
        .bg_new_item    = { 60, 15, 15 },
        .bg_loop        = { 50, 10, 20 },
        .bg_loop_shared = { 55, 20, 10 },

        .fg_title      = { 255, 80, 80 },
        .fg_dim        = { 120, 50, 50 },
        .fg_text       = { 220, 140, 140 },
        .fg_bright     = { 255, 180, 180 },
        .fg_muted      = { 90, 40, 40 },
        .fg_stream     = { 255, 110, 90 },
        .fg_fps        = { 255, 70, 70 },
        .fg_proc       = { 240, 130, 110 },
        .fg_stream_hdr = { 180, 70, 60 },
        .fg_fps_hdr    = { 170, 50, 50 },
        .fg_proc_hdr   = { 160, 80, 70 },
        .fg_active     = { 255, 90, 90 },
        .fg_idle       = { 130, 70, 70 },

        .anim_pulse_bg_min = { 60, 5, 5 },
        .anim_pulse_bg_max = { 130, 15, 15 },
        .anim_pulse_fg_min = { 140, 60, 60 },
        .anim_pulse_fg_max = { 255, 160, 160 },
        .fg_warn           = { 255, 160, 30 },
        .fg_error          = { 255, 30, 30 },
        .fg_zombie         = { 180, 80, 20 },

        .fg_conn        = { 160, 70, 70 },
        .fg_edge_active = { 255, 110, 110 },
        .fg_loop        = { 230, 90, 120 },
        .fg_loop_shared = { 255, 140, 50 },

        .grad_lo     = { 100, 30, 30 },
        .grad_hi     = { 255, 80, 80 },
        .grad_cpu_lo = { 140, 50, 50 },
        .grad_cpu_hi = { 255, 40, 40 },
    },

    /* -----------------------------------------------------
     * 2. High-Contrast CVD (Color-blind friendly)
     * ----------------------------------------------------- */
    {
        .id   = "accessible",
        .name = "High-Contrast CVD",
        .desc = "Colorblind-friendly high-contrast palette (Okabe-Ito)",

        .bg_terminal    = { 10, 12, 16 },
        .bg_panel       = { 20, 24, 32 },
        .bg_panel_alt   = { 15, 18, 25 },
        .bg_header      = { 32, 40, 54 },
        .bg_selected    = { 0, 70, 130 },
        .bg_related     = { 25, 45, 55 },
        .bg_frozen      = { 0, 95, 175 },
        .bg_hover       = { 35, 50, 70 },
        .bg_pid_match   = { 0, 120, 90 },
        .bg_stale       = { 65, 45, 10 },
        .bg_new_item    = { 0, 80, 70 },
        .bg_loop        = { 50, 35, 60 },
        .bg_loop_shared = { 60, 45, 15 },

        .fg_title      = { 120, 180, 255 },
        .fg_dim        = { 130, 135, 145 },
        .fg_text       = { 230, 235, 245 },
        .fg_bright     = { 255, 255, 255 },
        .fg_muted      = { 100, 105, 115 },
        .fg_stream     = { 86, 180, 233 },
        .fg_fps        = { 0, 114, 178 },
        .fg_proc       = { 213, 94, 0 },
        .fg_stream_hdr = { 60, 135, 180 },
        .fg_fps_hdr    = { 30, 90, 145 },
        .fg_proc_hdr   = { 170, 75, 10 },
        .fg_active     = { 86, 180, 233 },
        .fg_idle       = { 140, 145, 155 },

        .anim_pulse_bg_min = { 100, 40, 0 },
        .anim_pulse_bg_max = { 200, 80, 0 },
        .anim_pulse_fg_min = { 220, 180, 100 },
        .anim_pulse_fg_max = { 255, 255, 255 },
        .fg_warn           = { 240, 228, 66 },
        .fg_error          = { 213, 94, 0 },
        .fg_zombie         = { 204, 121, 167 },

        .fg_conn        = { 110, 140, 190 },
        .fg_edge_active = { 140, 205, 255 },
        .fg_loop        = { 204, 121, 167 },
        .fg_loop_shared = { 230, 159, 0 },

        .grad_lo     = { 0, 114, 178 },
        .grad_hi     = { 86, 180, 233 },
        .grad_cpu_lo = { 86, 180, 233 },
        .grad_cpu_hi = { 213, 94, 0 },
    },

    /* -----------------------------------------------------
     * 3. Paper Light (Daylight & publication figures)
     * ----------------------------------------------------- */
    {
        .id   = "light",
        .name = "Paper Light",
        .desc = "Clean light theme for daylight offices & publication figures",

        .bg_terminal    = { 242, 244, 248 },
        .bg_panel       = { 255, 255, 255 },
        .bg_panel_alt   = { 247, 248, 251 },
        .bg_header      = { 225, 230, 238 },
        .bg_selected    = { 200, 215, 245 },
        .bg_related     = { 215, 235, 220 },
        .bg_frozen      = { 170, 200, 245 },
        .bg_hover       = { 232, 238, 248 },
        .bg_pid_match   = { 190, 240, 190 },
        .bg_stale       = { 250, 235, 200 },
        .bg_new_item    = { 210, 240, 225 },
        .bg_loop        = { 240, 225, 245 },
        .bg_loop_shared = { 250, 230, 205 },

        .fg_title      = { 25, 65, 140 },
        .fg_dim        = { 115, 125, 140 },
        .fg_text       = { 35, 42, 54 },
        .fg_bright     = { 10, 15, 25 },
        .fg_muted      = { 150, 160, 175 },
        .fg_stream     = { 0, 125, 145 },
        .fg_fps        = { 30, 80, 180 },
        .fg_proc       = { 120, 50, 170 },
        .fg_stream_hdr = { 40, 140, 160 },
        .fg_fps_hdr    = { 50, 100, 190 },
        .fg_proc_hdr   = { 130, 70, 180 },
        .fg_active     = { 20, 140, 40 },
        .fg_idle       = { 110, 120, 135 },

        .anim_pulse_bg_min = { 255, 210, 210 },
        .anim_pulse_bg_max = { 255, 150, 150 },
        .anim_pulse_fg_min = { 160, 20, 20 },
        .anim_pulse_fg_max = { 100, 0, 0 },
        .fg_warn           = { 185, 105, 0 },
        .fg_error          = { 200, 30, 30 },
        .fg_zombie         = { 150, 80, 20 },

        .fg_conn        = { 70, 95, 140 },
        .fg_edge_active = { 20, 70, 180 },
        .fg_loop        = { 150, 40, 180 },
        .fg_loop_shared = { 180, 90, 10 },

        .grad_lo     = { 140, 180, 230 },
        .grad_hi     = { 30, 80, 180 },
        .grad_cpu_lo = { 30, 150, 50 },
        .grad_cpu_hi = { 210, 40, 40 },
    },

    /* -----------------------------------------------------
     * 4. Nordic Slate (Cool arctic dark palette)
     * ----------------------------------------------------- */
    {
        .id   = "nordic",
        .name = "Nordic Slate",
        .desc = "Cool gray & frost blue arctic theme",

        .bg_terminal    = { 46, 52, 64 },
        .bg_panel       = { 59, 66, 82 },
        .bg_panel_alt   = { 52, 58, 73 },
        .bg_header      = { 67, 76, 94 },
        .bg_selected    = { 84, 102, 130 },
        .bg_related     = { 65, 88, 85 },
        .bg_frozen      = { 94, 129, 172 },
        .bg_hover       = { 76, 86, 106 },
        .bg_pid_match   = { 70, 120, 90 },
        .bg_stale       = { 95, 80, 55 },
        .bg_new_item    = { 75, 105, 95 },
        .bg_loop        = { 85, 70, 95 },
        .bg_loop_shared = { 95, 75, 60 },

        .fg_title      = { 136, 192, 208 },
        .fg_dim        = { 140, 150, 170 },
        .fg_text       = { 229, 233, 240 },
        .fg_bright     = { 236, 239, 244 },
        .fg_muted      = { 105, 115, 130 },
        .fg_stream     = { 143, 188, 187 },
        .fg_fps        = { 129, 161, 193 },
        .fg_proc       = { 180, 142, 173 },
        .fg_stream_hdr = { 115, 155, 154 },
        .fg_fps_hdr    = { 100, 130, 160 },
        .fg_proc_hdr   = { 145, 115, 140 },
        .fg_active     = { 163, 190, 140 },
        .fg_idle       = { 135, 145, 160 },

        .anim_pulse_bg_min = { 100, 45, 55 },
        .anim_pulse_bg_max = { 191, 97, 106 },
        .anim_pulse_fg_min = { 200, 140, 150 },
        .anim_pulse_fg_max = { 255, 230, 235 },
        .fg_warn           = { 235, 203, 139 },
        .fg_error          = { 191, 97, 106 },
        .fg_zombie         = { 208, 135, 112 },

        .fg_conn        = { 115, 135, 160 },
        .fg_edge_active = { 136, 192, 208 },
        .fg_loop        = { 180, 142, 173 },
        .fg_loop_shared = { 208, 135, 112 },

        .grad_lo     = { 94, 129, 172 },
        .grad_hi     = { 136, 192, 208 },
        .grad_cpu_lo = { 163, 190, 140 },
        .grad_cpu_hi = { 191, 97, 106 },
    },

    /* -----------------------------------------------------
     * 5. Dracula (Gothic dark slate with vibrant neon accents)
     * ----------------------------------------------------- */
    {
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
    },

    /* -----------------------------------------------------
     * 6. Solarized Dark (Precision cyan & amber dark palette)
     * ----------------------------------------------------- */
    {
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
    },

    /* -----------------------------------------------------
     * 7. Solarized Light (Warm cream & cyan light palette)
     * ----------------------------------------------------- */
    {
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
    },

    /* -----------------------------------------------------
     * 8. Monokai Pro (Warm charcoal with radiant neon accents)
     * ----------------------------------------------------- */
    {
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
    },

    /* -----------------------------------------------------
     * 9. Matrix Phosphor (Phosphor green on pure black)
     * ----------------------------------------------------- */
    {
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
    },
};

static const int ov_num_themes = (int) (sizeof(ov_themes) / sizeof(ov_themes[0]));

static int ov_theme_active_idx = 0;
const ov_theme_t *ov_active_theme = &ov_themes[0];

/* =========================================================
 * Public API implementation
 * ========================================================= */

/**
 * @brief Return total number of registered themes.
 */
int ov_theme_count(void)
{
    return ov_num_themes;
}

/**
 * @brief Retrieve a theme by index.
 *
 * @param index Theme index (0 to ov_theme_count() - 1)
 * @return const ov_theme_t* Pointer to theme or NULL if invalid
 */
const ov_theme_t *ov_theme_get(int index)
{
    if (index < 0 || index >= ov_num_themes)
    {
        return NULL;
    }
    return &ov_themes[index];
}

/**
 * @brief Retrieve active theme.
 */
const ov_theme_t *ov_theme_get_active(void)
{
    return ov_active_theme;
}

/**
 * @brief Retrieve active theme index.
 */
int ov_theme_get_active_index(void)
{
    return ov_theme_active_idx;
}

/**
 * @brief Find a theme by identifier or name.
 *
 * @param id Identifier string (e.g. "dark", "night", "light", "accessible", "nordic")
 * @return int Theme index or -1 if not found
 */
int ov_theme_find_by_id(const char *id)
{
    if (id == NULL || id[0] == '\0')
    {
        return -1;
    }

    for (int i = 0; i < ov_num_themes; i++)
    {
        if (strcasecmp(ov_themes[i].id, id) == 0 ||
            strcasecmp(ov_themes[i].name, id) == 0)
        {
            return i;
        }
    }

    /* Aliases */
    if (strcasecmp(id, "red") == 0 || strcasecmp(id, "obs-red") == 0)
    {
        return 1; /* night */
    }
    if (strcasecmp(id, "cvd") == 0 || strcasecmp(id, "colorblind") == 0)
    {
        return 2; /* accessible */
    }
    if (strcasecmp(id, "paper") == 0 || strcasecmp(id, "day") == 0)
    {
        return 3; /* light */
    }
    if (strcasecmp(id, "nord") == 0)
    {
        return 4; /* nordic */
    }
    if (strcasecmp(id, "drac") == 0)
    {
        return 5; /* dracula */
    }
    if (strcasecmp(id, "sol-dark") == 0 || strcasecmp(id, "soldark") == 0)
    {
        return 6; /* solarized-dark */
    }
    if (strcasecmp(id, "sol-light") == 0 || strcasecmp(id, "sollight") == 0)
    {
        return 7; /* solarized-light */
    }
    if (strcasecmp(id, "monokai-pro") == 0)
    {
        return 8; /* monokai */
    }
    if (strcasecmp(id, "green") == 0 || strcasecmp(id, "cyber") == 0 ||
        strcasecmp(id, "hacker") == 0)
    {
        return 9; /* matrix */
    }

    return -1;
}

/**
 * @brief Set the active theme by index.
 *
 * @param index Target theme index
 */
void ov_theme_set(int index)
{
    if (index < 0 || index >= ov_num_themes)
    {
        return;
    }

    ov_theme_active_idx = index;
    ov_active_theme     = &ov_themes[index];

    ov_rgb_t tbg   = ov_active_theme->bg_terminal;
    ov__default_bg = OV_COLOR_TRUE | ((tbg.r & 0xFF) << 16) | ((tbg.g & 0xFF) << 8) |
                     (tbg.b & 0xFF);

    /* Invalidate front delta cache so full screen repaints with new palette */
    ov_buf_force_clear();
}

/**
 * @brief Cycle to the next theme.
 */
void ov_theme_cycle(void)
{
    int next = (ov_theme_active_idx + 1) % ov_num_themes;
    ov_theme_set(next);
}

/**
 * @brief Initialize theme subsystem from CLI arg or environment.
 *
 * Defaults to the standard dark color theme (index 0).
 *
 * @param preferred_theme CLI argument or NULL
 */
void ov_theme_init(const char *preferred_theme)
{
    int idx = -1;

    /* 1. CLI argument override */
    if (preferred_theme != NULL && preferred_theme[0] != '\0')
    {
        idx = ov_theme_find_by_id(preferred_theme);
    }

    /* 2. Environment variable MILK_CTRL_THEME */
    if (idx < 0)
    {
        const char *env_theme = getenv("MILK_CTRL_THEME");
        if (env_theme != NULL && env_theme[0] != '\0')
        {
            idx = ov_theme_find_by_id(env_theme);
        }
    }

    /* 3. Default to index 0 (Default Dark) */
    if (idx < 0)
    {
        idx = 0;
    }

    ov_theme_active_idx = idx;
    ov_active_theme     = &ov_themes[idx];

    ov_rgb_t tbg   = ov_active_theme->bg_terminal;
    ov__default_bg = OV_COLOR_TRUE | ((tbg.r & 0xFF) << 16) | ((tbg.g & 0xFF) << 8) |
                     (tbg.b & 0xFF);
}
