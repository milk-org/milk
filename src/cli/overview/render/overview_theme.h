// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_theme.h
 * @brief Theme definitions and palette management for milk-CTRL.
 */

#ifndef OVERVIEW_THEME_H
#define OVERVIEW_THEME_H

#include "overview_ansi.h"
#include "overview_data.h"

/* RGB color struct */

typedef struct
{
    int r;
    int g;
    int b;
} ov_rgb_t;

/* Theme structure & palette */

typedef struct
{
    const char *id;
    const char *name;
    const char *desc;

    /* Panel backgrounds */
    ov_rgb_t bg_terminal;
    ov_rgb_t bg_panel;
    ov_rgb_t bg_panel_alt;
    ov_rgb_t bg_header;
    ov_rgb_t bg_selected;
    ov_rgb_t bg_related;
    ov_rgb_t bg_frozen;
    ov_rgb_t bg_hover;
    ov_rgb_t bg_pid_match;
    ov_rgb_t bg_stale;
    ov_rgb_t bg_new_item;
    ov_rgb_t bg_loop;
    ov_rgb_t bg_loop_shared;

    /* Foreground — text */
    ov_rgb_t fg_title;
    ov_rgb_t fg_dim;
    ov_rgb_t fg_text;
    ov_rgb_t fg_bright;
    ov_rgb_t fg_muted;

    /* Foreground — node types */
    ov_rgb_t fg_stream;
    ov_rgb_t fg_fps;
    ov_rgb_t fg_proc;

    /* Dimmed accent colors for column headers */
    ov_rgb_t fg_stream_hdr;
    ov_rgb_t fg_fps_hdr;
    ov_rgb_t fg_proc_hdr;

    /* Foreground — status */
    ov_rgb_t fg_active;
    ov_rgb_t fg_idle;

    /* Animation pulse parameters & status fg */
    ov_rgb_t anim_pulse_bg_min;
    ov_rgb_t anim_pulse_bg_max;
    ov_rgb_t anim_pulse_fg_min;
    ov_rgb_t anim_pulse_fg_max;
    ov_rgb_t fg_warn;
    ov_rgb_t fg_error;
    ov_rgb_t fg_zombie;

    /* Foreground — graph & loops */
    ov_rgb_t fg_conn;
    ov_rgb_t fg_edge_active;
    ov_rgb_t fg_loop;
    ov_rgb_t fg_loop_shared;

    /* Gradient endpoints for bars/sparklines */
    ov_rgb_t grad_lo;
    ov_rgb_t grad_hi;
    ov_rgb_t grad_cpu_lo;
    ov_rgb_t grad_cpu_hi;
} ov_theme_t;

extern const ov_theme_t *ov_active_theme;

/* Theme API declarations */
int               ov_theme_count(void);
const ov_theme_t *ov_theme_get(int index);
const ov_theme_t *ov_theme_get_active(void);
int               ov_theme_get_active_index(void);
int               ov_theme_find_by_id(const char *id);
void              ov_theme_set(int index);
void              ov_theme_cycle(void);
void              ov_theme_init(const char *preferred_theme);

/* Semantic color palette (maps to active theme) */

/* Panel backgrounds */
#define OV_BG_TERMINAL (ov_active_theme->bg_terminal)
#define OV_BG_PANEL (ov_active_theme->bg_panel)
#define OV_BG_PANEL_ALT (ov_active_theme->bg_panel_alt)
#define OV_BG_HEADER (ov_active_theme->bg_header)
#define OV_BG_SELECTED (ov_active_theme->bg_selected)
#define OV_BG_RELATED (ov_active_theme->bg_related)
#define OV_BG_FROZEN (ov_active_theme->bg_frozen)
#define OV_BG_HOVER (ov_active_theme->bg_hover)
#define OV_BG_PID_MATCH (ov_active_theme->bg_pid_match)
#define OV_BG_STALE (ov_active_theme->bg_stale)
#define OV_BG_NEW_ITEM (ov_active_theme->bg_new_item)
#define OV_BG_LOOP (ov_active_theme->bg_loop)
#define OV_BG_LOOP_SHARED (ov_active_theme->bg_loop_shared)

/* Foreground — text */
#define OV_FG_TITLE (ov_active_theme->fg_title)
#define OV_FG_DIM (ov_active_theme->fg_dim)
#define OV_FG_TEXT (ov_active_theme->fg_text)
#define OV_FG_BRIGHT (ov_active_theme->fg_bright)
#define OV_FG_MUTED (ov_active_theme->fg_muted)

/* Foreground — node types */
#define OV_FG_STREAM (ov_active_theme->fg_stream)
#define OV_FG_FPS (ov_active_theme->fg_fps)
#define OV_FG_PROC (ov_active_theme->fg_proc)

/* Dimmed accent colors for column headers */
#define OV_FG_STREAM_HDR (ov_active_theme->fg_stream_hdr)
#define OV_FG_FPS_HDR (ov_active_theme->fg_fps_hdr)
#define OV_FG_PROC_HDR (ov_active_theme->fg_proc_hdr)

/* Foreground — status */
#define OV_FG_ACTIVE (ov_active_theme->fg_active)
#define OV_FG_IDLE (ov_active_theme->fg_idle)

/* Animation Parameters */
#define OV_ANIM_PULSE_SPEED 0.15f
#define OV_ANIM_PULSE_BG_MIN (ov_active_theme->anim_pulse_bg_min)
#define OV_ANIM_PULSE_BG_MAX (ov_active_theme->anim_pulse_bg_max)
#define OV_ANIM_PULSE_FG_MIN (ov_active_theme->anim_pulse_fg_min)
#define OV_ANIM_PULSE_FG_MAX (ov_active_theme->anim_pulse_fg_max)
#define OV_FG_WARN (ov_active_theme->fg_warn)
#define OV_FG_ERROR (ov_active_theme->fg_error)
#define OV_FG_ZOMBIE (ov_active_theme->fg_zombie)

/* Foreground — graph & loops */
#define OV_FG_CONN (ov_active_theme->fg_conn)
#define OV_FG_EDGE_ACTIVE (ov_active_theme->fg_edge_active)
#define OV_FG_LOOP (ov_active_theme->fg_loop)
#define OV_FG_LOOP_SHARED (ov_active_theme->fg_loop_shared)

/* Gradient endpoints for bars/sparklines */
#define OV_GRAD_LO (ov_active_theme->grad_lo)
#define OV_GRAD_HI (ov_active_theme->grad_hi)

#define OV_GRAD_CPU_LO (ov_active_theme->grad_cpu_lo)
#define OV_GRAD_CPU_HI (ov_active_theme->grad_cpu_hi)

/* Borders & box-drawing */

#define OV_BOX_TL "╭"
#define OV_BOX_TR "╮"
#define OV_BOX_BL "╰"
#define OV_BOX_BR "╯"
#define OV_BOX_H "─"
#define OV_BOX_V "│"

/* Double borders for focused panels */
#define OV_BOX_TL_D "╔"
#define OV_BOX_TR_D "╗"
#define OV_BOX_BL_D "╚"
#define OV_BOX_BR_D "╝"
#define OV_BOX_H_D "═"
#define OV_BOX_V_D "║"
#define OV_BOX_LT "├"
#define OV_BOX_RT "┤"
#define OV_BOX_TB "┬"
#define OV_BOX_BT "┴"
#define OV_BOX_X "┼"

/* Arrow characters */
#define OV_ARROW_R "→"
#define OV_ARROW_L "←"
#define OV_ARROW_D "↓"
#define OV_ARROW_U "↑"
#define OV_TRI_R "▶"
#define OV_TRI_L "◀"
#define OV_TRI_D "▼"
#define OV_TRI_U "▲"
#define OV_BULLET "●"
#define OV_DIAMOND "◆"

/* LCARS block elements */
#define OV_LCARS_LEFT "▌"
#define OV_LCARS_RIGHT "▐"

/* Sparkline block characters (1/8 to full) */
static const char *OV_SPARK_CHARS[] = { " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" };
#define OV_SPARK_LEVELS 9

#include "overview_theme_draw.h"
#include "overview_theme_panel.h"

#endif /* OVERVIEW_THEME_H */
