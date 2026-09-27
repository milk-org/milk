// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_help_data.c
 * @brief   Command reference database and introductory guide data for milk-CTRL.
 */

#include "overview_help_data.h"

/* clang-format off */
const help_entry_t g_help_entries[] =
{
#include "overview_help_data_nav.inc"
#include "overview_help_data_panels.inc"
};
/* clang-format on */

const int g_help_total = (int) (sizeof(g_help_entries) / sizeof(g_help_entries[0]));

/* clang-format off */
const intro_item_t g_intro_items[] = {
    { IL_HEADER,    "■ WHAT IS milk-CTRL?", NULL },
    { IL_TEXT,      NULL, "milk-CTRL is the unified, real-time diagnostic and control dashboard" },
    { IL_TEXT,      NULL, "for the milk (Modular Image Processing Library Kernel) framework." },
    { IL_TEXT,      NULL, "It provides a high-performance, flicker-free terminal interface" },
    { IL_TEXT,      NULL, "specially designed for mission-critical astronomical adaptive optics" },
    { IL_TEXT,      NULL, "(AO), wavefront control, and image processing pipelines." },
    { IL_BLANK,     NULL, NULL },
    { IL_TEXT,      NULL, "Built on zero-copy shared memory, milk-CTRL aggregates live system" },
    { IL_TEXT,      NULL, "state into a single dashboard without adding IPC overhead or latency." },
    { IL_BLANK,     NULL, NULL },

    { IL_HEADER,    "■ THE THREE CORE PILLARS", NULL },
    { IL_SUBHEADER, "1. ImageStreamIO (Streams — STRM / F3)", NULL },
    { IL_BULLET,    "• Zero-Copy SHM: ", "Circular buffers & image arrays in /milk/shm/." },
    { IL_BULLET,    "• Ultra-High Speed: ", "Transfers multi-dimensional data up to 10+ kHz." },
    { IL_BULLET,    "• Data Diversity: ", "Supports FLOAT, DOUBLE, UINT16, INT32, etc." },
    { IL_BULLET,    "• Synchronization: ", "POSIX read/write semaphores coordinate clients." },
    { IL_BLANK,     NULL, NULL },

    { IL_SUBHEADER, "2. FPS — Function Processing System (FPS — FPS / F5)", NULL },
    { IL_BULLET,    "• Compute Units: ", "Modular compute engines for algorithms & AO." },
    { IL_BULLET,    "• Tmux Isolation: ", "Runs compute tasks in isolated tmux sessions." },
    { IL_BULLET,    "• Dual Loops: ", "Separates parameter config (conf) from processing (run)." },
    { IL_BULLET,    "• Parameter Trees: ", "Hierarchical parameter directories editable live." },
    { IL_BLANK,     NULL, NULL },

    { IL_SUBHEADER, "3. processinfo (Processes — PROC / F4)", NULL },
    { IL_BULLET,    "• Telemetry: ", "Heartbeat registration, loop frequencies (Hz) & jitter." },
    { IL_BULLET,    "• Resources: ", "CPU core pinning masks, RSS memory, & cycle times." },
    { IL_BULLET,    "• Lifecycle: ", "Tracks RUNNING, PAUSED, STOPPED, & CRASHED states." },
    { IL_BLANK,     NULL, NULL },

    { IL_HEADER,    "■ WHAT CAN milk-CTRL DO?", NULL },
    { IL_BULLET,    "• Multi-Panel (F2): ", "View Streams, Procs, FPS, & Graph together." },
    { IL_BULLET,    "• Full Monitors (F3-F5): ", "Deep-dive into streams, telemetry, or FPS." },
    { IL_BULLET,    "• Data Lineage (F6): ", "Maps upstream producers & downstream consumers." },
    { IL_BULLET,    "• Feedback Loops (LOOPS/F7): ", "Detects closed directed loops (WFS -> DM)." },
    { IL_BULLET,    "• Control Mode ('c'): ", "Start/stop FPS, signal procs, or delete streams." },
    { IL_BULLET,    "• Parameter Editing: ", "Browse and edit FPS variables with limits check." },
    { IL_BULLET,    "• Filtering & Search: ", "Regex filter ('/'), toggle ('f'), freeze (SPACE)." },
    { IL_BULLET,    "• Snapshots & Logging: ", "Save state snapshots ('W') & command ring ('G')." },
    { IL_BLANK,     NULL, NULL },

    { IL_HEADER,    "■ QUICK START & KEYSTROKES", NULL },
    { IL_KEY,       "  1 / i", "Toggle this Introduction guide on or off" },
    { IL_KEY,       "  2 / k", "Switch to the Controls & Keybindings reference" },
    { IL_KEY,       "  F2 - F7", "Switch view panels (DASH, STRM, PROC, FPS, CONN, LOOPS)" },
    { IL_KEY,       "  TAB", "Cycle active panel focus (Streams -> Procs -> FPS -> Graph)" },
    { IL_KEY,       "  ↑ / ↓ (j / k)", "Navigate items in focused list, or scroll this guide" },
    { IL_KEY,       "  c", "Toggle Control Mode ON to enable management actions" },
    { IL_KEY,       "  /", "Search topics in help, or regex filter in dashboard" },
    { IL_KEY,       "  f", "Toggle regex filter ON/OFF for focused panel" },
    { IL_KEY,       "  SPACE", "Freeze selection highlight during rapid live updates" },
    { IL_KEY,       "  ESC", "Close help overlay or exit current prompt" },
    { IL_KEY,       "  q / x", "Quit milk-CTRL cleanly" },
};
/* clang-format on */

const int g_intro_total = (int) (sizeof(g_intro_items) / sizeof(g_intro_items[0]));

/**
 * ov_help_nb_sections - get the total number of help sections.
 *
 * Return: Number of help sections (HS_COUNT).
 */
int ov_help_nb_sections(void)
{
    return HS_COUNT;
}

/**
 * ov_help_section_name - get human-readable name of a section.
 * @sec: Section index.
 *
 * Return: Short section name string.
 */
const char *ov_help_section_name(
    int sec)
{
    switch (sec)
    {
    case HS_INTRO:
        return "Intro";
    case HS_NAV:
        return "Global & Nav";
    case HS_STREAMS:
        return "Streams";
    case HS_PROCS:
        return "Processes";
    case HS_FPS:
        return "FPS";
    case HS_GRAPH:
        return "Graph";
    case HS_CMDLOG:
        return "CmdLog";
    case HS_MOUSE:
        return "Mouse";
    case HS_COLORS:
        return "Colors";
    default:
        return "Help";
    }
}

/**
 * ov_help_focus_section - map layout focus to its corresponding help section.
 * @focus: Active panel focus enum.
 *
 * Return: Corresponding section index (HS_STREAMS, HS_PROCS, etc.).
 */
int ov_help_focus_section(
    ov_focus_t focus)
{
    switch (focus)
    {
    case OV_FOCUS_STREAMS:
        return HS_STREAMS;
    case OV_FOCUS_PROCS:
        return HS_PROCS;
    case OV_FOCUS_FPS:
        return HS_FPS;
    case OV_FOCUS_GRAPH:
        return HS_GRAPH;
    default:
        return HS_NAV;
    }
}

/**
 * ov_help_section_tag - return short 4-letter panel tag for search results.
 * @sec: Section index.
 *
 * Return: Short uppercase badge string.
 */
const char *ov_help_section_tag(
    int sec)
{
    switch (sec)
    {
    case HS_INTRO:
        return "INTR";
    case HS_NAV:
        return "NAV";
    case HS_STREAMS:
        return "STRM";
    case HS_PROCS:
        return "PROC";
    case HS_FPS:
        return "FPS";
    case HS_GRAPH:
        return "CONN";
    case HS_CMDLOG:
        return "LOG";
    case HS_MOUSE:
        return "MOUS";
    case HS_COLORS:
        return "COLR";
    default:
        return "HELP";
    }
}

/**
 * ov_help_section_color - return semantic theme color for section badge.
 * @sec: Section index.
 *
 * Return: Theme ov_rgb_t color.
 */
ov_rgb_t ov_help_section_color(
    int sec)
{
    switch (sec)
    {
    case HS_INTRO:
        return (ov_rgb_t) { 255, 215, 80 };
    case HS_NAV:
        return OV_FG_TITLE;
    case HS_STREAMS:
        return OV_FG_STREAM;
    case HS_PROCS:
        return OV_FG_PROC;
    case HS_FPS:
        return OV_FG_FPS;
    case HS_GRAPH:
        return OV_FG_CONN;
    case HS_CMDLOG:
        return (ov_rgb_t) { 180, 210, 170 };
    case HS_MOUSE:
        return OV_FG_WARN;
    case HS_COLORS:
        return (ov_rgb_t) { 230, 130, 255 };
    default:
        return OV_FG_DIM;
    }
}
