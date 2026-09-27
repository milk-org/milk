// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_HELP_DATA_H
#define OVERVIEW_HELP_DATA_H

/**
 * @file overview_help_data.h
 * @brief Command reference database and introductory guide data for milk-CTRL
 */

#include "overview_defs.h"
#include "overview_theme.h"
#include "overview_layout.h"

/** Flag values for help entries */
#define HF_SECTION   1 /* Section header row                         */
#define HF_ENTRY     2 /* Standard keystroke command entry           */
#define HF_CTRL_MODE 4 /* Requires Control Mode ON (press 'c')       */
#define HF_COLORS    8 /* Render as theme color legend               */

/** Structure representing a single help topic or keystroke entry */
typedef struct
{
    const char *key;     /* Keystroke label (e.g. "F2 - F6", "k", "DEL") */
    const char *label;   /* Concise summary for list view                */
    const char *detail;  /* In-depth explanation for detail pane         */
    int         flags;   /* Bitmask of HF_* flags                        */
    int         section; /* Parent section index                         */
} help_entry_t;

/** Section indices matching GUI panels */
enum
{
    HS_INTRO = 0, /* Introduction & Overview    */
    HS_NAV,       /* Global & Navigation        */
    HS_STREAMS,   /* Streams Panel (STRM)       */
    HS_PROCS,     /* Processes Panel (PROC)     */
    HS_FPS,       /* FPS Panel (FPS)            */
    HS_GRAPH,     /* Graph & Lineage (CONN)     */
    HS_CMDLOG,    /* Command Log & Display      */
    HS_MOUSE,     /* Mouse Interactions         */
    HS_COLORS,    /* Theme & Colors             */
    HS_COUNT
};

/** Line types for introductory guide */
typedef enum
{
    IL_BLANK = 0,
    IL_HEADER,
    IL_SUBHEADER,
    IL_BULLET,
    IL_TEXT,
    IL_KEY
} intro_line_type_t;

/** Structure representing a line in the introductory guide */
typedef struct
{
    intro_line_type_t type;
    const char       *prefix;
    const char       *text;
} intro_item_t;

/* Global data references */
extern const help_entry_t g_help_entries[];
extern const int          g_help_total;

extern const intro_item_t g_intro_items[];
extern const int          g_intro_total;

/**
 * ov_help_nb_sections - get the total number of help sections.
 *
 * Return: Number of sections (HS_COUNT).
 */
int ov_help_nb_sections(void);

/**
 * ov_help_section_name - get human-readable name of a section.
 * @sec: Section index.
 *
 * Return: Section title string.
 */
const char *ov_help_section_name(
    int sec);

/**
 * ov_help_focus_section - map layout focus to its corresponding help section.
 * @focus: Active panel focus enum.
 *
 * Return: Corresponding section index (HS_STREAMS, HS_PROCS, etc.).
 */
int ov_help_focus_section(
    ov_focus_t focus);

/**
 * ov_help_section_tag - get short 3-4 letter badge for section.
 * @sec: Section index.
 *
 * Return: Badge string (e.g. "STRM", "PROC", "FPS").
 */
const char *ov_help_section_tag(
    int sec);

/**
 * ov_help_section_color - get semantic theme color for section.
 * @sec: Section index.
 *
 * Return: ov_rgb_t theme color.
 */
ov_rgb_t ov_help_section_color(
    int sec);

#endif /* OVERVIEW_HELP_DATA_H */
