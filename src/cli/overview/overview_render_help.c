// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_help.c
 * @brief   Structured, panel-grouped help overlay with detailed contextual help
 *
 * Organizes help into sections that match the milk-CTRL GUI panels.
 * Keystrokes are rendered with standardized bold formatting and a distinct
 * shade for commands requiring Control Mode ON. When an item is selected,
 * a dedicated detail pane displays in-depth explanations and live target
 * information for the currently selected GUI item.
 */

#include "overview_render_internal.h"

#include "overview_help_data.h"

/**
 * help_is_expanded - check if a section is expanded.
 * @lay: layout state
 * @sec: section index (HS_NAV .. HS_COLORS)
 *
 * Return: 1 if expanded, 0 if collapsed.
 */
static inline int help_is_expanded(const OV_LAYOUT *lay, int sec)
{
    return (lay->help_expand >> sec) & 1;
}

typedef struct
{
    int index; /* Index in g_help_entries[] array */
    int score; /* Composite relevance score */
} help_search_match_t;

/**
 * compare_search_matches - qsort comparator for help search results by relevance score.
 * @a: Pointer to first help_search_match_t.
 * @b: Pointer to second help_search_match_t.
 *
 * Return: Negative if a > b (descending score), positive if b > a, or tie-break on index.
 */
static int compare_search_matches(const void *a, const void *b)
{
    const help_search_match_t *ma = (const help_search_match_t *) a;
    const help_search_match_t *mb = (const help_search_match_t *) b;
    if (mb->score != ma->score)
    {
        return mb->score - ma->score; /* descending score */
    }
    return ma->index - mb->index; /* stable tie-breaker */
}

/**
 * help_score_entry - score a help entry against a search query.
 * @entry: help entry to evaluate
 * @query: user search string
 *
 * Supports multi-token queries where all tokens must match (AND-logic).
 *
 * Return: score >= 0 (0 means no match).
 */
static int help_score_entry(const help_entry_t *entry, const char *query)
{
    if (entry == NULL || query == NULL || query[0] == '\0')
    {
        return 0;
    }

    char qbuf[64];
    strncpy(qbuf, query, sizeof(qbuf) - 1);
    qbuf[sizeof(qbuf) - 1] = '\0';

    char *tokens[8];
    int   ntok    = 0;
    char *saveptr = NULL;
    char *tok     = strtok_r(qbuf, " \t", &saveptr);
    while (tok != NULL && ntok < 8)
    {
        tokens[ntok++] = tok;
        tok            = strtok_r(NULL, " \t", &saveptr);
    }

    if (ntok == 0)
    {
        return 0;
    }

    int         total_score = 0;
    const char *sec_name    = ov_help_section_name(entry->section);
    const char *sec_tag     = ov_help_section_tag(entry->section);

    for (int t = 0; t < ntok; t++)
    {
        const char *w    = tokens[t];
        int         wlen = (int) strlen(w);
        if (wlen == 0)
        {
            continue;
        }

        int tok_score = 0;

        /* 1. Keystroke exact or prefix match */
        if (entry->key != NULL)
        {
            if (strcasecmp(entry->key, w) == 0)
            {
                tok_score += 350; /* Exact match on key, e.g. "k" or "F2" */
            }
            else if (strncasecmp(entry->key, w, (size_t) wlen) == 0)
            {
                tok_score += 180; /* Key prefix match */
            }
            else if (strcasestr(entry->key, w) != NULL)
            {
                tok_score += 90;
            }
        }

        /* 2. Label match (primary title summary) */
        if (entry->label != NULL)
        {
            const char *p = strcasestr(entry->label, w);
            if (p != NULL)
            {
                int at_boundary = (p == entry->label || *(p - 1) == ' ' || *(p - 1) == '/' ||
                                   *(p - 1) == '(' || *(p - 1) == '[' || *(p - 1) == '-');
                if (at_boundary)
                {
                    if (p[wlen] == '\0' || p[wlen] == ' ' || p[wlen] == '/' || p[wlen] == ')' ||
                        p[wlen] == ']')
                    {
                        tok_score += 160;
                    }
                    else
                    {
                        tok_score += 120;
                    }
                }
                else
                {
                    tok_score += 60;
                }
            }
        }

        /* 3. Section/topic match */
        if (sec_name != NULL && strcasestr(sec_name, w) != NULL)
        {
            tok_score += 70;
        }
        if (sec_tag != NULL && strcasecmp(sec_tag, w) == 0)
        {
            tok_score += 80;
        }

        /* 4. Detail documentation match */
        if (entry->detail != NULL)
        {
            const char *p = strcasestr(entry->detail, w);
            if (p != NULL)
            {
                tok_score += 35;
            }
        }

        /* Every token must match somewhere (AND logic) */
        if (tok_score == 0)
        {
            return 0;
        }

        total_score += tok_score;
    }

    /* Command entries get a priority boost over section headers */
    if (entry->flags & HF_ENTRY)
    {
        total_score += 15;
    }

    return total_score;
}

/**
 * help_visible_rows - count visible rows and populate mapping array.
 * @lay: layout state (for expand bitmask or active search query)
 * @map: output array mapping visible row index to g_help_entries[] index
 *
 * Return: number of visible rows.
 */
static int help_visible_rows(const OV_LAYOUT *lay, int *map)
{
    /* If search query is non-empty, populate map with ranked search matches */
    if (lay->help_search[0] != '\0')
    {
        help_search_match_t matches[128];
        int                 n_matches = 0;

        for (int i = 0; i < g_help_total; i++)
        {
            int s = help_score_entry(&g_help_entries[i], lay->help_search);
            if (s > 0 && n_matches < 128)
            {
                matches[n_matches].index = i;
                matches[n_matches].score = s;
                n_matches++;
            }
        }

        if (n_matches > 1)
        {
            qsort(matches, (size_t) n_matches, sizeof(help_search_match_t), compare_search_matches);
        }

        for (int i = 0; i < n_matches; i++)
        {
            map[i] = matches[i].index;
        }
        return n_matches;
    }

    int vis = 0;
    for (int i = 0; i < g_help_total; i++)
    {
        if (g_help_entries[i].flags & HF_SECTION)
        {
            map[vis++] = i;
        }
        else if (g_help_entries[i].flags & (HF_ENTRY | HF_COLORS))
        {
            if (help_is_expanded(lay, g_help_entries[i].section))
            {
                map[vis++] = i;
            }
        }
    }
    return vis;
}

/**
 * ov_help_visible_count - public helper returning number of visible rows.
 * @lay: layout state
 *
 * Return: visible row count.
 */
int ov_help_visible_count(const OV_LAYOUT *lay)
{
    int map[128];
    return help_visible_rows(lay, map);
}

/**
 * ov_help_section_first_vis_row - find visible row index for a section header.
 * @lay: layout state
 * @sec: section index
 *
 * Return: visible row index in map[], or 0 if not found.
 */
int ov_help_section_first_vis_row(const OV_LAYOUT *lay, int sec)
{
    int map[128];
    int nvis = help_visible_rows(lay, map);
    for (int vr = 0; vr < nvis; vr++)
    {
        int idx = map[vr];
        if (g_help_entries[idx].section == sec && (g_help_entries[idx].flags & HF_SECTION))
        {
            return vr;
        }
    }
    return 0;
}

/**
 * ov_help_open - open help overlay with contextual initial focus and expansion.
 * @lay: layout state (modified)
 */
void ov_help_open(OV_LAYOUT *lay)
{
    lay->show_help          = 1;
    lay->help_expand        = 0;
    lay->filter_editing     = 0;
    lay->help_search[0]     = '\0';
    lay->help_search_active = 0;
    lay->help_search_cursor = 0;
    lay->help_mode          = 0;
    lay->help_intro_scroll  = 0;
    lay->help_sel           = 0;
}

/**
 * ov_help_toggle_at - toggle expansion for section header at visible row.
 * @lay:     layout state (help_expand bitmask modified)
 * @vis_row: 0-based visible row index
 *
 * Return: section index if toggled, or -1 if row is not a section header.
 */
int ov_help_toggle_at(OV_LAYOUT *lay, int vis_row)
{
    if (lay->help_search[0] != '\0')
    {
        return -1;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    if (vis_row < 0 || vis_row >= nvis)
    {
        return -1;
    }

    int idx = map[vis_row];
    if (!(g_help_entries[idx].flags & HF_SECTION))
    {
        return -1;
    }

    int sec = g_help_entries[idx].section;
    lay->help_expand ^= (1U << sec);
    return sec;
}

/**
 * ov_help_expand_at - expand or collapse section at visible row.
 * @lay:     layout state (help_expand bitmask modified)
 * @vis_row: 0-based visible row index
 * @expand:  1 to expand, 0 to collapse
 *
 * Return: section index if modified, or -1 if row is not eligible.
 */
int ov_help_expand_at(OV_LAYOUT *lay, int vis_row, int expand)
{
    if (lay->help_search[0] != '\0')
    {
        return -1;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    if (vis_row < 0 || vis_row >= nvis)
    {
        return -1;
    }

    int idx = map[vis_row];
    int sec = g_help_entries[idx].section;

    if (expand)
    {
        if (g_help_entries[idx].flags & HF_SECTION)
        {
            if (!help_is_expanded(lay, sec))
            {
                lay->help_expand |= (1U << sec);
                return sec;
            }
            else
            {
                int new_nvis = ov_help_visible_count(lay);
                if (lay->help_sel + 1 < new_nvis)
                {
                    lay->help_sel++;
                }
                return sec;
            }
        }
    }
    else
    {
        if (g_help_entries[idx].flags & HF_SECTION)
        {
            if (help_is_expanded(lay, sec))
            {
                lay->help_expand &= ~(1U << sec);
                return sec;
            }
        }
        else
        {
            /* On child item: collapse parent section and land on section header */
            lay->help_expand &= ~(1U << sec);
            lay->help_sel = ov_help_section_first_vis_row(lay, sec);
            return sec;
        }
    }

    return -1;
}

/**
 * ov_help_get_rect - compute bounding box for help overlay.
 * @lay: layout state
 * @pr:  output top row (1-based)
 * @pc:  output left column (1-based)
 * @ph:  output height in rows
 * @pw:  output width in columns
 *
 * Takes the whole available terminal space: starts below the dedicated tab bar
 * (row 3) and extends across the entire terminal width to the row above the status bar.
 */
static void ov_help_get_rect(const OV_LAYOUT *lay, int *pr, int *pc, int *ph, int *pw)
{
    int W = lay->term_cols;
    int H = lay->term_rows;

    *pc = 1;
    *pw = W;

    if (H <= 6)
    {
        *pr = 1;
        *ph = H;
    }
    else
    {
        *pr = 3;
        *ph = H - 3;
    }
}

/**
 * ov_help_handle_click - handle mouse click within help overlay.
 * @lay: layout state
 * @mr:  clicked terminal row (1-based)
 * @mc:  clicked terminal column (1-based)
 *
 * Return: 1 if click was handled inside help overlay, 0 otherwise.
 */
int ov_help_handle_click(OV_LAYOUT *lay, int mr, int mc)
{
    int pr, pc, ph, pw;
    ov_help_get_rect(lay, &pr, &pc, &ph, &pw);

    /* Click outside popup -> close help overlay */
    if (mr < pr || mr >= pr + ph || mc < pc || mc >= pc + pw)
    {
        lay->show_help = 0;
        ov_buf_force_clear();
        if (mr == lay->r_tabs.row)
        {
            int tx = 1;
            for (int v = 0; v < OV_VIEW_COUNT; v++)
            {
                int tw = (int) strlen(ov_view_label((ov_view_t) v)) + 9;
                if (mc >= tx && mc < tx + tw)
                {
                    lay->view = (ov_view_t) v;
                    break;
                }
                tx += tw;
            }
        }
        return 1;
    }

    /* Click close button area on top border or header row */
    if ((mr == pr && mc >= pc + pw - 6) || (mr == pr + 1 && mc >= pc + pw - 6))
    {
        lay->show_help = 0;
        ov_buf_force_clear();
        return 1;
    }

    /* Click on header row (pr + 1): Mode selector tabs */
    if (mr == pr + 1)
    {
        int tab1_w = (pw >= 100) ? 26 : 14;
        int tab2_w = (pw >= 100) ? 32 : 20;
        if (mc >= pc + 2 && mc < pc + 2 + tab1_w)
        {
            lay->help_mode         = 1;
            lay->help_intro_scroll = 0;
            return 1;
        }
        if (mc >= pc + 2 + tab1_w && mc < pc + 2 + tab1_w + tab2_w)
        {
            lay->help_mode = 0;
            return 1;
        }
    }

    /* If in Intro mode (help_mode == 1), clicks in body do not alter list */
    if (lay->help_mode == 1)
    {
        return 1;
    }

    /* Click on search bar area on row pr + 2 (in Controls mode) */
    if (mr == pr + 2 && mc >= pc + 1 && mc < pc + pw - 1)
    {
        if (lay->help_search[0] != '\0' && mc >= pc + pw - 16)
        {
            /* Clicked on [ESC: clear] button */
            lay->help_search[0]     = '\0';
            lay->help_search_cursor = 0;
            lay->help_search_active = 0;
            lay->help_sel           = 0;
        }
        else
        {
            /* Clicked on search input box */
            lay->help_search_active = 1;
            lay->help_search_cursor = (int) strlen(lay->help_search);
        }
        return 1;
    }

    /* Detail pane height and split line */
    int detail_h = (ph >= 36) ? 10 : ((ph >= 28) ? 8 : ((ph >= 22) ? 6 : 5));
    int split_r  = (pr + ph - 1) - detail_h;
    int list_top = pr + 4;
    int list_h   = split_r - list_top;

    /* Check if click is inside the list area */
    if (mr >= list_top && mr < split_r)
    {
        int map[128];
        int nvis = help_visible_rows(lay, map);

        int sel = lay->help_sel;
        if (sel < 0)
        {
            sel = 0;
        }
        if (sel >= nvis)
        {
            sel = nvis - 1;
        }

        int scroll = 0;
        if (sel >= list_h)
        {
            scroll = sel - list_h + 1;
        }

        int vis_row = (mr - list_top) + scroll;
        if (vis_row >= 0 && vis_row < nvis)
        {
            int idx = map[vis_row];
            if (g_help_entries[idx].section == HS_INTRO &&
                (lay->help_sel == vis_row || mr == list_top))
            {
                /* Clicking on Introduction entry/header opens full intro guide */
                lay->help_mode         = 1;
                lay->help_intro_scroll = 0;
                return 1;
            }
            if (lay->help_search[0] == '\0' && lay->help_sel == vis_row)
            {
                /* Clicking selected header toggles expansion */
                ov_help_toggle_at(lay, vis_row);
            }
            else
            {
                lay->help_sel = vis_row;
            }
            return 1;
        }
    }

    return 0;
}

/**
 * ov_help_print_wrapped - word-wrap and print text within a specified bounding box.
 * @text:      text string to print
 * @row:       starting row
 * @col:       left column
 * @max_w:     maximum visible column width
 * @max_lines: maximum lines to print
 * @fg:        foreground color
 * @bg:        background color
 */
static void ov_help_print_wrapped(const char *text,
                                  int         row,
                                  int         col,
                                  int         max_w,
                                  int         max_lines,
                                  ov_rgb_t    fg,
                                  ov_rgb_t    bg)
{
    if (text == NULL || max_w <= 0 || max_lines <= 0)
    {
        return;
    }

    const char *p             = text;
    int         lines_printed = 0;

    while (*p != '\0' && lines_printed < max_lines)
    {
        /* Skip leading whitespace on new line */
        while (*p == ' ')
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }

        int len        = 0;
        int last_space = -1;
        while (p[len] != '\0' && p[len] != '\n' && len < max_w)
        {
            if (p[len] == ' ')
            {
                last_space = len;
            }
            len++;
        }

        int line_len = len;
        if (p[len] == '\n')
        {
            line_len = len;
            len++;
        }
        else if (p[len] != '\0' && last_space > 0)
        {
            line_len = last_space;
            len      = last_space + 1;
        }

        ov_buf_pos(row + lines_printed, col);
        ov_theme_bg(bg);
        ov_theme_fg(fg);
        ov_buf_printf("%.*s", line_len, p);

        int pad = max_w - line_len;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }

        p += len;
        lines_printed++;
    }

    /* Clear any remaining allocated lines */
    for (int r = lines_printed; r < max_lines; r++)
    {
        ov_buf_pos(row + r, col);
        ov_theme_bg(bg);
        ov_buf_hline(' ', max_w);
    }
}

/**
 * ov_help_render_detail - render in-depth help and target info for selected item.
 * @lay:      layout state
 * @m:        system model
 * @entry:    currently highlighted help entry
 * @split_r:  row of separator line
 * @pc:       left column of help box
 * @pw:       width of help box
 * @detail_h: height of detail pane
 */
static void ov_help_render_detail(const OV_LAYOUT    *lay,
                                  const OV_MODEL     *m,
                                  const help_entry_t *entry,
                                  int                 split_r,
                                  int                 pc,
                                  int                 pw,
                                  int                 detail_h)
{
    int inner_w = pw - 4;
    int col     = pc + 2;

    /* Row 1: Header (Key / Command title + Status badges) */
    ov_buf_pos(split_r + 1, col);
    ov_theme_bg(OV_BG_PANEL);

    if (entry->flags & HF_SECTION)
    {
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        if (entry->section == HS_INTRO)
        {
            ov_buf_printf("■ INTRODUCTION: %s", entry->label);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            const char *hint = "[Press 1, 'i', or ENTER to open full interactive guide]";
            int         rem  = inner_w - (16 + (int) strlen(entry->label) + (int) strlen(hint));
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("%s", hint);
        }
        else
        {
            ov_buf_printf("■ PANEL OVERVIEW: %s", entry->label);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            const char *hint = help_is_expanded(lay, entry->section)
                                   ? "[Press ← / ENTER to collapse]"
                                   : "[Press → / ENTER to expand]";
            int         rem  = inner_w - (18 + (int) strlen(entry->label) + (int) strlen(hint));
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%s", hint);
        }
    }
    else if (entry->flags & HF_ENTRY)
    {
        ov_buf_bold();
        if (entry->flags & HF_CTRL_MODE)
        {
            if (lay->ctrl_mode)
            {
                ov_buf_fg(255, 95, 75);
                ov_buf_printf("Key: [ %s ]", entry->key);
                ov_buf_fg(255, 80, 80);
                ov_buf_printf("  ⚡ CONTROL MODE: ON (Ready)");
            }
            else
            {
                ov_buf_fg(205, 140, 50);
                ov_buf_printf("Key: [ %s ]", entry->key);
                ov_buf_fg(205, 140, 50);
                ov_buf_printf("  🔒 REQUIRES CONTROL MODE (Press 'c')");
            }
        }
        else
        {
            ov_buf_fg(130, 205, 255);
            ov_buf_printf("Key: [ %s ]", entry->key);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("  (Standard Command)");
        }
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);

        /* Section scope badge on right */
        const char *sname   = ov_help_section_name(entry->section);
        int         key_len = (int) strlen(entry->key) + 9;
        int         badge_l = (int) strlen(sname) + 10;
        int         rem     = inner_w - (key_len + 35 + badge_l);
        if (rem > 0)
        {
            ov_buf_hline(' ', rem);
        }
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("[Panel: %s]", sname);
    }
    else
    {
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("■ %s", entry->label);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_hline(' ', inner_w - (2 + (int) strlen(entry->label)));
    }

    /* Rows 2..N: Word-wrapped descriptive detail */
    int text_lines = detail_h - 3;
    if (text_lines > 0)
    {
        ov_help_print_wrapped(entry->detail, split_r + 2, col, inner_w, text_lines, OV_FG_TEXT,
                              OV_BG_PANEL);
    }

    /* Last row: Live Target info for the currently selected GUI item */
    int target_row = split_r + detail_h - 1;
    ov_buf_pos(target_row, col);
    ov_theme_bg(OV_BG_PANEL);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_bold();
    ov_buf_printf("Target: ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    int  target_avail = inner_w - 8;
    char tbuf[160];
    tbuf[0]            = '\0';
    ov_rgb_t target_fg = OV_FG_DIM;

    if (entry->section == HS_INTRO)
    {
        target_fg = OV_FG_TITLE;
        snprintf(tbuf, sizeof(tbuf),
                 "Intro Guide: Press '1', 'i', or ENTER to open the interactive introduction");
    }
    else if (entry->section == HS_STREAMS)
    {
        int si = ov_get_selected_stream_idx(lay, m);
        if (si >= 0 && si < m->nb_streams)
        {
            const OV_STREAM *s = &m->streams[si];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t) { 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will delete stream '%s' (%s %s) from SHM", s->name,
                         s->size_str, render_dtype(s->datatype));
            }
            else
            {
                target_fg = OV_FG_STREAM;
                snprintf(tbuf, sizeof(tbuf), "Selected stream '%s' (%s %s, %.1f Hz)", s->name,
                         s->size_str, render_dtype(s->datatype), s->update_hz);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No stream selected (Streams panel is empty)");
        }
    }
    else if (entry->section == HS_PROCS)
    {
        int pi = ov_get_selected_proc_idx(lay, m);
        if (pi >= 0 && pi < m->nb_procs)
        {
            const OV_PROC *p = &m->procs[pi];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t) { 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will signal process '%s' (PID %d, %s)", p->name,
                         (int) p->PID, p->statusmsg);
            }
            else
            {
                target_fg = OV_FG_PROC;
                snprintf(tbuf, sizeof(tbuf), "Selected process '%s' (PID %d, CPU %.1f%%, %s)",
                         p->name, (int) p->PID, p->cpu_used, p->statusmsg);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No process selected (Processes panel is empty)");
        }
    }
    else if (entry->section == HS_FPS)
    {
        int fi = ov_get_selected_fps_idx(lay, m);
        if (fi >= 0 && fi < m->nb_fps)
        {
            const OV_FPS *f = &m->fps[fi];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t) { 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will control FPS module '%s' (run=%s, conf=%s)",
                         f->name, f->run_alive ? "ON" : "OFF", f->conf_alive ? "ON" : "OFF");
            }
            else
            {
                target_fg = OV_FG_FPS;
                snprintf(tbuf, sizeof(tbuf),
                         "Selected FPS module '%s' (run=%s, conf=%s, RSS: %ld KB)", f->name,
                         f->run_alive ? "ON" : "OFF", f->conf_alive ? "ON" : "OFF",
                         (long) f->mem_rss_kb);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No FPS module selected");
        }
    }
    else if (entry->section == HS_GRAPH)
    {
        if (m != NULL && lay->sel_graph >= 0 && lay->sel_graph < m->nb_nodes)
        {
            const OV_NODE *n = &m->nodes[lay->sel_graph];
            target_fg        = OV_FG_CONN;
            snprintf(tbuf, sizeof(tbuf), "Selected node '%s' (type: %s)", n->name,
                     (n->type == OV_NODE_STREAM) ? "Stream"
                                                 : ((n->type == OV_NODE_PROC) ? "Process" : "FPS"));
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "Node graph cursor active");
        }
    }
    else
    {
        /* Global & Display sections: show active GUI selection */
        int sel_s = ov_get_selected_stream_idx(lay, m);
        int sel_p = ov_get_selected_proc_idx(lay, m);
        int sel_f = ov_get_selected_fps_idx(lay, m);
        if (lay->focus == OV_FOCUS_STREAMS && m && sel_s >= 0 && sel_s < m->nb_streams)
        {
            target_fg = OV_FG_STREAM;
            snprintf(tbuf, sizeof(tbuf), "Active selection: Stream '%s' (Panel: Streams)",
                     m->streams[sel_s].name);
        }
        else if (lay->focus == OV_FOCUS_PROCS && m && sel_p >= 0 && sel_p < m->nb_procs)
        {
            target_fg = OV_FG_PROC;
            snprintf(tbuf, sizeof(tbuf), "Active selection: Process '%s' (PID %d) (Panel: Procs)",
                     m->procs[sel_p].name, (int) m->procs[sel_p].PID);
        }
        else if (lay->focus == OV_FOCUS_FPS && m && sel_f >= 0 && sel_f < m->nb_fps)
        {
            target_fg = OV_FG_FPS;
            snprintf(tbuf, sizeof(tbuf), "Active selection: FPS '%s' (Panel: FPS)",
                     m->fps[sel_f].name);
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "Global action — applies across all dashboard views");
        }
    }

    ov_theme_fg(target_fg);
    ov_buf_printf("%s", tbuf);
    int chars_used = (int) strlen(tbuf);
    int target_pad = target_avail - chars_used;
    if (target_pad > 0)
    {
        ov_buf_hline(' ', target_pad);
    }
}

/* clang-format off */
/**
 * ov_help_render_intro - render full interactive introduction guide to milk-CTRL.
 * @lay: layout state
 * @pr:  top row (1-based)
 * @pc:  left column (1-based)
 * @ph:  height in rows
 * @pw:  width in columns
 */
static void ov_help_render_intro(const OV_LAYOUT *lay, int pr, int pc, int ph, int pw)
{
    const char *title =
        (pw >= 84) ? "ABOUT milk-CTRL (1/i: Intro • 2/k: Controls • ↑↓: Scroll • ESC: Close)"
                   : ((pw >= 52) ? "ABOUT milk-CTRL (1/i: Intro • 2/k: Controls • ESC: Close)"
                                 : "ABOUT milk-CTRL");
    ov_draw_panel_border(pr, pc, ph, pw, title, OV_FG_TITLE, 1, 0);

    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    int inner_w = pw - 4;
    int col     = pc + 2;

    /* Header Row (pr + 1): Mode selector tabs */
    ov_buf_pos(pr + 1, col);
    ov_theme_bg(OV_BG_PANEL);

    int tab1_w = (pw >= 100) ? 26 : 14;
    int tab2_w = (pw >= 100) ? 32 : 20;

    /* Tab 1 (Active): Intro & Overview */
    ov_buf_bg(240, 175, 20);
    ov_buf_fg(20, 20, 25);
    ov_buf_bold();
    ov_buf_printf("%s", (pw >= 100) ? " [▶ 1: INTRO & OVERVIEW ◀] " : " [▶ 1: INTRO ◀] ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_printf(" ");

    /* Tab 2 (Inactive): Keystrokes & Controls */
    ov_theme_bg(OV_BG_PANEL_ALT);
    ov_theme_fg(OV_FG_MUTED);
    ov_buf_bold();
    ov_buf_printf("%s", (pw >= 100) ? " [ 2: KEYSTROKES & CONTROLS ] " : " [2: CONTROLS] ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    /* Close hint on right */
    const char *close_hint = "[ESC: Close Help] [X] ";
    int         used_hdr   = tab1_w + 1 + tab2_w;
    int         rem_hdr    = inner_w - used_hdr - (int) strlen(close_hint);
    if (rem_hdr > 0)
    {
        ov_buf_hline(' ', rem_hdr);
    }
    ov_buf_fg(255, 120, 100);
    ov_buf_bold();
    ov_buf_printf("%s", close_hint);
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    /* Row pr + 2: Subtitle */
    ov_buf_pos(pr + 2, col);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("Unified Real-Time Dashboard for Shared Memory, Telemetry, and AO Pipelines");
    int sub_rem = inner_w - 75;
    if (sub_rem > 0)
    {
        ov_buf_hline(' ', sub_rem);
    }

    /* Row pr + 3: Divider */
    ov_buf_pos(pr + 3, pc);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("├");
    for (int c = pc + 1; c < pc + pw - 1; c++)
    {
        ov_buf_printf("─");
    }
    ov_buf_printf("┤");

    /* Body viewport calculation */
    int body_top    = pr + 4;
    int body_bot    = pr + ph - 2;
    int body_h      = body_bot - body_top + 1;
    int total_lines = (int) (g_intro_total);
    int max_scroll  = (total_lines > body_h) ? (total_lines - body_h) : 0;

    int scroll = lay->help_intro_scroll;
    if (scroll < 0)
    {
        scroll = 0;
    }
    if (scroll > max_scroll)
    {
        scroll = max_scroll;
    }

    for (int r = 0; r < body_h; r++)
    {
        int row_idx = r + scroll;
        int cur_row = body_top + r;
        ov_buf_pos(cur_row, col);
        ov_theme_bg(OV_BG_PANEL);

        if (row_idx >= total_lines)
        {
            ov_buf_hline(' ', inner_w);
            continue;
        }

        const intro_item_t *item        = &g_intro_items[row_idx];
        int                 printed_len = 0;

        switch (item->type)
        {
        case IL_HEADER:
            ov_buf_bold();
            ov_theme_bg(OV_BG_PANEL_ALT);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf(" %s ", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            printed_len = (int) strlen(item->prefix) + 2;
            break;

        case IL_SUBHEADER:
            ov_buf_bold();
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("  %s", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            printed_len = (int) strlen(item->prefix) + 2;
            break;

        case IL_BULLET:
            ov_buf_bold();
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf("    %s", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf("%s", item->text ? item->text : "");
            printed_len =
                (int) strlen(item->prefix) + 4 + (item->text ? (int) strlen(item->text) : 0);
            break;

        case IL_TEXT:
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf("  %s", item->text ? item->text : "");
            printed_len = (item->text ? (int) strlen(item->text) : 0) + 2;
            break;

        case IL_KEY:
            ov_buf_bold();
            ov_buf_fg(130, 205, 255);
            ov_buf_printf("  %-16s", item->prefix ? item->prefix : "");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf(" %s", item->text ? item->text : "");
            printed_len = 2 + 16 + 1 + (item->text ? (int) strlen(item->text) : 0);
            break;

        case IL_BLANK:
        default:
            printed_len = 0;
            break;
        }

        int pad = inner_w - printed_len;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }
    }

    if (scroll > 0)
    {
        ov_buf_pos(body_top, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▲");
    }
    if (scroll < max_scroll)
    {
        ov_buf_pos(body_bot, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▼");
    }

    /* Bottom divider */
    {
        ov_buf_pos(body_bot + 1, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├─");

        char bstatus[80];
        if (max_scroll > 0)
        {
            snprintf(bstatus, sizeof(bstatus),
                     " [↑↓ / PgUp/PgDn: Scroll (%d/%d) • 2/k: Controls • ESC: Close] ", scroll + 1,
                     total_lines);
        }
        else
        {
            snprintf(bstatus, sizeof(bstatus), " [2 / k: Controls Reference • ESC: Close] ");
        }

        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("%s", bstatus);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);

        int used_b = 2 + (int) strlen(bstatus);
        int rem_b  = (pw - 2) - used_b;
        if (rem_b > 0)
        {
            for (int i = 0; i < rem_b; i++)
            {
                ov_buf_printf("─");
            }
        }
        ov_buf_printf("┤");
    }

    ov_buf_reset_attr();
}

/* ---- Render entry point ---- */

/**
 * ov_render_help - render the help overlay panel.
 * @lay: layout state
 * @m:   system model
 */
void ov_render_help(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int pr, pc, ph, pw;
    ov_help_get_rect(lay, &pr, &pc, &ph, &pw);

    if (lay->help_mode == 1)
    {
        ov_help_render_intro(lay, pr, pc, ph, pw);
        return;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    /* Draw outer panel border */
    const char *title =
        (pw >= 84) ? "HELP & CONTROLS (1/i: Intro • 2/k: Controls • ↑↓ nav • / search • ESC close)"
                   : ((pw >= 54) ? "HELP & CONTROLS (1/i: Intro • / search • ESC close)" : "HELP");
    ov_draw_panel_border(pr, pc, ph, pw, title, OV_FG_BRIGHT, 1, 0);

    /* Clear interior background */
    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    int inner_w = pw - 4;

    /* Row 1: Header Mode Selector and Context */
    {
        ov_buf_pos(pr + 1, pc + 2);
        ov_theme_bg(OV_BG_PANEL);

        int tab1_w = (pw >= 100) ? 26 : 14;
        int tab2_w = (pw >= 100) ? 32 : 20;

        /* Tab 1 (Inactive): Intro & Overview */
        ov_theme_bg(OV_BG_PANEL_ALT);
        ov_theme_fg(OV_FG_MUTED);
        ov_buf_bold();
        ov_buf_printf("%s", (pw >= 100) ? " [ 1: INTRO & OVERVIEW ] " : " [1: INTRO] ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf(" ");

        /* Tab 2 (Active): Keystrokes & Controls */
        ov_buf_bg(240, 175, 20);
        ov_buf_fg(20, 20, 25);
        ov_buf_bold();
        ov_buf_printf("%s",
                      (pw >= 100) ? " [▶ 2: KEYSTROKES & CONTROLS ◀] " : " [▶ 2: CONTROLS ◀] ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("  ");

        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("Context: ");
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);

        const char *vname = (lay->view == OV_VIEW_DASHBOARD) ? "DASH"
                            : (lay->view == OV_VIEW_STREAMS) ? "STRM"
                            : (lay->view == OV_VIEW_PROCS)   ? "PROC"
                            : (lay->view == OV_VIEW_FPS)     ? "FPS"
                                                             : "CONN";
        const char *pname = (lay->focus == OV_FOCUS_STREAMS) ? "Streams"
                            : (lay->focus == OV_FOCUS_PROCS) ? "Processes"
                            : (lay->focus == OV_FOCUS_FPS)   ? "FPS"
                                                             : "Graph";
        ov_buf_printf("[%s / %s]", vname, pname);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);

        int sel_s = ov_get_selected_stream_idx(lay, m);
        int sel_p = ov_get_selected_proc_idx(lay, m);
        int sel_f = ov_get_selected_fps_idx(lay, m);

        if (lay->focus == OV_FOCUS_STREAMS && m && sel_s >= 0 && sel_s < m->nb_streams)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_STREAM);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->streams[sel_s].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
        else if (lay->focus == OV_FOCUS_PROCS && m && sel_p >= 0 && sel_p < m->nb_procs)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->procs[sel_p].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
        else if (lay->focus == OV_FOCUS_FPS && m && sel_f >= 0 && sel_f < m->nb_fps)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->fps[sel_f].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }

        /* Control Mode badge on right side of header row */
        int ctrl_badge_col = pc + pw - 27;
        if (ctrl_badge_col > pc + tab1_w + tab2_w + 35)
        {
            ov_buf_pos(pr + 1, ctrl_badge_col);
            if (lay->ctrl_mode)
            {
                ov_buf_bg(220, 40, 40);
                ov_buf_fg(255, 255, 255);
                ov_buf_bold();
                ov_buf_printf("  CONTROL MODE: ON   ");
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL_ALT);
                ov_theme_fg(OV_FG_DIM);
                ov_buf_bold();
                ov_buf_printf("  CONTROL MODE: OFF  ");
            }
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
    }

    /* Row 2: Search Bar or Search Feature Notification Note */
    {
        ov_buf_pos(pr + 2, pc + 2);
        ov_theme_bg(OV_BG_PANEL);

        if (lay->help_search_active || lay->help_search[0] != '\0')
        {
            ov_buf_bold();
            ov_buf_fg(255, 220, 100);
            ov_buf_printf("Search: ");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            /* Search input box */
            if (lay->help_search_active)
            {
                ov_theme_bg(OV_BG_SELECTED);
                ov_theme_fg(OV_FG_BRIGHT);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL_ALT);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_bold();

            int qbox_w = 26;
            if (qbox_w > pw - 38)
            {
                qbox_w = pw - 38;
            }
            if (qbox_w < 12)
            {
                qbox_w = 12;
            }

            char qdisp[48];
            snprintf(qdisp, sizeof(qdisp), "%s%s", lay->help_search,
                     lay->help_search_active ? "█" : "");
            ov_buf_printf(" %-*.*s ", qbox_w, qbox_w, qdisp);

            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);

            /* Matches count */
            char count_str[32];
            if (lay->help_search[0] == '\0')
            {
                snprintf(count_str, sizeof(count_str), " (type query)");
            }
            else
            {
                snprintf(count_str, sizeof(count_str), " (%d match%s)", nvis,
                         (nvis == 1) ? "" : "es");
            }
            ov_buf_printf("%s", count_str);

            /* Clear / cancel button */
            const char *btn_str =
                (lay->help_search[0] == '\0') ? " [ESC: cancel] " : " [ESC: clear] ";
            int used = 8 + qbox_w + 2 + (int) strlen(count_str);
            int rem  = inner_w - used - (int) strlen(btn_str);
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_buf_fg(255, 120, 100);
            ov_buf_printf("%s", btn_str);
        }
        else
        {
            /* 1-line note notifying users of the search feature */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("Search: ");
            ov_buf_bold();
            ov_buf_fg(255, 220, 100);
            ov_buf_printf("[/]");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(
                " Press '/' to search topics & commands (e.g. \"kill\", \"stream\", \"fps\")");

            int used = 8 + 3 + 68;
            int rem  = inner_w - used;
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
        }
    }

    /* Row 3: Top divider */
    {
        ov_buf_pos(pr + 3, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├");
        for (int c = pc + 1; c < pc + pw - 1; c++)
        {
            ov_buf_printf("─");
        }
        ov_buf_printf("┤");
    }

    /* Detailed Help split calculation */
    int detail_h = (ph >= 36) ? 10 : ((ph >= 28) ? 8 : ((ph >= 22) ? 6 : 5));
    int split_r  = (pr + ph - 1) - detail_h;
    int list_top = pr + 4;
    int list_h   = split_r - list_top;
    if (list_h < 4)
    {
        list_h   = 4;
        split_r  = list_top + list_h;
        detail_h = (pr + ph - 1) - split_r;
    }

    /* Cursor bounds check */
    int sel = lay->help_sel;
    if (sel < 0)
    {
        sel = 0;
    }
    if (sel >= nvis)
    {
        sel = nvis - 1;
    }

    /* Scroll so cursor is within list viewport */
    int scroll = 0;
    if (sel >= list_h)
    {
        scroll = sel - list_h + 1;
    }

    /* Render visible rows in list area */
    if (lay->help_search[0] != '\0' && nvis == 0)
    {
        ov_buf_pos(list_top + 1, pc + 4);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("No matching commands found for \"%s\"", lay->help_search);
        ov_buf_pos(list_top + 2, pc + 4);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("Try searching: stream, proc, fps, kill, sort, filter, view");
    }

    for (int vr = 0; vr < list_h && vr + scroll < nvis; vr++)
    {
        int                 idx    = map[vr + scroll];
        const help_entry_t *h      = &g_help_entries[idx];
        int                 row    = list_top + vr;
        int                 is_sel = ((vr + scroll) == sel);

        ov_buf_pos(row, pc + 2);
        if (is_sel)
        {
            ov_theme_bg(OV_BG_SELECTED);
        }
        else
        {
            ov_theme_bg(OV_BG_PANEL);
        }

        if (lay->help_search[0] != '\0')
        {
            /* Search match row with section badge */
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf(" ▶ ");
            }
            else
            {
                ov_buf_printf("   ");
            }

            /* Section badge */
            ov_buf_bold();
            ov_theme_fg(ov_help_section_color(h->section));
            ov_buf_printf("[%-4s] ", ov_help_section_tag(h->section));
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_theme_bg(OV_BG_SELECTED);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
            }

            /* Keystroke or category indicator */
            ov_buf_bold();
            if (h->flags & HF_SECTION)
            {
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_printf("%-13s", "Topic");
                ov_buf_printf("   ");
            }
            else if (h->flags & HF_CTRL_MODE)
            {
                if (lay->ctrl_mode)
                {
                    ov_buf_fg(255, 95, 75);
                    ov_buf_printf("%-13s", h->key ? h->key : "");
                    ov_buf_fg(255, 80, 80);
                    ov_buf_printf(" ⚡ ");
                }
                else
                {
                    ov_buf_fg(200, 140, 50);
                    ov_buf_printf("%-13s", h->key ? h->key : "");
                    ov_buf_fg(160, 115, 45);
                    ov_buf_printf(" 🔒 ");
                }
            }
            else
            {
                ov_buf_fg(130, 205, 255);
                ov_buf_printf("%-13s", h->key ? h->key : "");
                ov_buf_printf("   ");
            }

            /* Summary label */
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_theme_bg(OV_BG_SELECTED);
                ov_theme_fg(OV_FG_BRIGHT);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_printf("%s", h->label);

            int used = 3 + 7 + 13 + 3 + (int) strlen(h->label);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }
        else if (h->flags & HF_SECTION)
        {
            int         expanded = help_is_expanded(lay, h->section);
            const char *chev     = expanded ? "▾" : "▸";

            if (is_sel)
            {
                ov_theme_bg(OV_BG_SELECTED);
                ov_buf_bold();
                ov_theme_fg(OV_FG_BRIGHT);
                ov_buf_printf("▶ %s %s", chev, h->label);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL_ALT);
                ov_buf_bold();
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_printf("  %s %s", chev, h->label);
            }

            /* Count child entries in section */
            int nchildren = 0;
            for (int k = 0; k < g_help_total; k++)
            {
                if (g_help_entries[k].section == h->section &&
                    !(g_help_entries[k].flags & HF_SECTION))
                {
                    nchildren++;
                }
            }

            char tag[32];
            snprintf(tag, sizeof(tag), "(%d keys)", nchildren);
            int used = 4 + (int) strlen(h->label) + 1 + (int) strlen(tag);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" %s", tag);
        }
        else if (h->flags & HF_COLORS)
        {
            ov_theme_fg(OV_FG_DIM);
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf("    ▶ ");
            }
            else
            {
                ov_buf_printf("      ");
            }

            ov_theme_fg(OV_FG_STREAM);
            ov_buf_printf("● stream ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_printf("● proc ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_printf("● fps ");
            ov_theme_fg(OV_FG_CONN);
            ov_buf_printf("● conn ");
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf("● active ");
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("● warn ");
            ov_theme_fg(OV_FG_ERROR);
            ov_buf_printf("● error");

            int used = 6 + 53;
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }
        else
        {
            /* Standard keystroke entry with tab offset */
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf("    ▶ ");
            }
            else
            {
                ov_buf_printf("      ");
            }

            /* Keystroke column in standard bold font */
            ov_buf_bold();
            if (h->flags & HF_CTRL_MODE)
            {
                if (lay->ctrl_mode)
                {
                    ov_buf_fg(255, 95, 75);
                    ov_buf_printf("%-13s", h->key);
                    ov_buf_fg(255, 80, 80);
                    ov_buf_printf(" ⚡ ");
                }
                else
                {
                    ov_buf_fg(200, 140, 50);
                    ov_buf_printf("%-13s", h->key);
                    ov_buf_fg(160, 115, 45);
                    ov_buf_printf(" 🔒 ");
                }
            }
            else
            {
                ov_buf_fg(130, 205, 255);
                ov_buf_printf("%-13s", h->key);
                ov_buf_printf("    ");
            }

            /* Summary label */
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_theme_bg(OV_BG_SELECTED);
                ov_theme_fg(OV_FG_BRIGHT);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_printf("%s", h->label);

            int used = 6 + 13 + 4 + (int) strlen(h->label);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }

        ov_buf_reset_attr();
    }

    /* Scroll indicators */
    if (scroll > 0)
    {
        ov_buf_pos(list_top, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▲");
    }
    if (scroll + list_h < nvis)
    {
        ov_buf_pos(split_r - 1, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▼");
    }

    /* Divider above Detailed Help */
    {
        ov_buf_pos(split_r, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├─");
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf(" DETAILED HELP ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);

        int         used_div = 17;
        const char *hint     = (lay->help_search[0] != '\0')
                                   ? "[↑↓ nav • ESC clear search]"
                                   : "[↑↓ nav • →/← expand • [/] search • ESC close]";
        int         hint_len = (int) strlen(hint);
        int         div_pad  = (pw - 2) - used_div - hint_len - 1;
        if (div_pad > 0)
        {
            for (int i = 0; i < div_pad; i++)
            {
                ov_buf_printf("─");
            }
            ov_buf_printf(" %s─┤", hint);
        }
        else
        {
            for (int i = 0; i < (pw - 2) - used_div; i++)
            {
                ov_buf_printf("─");
            }
            ov_buf_printf("┤");
        }
    }

    /* Render detailed help for currently selected item */
    if (sel >= 0 && sel < nvis)
    {
        ov_help_render_detail(lay, m, &g_help_entries[map[sel]], split_r, pc, pw, detail_h);
    }

    ov_buf_reset_attr();
}
