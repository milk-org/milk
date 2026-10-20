// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_filter.c
 * @brief   Pattern filtering and panel filter state management for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <string.h>

#define OV_FILTER_CACHE_SIZE 4

typedef struct
{
    char     pattern[64];
    regex_t  re;
    int      state; /* 0 empty, 1 compiled, -1 regcomp failed (negative entry, nothing to free) */
    uint64_t used;  /* last-use tick, for LRU eviction */
} ov_filter_cache_entry_t;

static ov_filter_cache_entry_t s_filter_cache[OV_FILTER_CACHE_SIZE];
static uint64_t                s_filter_tick;
static unsigned long           s_filter_compiles;

/** ov_filter_compile_count - number of regcomp() calls so far (cache behaviour checks). */
unsigned long ov_filter_compile_count(void)
{
    return s_filter_compiles;
}

/**
 * get_cached_regex - compiled @pattern from a small LRU cache. Failed patterns are cached too, so an
 * invalid filter is not recompiled every frame and does not evict the other panels' regexes.
 * Return: the regex with *@out_reg_ok = 1, or NULL with 0 when @pattern does not compile.
 */
static regex_t *get_cached_regex(const char *pattern, int *out_reg_ok)
{
    int slot = 0;
    for (int i = 0; i < OV_FILTER_CACHE_SIZE; i++)
    {
        ov_filter_cache_entry_t *e = &s_filter_cache[i];
        if (e->state != 0 && strcmp(e->pattern, pattern) == 0)
        {
            e->used     = ++s_filter_tick;
            *out_reg_ok = (e->state == 1);
            return (e->state == 1) ? &e->re : NULL;
        }
        if (e->used < s_filter_cache[slot].used)
        {
            slot = i; /* least recently used (empty slots have used == 0) */
        }
    }

    ov_filter_cache_entry_t *e = &s_filter_cache[slot];
    if (e->state == 1)
    {
        regfree(&e->re);
    }
    snprintf(e->pattern, sizeof(e->pattern), "%s", pattern);
    s_filter_compiles++;
    e->state    = (regcomp(&e->re, pattern, REG_EXTENDED | REG_NOSUB | REG_ICASE) == 0) ? 1 : -1;
    e->used     = ++s_filter_tick;
    *out_reg_ok = (e->state == 1);
    return (e->state == 1) ? &e->re : NULL;
}

/**
 * ov_filter_build - filter a string list using POSIX regex or case-insensitive substring matching.
 * @pattern: Regex or substring pattern.
 * @names:   Array of name strings to test.
 * @count:   Number of items in names array.
 * @out:     Output buffer to receive indices of matching items.
 * @max_out: Maximum capacity of out buffer.
 *
 * Return: Number of matching indices written to out buffer.
 */
int ov_filter_build(const char *pattern, const char **names, int count, int *out, int max_out)
{
    if (pattern == NULL || pattern[0] == '\0')
    {
        /* No filter — all items match */
        int n = count < max_out ? count : max_out;
        for (int i = 0; i < n; i++)
        {
            out[i] = i;
        }
        return n;
    }

    int      reg_ok = 0;
    regex_t *re     = get_cached_regex(pattern, &reg_ok);
    int      n      = 0;

    if (reg_ok && re != NULL)
    {
        for (int i = 0; i < count && n < max_out; i++)
        {
            if (names[i] != NULL && regexec(re, names[i], 0, NULL, 0) == 0)
            {
                out[n++] = i;
            }
        }
    }
    else
    {
        /* Fallback: case-insensitive literal substring search */
        for (int i = 0; i < count && n < max_out; i++)
        {
            if (names[i] != NULL && strcasestr(names[i], pattern) != NULL)
            {
                out[n++] = i;
            }
        }
    }

    return n;
}

/**
 * ov_get_effective_filter_panel - determine active panel target for filtering.
 * @lay: Pointer to layout structure.
 *
 * Return: Target panel focus enum.
 */
ov_focus_t ov_get_effective_filter_panel(const OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return OV_FOCUS_STREAMS;
    }
    if (lay->view == OV_VIEW_STREAMS)
    {
        return OV_FOCUS_STREAMS;
    }
    if (lay->view == OV_VIEW_PROCS)
    {
        return OV_FOCUS_PROCS;
    }
    if (lay->view == OV_VIEW_FPS)
    {
        return OV_FOCUS_FPS;
    }
    if (lay->focus == OV_FOCUS_STREAMS || lay->focus == OV_FOCUS_PROCS ||
        lay->focus == OV_FOCUS_FPS)
    {
        return lay->focus;
    }
    return OV_FOCUS_GRAPH;
}

/**
 * ov_has_panel_filter - check if a specific panel has a defined filter string.
 * @lay:   Pointer to layout structure.
 * @panel: Panel focus enum.
 *
 * Return: 1 if non-empty filter exists, 0 otherwise.
 */
int ov_has_panel_filter(const OV_LAYOUT *lay, ov_focus_t panel)
{
    if (lay == NULL)
    {
        return 0;
    }
    if (panel == OV_FOCUS_STREAMS)
    {
        return (lay->filter_stream[0] != '\0');
    }
    if (panel == OV_FOCUS_PROCS)
    {
        return (lay->filter_proc[0] != '\0');
    }
    if (panel == OV_FOCUS_FPS)
    {
        return (lay->filter_fps[0] != '\0');
    }
    return 0;
}

/**
 * ov_is_panel_filter_active - check if a panel filter is currently enabled.
 * @lay:   Pointer to layout structure.
 * @panel: Panel focus enum.
 *
 * Return: 1 if active, 0 otherwise.
 */
int ov_is_panel_filter_active(const OV_LAYOUT *lay, ov_focus_t panel)
{
    if (lay == NULL)
    {
        return 0;
    }
    if (panel == OV_FOCUS_STREAMS)
    {
        return lay->filter_stream_active && (lay->filter_stream[0] != '\0');
    }
    if (panel == OV_FOCUS_PROCS)
    {
        return lay->filter_proc_active && (lay->filter_proc[0] != '\0');
    }
    if (panel == OV_FOCUS_FPS)
    {
        return lay->filter_fps_active && (lay->filter_fps[0] != '\0');
    }
    return 0;
}

/**
 * ov_get_panel_filter_pattern - get configured filter string for a panel.
 * @lay:   Pointer to layout structure.
 * @panel: Panel focus enum.
 *
 * Return: Pointer to filter string or empty string.
 */
const char *ov_get_panel_filter_pattern(const OV_LAYOUT *lay, ov_focus_t panel)
{
    if (lay == NULL)
    {
        return "";
    }
    if (panel == OV_FOCUS_STREAMS)
    {
        return lay->filter_stream;
    }
    if (panel == OV_FOCUS_PROCS)
    {
        return lay->filter_proc;
    }
    if (panel == OV_FOCUS_FPS)
    {
        return lay->filter_fps;
    }
    return "";
}

/**
 * ov_get_active_filter_for - get active filter pattern string for a panel.
 * @lay:   Pointer to layout structure.
 * @panel: Panel focus enum.
 *
 * Return: Pointer to pattern string if active, or empty string.
 */
const char *ov_get_active_filter_for(const OV_LAYOUT *lay, ov_focus_t panel)
{
    if (!ov_is_panel_filter_active(lay, panel))
    {
        return "";
    }
    return ov_get_panel_filter_pattern(lay, panel);
}

/**
 * ov_clear_panel_filter - reset filter string and active flag for a panel.
 * @lay:   Pointer to layout structure.
 * @panel: Panel focus enum.
 */
void ov_clear_panel_filter(OV_LAYOUT *lay, ov_focus_t panel)
{
    if (lay == NULL)
    {
        return;
    }
    if (panel == OV_FOCUS_STREAMS)
    {
        lay->filter_stream[0]     = '\0';
        lay->filter_stream_active = 0;
    }
    else if (panel == OV_FOCUS_PROCS)
    {
        lay->filter_proc[0]     = '\0';
        lay->filter_proc_active = 0;
    }
    else if (panel == OV_FOCUS_FPS)
    {
        lay->filter_fps[0]     = '\0';
        lay->filter_fps_active = 0;
    }
    lay->filter_active =
        (lay->filter_stream_active || lay->filter_proc_active || lay->filter_fps_active);
}

/**
 * ov_has_filter - check if any panel has a non-empty filter pattern.
 * @lay: Pointer to layout structure.
 *
 * Return: 1 if any pattern is set, 0 otherwise.
 */
int ov_has_filter(const OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return 0;
    }
    return (lay->filter_stream[0] != '\0' || lay->filter_proc[0] != '\0' ||
            lay->filter_fps[0] != '\0');
}

/**
 * ov_is_filter_active - check if filter is active for focused panel or any panel.
 * @lay: Pointer to layout structure.
 *
 * Return: 1 if active, 0 otherwise.
 */
int ov_is_filter_active(const OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return 0;
    }
    ov_focus_t panel = ov_get_effective_filter_panel(lay);
    if (panel == OV_FOCUS_STREAMS || panel == OV_FOCUS_PROCS || panel == OV_FOCUS_FPS)
    {
        return ov_is_panel_filter_active(lay, panel);
    }
    return (ov_is_panel_filter_active(lay, OV_FOCUS_STREAMS) ||
            ov_is_panel_filter_active(lay, OV_FOCUS_PROCS) ||
            ov_is_panel_filter_active(lay, OV_FOCUS_FPS));
}

/**
 * ov_get_filter_pattern - get configured filter pattern for focused panel.
 * @lay: Pointer to layout structure.
 *
 * Return: Pointer to filter pattern string or empty string.
 */
const char *ov_get_filter_pattern(const OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return "";
    }
    ov_focus_t panel = ov_get_effective_filter_panel(lay);
    if (panel == OV_FOCUS_STREAMS && lay->filter_stream[0] != '\0')
    {
        return lay->filter_stream;
    }
    if (panel == OV_FOCUS_PROCS && lay->filter_proc[0] != '\0')
    {
        return lay->filter_proc;
    }
    if (panel == OV_FOCUS_FPS && lay->filter_fps[0] != '\0')
    {
        return lay->filter_fps;
    }
    if (lay->filter_stream[0] != '\0')
    {
        return lay->filter_stream;
    }
    if (lay->filter_proc[0] != '\0')
    {
        return lay->filter_proc;
    }
    if (lay->filter_fps[0] != '\0')
    {
        return lay->filter_fps;
    }
    return "";
}

/**
 * ov_get_active_filter - get active filter string for currently focused panel.
 * @lay: Pointer to layout structure.
 *
 * Return: Pointer to active pattern string or empty string.
 */
const char *ov_get_active_filter(const OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return "";
    }
    ov_focus_t panel = ov_get_effective_filter_panel(lay);
    return ov_get_active_filter_for(lay, panel);
}

/**
 * ov_clear_all_filters - clear all filter strings and reset filter state.
 * @lay: Pointer to layout structure.
 */
void ov_clear_all_filters(OV_LAYOUT *lay)
{
    if (lay == NULL)
    {
        return;
    }
    lay->filter[0]            = '\0';
    lay->filter_stream[0]     = '\0';
    lay->filter_proc[0]       = '\0';
    lay->filter_fps[0]        = '\0';
    lay->filter_stream_active = 0;
    lay->filter_proc_active   = 0;
    lay->filter_fps_active    = 0;
    lay->filter_active        = 0;
    lay->filter_editing       = 0;
    lay->filter_cursor        = 0;
    lay->filter_jump          = 0;
}
