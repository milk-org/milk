// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_render_internal.h"
/**
 * @brief Build filtered index array based on regular expression.
 *
 * Compiles @pattern with POSIX extended case-insensitive regex.
 * If regex compilation fails (e.g. incomplete pattern), falls back
 * to case-insensitive substring search.
 * When a pattern is provided, strictly matching indices are returned.
 *
 * @param pattern  Pattern string to filter by (empty = match all)
 * @param names    Array of string names
 * @param count    Total count of names
 * @param out      Output array for matching indices
 * @param max_out  Maximum capacity of output array
 * @return Number of matching entries written to @out
 */
#define OV_FILTER_CACHE_SIZE 4

typedef struct
{
    char    pattern[64];
    regex_t re;
    int     valid;
} ov_filter_cache_entry_t;

static ov_filter_cache_entry_t s_filter_cache[OV_FILTER_CACHE_SIZE];
static int                     s_filter_cache_init = 0;

/**
 * get_cached_regex - lookup or compile regular expression in circular LRU cache.
 * @pattern:    Regular expression pattern string.
 * @out_reg_ok: Output flag set to 1 if regex compilation succeeded, 0 otherwise.
 *
 * Return: Pointer to compiled regex_t, or NULL if compilation failed.
 */
static regex_t *get_cached_regex(
    const char *pattern,
    int        *out_reg_ok)
{
    if (!s_filter_cache_init)
    {
        memset(s_filter_cache, 0, sizeof(s_filter_cache));
        s_filter_cache_init = 1;
    }

    /* Check cache hit */
    for (int i = 0; i < OV_FILTER_CACHE_SIZE; i++)
    {
        if (s_filter_cache[i].valid && strcmp(s_filter_cache[i].pattern, pattern) == 0)
        {
            *out_reg_ok = 1;
            return &s_filter_cache[i].re;
        }
    }

    /* Cache miss: evict slot round-robin */
    static int next_slot = 0;
    int        slot      = next_slot;
    next_slot            = (next_slot + 1) % OV_FILTER_CACHE_SIZE;

    if (s_filter_cache[slot].valid)
    {
        regfree(&s_filter_cache[slot].re);
        s_filter_cache[slot].valid = 0;
    }

    strncpy(s_filter_cache[slot].pattern, pattern, sizeof(s_filter_cache[slot].pattern) - 1);
    s_filter_cache[slot].pattern[sizeof(s_filter_cache[slot].pattern) - 1] = '\0';

    if (regcomp(&s_filter_cache[slot].re, pattern, REG_EXTENDED | REG_NOSUB | REG_ICASE) == 0)
    {
        s_filter_cache[slot].valid = 1;
        *out_reg_ok                = 1;
        return &s_filter_cache[slot].re;
    }

    *out_reg_ok = 0;
    return NULL;
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
int ov_filter_build(
    const char  *pattern,
    const char **names,
    int          count,
    int         *out,
    int          max_out)
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
 * @brief Get effective panel for filtering based on current focus or view.
 *
 * @param[in] lay Layout structure
 * @return Panel focus enum (OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS, or OV_FOCUS_GRAPH)
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
 * @brief Check if a specific panel has a defined filter pattern string.
 *
 * @param[in] lay   Layout structure
 * @param[in] panel Panel focus enum
 * @return 1 if defined, 0 otherwise
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
 * @brief Check if a specific panel's filter is currently active/enabled.
 *
 * @param[in] lay   Layout structure
 * @param[in] panel Panel focus enum
 * @return 1 if active, 0 otherwise
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
 * @brief Get the configured filter pattern for a specific panel.
 *
 * @param[in] lay   Layout structure
 * @param[in] panel Panel focus enum
 * @return Pointer to filter pattern string, or "" if none
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
 * @brief Get the active filter pattern string for a specific panel.
 *
 * @param[in] lay   Layout structure
 * @param[in] panel Panel focus enum
 * @return Pointer to active filter string, or "" if inactive/empty
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
 * @brief Clear filter string and reset active state for a specific panel.
 *
 * @param[in,out] lay   Layout structure
 * @param[in]     panel Panel focus enum
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
 * @brief Check if any regular expression filter pattern is defined in layout.
 *
 * @param[in] lay Layout structure
 * @return 1 if any filter pattern is set, 0 otherwise
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
 * @brief Check if filter is active for the current focused panel (or any panel).
 *
 * @param[in] lay Layout structure
 * @return 1 if active, 0 otherwise
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
 * @brief Get the configured filter pattern for the current focused panel.
 *
 * @param[in] lay Layout structure
 * @return Pointer to filter pattern string, or "" if none
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
 * @brief Get the active filter pattern string for the current focused panel.
 *
 * @param[in] lay Layout structure
 * @return Pointer to active filter string, or "" if filter is inactive/empty
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
 * @brief Clear all filter strings and reset filter state across all panels.
 *
 * @param[in,out] lay Layout structure
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

/**
 * @brief Filter stream indices by regex, freeze relations, and loop isolation.
 *
 * @param[in]  lay      Layout structure
 * @param[in]  m        Current model
 * @param[in]  rel      Related items bitset (optional, used in freeze mode)
 * @param[out] fidx     Output array for filtered model stream indices
 * @param[in]  max_fidx Maximum capacity of fidx
 * @return Number of matching stream indices written to fidx
 */
int ov_filter_streams(const OV_LAYOUT  *lay,
                      const OV_MODEL   *m,
                      const OV_RELATED *rel,
                      int              *fidx,
                      int               max_fidx)
{
    if (lay == NULL || m == NULL || m->nb_streams <= 0 || fidx == NULL || max_fidx <= 0)
    {
        return 0;
    }

    const char *names[OV_MAX_STREAMS];
    int         total = (m->nb_streams < max_fidx) ? m->nb_streams : max_fidx;
    for (int i = 0; i < m->nb_streams; i++)
    {
        names[i] = m->streams[i].name;
    }

    const char *filt   = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
    int         filt_n = 0;
    if (filt != NULL && filt[0] != '\0')
    {
        filt_n = ov_filter_build(filt, names, m->nb_streams, fidx, max_fidx);
    }
    else
    {
        filt_n = total;
        for (int i = 0; i < filt_n; i++)
        {
            fidx[i] = i;
        }
    }

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_STREAMS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->streams, fidx[i]))
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    if (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        int      new_filt_n  = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (m->streams[fidx[i]].loop_mask & active_mask)
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    return filt_n;
}

/**
 * @brief Filter process indices by regex, freeze relations, and loop isolation.
 *
 * @param[in]  lay      Layout structure
 * @param[in]  m        Current model
 * @param[in]  rel      Related items bitset (optional, used in freeze mode)
 * @param[out] fidx     Output array for filtered model process indices
 * @param[in]  max_fidx Maximum capacity of fidx
 * @return Number of matching process indices written to fidx
 */
int ov_filter_procs(const OV_LAYOUT  *lay,
                    const OV_MODEL   *m,
                    const OV_RELATED *rel,
                    int              *fidx,
                    int               max_fidx)
{
    if (lay == NULL || m == NULL || m->nb_procs <= 0 || fidx == NULL || max_fidx <= 0)
    {
        return 0;
    }

    const char *names[OV_MAX_PROCS];
    int         total = (m->nb_procs < max_fidx) ? m->nb_procs : max_fidx;
    for (int i = 0; i < m->nb_procs; i++)
    {
        names[i] = m->procs[i].name;
    }

    const char *filt   = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
    int         filt_n = 0;
    if (filt != NULL && filt[0] != '\0')
    {
        filt_n = ov_filter_build(filt, names, m->nb_procs, fidx, max_fidx);
    }
    else
    {
        filt_n = total;
        for (int i = 0; i < filt_n; i++)
        {
            fidx[i] = i;
        }
    }

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_PROCS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->procs, fidx[i]))
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    if (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        int      new_filt_n  = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (m->procs[fidx[i]].loop_mask & active_mask)
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    return filt_n;
}

/**
 * @brief Filter FPS indices by regex, freeze relations, and loop isolation.
 *
 * @param[in]  lay      Layout structure
 * @param[in]  m        Current model
 * @param[in]  rel      Related items bitset (optional, used in freeze mode)
 * @param[out] fidx     Output array for filtered model FPS indices
 * @param[in]  max_fidx Maximum capacity of fidx
 * @return Number of matching FPS indices written to fidx
 */
int ov_filter_fps(const OV_LAYOUT  *lay,
                  const OV_MODEL   *m,
                  const OV_RELATED *rel,
                  int              *fidx,
                  int               max_fidx)
{
    if (lay == NULL || m == NULL || m->nb_fps <= 0 || fidx == NULL || max_fidx <= 0)
    {
        return 0;
    }

    const char *names[OV_MAX_FPS];
    int         total = (m->nb_fps < max_fidx) ? m->nb_fps : max_fidx;
    for (int i = 0; i < m->nb_fps; i++)
    {
        names[i] = m->fps[i].name;
    }

    const char *filt   = ov_get_active_filter_for(lay, OV_FOCUS_FPS);
    int         filt_n = 0;
    if (filt != NULL && filt[0] != '\0')
    {
        filt_n = ov_filter_build(filt, names, m->nb_fps, fidx, max_fidx);
    }
    else
    {
        filt_n = total;
        for (int i = 0; i < filt_n; i++)
        {
            fidx[i] = i;
        }
    }

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_FPS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->fps, fidx[i]))
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    if (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        int      new_filt_n  = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (m->fps[fidx[i]].loop_mask & active_mask)
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    return filt_n;
}

/**
 * @brief Resolve the selected stream row to its index in OV_MODEL.
 *
 * @param[in] lay Layout structure
 * @param[in] m   Current model
 * @return Model stream index in 0..nb_streams-1, or -1 if none
 */
int ov_get_selected_stream_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay == NULL || m == NULL || m->nb_streams <= 0)
    {
        return -1;
    }
    int ssel = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    if (ssel < 0)
    {
        return -1;
    }
    int fidx[OV_MAX_STREAMS];
    int n = ov_filter_streams(lay, m, NULL, fidx, OV_MAX_STREAMS);
    if (ssel < n)
    {
        return fidx[ssel];
    }
    return -1;
}

/**
 * @brief Resolve the selected process row to its index in OV_MODEL.
 *
 * @param[in] lay Layout structure
 * @param[in] m   Current model
 * @return Model process index in 0..nb_procs-1, or -1 if none
 */
int ov_get_selected_proc_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay == NULL || m == NULL || m->nb_procs <= 0)
    {
        return -1;
    }
    int psel = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
    if (psel < 0)
    {
        return -1;
    }
    int fidx[OV_MAX_PROCS];
    int n = ov_filter_procs(lay, m, NULL, fidx, OV_MAX_PROCS);
    if (psel < n)
    {
        return fidx[psel];
    }
    return -1;
}

/**
 * @brief Resolve the selected FPS row to its index in OV_MODEL.
 *
 * @param[in] lay Layout structure
 * @param[in] m   Current model
 * @return Model FPS index in 0..nb_fps-1, or -1 if none
 */
int ov_get_selected_fps_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay == NULL || m == NULL || m->nb_fps <= 0)
    {
        return -1;
    }
    int fsel = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;
    if (fsel < 0)
    {
        return -1;
    }
    int fidx[OV_MAX_FPS];
    int n = ov_filter_fps(lay, m, NULL, fidx, OV_MAX_FPS);
    if (fsel < n)
    {
        return fidx[fsel];
    }
    return -1;
}

/* =========================================================
 * Cross-panel relation highlight
 * ========================================================= */

/*
 * Bitset helpers and OV_RELATED defined in
 * overview_render_internal.h
 */

/**
 * bset - set bit at index in 64-bit word bitset array.
 * @words: Pointer to 64-bit integer bitset words.
 * @idx:   0-based bit index to set.
 */
void bset(uint64_t *words, int idx)
{
    words[idx / BITS_PER_WORD] |= (UINT64_C(1) << (idx % BITS_PER_WORD));
}

/**
 * bget - test bit at index in 64-bit word bitset array.
 * @words: Pointer to 64-bit integer bitset words.
 * @idx:   0-based bit index to test.
 *
 * Return: 1 if bit is set, 0 otherwise.
 */
int bget(const uint64_t *words, int idx)
{
    return (words[idx / BITS_PER_WORD] >> (idx % BITS_PER_WORD)) & 1;
}

/**
 * ov_compute_related - identify all items connected to the active selection across all panels.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 * @out: Output structure populated with highlight bitsets and selected IDs.
 */
void ov_compute_related(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    OV_RELATED      *out)
{
    memset(out, 0, sizeof(*out));
    /* fps_param_mask initialised to 0 by memset — no matches yet */

    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    if (lay->mouse_hover && lay->hover_idx >= 0 && lay->hover_view != -1)
    {
        focus = lay->hover_view;
    }

    int sel_stream_idx = (lay->mouse_hover && lay->hover_global_stream >= 0)
                             ? lay->hover_global_stream
                             : ov_get_selected_stream_idx(lay, m);

    /* Determine the graph node index of the selected item */
    int sel_node = -1;
    if (focus == OV_FOCUS_STREAMS)
    {
        if (sel_stream_idx >= 0 && sel_stream_idx < m->nb_streams)
        {
            sel_node = m->streams[sel_stream_idx].node_idx;
        }
    }
    else if (focus == OV_FOCUS_FPS)
    {
        int model_idx = (lay->mouse_hover && lay->hover_global_fps >= 0)
                            ? lay->hover_global_fps
                            : ov_get_selected_fps_idx(lay, m);
        if (model_idx >= 0 && model_idx < m->nb_fps)
        {
            sel_node = m->fps[model_idx].node_idx;
        }
    }
    else if (focus == OV_FOCUS_PROCS)
    {
        int model_idx = (lay->mouse_hover && lay->hover_global_proc >= 0)
                            ? lay->hover_global_proc
                            : ov_get_selected_proc_idx(lay, m);
        if (model_idx >= 0 && model_idx < m->nb_procs)
        {
            sel_node     = m->procs[model_idx].node_idx;
            out->sel_pid = m->procs[model_idx].PID;
        }
    }

    if (sel_node < 0)
    {
        return;
    }

    /* Walk all edges; mark neighbours of sel_node */
    for (int ei = 0; ei < m->nb_edges; ei++)
    {
        const OV_EDGE *e        = &m->edges[ei];
        int            other    = -1;
        int            is_write = 0; /* 1 = proc writes stream */
        /* For FPS edges: which end is the stream node? */
        int fps_is_src = 0;

        if (e->src_node == sel_node)
        {
            other = e->tgt_node;
            is_write =
                (e->type == OV_EDGE_PROC_WRITES_STREAM) || (e->type == OV_EDGE_FPS_OUTPUT_STREAM);
            fps_is_src = 0; /* FPS is tgt when stream→FPS */
        }
        else if (e->tgt_node == sel_node)
        {
            other = e->src_node;
            is_write =
                (e->type == OV_EDGE_PROC_WRITES_STREAM) || (e->type == OV_EDGE_FPS_OUTPUT_STREAM);
            fps_is_src = 1; /* FPS is src when FPS→stream */
        }

        if (other < 0 || other >= m->nb_nodes)
        {
            continue;
        }

        const OV_NODE *n = &m->nodes[other];
        if (n->type == OV_NODE_STREAM && n->index >= 0 && n->index < m->nb_streams)
        {
            bset(out->streams, n->index);
            if (is_write)
            {
                bset(out->stream_written, n->index);
            }
        }
        else if (n->type == OV_NODE_FPS && n->index >= 0 && n->index < m->nb_fps)
        {
            int fi = n->index;
            bset(out->fps, fi);
            if (is_write)
            {
                bset(out->fps_writes, fi);
            }

            /* Find all stream params of this FPS that match sel_node.
             * Only meaningful when the selection is a stream.
             * OR all matching indices into the bitmask. */
            if (focus == OV_FOCUS_STREAMS && sel_stream_idx >= 0 && sel_stream_idx < m->nb_streams)
            {
                const char   *sname = m->streams[sel_stream_idx].name;
                const OV_FPS *f     = &m->fps[fi];
                for (int sp = 0; sp < f->nb_stream_params; sp++)
                {
                    if (strcmp(f->stream_param_value[sp], sname) == 0)
                    {
                        out->fps_param_mask[fi] |= (UINT32_C(1) << sp);
                    }
                } /* for sp */
            } /* if FOCUS_STREAMS */

            (void) fps_is_src; /* suppress unused-var warning */
        }
        else if (n->type == OV_NODE_PROC && n->index >= 0 && n->index < m->nb_procs)
        {
            bset(out->procs, n->index);
            if (is_write)
            {
                bset(out->proc_writes, n->index);
            }
        }
    } /* for ei */
}
