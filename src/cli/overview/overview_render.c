/**
 * @file    overview_render.c
 * @brief   Shared render utilities and orchestrator
 *          for milkCTRL
 *
 * Panel-specific rendering lives in:
 *   - overview_render_streams.c
 *   - overview_render_procs.c
 *   - overview_render_fps.c
 */

#include "overview_render_internal.h"


extern float ov_scan_get_interval(void);

/* =========================================================
 * Persistent sort ordering caches
 * ========================================================= */
static char g_stream_order[OV_MAX_STREAMS][80];
static int  g_nb_stream_order = 0;

static char g_proc_order[OV_MAX_PROCS][80];
static int  g_nb_proc_order = 0;

static char g_fps_order[OV_MAX_FPS][80];
static int  g_nb_fps_order = 0;

static const OV_MODEL *g_last_model = NULL;

static int get_stream_rank(const char *name)
{
    for (int i = 0; i < g_nb_stream_order; i++)
    {
        if (strncmp(g_stream_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

static int get_proc_rank(const char *name)
{
    for (int i = 0; i < g_nb_proc_order; i++)
    {
        if (strncmp(g_proc_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

static int get_fps_rank(const char *name)
{
    for (int i = 0; i < g_nb_fps_order; i++)
    {
        if (strncmp(g_fps_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

static int sort_stream_by_rank(const void *a, const void *b)
{
    int ra = get_stream_rank(((const OV_STREAM *)a)->name);
    int rb = get_stream_rank(((const OV_STREAM *)b)->name);
    if (ra != rb)
    {
        return ra - rb;
    }
    return strcmp(((const OV_STREAM *)a)->name, ((const OV_STREAM *)b)->name);
}

static int sort_proc_by_rank(const void *a, const void *b)
{
    int ra = get_proc_rank(((const OV_PROC *)a)->name);
    int rb = get_proc_rank(((const OV_PROC *)b)->name);
    if (ra != rb)
    {
        return ra - rb;
    }
    return strcmp(((const OV_PROC *)a)->name, ((const OV_PROC *)b)->name);
}

static int sort_fps_by_rank(const void *a, const void *b)
{
    int ra = get_fps_rank(((const OV_FPS *)a)->name);
    int rb = get_fps_rank(((const OV_FPS *)b)->name);
    if (ra != rb)
    {
        return ra - rb;
    }
    return strcmp(((const OV_FPS *)a)->name, ((const OV_FPS *)b)->name);
}

static void ov_apply_rank_sort(OV_MODEL *mm)
{
    if (g_nb_stream_order > 0)
    {
        qsort(mm->streams, (size_t)mm->nb_streams, sizeof(OV_STREAM), sort_stream_by_rank);
    }
    if (g_nb_proc_order > 0)
    {
        qsort(mm->procs, (size_t)mm->nb_procs, sizeof(OV_PROC), sort_proc_by_rank);
    }
    if (g_nb_fps_order > 0)
    {
        qsort(mm->fps, (size_t)mm->nb_fps, sizeof(OV_FPS), sort_fps_by_rank);
    }
}

void render_pad_spaces(int chars_written, int panel_width);

/**
 * render_highlighted_name - print a string with regex match highlighting
 */
void render_highlighted_name(
    const char *name,
    int         max_len,
    regex_t    *re,
    int         has_re,
    ov_rgb_t    normal_fg,
    ov_rgb_t    row_bg)
{
    int len = (int) strlen(name);
    if (len > max_len) len = max_len;

    regmatch_t pm[1];
    if (has_re && regexec(re, name, 1, pm, 0) == 0)
    {
        int b_len = pm[0].rm_so;
        if (b_len > max_len) b_len = max_len;

        int m_len = pm[0].rm_eo - pm[0].rm_so;
        if (b_len + m_len > max_len) m_len = max_len - b_len;

        int a_len = len - (b_len + m_len);

        if (b_len > 0)
        {
            ov_buf_printf("%.*s", b_len, name);
        }
        if (m_len > 0)
        {
            ov_buf_bold();
            ov_buf_fg(255, 255, 255);
            ov_buf_printf("%.*s", m_len, name + b_len);
            ov_buf_reset_attr();
            ov_theme_bg(row_bg);
            ov_theme_fg(normal_fg);
        }
        if (a_len > 0)
        {
            ov_buf_printf("%.*s", a_len, name + b_len + m_len);
        }
    }
    else
    {
        ov_buf_printf("%.*s", len, name);
    }

    int pad = max_len - len;
    if (pad > 0)
    {
        ov_buf_hline(' ', pad);
    }
}

/**
 * ov_filter_build - build filtered index array.
 * @pattern:  regex pattern string (empty = match all)
 * @names:    array of name pointers
 * @count:    total item count
 * @out:      output index array (caller-allocated)
 * @max_out:  capacity of @out
 *
 * Return: number of matching indices written to @out.
 */
int ov_filter_build(
    const char  *pattern,
    const char **names,
    int          count,
    int         *out,
    int          max_out)
{
    if (pattern[0] == '\0')
    {
        /* No filter — all items match */
        int n = count < max_out ? count : max_out;
        for (int i = 0; i < n; i++)
        {
            out[i] = i;
        }
        return n;
    }

    regex_t re;
    if (regcomp(&re, pattern,
                REG_EXTENDED | REG_NOSUB
                | REG_ICASE) != 0)
    {
        /* Invalid regex — show all */
        int n = count < max_out ? count : max_out;
        for (int i = 0; i < n; i++)
        {
            out[i] = i;
        }
        return n;
    }

    int n = 0;
    for (int i = 0; i < count && n < max_out; i++)
    {
        if (regexec(&re, names[i], 0, NULL, 0) == 0)
        {
            out[n++] = i;
        }
    }
    regfree(&re);
    return n;
}

/* =========================================================
 * Cross-panel relation highlight
 * ========================================================= */

/*
 * Bitset helpers and OV_RELATED defined in
 * overview_render_internal.h
 */

void bset(uint64_t *words, int idx)
{
    words[idx / BITS_PER_WORD] |= (UINT64_C(1) << (idx % BITS_PER_WORD));
}

int bget(const uint64_t *words, int idx)
{
    return (words[idx / BITS_PER_WORD] >> (idx % BITS_PER_WORD)) & 1;
}

/* =========================================================
 * Stream lineage (ancestors / descendants)
 * ========================================================= */

/* Lineage types and BFS now in stream_graph.{h,c} */

/**
 * ov_compute_related - find items in other panels related to the selection.
 * @lay: current layout (holds focus + sel_* indices)
 * @m:   data model
 * @out: result bitsets (cleared on entry)
 *
 * Traverses graph edges from the selected node and marks all
 * reachable streams, FPS entries, and processes.
 */
static void ov_compute_related(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    OV_RELATED      *out)
{
    memset(out, 0, sizeof(*out));
    /* fps_param_mask initialised to 0 by memset — no matches yet */

    /* Use frozen selection when freeze is active */
    ov_focus_t focus   = lay->freeze
                         ? lay->freeze_focus
                         : lay->focus;
    int sel_stream_idx = lay->freeze
                         ? lay->freeze_sel_stream
                         : lay->sel_stream;
    int sel_proc_idx   = lay->freeze
                         ? lay->freeze_sel_proc
                         : lay->sel_proc;
    int sel_fps_idx    = lay->freeze
                         ? lay->freeze_sel_fps
                         : lay->sel_fps;

    /* Determine the graph node index of the selected item */
    int sel_node = -1;
    if (focus == OV_FOCUS_STREAMS && sel_stream_idx >= 0
        && sel_stream_idx < m->nb_streams)
    {
        sel_node = m->streams[sel_stream_idx].node_idx;
    }
    else if (focus == OV_FOCUS_FPS && sel_fps_idx >= 0
             && sel_fps_idx < m->nb_fps)
    {
        sel_node = m->fps[sel_fps_idx].node_idx;
    }
    else if (focus == OV_FOCUS_PROCS && sel_proc_idx >= 0
             && sel_proc_idx < m->nb_procs)
    {
        sel_node = m->procs[sel_proc_idx].node_idx;
        out->sel_pid = m->procs[sel_proc_idx].PID;
    }

    if (sel_node < 0)
    {
        return;
    }

    /* Walk all edges; mark neighbours of sel_node */
    for (int ei = 0; ei < m->nb_edges; ei++)
    {
        const OV_EDGE *e = &m->edges[ei];
        int other    = -1;
        int is_write = 0; /* 1 = proc writes stream */
        /* For FPS edges: which end is the stream node? */
        int fps_is_src = 0;

        if (e->src_node == sel_node)
        {
            other      = e->tgt_node;
            is_write   = (e->type == OV_EDGE_PROC_WRITES_STREAM);
            fps_is_src = 0; /* FPS is tgt when stream→FPS */
        }
        else if (e->tgt_node == sel_node)
        {
            other      = e->src_node;
            is_write   = 0;
            fps_is_src = 1; /* FPS is src when FPS→stream */
        }

        if (other < 0 || other >= m->nb_nodes)
        {
            continue;
        }

        const OV_NODE *n = &m->nodes[other];
        if (n->type == OV_NODE_STREAM && n->index >= 0
            && n->index < m->nb_streams)
        {
            bset(out->streams, n->index);
        }
        else if (n->type == OV_NODE_FPS && n->index >= 0
                 && n->index < m->nb_fps)
        {
            int fi = n->index;
            bset(out->fps, fi);

            /* Find all stream params of this FPS that match sel_node.
             * Only meaningful when the selection is a stream.
             * OR all matching indices into the bitmask. */
            if (focus == OV_FOCUS_STREAMS
                && sel_stream_idx >= 0
                && sel_stream_idx < m->nb_streams)
            {
                const char *sname =
                    m->streams[sel_stream_idx].name;
                const OV_FPS *f = &m->fps[fi];
                for (int sp = 0; sp < f->nb_stream_params; sp++)
                {
                    if (strcmp(f->stream_param_value[sp],
                               sname) == 0)
                    {
                        out->fps_param_mask[fi] |=
                            (UINT32_C(1) << sp);
                    }
                } /* for sp */
            } /* if FOCUS_STREAMS */

            (void) fps_is_src; /* suppress unused-var warning */
        }
        else if (n->type == OV_NODE_PROC && n->index >= 0
                 && n->index < m->nb_procs)
        {
            bset(out->procs, n->index);
            if (is_write)
            {
                bset(out->proc_writes, n->index);
            }
        }
    } /* for ei */
}

const char *render_dtype(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:   return "UI8";
    case _DATATYPE_INT8:    return "SI8";
    case _DATATYPE_UINT16:  return "U16";
    case _DATATYPE_INT16:   return "S16";
    case _DATATYPE_UINT32:  return "U32";
    case _DATATYPE_INT32:   return "S32";
    case _DATATYPE_UINT64:  return "U64";
    case _DATATYPE_INT64:   return "S64";
    case _DATATYPE_FLOAT:   return "F32";
    case _DATATYPE_DOUBLE:  return "F64";
    default:                return "???";
    }
}

/**
 * dtype_bytesize - bytes per element for a datatype.
 */
int dtype_bytesize(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
    case _DATATYPE_INT8:    return 1;
    case _DATATYPE_UINT16:
    case _DATATYPE_INT16:   return 2;
    case _DATATYPE_UINT32:
    case _DATATYPE_INT32:
    case _DATATYPE_FLOAT:   return 4;
    case _DATATYPE_UINT64:
    case _DATATYPE_INT64:
    case _DATATYPE_DOUBLE:  return 8;
    default:                return 1;
    }
}

static const char *view_label(ov_view_t v)
{
    switch (v)
    {
    case OV_VIEW_DASHBOARD: return "DASH";
    case OV_VIEW_GRAPH:     return "GRAPH";
    case OV_VIEW_STREAMS:   return "STRM";
    case OV_VIEW_PROCS:     return "PROC";
    case OV_VIEW_FPS:       return "FPS";
    default:                return "";
    }
}

void clear_row(
    int row,
    int col,
    int width,
    ov_rgb_t bg)
{
    ov_buf_pos(row, col);
    ov_theme_bg(bg);
    ov_buf_hline(' ', width);
}

/* Pad the remainder of a panel's interior row */
void render_pad_spaces(int chars_written, int panel_width)
{
    int remain = (panel_width - 2) - chars_written;
    if (remain > 0)
    {
        ov_buf_hline(' ', remain);
    }
}

/**
 * render_scroll_indicators - draw ▲N / ▼N on panel borders.
 * @r:         panel rect
 * @scroll:    current scroll offset (first visible index)
 * @max_rows:  visible rows in the panel body
 * @total:     total item count
 * @accent:    panel accent color for the arrow
 *
 * Draws indicators on the top and bottom border lines of
 * the panel to show how many items are hidden above / below.
 */
void render_scroll_indicators(
    OV_RECT  r,
    int      scroll,
    int      max_rows,
    int      total,
    ov_rgb_t accent)
{
    int above = scroll;
    int below = total - scroll - max_rows;
    if (below < 0)
    {
        below = 0;
    }

    /* Top border: "▲ N more" right-aligned inside border */
    if (above > 0)
    {
        char buf[32];
        int n = snprintf(buf, sizeof(buf), " ▲%d ", above);
        /* display width: space + ▲(1col) + digits + space */
        int dw = 3;
        {
            int tmp = above;
            while (tmp > 0) { dw++; tmp /= 10; }
        }
        int col = r.col + r.width - dw - 2;
        if (col > r.col + 2)
        {
            ov_buf_pos(r.row, col);
            ov_theme_fg(accent);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", buf);
            (void) n;
        }
    }

    /* Bottom border: "▼ N more" right-aligned inside border */
    if (below > 0)
    {
        char buf[32];
        int n = snprintf(buf, sizeof(buf), " ▼%d ", below);
        int dw = 3;
        {
            int tmp = below;
            while (tmp > 0) { dw++; tmp /= 10; }
        }
        int brow = r.row + r.height - 1;
        int col  = r.col + r.width - dw - 2;
        if (col > r.col + 2)
        {
            ov_buf_pos(brow, col);
            ov_theme_fg(accent);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", buf);
            (void) n;
        }
    }
}

static double get_cpu_usage(void)
{
    static struct rusage last_usage;
    static struct timespec last_time;
    static int initialized = 0;
    static double smoothed_cpu = 0.0;
    
    struct rusage current_usage;
    struct timespec current_time;
    
    getrusage(RUSAGE_SELF, &current_usage);
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    
    if (!initialized) {
        last_usage = current_usage;
        last_time = current_time;
        initialized = 1;
        return 0.0;
    }
    
    double dt = (current_time.tv_sec - last_time.tv_sec) +
                (current_time.tv_nsec - last_time.tv_nsec) / 1e9;
                
    if (dt >= 0.5) { /* update every 0.5s */
        double d_utime = (current_usage.ru_utime.tv_sec - last_usage.ru_utime.tv_sec) +
                         (current_usage.ru_utime.tv_usec - last_usage.ru_utime.tv_usec) / 1e6;
        double d_stime = (current_usage.ru_stime.tv_sec - last_usage.ru_stime.tv_sec) +
                         (current_usage.ru_stime.tv_usec - last_usage.ru_stime.tv_usec) / 1e6;
                         
        double inst_cpu = 100.0 * (d_utime + d_stime) / dt;
        smoothed_cpu = inst_cpu;
        last_usage = current_usage;
        last_time = current_time;
    }
    return smoothed_cpu;
}

void ov_render_header(
    OV_LAYOUT       *lay,
    const OV_MODEL  *m)
{
    /* Advance blink counter each frame */
    lay->ctrl_blink++;

    OV_RECT r = lay->r_header;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);

    ov_theme_fg(OV_FG_BRIGHT);
    ov_buf_bold();
    ov_buf_printf(" %s milkCTRL ", OV_BULLET);
    ov_buf_reset_attr();

    ov_theme_bg(OV_BG_HEADER);

    /* Blinking [CTRL] badge — visible when ctrl_mode is ON */
    int ctrl_w = 0;
    if (lay->ctrl_mode)
    {
        /* Blink: show solid for 4 frames, dim for 2 frames (6-frame cycle) */
        int blink_on = ((lay->ctrl_blink / 4) % 2 == 0);
        if (blink_on)
        {
            ov_buf_bg(180, 20, 20);    /* deep red background */
            ov_buf_fg(255, 220, 220);  /* light text */
            ov_buf_bold();
            ov_buf_printf(" CTRL ");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
        }
        else
        {
            ov_buf_bg(80, 10, 10);     /* dim red background */
            ov_buf_fg(160, 80, 80);    /* dim text */
            ov_buf_printf(" CTRL ");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
        }
        ctrl_w = 6; /* visual width of " CTRL " */
    }

    ov_theme_fg(OV_FG_STREAM);
    int c2 = snprintf(NULL, 0, " %d stm", m->nb_streams);
    ov_buf_printf(" %d stm", m->nb_streams);

    ov_theme_fg(OV_FG_PROC);
    int c3 = snprintf(NULL, 0, " %d prc", m->nb_procs);
    ov_buf_printf(" %d prc", m->nb_procs);

    ov_theme_fg(OV_FG_FPS);
    int c4 = snprintf(NULL, 0, " %d fps", m->nb_fps);
    ov_buf_printf(" %d fps", m->nb_fps);

    ov_theme_fg(OV_FG_CONN);
    int c5 = snprintf(NULL, 0, " %d edg", m->nb_edges);
    ov_buf_printf(" %d edg", m->nb_edges);

    ov_theme_fg(OV_FG_DIM);
    double cpu_pct = get_cpu_usage();
    int c6 = snprintf(NULL, 0, "  CPU: %4.1f%%", cpu_pct);
    ov_buf_printf("  CPU: %4.1f%%", cpu_pct);

    /* c1 visual length: 17 chars for " ● milkCTRL " */
    int chars_left = 17 + ctrl_w + c2 + c3 + c4 + c5 + c6;

    int tabs_width = 0;
    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        tabs_width += (int) strlen(view_label((ov_view_t) v)) + 5;
    }

    int pad = r.width - tabs_width - chars_left;
    if (pad < 1)
    {
        pad = 1;
    }
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_hline(' ', pad);

    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        if (v == (int) lay->view)
        {
            ov_theme_bg(OV_BG_SELECTED);
            ov_theme_fg(OV_FG_BRIGHT);
            ov_buf_bold();
        }
        else
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_DIM);
        }
        ov_buf_printf(" F%d:%s ", v + 2, view_label((ov_view_t) v));
        ov_buf_reset_attr();
    }

    ov_theme_bg(OV_BG_HEADER);
}

void ov_render_frame(
    OV_LAYOUT       *lay,
    const OV_MODEL  *m)
{
    ov_buf_reset();

    /* One-shot sort: only runs once when the user
     * presses S or s.  Order stays frozen until
     * the user explicitly presses S/s again. */
    if (lay->sort_pending)
    {
        OV_MODEL *mm = (OV_MODEL *)(uintptr_t) m;
        ov_sort_streams(mm,
                        lay->sort_key_stream,
                        lay->sort_dir_stream);
        ov_sort_procs(mm,
                      lay->sort_key_proc,
                      lay->sort_dir_proc);
        ov_sort_fps(mm,
                    lay->sort_key_fps,
                    lay->sort_dir_fps);
        
        g_nb_stream_order = mm->nb_streams;
        for (int i = 0; i < mm->nb_streams; i++)
        {
            strncpy(g_stream_order[i], mm->streams[i].name, 79);
            g_stream_order[i][79] = '\0';
        }
        
        g_nb_proc_order = mm->nb_procs;
        for (int i = 0; i < mm->nb_procs; i++)
        {
            strncpy(g_proc_order[i], mm->procs[i].name, 79);
            g_proc_order[i][79] = '\0';
        }
        
        g_nb_fps_order = mm->nb_fps;
        for (int i = 0; i < mm->nb_fps; i++)
        {
            strncpy(g_fps_order[i], mm->fps[i].name, 79);
            g_fps_order[i][79] = '\0';
        }

        lay->sort_pending = 0;
        g_last_model = m;
    }
    else if (m != g_last_model)
    {
        /* A new scan model arrived. Re-apply the saved order so items don't shuffle. */
        OV_MODEL *mm = (OV_MODEL *)(uintptr_t) m;
        ov_apply_rank_sort(mm);
        g_last_model = m;
    }

    /* Always patch graph node .index fields to
     * reflect current array positions — needed
     * whether we just sorted or scan rebuilt the
     * model.  Each item's node_idx still points
     * to its graph node; update the reverse link. */
    {
        OV_MODEL *mm = (OV_MODEL *)(uintptr_t) m;
        for (int i = 0; i < mm->nb_streams; i++)
        {
            int ni = mm->streams[i].node_idx;
            if (ni >= 0 && ni < mm->nb_nodes)
            {
                mm->nodes[ni].index = i;
            }
        }
        for (int i = 0; i < mm->nb_fps; i++)
        {
            int ni = mm->fps[i].node_idx;
            if (ni >= 0 && ni < mm->nb_nodes)
            {
                mm->nodes[ni].index = i;
            }
        }
        for (int i = 0; i < mm->nb_procs; i++)
        {
            int ni = mm->procs[i].node_idx;
            if (ni >= 0 && ni < mm->nb_nodes)
            {
                mm->nodes[ni].index = i;
            }
        }
    }

    /* Compute cross-panel relation set once per frame */
    OV_RELATED rel;
    ov_compute_related(lay, m, &rel);

    /* Start frame: begin synchronized update, then cursor home */


    ov_render_header(lay, m);

    /* To prevent flickering on terminals that do not support synchronized updates,
     * we skip rendering the background panels when the help overlay is active.
     * The existing background is preserved on the terminal's screen. */
    if (!lay->show_help)
    {
        switch (lay->view)
        {
        case OV_VIEW_DASHBOARD:
            ov_render_preview_line(lay, m);
            ov_render_streams_panel(lay, m, &rel);
            ov_render_procs_panel(lay, m, &rel);
            ov_render_fps_panel(lay, m, &rel);
            /* Detail pane replaces graph when active */
            if (!lay->detail_mode
                || !ov_render_detail_panel(lay, m))
            {
                ov_render_graph_panel(lay, m);
            }
            break;
        case OV_VIEW_GRAPH:
            ov_render_graph_panel(lay, m);
            break;
        case OV_VIEW_STREAMS:
            ov_render_streams_panel(lay, m, &rel);
            break;
        case OV_VIEW_PROCS:
            ov_render_procs_panel(lay, m, &rel);
            break;
        case OV_VIEW_FPS:
            ov_render_fps_panel(lay, m, &rel);
            break;
        default:
            break;
        }
    }

    if (lay->show_help)
    {
        ov_render_help(lay);
    }

    ov_render_cmdlog(lay);
    ov_render_status(lay, m);

    /* End frame */

    ov_buf_flush_delta(lay->term_rows, lay->term_cols);
}
