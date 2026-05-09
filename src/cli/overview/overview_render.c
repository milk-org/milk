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

/* All overview headers included via
 * overview_render_internal.h */

/* forward declarations for scan API */
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

/**
 * ov_render_preview_line - full-width preview of
 *     the currently selected item (dashboard only).
 *
 * Renders on row 2, between header and panels.
 * Shows untruncated fields for the focused panel's
 * selected item.
 */
static void ov_render_preview_line(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    int W = lay->term_cols;

    ov_buf_pos(2, 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_hline(' ', W);
    ov_buf_pos(2, 1);

    /* Use frozen selection when freeze is active */
    ov_focus_t focus = lay->freeze
                       ? lay->freeze_focus
                       : lay->focus;
    int ssel = lay->freeze
               ? lay->freeze_sel_stream
               : lay->sel_stream;
    int psel = lay->freeze
               ? lay->freeze_sel_proc
               : lay->sel_proc;
    int fsel = lay->freeze
               ? lay->freeze_sel_fps
               : lay->sel_fps;

    char line[512];
    int  len = 0;
    ov_rgb_t label_color = OV_FG_DIM;

    switch (focus)
    {
    case OV_FOCUS_STREAMS:
    {
        label_color = OV_FG_STREAM;
        if (ssel < 0 || ssel >= m->nb_streams)
        {
            break;
        }
        const OV_STREAM *s = &m->streams[ssel];
        char szb[32];
        if (s->naxis == 1)
        {
            snprintf(szb, sizeof(szb), "%u",
                (unsigned) s->size[0]);
        }
        else if (s->naxis == 2)
        {
            snprintf(szb, sizeof(szb), "%ux%u",
                (unsigned) s->size[0],
                (unsigned) s->size[1]);
        }
        else
        {
            snprintf(szb, sizeof(szb),
                "%ux%ux%u",
                (unsigned) s->size[0],
                (unsigned) s->size[1],
                (unsigned) s->size[2]);
        }
        len = snprintf(line, sizeof(line),
            " STM  %s  %s %s"
            "  Hz:%.1f  ino:%lu"
            "  own:%d  cnt:%lu"
            "  wpid:%d  sem:%d",
            s->name,
            render_dtype(s->datatype),
            szb,
            s->update_hz,
            (unsigned long) s->inode,
            (int) s->ownerPID,
            (unsigned long) s->cnt0,
            (int) s->write_pid,
            s->nb_sem);
        break;
    }
    case OV_FOCUS_PROCS:
    {
        label_color = OV_FG_PROC;
        if (psel < 0 || psel >= m->nb_procs)
        {
            break;
        }
        const OV_PROC *p = &m->procs[psel];
        const char *sl;
        switch (p->loopstat)
        {
        case 0:  sl = "IDLE"; break;
        case 1:  sl = "RUN";  break;
        case 2:  sl = "PAUS"; break;
        case 3:  sl = "TERM"; break;
        case 4:  sl = "ERR";  break;
        default: sl = "??";   break;
        }
        len = snprintf(line, sizeof(line),
            " PRC  %s  PID:%d  %s"
            "  Hz:%.1f  trig:%s"
            "  sem:%d  loop:%ld"
            "  miss:%d  prio:%d",
            p->name, (int) p->PID, sl,
            p->loop_hz,
            p->trigstreamname[0]
                ? p->trigstreamname : "-",
            p->triggersem,
            (long) p->loopcnt,
            p->triggermissed,
            p->rt_priority);
        break;
    }
    case OV_FOCUS_FPS:
    {
        label_color = OV_FG_FPS;
        if (fsel < 0 || fsel >= m->nb_fps)
        {
            break;
        }
        const OV_FPS *f = &m->fps[fsel];
        len = snprintf(line, sizeof(line),
            " FPS  %s  C:%s R:%s"
            "  st:%08X  cpid:%d  rpid:%d"
            "  %s",
            f->name,
            f->conf_alive ? "Y" : "-",
            f->run_alive  ? "Y" : "-",
            f->md_status,
            (int) f->confpid,
            (int) f->runpid,
            f->description);
        break;
    }
    default:
        break;
    }

    if (len > 0)
    {
        /* Label badge */
        ov_theme_bg(label_color);
        ov_buf_fg(0, 0, 0);
        ov_buf_bold();
        ov_buf_printf("%.*s", 5, line);
        ov_buf_reset_attr();

        /* Remaining content */
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_TEXT);
        int rem = len - 5;
        if (rem > W - 5)
        {
            rem = W - 5;
        }
        if (rem > 0)
        {
            ov_buf_printf("%.*s", rem, line + 5);
        }
    }

    /* [FREEZE] badge on right edge when frozen */
    if (lay->freeze)
    {
        const char *badge = " FREEZE ";
        int bw = 8;
        int col = W - bw + 1;
        if (col > 1)
        {
            ov_buf_pos(2, col);
            ov_buf_bg(60, 130, 200);
            ov_buf_fg(255, 255, 255);
            ov_buf_bold();
            ov_buf_printf("%s", badge);
        }
    }

    ov_buf_reset_attr();
}


void ov_render_graph_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    OV_RECT r = lay->r_graph;
    ov_draw_panel_border(r.row, r.col, r.height, r.width, "CONNECTIONS", OV_FG_CONN, lay->focus == OV_FOCUS_GRAPH);

    int max_rows = r.height - 3;
    int row = r.row + 1;

    /* Render Header */
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);
    char htext[256];
    int hlen = snprintf(htext, sizeof(htext), " %-12s %-4s %-12s %-6s %-16s %-4s %-5s %-4s", 
                        "PROCESS", "PID", "STREAM", "STATUS", "LOOPCNT", "MISS", "MEM", "PR");
    ov_buf_printf("%s", htext);
    render_pad_spaces(hlen, r.width);
    row++;

    if (m->nb_edges == 0)
    {
        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("  ");
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("No connections detected");
        render_pad_spaces(2 + 23, r.width);
        row++;
        for (int i = 1; i < max_rows; i++, row++) {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return;
    }

    int rendered_rows = 0;
    for (int ei = 0; ei < m->nb_edges && rendered_rows < max_rows; ei++)
    {
        const OV_EDGE *e = &m->edges[ei];
        if (e->src_node < 0 || e->src_node >= m->nb_nodes || e->tgt_node < 0 || e->tgt_node >= m->nb_nodes) continue;

        const OV_NODE *src = &m->nodes[e->src_node];
        const OV_NODE *tgt = &m->nodes[e->tgt_node];

        ov_buf_pos(row, r.col + 1);
        
        int is_sel = (ei == lay->sel_graph && lay->focus == OV_FOCUS_GRAPH);
        ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : OV_BG_PANEL;
        ov_theme_bg(row_bg);
        ov_buf_printf(" ");

        /* Detailed rendering with color */
        ov_rgb_t sc;
        const char *styp = "UNK";
        switch (src->type) {
        case OV_NODE_STREAM: sc = OV_FG_STREAM; styp = "STRM"; break;
        case OV_NODE_FPS:    sc = OV_FG_FPS;    styp = "FPS "; break;
        case OV_NODE_PROC:   sc = OV_FG_PROC;   styp = "PROC"; break;
        }

        ov_rgb_t tc;
        const char *ttyp = "UNK";
        switch (tgt->type) {
        case OV_NODE_STREAM: tc = OV_FG_STREAM; ttyp = "STRM"; break;
        case OV_NODE_FPS:    tc = OV_FG_FPS;    ttyp = "FPS "; break;
        case OV_NODE_PROC:   tc = OV_FG_PROC;   ttyp = "PROC"; break;
        }

        const char *etype = "UNKNOWN";
        switch (e->type) {
        case OV_EDGE_PROC_WRITES_STREAM:   etype = "PROC_WRITES_STRM"; break;
        case OV_EDGE_STREAM_TRIGGERS_PROC: etype = "STRM_TRIGS_PROC "; break;
        case OV_EDGE_FPS_RUNS_PROC:        etype = "FPS_RUNS_PROC   "; break;
        case OV_EDGE_FPS_INPUT_STREAM:     etype = "FPS_INPUT_STRM  "; break;
        case OV_EDGE_FPS_OUTPUT_STREAM:    etype = "FPS_OUTPUT_STRM "; break;
        case OV_EDGE_PROC_TRIGGER_STREAM:  etype = "PROC_TRIGS_STRM "; break;
        }

        int printed = 1;
        int avail = r.width - 2;

        #define GRAPH_FIELD(color, fmt, ...)         \
        do {                                         \
            char _fb[128];                           \
            int _fl = snprintf(                      \
                _fb, sizeof(_fb), fmt,               \
                ##__VA_ARGS__);                      \
            int _vis = _fl;                          \
            int _max = avail - printed;              \
            if (_vis > _max) _vis = _max;            \
            if (_vis > 0) {                          \
                ov_theme_fg(color);                  \
                ov_buf_printf("%.*s", _vis, _fb);    \
                printed += _vis;                     \
            }                                        \
        } while(0)

        GRAPH_FIELD(sc, "%-12.12s", src->name);
        
        /* Box drawing character length workaround */
        if (avail - printed >= 4) {
            ov_theme_fg(OV_FG_CONN);
            ov_buf_printf(" %s%s ", OV_BOX_H, OV_TRI_R);
            printed += 4;
        }

        GRAPH_FIELD(tc, "%-12.12s", tgt->name);
        GRAPH_FIELD(OV_FG_DIM, " [%.6s]", e->label);
        GRAPH_FIELD(OV_FG_TEXT, " %-16.16s ", etype);
        GRAPH_FIELD(sc, "%-4s ", styp);
        GRAPH_FIELD(tc, "%-4s", ttyp);

        #undef GRAPH_FIELD

        render_pad_spaces(printed, r.width);
        row++;
        rendered_rows++;
    }
    
    for (; rendered_rows < max_rows; rendered_rows++, row++)
    {
        clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    
    ov_buf_reset_attr();
}

void ov_render_status(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    OV_RECT r = lay->r_status;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);

    double interval = (double) ov_scan_get_interval();
    double rate_hz  = (interval > 0.0) ? (1.0 / interval) : 0.0;

    /* Context-sensitive ctrl hints */
    const char *ctrl_hint = "";
    if (lay->ctrl_mode)
    {
        switch (lay->focus)
        {
        case OV_FOCUS_FPS:
            ctrl_hint = "  r:run s:conf";
            break;
        case OV_FOCUS_STREAMS:
            ctrl_hint = "  d:delete";
            break;
        case OV_FOCUS_PROCS:
            ctrl_hint = "  k:kill";
            break;
        default:
            ctrl_hint = "";
            break;
        }
    }

    /* Sort key label */
    const char *sort_label = "";
    switch (lay->focus)
    {
    case OV_FOCUS_STREAMS:
        switch (lay->sort_key_stream)
        {
        case 1:  sort_label = " [sort:typ]";   break;
        case 2:  sort_label = " [sort:size]";  break;
        case 3:  sort_label = " [sort:Hz]";    break;
        case 4:  sort_label = " [sort:MB/s]";  break;
        case 5:  sort_label = " [sort:inode]"; break;
        case 6:  sort_label = " [sort:count]"; break;
        default: sort_label = " [sort:name]";  break;
        }
        break;
    case OV_FOCUS_PROCS:
        switch (lay->sort_key_proc)
        {
        case 1:  sort_label = " [sort:PID]";  break;
        case 2:  sort_label = " [sort:stat]"; break;
        case 3:  sort_label = " [sort:Hz]";   break;
        case 4:  sort_label = " [sort:MEM]";  break;
        default: sort_label = " [sort:name]"; break;
        }
        break;
    case OV_FOCUS_FPS:
        switch (lay->sort_key_fps)
        {
        case 1:  sort_label = " [sort:alive]"; break;
        case 2:  sort_label = " [sort:MEM]";   break;
        default: sort_label = " [sort:name]";  break;
        }
        break;
    default:
        break;
    }

    const char *detail_label =
        lay->detail_mode ? " [DETAIL]" : "";

    /* Filter editing prompt overrides normal status */
    if (lay->filter_editing)
    {
        const char *fstr = "";
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            fstr = lay->filter_stream;
            break;
        case OV_FOCUS_PROCS:
            fstr = lay->filter_proc;
            break;
        case OV_FOCUS_FPS:
            fstr = lay->filter_fps;
            break;
        default:
            break;
        }
        ov_theme_fg(OV_FG_TEXT);
        int np = snprintf(
            NULL, 0, " /%s", fstr);
        ov_buf_printf(" /%s", fstr);

        /* Blinking cursor */
        ov_buf_fg(255, 200, 50);
        ov_buf_printf("█");
        np += 1;

        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("  (ENTER=accept ESC=cancel)");
        np += 26;

        int pad = r.width - np - 1;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }
        ov_buf_reset_attr();
        return;
    }

    int n1 = 0;
    if (lay->paused)
    {
        ov_buf_fg(255, 50, 50);
        n1 = snprintf(NULL, 0, " [PAUSED] ");
        ov_buf_printf(" [PAUSED] ");
        ov_theme_fg(OV_FG_DIM);
    }
    else
    {
        n1 = snprintf(NULL, 0, " scan:%.0fms %.1fHz", m->scan_time_ms, rate_hz);
        ov_buf_printf(" scan:%.0fms %.1fHz", m->scan_time_ms, rate_hz);
    }

    int n_hints = snprintf(NULL, 0,
        "%s%s%s  +/- TAB D S/s / p c h q",
        ctrl_hint, sort_label, detail_label);
    ov_buf_printf(
        "%s%s%s  +/- TAB D S/s / p c h q",
        ctrl_hint, sort_label, detail_label);
    n1 += n_hints;

    time_t now = time(NULL);
    struct tm *tm_ptr = localtime(&now);
    char tstr[16];
    int n2 = strftime(tstr, sizeof(tstr), "%H:%M:%S", tm_ptr);

    /* Leave 1 char at the end to prevent terminal auto-scroll on last row */
    int pad = r.width - n1 - n2 - 1;
    if (pad > 0)
    {
        ov_buf_hline(' ', pad);
    }
    ov_theme_fg(OV_FG_TEXT);
    ov_buf_printf("%s ", tstr);
    ov_buf_reset_attr();
}

/* ov_render_help
 * Draws a centered help overlay.
 *
 * Algorithm:
 *   1. Enumerate all lines into a static table so pw/ph are exact.
 *   2. Fill the entire terminal with a dim translucent background
 *      BEFORE drawing the box — every terminal cell is written exactly
 *      once, eliminating the panel-under-help flicker.
 *   3. Draw border + content.
 */
void ov_render_help(const OV_LAYOUT *lay)
{
    /* ---- Content table ---- */
    static const struct
    {
        const char *text;
        int         color_row; /* 1 = render as colour-legend row */
    } lines[] =
    {
        { "Navigation",                              0 },
        { "  F2-F6 / ^Left/^Right  Switch views",    0 },
        { "  TAB      Cycle panel focus",            0 },
        { "  UP/DOWN  Navigate list",                0 },
        { "  Left/Right  Panel focus / scroll",      0 },
        { "  PgUp/Dn  Scroll page",                  0 },
        { "  Home/End  Jump to top/bottom",          0 },
        { "Sorting",                                 0 },
        { "  </>      Change sort column",           0 },
        { "  [        Toggle sort direction",        0 },
        { "  S        Sort by activity (Hz)",        0 },
        { "  s        Sort by name",                 0 },
        { "Display",                                 0 },
        { "  +/-      Adjust scan rate",             0 },
        { "  D        Toggle detail pane",           0 },
        { "  L        Toggle lineage mode",          0 },
        { "  p        Pause/resume display",         0 },
        { "  SPACE    Freeze selection highlight",   0 },
        { "  /        Filter (regex search)",        0 },
        { "  W        Export snapshot to file",      0 },
        { "  h        Toggle this help",             0 },
        { "  q        Quit",                         0 },
        { "Control mode  (c to toggle)",             0 },
        { "  FPS:  r=run  s=conf",                   0 },
        { "  STRM: d=delete stream",                 0 },
        { "Process signals  (PROCS & FPS panels)",   0 },
        { "  k  graceful kill (SIGTERM)",            0 },
        { "  K  immediate kill (SIGKILL)",           0 },
        { "  x  pause/resume (SIGSTOP/SIGCONT)",     0 },
        { "Detail View (ENTER or D)",                0 },
        { "  Toggles detail pane for selected item", 0 },
        { "Columns",                                 0 },
        { "  STRM: MB/s throughput, total in footer",0 },
        { "  PROC: DUTY% (exec/iter), CPU%, MEM",    0 },
        { "  FPS:  MEM (RSS)",                       0 },
        { "Mouse",                                     0 },
        { "  Click=select  DblClick=detail",           0 },
        { "  Scroll wheel=navigate list",              0 },
        { "",                                          0 },
        { "Colors:  ",                                 1 },
    };
    static const int NL = (int)(sizeof(lines) / sizeof(lines[0]));

    /* Compute exact box dimensions from content */
    int content_w = 0;
    for (int i = 0; i < NL; i++)
    {
        int l = (int) strlen(lines[i].text);
        if (l > content_w)
        {
            content_w = l;
        }
    }
    /* Add " stream proc fps" for the color legend row */
    int legend_extra = (int) strlen(" stream proc fps");
    int last_text_w  =
        (int) strlen(lines[NL - 1].text) + legend_extra;
    if (last_text_w > content_w)
    {
        content_w = last_text_w;
    }

    /* Box: 2-char left pad + content + 2-char right pad + 2 borders */
    int pw = content_w + 4 + 2;
    /* Box height: 2 borders + 1 top margin + NL lines + 1 bottom margin */
    int ph = NL + 4;

    int W  = lay->term_cols;
    int H  = lay->term_rows;

    int pr = (H - ph) / 2;
    int pc = (W - pw) / 2;
    if (pr < 1) { pr = 1; }
    if (pc < 1) { pc = 1; }

    /* ---- Step 1: dim overlay for the whole terminal ---- */
    /* REMOVED: no longer overwriting the whole terminal so background
     * panels remain visible. */

    /* ---- Step 2: border + interior ---- */
    ov_draw_panel_border(pr, pc, ph, pw, "HELP", OV_FG_BRIGHT, 1);

    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    /* ---- Step 3: text content ---- */
    int row = pr + 2;
    for (int i = 0; i < NL; i++)
    {
        ov_buf_pos(row, pc + 3);
        ov_theme_bg(OV_BG_PANEL);

        /* Section headers: non-empty lines that
         * don't start with a space */
        const char *t = lines[i].text;
        int is_section = (t[0] != '\0'
                          && t[0] != ' '
                          && !lines[i].color_row);
        if (is_section)
        {
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
        }
        else
        {
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }

        if (lines[i].color_row)
        {
            /* Color legend: print label then colored words */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%s", lines[i].text);
            ov_theme_fg(OV_FG_STREAM);
            ov_buf_printf("stream ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_printf("proc ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_printf("fps");
        }
        else
        {
            ov_buf_printf("%-*s", content_w, lines[i].text);
        }

        ov_buf_reset_attr();
        row++;
    }

    ov_buf_reset_attr();
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

    ov_render_status(lay, m);

    /* End frame */

    ov_buf_flush_delta(lay->term_rows, lay->term_cols);
}
