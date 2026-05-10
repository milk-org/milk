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
