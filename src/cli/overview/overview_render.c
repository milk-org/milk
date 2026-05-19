/**
 * @file    overview_render.c
 * @brief   Shared render utilities and orchestrator
 *          for milk-CTRL
 *
 * Panel-specific rendering lives in:
 *   - overview_render_streams.c
 *   - overview_render_procs.c
 *   - overview_render_fps.c
 */

#include "overview_render_internal.h"
#include <math.h>


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

/**
 * @brief Compute display rank for a stream.
 *
 * Returns a score for priority-based ordering.
 */
static int get_stream_rank(const char *name)
{
    for(int i = 0; i < g_nb_stream_order; i++)
    {
        if(strncmp(g_stream_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

/**
 * @brief Compute display rank for a process.
 */
static int get_proc_rank(const char *name)
{
    for(int i = 0; i < g_nb_proc_order; i++)
    {
        if(strncmp(g_proc_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

/**
 * @brief Compute display rank for an FPS instance.
 */
static int get_fps_rank(const char *name)
{
    for(int i = 0; i < g_nb_fps_order; i++)
    {
        if(strncmp(g_fps_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

static int sort_stream_by_rank(
    const void *a,
    const void *b)
{
    int ra = get_stream_rank(((const OV_STREAM *)a)->name);
    int rb = get_stream_rank(((const OV_STREAM *)b)->name);
    if(ra != rb)
    {
        return ra - rb;
    }
    return strcmp(((const OV_STREAM *)a)->name, ((const OV_STREAM *)b)->name);
}

static int sort_proc_by_rank(
    const void *a,
    const void *b)
{
    int ra = get_proc_rank(((const OV_PROC *)a)->name);
    int rb = get_proc_rank(((const OV_PROC *)b)->name);
    if(ra != rb)
    {
        return ra - rb;
    }
    return strcmp(((const OV_PROC *)a)->name, ((const OV_PROC *)b)->name);
}

static int sort_fps_by_rank(
    const void *a,
    const void *b)
{
    int ra = get_fps_rank(((const OV_FPS *)a)->name);
    int rb = get_fps_rank(((const OV_FPS *)b)->name);
    if(ra != rb)
    {
        return ra - rb;
    }
    return strcmp(((const OV_FPS *)a)->name, ((const OV_FPS *)b)->name);
}

static void ov_apply_rank_sort(OV_MODEL *mm)
{
    if(g_nb_stream_order > 0)
    {
        qsort(mm->streams, (size_t)mm->nb_streams, sizeof(OV_STREAM), sort_stream_by_rank);
    }
    if(g_nb_proc_order > 0)
    {
        qsort(mm->procs, (size_t)mm->nb_procs, sizeof(OV_PROC), sort_proc_by_rank);
    }
    if(g_nb_fps_order > 0)
    {
        qsort(mm->fps, (size_t)mm->nb_fps, sizeof(OV_FPS), sort_fps_by_rank);
    }
}

int ov_render_header_text(
    const char *text,
    int        hs,
    int        max_vis_width,
    ov_rgb_t   base_fg)
{
    int vis_col = 0;
    int printed = 0;
    int i = 0;

    while(text[i] != '\0' && printed < max_vis_width)
    {
        if(text[i] == '\x01')
        {
            if(vis_col >= hs)
            {
                ov_theme_fg(OV_FG_BRIGHT);
                ov_buf_bold();
                ov_buf_underline();
            }
            i++;
        }
        else if(text[i] == '\x02')
        {
            if(vis_col >= hs)
            {
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_HEADER);
                ov_theme_fg(base_fg);
            }
            i++;
        }
        else
        {
            int clen = 1;
            if((text[i] & 0xE0) == 0xC0)
            {
                clen = 2;
            }
            else if((text[i] & 0xF0) == 0xE0)
            {
                clen = 3;
            }
            else if((text[i] & 0xF8) == 0xF0)
            {
                clen = 4;
            }

            if(vis_col >= hs)
            {
                ov_buf_printf("%.*s", clen, text + i);
                printed++;
            }
            vis_col++;
            i += clen;
        }
    }
    return printed;
}

void render_pad_spaces(
    int chars_written,
    int panel_width);


static const char *view_label(ov_view_t v)
{
    switch(v)
    {
    case OV_VIEW_DASHBOARD: return "DASH";
    case OV_VIEW_STREAMS: return "STRM";
    case OV_VIEW_PROCS: return "PROC";
    case OV_VIEW_FPS: return "FPS";
    case OV_VIEW_GRAPH: return "CONN";
    default: return "";
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

    if(!initialized)
    {
        last_usage = current_usage;
        last_time = current_time;
        initialized = 1;
        return 0.0;
    }

    double dt = (current_time.tv_sec - last_time.tv_sec) +
                (current_time.tv_nsec - last_time.tv_nsec) / 1e9;

    if(dt >= 0.5)    /* update every 0.5s */
    {
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

static double get_bandwidth_usage(void)
{
    static struct timespec last_time;
    static uint64_t last_bytes = 0;
    static int initialized = 0;
    static double smoothed_bw = 0.0;

    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    if(!initialized)
    {
        last_time = current_time;
        last_bytes = ov__total_bytes_rendered;
        initialized = 1;
        return 0.0;
    }

    double dt = (current_time.tv_sec - last_time.tv_sec) +
                (current_time.tv_nsec - last_time.tv_nsec) / 1e9;

    if(dt >= 0.5)    /* update every 0.5s */
    {
        uint64_t d_bytes = ov__total_bytes_rendered - last_bytes;
        /* bandwidth in kB/s */
        double inst_bw = (double)d_bytes / 1024.0 / dt;
        smoothed_bw = inst_bw;
        last_bytes = ov__total_bytes_rendered;
        last_time = current_time;
    }
    return smoothed_bw;
}

void ov_render_header(
    OV_LAYOUT      *lay,
    const OV_MODEL *m)
{
    /* Advance blink counter each frame */
    lay->ctrl_blink++;

    OV_RECT r = lay->r_header;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);

    /* LCARS-style rounded end cap */
    ov_theme_fg(OV_GRAD_LO);
    ov_buf_printf("%s", OV_LCARS_LEFT);

    /* Gradient header text */
    ov_buf_bold();
    ov_buf_printf_gradient(OV_GRAD_LO, OV_GRAD_HI, " %s milk-CTRL ", OV_BULLET);
    ov_buf_reset_attr();

    /* LCARS-style rounded end cap (matching the gradient end) */
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_GRAD_HI);
    ov_buf_printf("%s ", OV_LCARS_RIGHT);

    /* Blinking badge — visible when ctrl_mode is ON, READ ONLY when OFF */
    int ctrl_w = 0;
    if(lay->ctrl_mode)
    {
        /* Software blinking badge for "CONTROL" (10 fps -> 2Hz blink) */
        if((lay->ctrl_blink % 10) < 5)
        {
            ov_buf_bg(OV_ANIM_PULSE_BG_MAX.r, OV_ANIM_PULSE_BG_MAX.g, OV_ANIM_PULSE_BG_MAX.b);
            ov_buf_fg(OV_ANIM_PULSE_FG_MAX.r, OV_ANIM_PULSE_FG_MAX.g, OV_ANIM_PULSE_FG_MAX.b);
        }
        else
        {
            ov_buf_bg(220, 40, 40);    /* vibrant red */
            ov_buf_fg(255, 255, 255);  /* white text */
        }
        ov_buf_bold();
        ov_buf_printf(" [c] CONTROL ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ctrl_w = 13; /* visual width of " [c] CONTROL " */
    }
    else
    {
        /* READ ONLY badge (green) */
        ov_buf_bg(20, 180, 20);    /* deep green background */
        ov_buf_fg(220, 255, 220);  /* light text */
        ov_buf_bold();
        ov_buf_printf(" [c] READ ONLY ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ctrl_w = 15; /* visual width of " [c] READ ONLY " */
    }

    ov_buf_printf(" ");
    int hover_w = 0;
    if(lay->mouse_hover)
    {
        /* Mouse hover active badge */
        ov_buf_bg(180, 180, 20);   /* deep yellow background */
        ov_buf_fg(255, 255, 220);  /* light text */
        ov_buf_bold();
        ov_buf_printf(" [m] HOVER: ON ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        hover_w = 16; /* visual width of "  [m] HOVER: ON " */
    }
    else
    {
        /* Mouse hover inactive badge */
        ov_buf_bg(80, 80, 80);     /* dim gray background */
        ov_buf_fg(200, 200, 200);  /* light gray text */
        ov_buf_bold();
        ov_buf_printf(" [m] HOVER: OFF ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        hover_w = 17; /* visual width of "  [m] HOVER: OFF " */
    }

    int chars_left = 17 + ctrl_w + hover_w;

    ov_theme_fg(OV_FG_STREAM);
    chars_left += snprintf(NULL, 0, " %d stm", m->nb_streams);
    ov_buf_printf(" %d stm", m->nb_streams);

    ov_theme_fg(OV_FG_PROC);
    chars_left += snprintf(NULL, 0, " %d prc", m->nb_procs);
    ov_buf_printf(" %d prc", m->nb_procs);

    ov_theme_fg(OV_FG_FPS);
    chars_left += snprintf(NULL, 0, " %d fps", m->nb_fps);
    ov_buf_printf(" %d fps", m->nb_fps);

    ov_theme_fg(OV_FG_CONN);
    chars_left += snprintf(NULL, 0, " %d edg", m->nb_edges);
    ov_buf_printf(" %d edg", m->nb_edges);

    ov_theme_fg(OV_FG_DIM);
    {
        double cpu_pct = get_cpu_usage();
        chars_left += snprintf(NULL, 0, "  CPU: %4.1f%%", cpu_pct);
        ov_buf_printf("  CPU: %4.1f%%", cpu_pct);
    }

    {
        double bw_kbs = get_bandwidth_usage();
        chars_left += snprintf(NULL, 0, "  BW: %4.1f kB/s", bw_kbs);
        ov_buf_printf("  BW: %4.1f kB/s", bw_kbs);
    }

    int tabs_width = 0;
    for(int v = 0; v < OV_VIEW_COUNT; v++)
    {
        tabs_width += (int) strlen(view_label((ov_view_t) v)) + 9;
    }

    int pad = r.width - tabs_width - chars_left;
    if(pad < 1)
    {
        pad = 1;
    }
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_hline(' ', pad);

    for(int v = 0; v < OV_VIEW_COUNT; v++)
    {
        if(v == (int) lay->view)
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf(" %s", OV_LCARS_LEFT);
            ov_theme_bg(OV_FG_TITLE);
            ov_theme_fg(OV_BG_TERMINAL);
            ov_buf_bold();
            ov_buf_printf(" F%d:%s ", v + 2, view_label((ov_view_t) v));
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf("%s ", OV_LCARS_RIGHT);
        }
        else
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" [");
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_bold();
            ov_buf_printf(" F%d:%s ", v + 2, view_label((ov_view_t) v));
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("] ");
        }
    }

    ov_theme_bg(OV_BG_HEADER);
}

static void ov_draw_tooltip(OV_LAYOUT *lay)
{
    if(!lay->mouse_hover || lay->hover_tooltip[0] == '\0')
    {
        return;
    }

    int len = (int) strlen(lay->hover_tooltip);
    if(len == 0)
    {
        return;
    }

    /* Try drawing above the cursor first */
    int tr = ov_mouse_row - 1;
    int tc = ov_mouse_col;

    /* Screen boundary clamping */
    if(tr < 0)
    {
        tr = ov_mouse_row + 1; /* flip below */
    }

    if(tc + len + 2 > lay->term_cols)
    {
        tc = lay->term_cols - len - 2;
    }
    if(tc < 0)
    {
        tc = 0;
    }

    ov_buf_pos(tr, tc);
    ov_theme_bg(OV_BG_HEADER); /* Pop out visually */
    ov_theme_fg(OV_FG_WARN);
    ov_buf_printf(" %s ", lay->hover_tooltip);

    /* Reset for next frame */
    lay->hover_tooltip[0] = '\0';
}


void ov_render_frame(
    OV_LAYOUT      *lay,
    const OV_MODEL *m)
{
    ov_buf_reset();

    /* Perform global hit-test to populate hover state */
    ov_hittest(lay, m, ov_mouse_row, ov_mouse_col);
    ov_hittest_resolve_globals(lay, m);

    /* One-shot sort: only runs once when the user
     * presses S or s.  Order stays frozen until
     * the user explicitly presses S/s again. */
    if(lay->sort_pending)
    {
        OV_MODEL *mm = (OV_MODEL *)(uintptr_t) m;

        /* Calculate ancestry depths before sorting */
        int8_t depths[OV_MAX_NODES];
        for(int i = 0; i < OV_MAX_NODES; i++)
        {
            depths[i] = 127;
        }

        int sel_node = -1;
        ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
        int sel_stream_idx = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
        int sel_proc_idx = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
        int sel_fps_idx = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;

        char saved_sel_stream[80] = {0};
        char saved_sel_proc[80] = {0};
        char saved_sel_fps[80] = {0};

        {
            const char *names[OV_MAX_NODES];
            int fidx[OV_MAX_NODES];

            /* Streams */
            for(int i = 0; i < mm->nb_streams; i++)
            {
                names[i] = mm->streams[i].name;
            }
            int fn = ov_filter_build(lay->filter_stream, names, mm->nb_streams, fidx, OV_MAX_NODES);
            if(lay->sel_stream >= 0 && lay->sel_stream < fn)
            {
                strncpy(saved_sel_stream, mm->streams[fidx[lay->sel_stream]].name, 79);
            }
            if(focus == OV_FOCUS_STREAMS && sel_stream_idx >= 0 && sel_stream_idx < fn)
            {
                sel_node = mm->streams[fidx[sel_stream_idx]].node_idx;
            }

            /* Procs */
            for(int i = 0; i < mm->nb_procs; i++)
            {
                names[i] = mm->procs[i].name;
            }
            fn = ov_filter_build(lay->filter_proc, names, mm->nb_procs, fidx, OV_MAX_NODES);
            if(lay->sel_proc >= 0 && lay->sel_proc < fn)
            {
                strncpy(saved_sel_proc, mm->procs[fidx[lay->sel_proc]].name, 79);
            }
            if(focus == OV_FOCUS_PROCS && sel_proc_idx >= 0 && sel_proc_idx < fn)
            {
                sel_node = mm->procs[fidx[sel_proc_idx]].node_idx;
            }

            /* FPS */
            for(int i = 0; i < mm->nb_fps; i++)
            {
                names[i] = mm->fps[i].name;
            }
            fn = ov_filter_build(lay->filter_fps, names, mm->nb_fps, fidx, OV_MAX_NODES);
            if(lay->sel_fps >= 0 && lay->sel_fps < fn)
            {
                strncpy(saved_sel_fps, mm->fps[fidx[lay->sel_fps]].name, 79);
            }
            if(focus == OV_FOCUS_FPS && sel_fps_idx >= 0 && sel_fps_idx < fn)
            {
                sel_node = mm->fps[fidx[sel_fps_idx]].node_idx;
            }
        }

        if(sel_node >= 0)
        {
            sg_mode_t smode = (focus == OV_FOCUS_FPS) ? SG_MODE_FPS : SG_MODE_FULL;
            sg_compute_node_depths(mm, sel_node, smode, depths);
        }
        ov_sort_set_depths(depths);

        ov_sort_streams(mm, lay->sort_key_stream,  lay->sort_dir_stream);
        ov_sort_procs(mm, lay->sort_key_proc, lay->sort_dir_proc);
        ov_sort_fps(mm, lay->sort_key_fps, lay->sort_dir_fps);

        g_nb_stream_order = mm->nb_streams;
        for(int i = 0; i < mm->nb_streams; i++)
        {
            strncpy(g_stream_order[i], mm->streams[i].name, 79);
            g_stream_order[i][79] = '\0';
        }

        g_nb_proc_order = mm->nb_procs;
        for(int i = 0; i < mm->nb_procs; i++)
        {
            strncpy(g_proc_order[i], mm->procs[i].name, 79);
            g_proc_order[i][79] = '\0';
        }

        g_nb_fps_order = mm->nb_fps;
        for(int i = 0; i < mm->nb_fps; i++)
        {
            strncpy(g_fps_order[i], mm->fps[i].name, 79);
            g_fps_order[i][79] = '\0';
        }

        {
            const char *names[OV_MAX_NODES];
            int fidx[OV_MAX_NODES];

            if(saved_sel_stream[0] != '\0')
            {
                for(int i = 0; i < mm->nb_streams; i++)
                {
                    names[i] = mm->streams[i].name;
                }
                int fn = ov_filter_build(lay->filter_stream, names, mm->nb_streams, fidx, OV_MAX_NODES);
                for(int i = 0; i < fn; i++)
                {
                    if(strcmp(saved_sel_stream, mm->streams[fidx[i]].name) == 0)
                    {
                        lay->sel_stream = i;
                        int page_h = lay->r_streams.height - 3;
                        if(page_h > 0)
                        {
                            if(lay->sel_stream < lay->scroll_stream)
                            {
                                lay->scroll_stream = lay->sel_stream;
                            }
                            if(lay->sel_stream >= lay->scroll_stream + page_h)
                            {
                                lay->scroll_stream = lay->sel_stream - page_h + 1;
                            }
                        }
                        break;
                    }
                }
            }

            if(saved_sel_proc[0] != '\0')
            {
                for(int i = 0; i < mm->nb_procs; i++)
                {
                    names[i] = mm->procs[i].name;
                }
                int fn = ov_filter_build(lay->filter_proc, names, mm->nb_procs, fidx, OV_MAX_NODES);
                for(int i = 0; i < fn; i++)
                {
                    if(strcmp(saved_sel_proc, mm->procs[fidx[i]].name) == 0)
                    {
                        lay->sel_proc = i;
                        int page_h = lay->r_procs.height - 3;
                        if(page_h > 0)
                        {
                            if(lay->sel_proc < lay->scroll_proc)
                            {
                                lay->scroll_proc = lay->sel_proc;
                            }
                            if(lay->sel_proc >= lay->scroll_proc + page_h)
                            {
                                lay->scroll_proc = lay->sel_proc - page_h + 1;
                            }
                        }
                        break;
                    }
                }
            }

            if(saved_sel_fps[0] != '\0')
            {
                for(int i = 0; i < mm->nb_fps; i++)
                {
                    names[i] = mm->fps[i].name;
                }
                int fn = ov_filter_build(lay->filter_fps, names, mm->nb_fps, fidx, OV_MAX_NODES);
                for(int i = 0; i < fn; i++)
                {
                    if(strcmp(saved_sel_fps, mm->fps[fidx[i]].name) == 0)
                    {
                        lay->sel_fps = i;
                        int page_h = lay->r_fps.height - 3;
                        if(page_h > 0)
                        {
                            if(lay->sel_fps < lay->scroll_fps)
                            {
                                lay->scroll_fps = lay->sel_fps;
                            }
                            if(lay->sel_fps >= lay->scroll_fps + page_h)
                            {
                                lay->scroll_fps = lay->sel_fps - page_h + 1;
                            }
                        }
                        break;
                    }
                }
            }
        }

        lay->sort_pending = 0;
        g_last_model = m;
    }
    else
    {
        /* A new scan model arrived. Re-apply the saved order so items don't shuffle. */
        OV_MODEL *mm = (OV_MODEL *)(uintptr_t) m;
        ov_apply_rank_sort(mm);
        g_last_model = m;
    }

    /* Enforce active selection tracking:
     * If the selected item no longer exists in the filtered list
     * (e.g. removed by an external process), reset the selection to 0.
     * Otherwise, clamp bounds and update the tracked name. */
    {
        const char *names[OV_MAX_NODES];
        int fidx[OV_MAX_NODES];

        /* Streams */
        for(int i = 0; i < m->nb_streams; i++)
        {
            names[i] = m->streams[i].name;
        }
        int fn = ov_filter_build(lay->filter_stream, names, m->nb_streams, fidx, OV_MAX_NODES);
        if(fn > 0)
        {
            if(lay->sel_name_stream[0] != '\0')
            {
                int still_exists = 0;
                for(int i = 0; i < fn; i++)
                {
                    if(strcmp(m->streams[fidx[i]].name, lay->sel_name_stream) == 0)
                    {
                        still_exists = 1;
                        lay->sel_stream = i;
                        int page_h = lay->r_streams.height - 3;
                        if(page_h > 0)
                        {
                            if(lay->sel_stream < lay->scroll_stream)
                            {
                                lay->scroll_stream = lay->sel_stream;
                            }
                            if(lay->sel_stream >= lay->scroll_stream + page_h)
                            {
                                lay->scroll_stream = lay->sel_stream - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if(!still_exists)
                {
                    lay->sel_stream = 0;
                }
            }
            if(lay->sel_stream >= fn)
            {
                lay->sel_stream = fn - 1;
            }
            if(lay->sel_stream < 0)
            {
                lay->sel_stream = 0;
            }
            strncpy(lay->sel_name_stream, m->streams[fidx[lay->sel_stream]].name, 79);
        }
        else
        {
            lay->sel_stream = 0;
            lay->sel_name_stream[0] = '\0';
        }

        /* Procs */
        for(int i = 0; i < m->nb_procs; i++)
        {
            names[i] = m->procs[i].name;
        }
        fn = ov_filter_build(lay->filter_proc, names, m->nb_procs, fidx, OV_MAX_NODES);
        if(fn > 0)
        {
            if(lay->sel_name_proc[0] != '\0')
            {
                int still_exists = 0;
                for(int i = 0; i < fn; i++)
                {
                    if(strcmp(m->procs[fidx[i]].name, lay->sel_name_proc) == 0 &&
                            m->procs[fidx[i]].PID == lay->sel_pid_proc)
                    {
                        still_exists = 1;
                        lay->sel_proc = i;
                        int page_h = lay->r_procs.height - 3;
                        if(page_h > 0)
                        {
                            if(lay->sel_proc < lay->scroll_proc)
                            {
                                lay->scroll_proc = lay->sel_proc;
                            }
                            if(lay->sel_proc >= lay->scroll_proc + page_h)
                            {
                                lay->scroll_proc = lay->sel_proc - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if(!still_exists)
                {
                    lay->sel_proc = 0;
                }
            }
            if(lay->sel_proc >= fn)
            {
                lay->sel_proc = fn - 1;
            }
            if(lay->sel_proc < 0)
            {
                lay->sel_proc = 0;
            }
            strncpy(lay->sel_name_proc, m->procs[fidx[lay->sel_proc]].name, 79);
            lay->sel_pid_proc = m->procs[fidx[lay->sel_proc]].PID;
        }
        else
        {
            lay->sel_proc = 0;
            lay->sel_name_proc[0] = '\0';
            lay->sel_pid_proc = 0;
        }

        /* FPS */
        for(int i = 0; i < m->nb_fps; i++)
        {
            names[i] = m->fps[i].name;
        }
        fn = ov_filter_build(lay->filter_fps, names, m->nb_fps, fidx, OV_MAX_NODES);
        if(fn > 0)
        {
            if(lay->sel_name_fps[0] != '\0')
            {
                int still_exists = 0;
                for(int i = 0; i < fn; i++)
                {
                    if(strcmp(m->fps[fidx[i]].name, lay->sel_name_fps) == 0)
                    {
                        still_exists = 1;
                        lay->sel_fps = i;
                        int page_h = lay->r_fps.height - 3;
                        if(page_h > 0)
                        {
                            if(lay->sel_fps < lay->scroll_fps)
                            {
                                lay->scroll_fps = lay->sel_fps;
                            }
                            if(lay->sel_fps >= lay->scroll_fps + page_h)
                            {
                                lay->scroll_fps = lay->sel_fps - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if(!still_exists)
                {
                    lay->sel_fps = 0;
                }
            }
            if(lay->sel_fps >= fn)
            {
                lay->sel_fps = fn - 1;
            }
            if(lay->sel_fps < 0)
            {
                lay->sel_fps = 0;
            }
            strncpy(lay->sel_name_fps, m->fps[fidx[lay->sel_fps]].name, 79);
        }
        else
        {
            lay->sel_fps = 0;
            lay->sel_name_fps[0] = '\0';
        }
    }

    /* Always patch graph node .index fields to
     * reflect current array positions — needed
     * whether we just sorted or scan rebuilt the
     * model.  Each item's node_idx still points
     * to its graph node; update the reverse link. */
    {
        OV_MODEL *mm = (OV_MODEL *)(uintptr_t) m;
        for(int i = 0; i < mm->nb_streams; i++)
        {
            int ni = mm->streams[i].node_idx;
            if(ni >= 0 && ni < mm->nb_nodes)
            {
                mm->nodes[ni].index = i;
            }
        }
        for(int i = 0; i < mm->nb_fps; i++)
        {
            int ni = mm->fps[i].node_idx;
            if(ni >= 0 && ni < mm->nb_nodes)
            {
                mm->nodes[ni].index = i;
            }
        }
        for(int i = 0; i < mm->nb_procs; i++)
        {
            int ni = mm->procs[i].node_idx;
            if(ni >= 0 && ni < mm->nb_nodes)
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
    if(!lay->show_help)
    {
        switch(lay->view)
        {
        case OV_VIEW_DASHBOARD: ov_render_preview_line(lay, m);
            ov_render_streams_panel(lay, m, &rel);
            ov_render_procs_panel(lay, m, &rel);
            ov_render_fps_panel(lay, m, &rel);
            int rendered = 0;
            if(lay->graph_tab_mode == 1)
            {
                rendered = ov_render_detail_panel(lay, m);
            }
            else if(lay->graph_tab_mode == 2)
            {
                rendered = ov_render_resources_panel(lay, m);
            }

            if(!rendered)
            {
                ov_render_graph_panel(lay, m);
            }
            break;
        case OV_VIEW_GRAPH: ov_render_graph_panel(lay, m);
            break;
        case OV_VIEW_STREAMS: ov_render_streams_panel(lay, m, &rel);
            break;
        case OV_VIEW_PROCS: ov_render_procs_panel(lay, m, &rel);
            break;
        case OV_VIEW_FPS: ov_render_fps_panel(lay, m, &rel);
            if(lay->sel_fps >= 0
                    && lay->sel_fps < m->nb_fps
                    && m->fps[lay->sel_fps].nb_disp_params > 0)
            {
                ov_render_fps_params_panel(lay, m);
            }
            else
            {
                /* No params: draw empty right panel */
                ov_draw_panel_border(
                    lay->r_fps_params.row,
                    lay->r_fps_params.col,
                    lay->r_fps_params.height, lay->r_fps_params.width, "PARAMS", OV_FG_DIM, 0, 0);
            }
            break;
        default: break;
        }
    }

    if(lay->show_help)
    {
        ov_render_help(lay);
    }

    ov_render_cmdlog(lay);
    ov_render_status(lay, m);

    /* Highlight movable edges if hovering */
    if(lay->mouse_hover && !lay->show_help)
    {
        ov_theme_fg(OV_FG_WARN);
        ov_theme_bg(OV_BG_TERMINAL);
        ov_buf_bold();

        if(lay->cmdlog_split_hover)
        {
            int cmdlog_top = (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
            if(cmdlog_top > 1)
            {
                ov_buf_pos(cmdlog_top - 1, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
            }
        }

        if(lay->view == OV_VIEW_DASHBOARD)
        {
            if(lay->dash_split_h_hover)
            {
                int r = lay->r_streams.row + lay->r_streams.height - 1;
                ov_buf_pos(r, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
                ov_buf_pos(r + 1, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
            }
            if(lay->dash_split_v_hover)
            {
                int c = lay->r_streams.width;
                for(int rr = lay->r_streams.row; rr < lay->r_fps.row + lay->r_fps.height; rr++)
                {
                    ov_buf_pos(rr, c);
                    ov_buf_printf("%s", OV_BOX_V_D);
                    ov_buf_pos(rr, c + 1);
                    ov_buf_printf("%s", OV_BOX_V_D);
                }
            }
        }
        else if(lay->view == OV_VIEW_FPS)
        {
            if(lay->fps_split_hover)
            {
                int c = lay->r_fps_list.width;
                for(int rr = lay->r_fps_list.row; rr < lay->r_fps_list.row + lay->r_fps_list.height; rr++)
                {
                    ov_buf_pos(rr, c);
                    ov_buf_printf("%s", OV_BOX_V_D);
                    ov_buf_pos(rr, c + 1);
                    ov_buf_printf("%s", OV_BOX_V_D);
                }
            }
        }

        ov_buf_reset_attr();
    }

    /* End frame */
    ov_draw_tooltip(lay);

    ov_buf_flush_delta(lay->term_rows, lay->term_cols);
}
