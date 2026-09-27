// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
#include "overview_render_fps_params.h"
#include "overview_render_loops.h"
#include "milk_config.h"
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
    for (int i = 0; i < g_nb_stream_order; i++)
    {
        if (strncmp(g_stream_order[i], name, 80) == 0)
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
    for (int i = 0; i < g_nb_proc_order; i++)
    {
        if (strncmp(g_proc_order[i], name, 80) == 0)
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
    for (int i = 0; i < g_nb_fps_order; i++)
    {
        if (strncmp(g_fps_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

typedef struct
{
    int         rank;
    int         orig_idx;
    const char *name;
} ov_sort_rank_entry_t;

static int sort_entry_by_rank(const void *a, const void *b)
{
    const ov_sort_rank_entry_t *ea = (const ov_sort_rank_entry_t *) a;
    const ov_sort_rank_entry_t *eb = (const ov_sort_rank_entry_t *) b;
    if (ea->rank != eb->rank)
    {
        return ea->rank - eb->rank;
    }
    return strcmp(ea->name, eb->name);
}

static void ov_apply_rank_sort(OV_MODEL *mm)
{
    if (g_nb_stream_order > 0 && mm->nb_streams > 1)
    {
        static ov_sort_rank_entry_t entries[OV_MAX_STREAMS];
        for (int i = 0; i < mm->nb_streams; i++)
        {
            entries[i].orig_idx = i;
            entries[i].name     = mm->streams[i].name;
            entries[i].rank     = get_stream_rank(mm->streams[i].name);
        }
        qsort(entries, (size_t) mm->nb_streams, sizeof(ov_sort_rank_entry_t), sort_entry_by_rank);

        static OV_STREAM temp_streams[OV_MAX_STREAMS];
        memcpy(temp_streams, mm->streams, (size_t) mm->nb_streams * sizeof(OV_STREAM));
        for (int i = 0; i < mm->nb_streams; i++)
        {
            mm->streams[i] = temp_streams[entries[i].orig_idx];
        }
    }

    if (g_nb_proc_order > 0 && mm->nb_procs > 1)
    {
        static ov_sort_rank_entry_t entries[OV_MAX_PROCS];
        for (int i = 0; i < mm->nb_procs; i++)
        {
            entries[i].orig_idx = i;
            entries[i].name     = mm->procs[i].name;
            entries[i].rank     = get_proc_rank(mm->procs[i].name);
        }
        qsort(entries, (size_t) mm->nb_procs, sizeof(ov_sort_rank_entry_t), sort_entry_by_rank);

        static OV_PROC temp_procs[OV_MAX_PROCS];
        memcpy(temp_procs, mm->procs, (size_t) mm->nb_procs * sizeof(OV_PROC));
        for (int i = 0; i < mm->nb_procs; i++)
        {
            mm->procs[i] = temp_procs[entries[i].orig_idx];
        }
    }

    if (g_nb_fps_order > 0 && mm->nb_fps > 1)
    {
        static ov_sort_rank_entry_t entries[OV_MAX_FPS];
        for (int i = 0; i < mm->nb_fps; i++)
        {
            entries[i].orig_idx = i;
            entries[i].name     = mm->fps[i].name;
            entries[i].rank     = get_fps_rank(mm->fps[i].name);
        }
        qsort(entries, (size_t) mm->nb_fps, sizeof(ov_sort_rank_entry_t), sort_entry_by_rank);

        static OV_FPS temp_fps[OV_MAX_FPS];
        memcpy(temp_fps, mm->fps, (size_t) mm->nb_fps * sizeof(OV_FPS));
        for (int i = 0; i < mm->nb_fps; i++)
        {
            mm->fps[i] = temp_fps[entries[i].orig_idx];
        }
    }
}

int ov_render_header_text(const char *text, int hs, int max_vis_width, ov_rgb_t base_fg)
{
    int vis_col = 0;
    int printed = 0;
    int i       = 0;

    while (text[i] != '\0' && printed < max_vis_width)
    {
        if (text[i] == '\x01')
        {
            if (vis_col >= hs)
            {
                ov_theme_fg(OV_FG_BRIGHT);
                ov_buf_bold();
                ov_buf_underline();
            }
            i++;
        }
        else if (text[i] == '\x02')
        {
            if (vis_col >= hs)
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
            if ((text[i] & 0xE0) == 0xC0)
            {
                clen = 2;
            }
            else if ((text[i] & 0xF0) == 0xE0)
            {
                clen = 3;
            }
            else if ((text[i] & 0xF8) == 0xF0)
            {
                clen = 4;
            }

            if (vis_col >= hs)
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

void render_pad_spaces(int chars_written, int panel_width);


static const char *view_label(ov_view_t v)
{
    switch (v)
    {
    case OV_VIEW_DASHBOARD:
        return "DASH";
    case OV_VIEW_STREAMS:
        return "STRM";
    case OV_VIEW_PROCS:
        return "PROC";
    case OV_VIEW_FPS:
        return "FPS";
    case OV_VIEW_GRAPH:
        return "CONN";
    case OV_VIEW_LOOPS:
        return "LOOPS";
    default:
        return "";
    }
}

static double get_cpu_usage(void)
{
    static struct rusage   last_usage;
    static struct timespec last_time;
    static int             initialized  = 0;
    static double          smoothed_cpu = 0.0;

    struct rusage   current_usage;
    struct timespec current_time;

    getrusage(RUSAGE_SELF, &current_usage);
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    if (!initialized)
    {
        last_usage  = current_usage;
        last_time   = current_time;
        initialized = 1;
        return 0.0;
    }

    double dt =
        (current_time.tv_sec - last_time.tv_sec) + (current_time.tv_nsec - last_time.tv_nsec) / 1e9;

    if (dt >= 0.5) /* update every 0.5s */
    {
        double d_utime = (current_usage.ru_utime.tv_sec - last_usage.ru_utime.tv_sec) +
                         (current_usage.ru_utime.tv_usec - last_usage.ru_utime.tv_usec) / 1e6;
        double d_stime = (current_usage.ru_stime.tv_sec - last_usage.ru_stime.tv_sec) +
                         (current_usage.ru_stime.tv_usec - last_usage.ru_stime.tv_usec) / 1e6;

        double inst_cpu = 100.0 * (d_utime + d_stime) / dt;
        smoothed_cpu    = inst_cpu;
        last_usage      = current_usage;
        last_time       = current_time;
    }
    return smoothed_cpu;
}

static double get_bandwidth_usage(void)
{
    static struct timespec last_time;
    static uint64_t        last_bytes  = 0;
    static int             initialized = 0;
    static double          smoothed_bw = 0.0;

    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    if (!initialized)
    {
        last_time   = current_time;
        last_bytes  = ov__total_bytes_rendered;
        initialized = 1;
        return 0.0;
    }

    double dt =
        (current_time.tv_sec - last_time.tv_sec) + (current_time.tv_nsec - last_time.tv_nsec) / 1e9;

    if (dt >= 0.5) /* update every 0.5s */
    {
        uint64_t d_bytes = ov__total_bytes_rendered - last_bytes;
        /* bandwidth in kB/s */
        double inst_bw = (double) d_bytes / 1024.0 / dt;
        smoothed_bw    = inst_bw;
        last_bytes     = ov__total_bytes_rendered;
        last_time      = current_time;
    }
    return smoothed_bw;
}

void ov_render_header(OV_LAYOUT *lay, const OV_MODEL *m)
{
    /* Advance blink counter each frame */
    lay->ctrl_blink++;

    OV_RECT r = lay->r_header;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);

    /* ── Heartbeat: fast-pulsing indicator ── */
    {
        int beat = lay->ctrl_blink % 2;
        if (beat == 0)
        {
            /* Bright beat — vivid red */
            ov_buf_fg(255, 50, 50);
            ov_buf_bold();
            ov_buf_printf("\xe2\x99\xa5"); /* ♥ */
            ov_buf_reset_attr();
        }
        else
        {
            /* Dim beat — dark red */
            ov_buf_fg(100, 30, 30);
            ov_buf_printf("\xe2\x99\xa5"); /* ♥ */
        }
        ov_theme_bg(OV_BG_HEADER);
    }

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

    /* Version / commit tracking */
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("[%s] ", MILK_GIT_COMMIT);

    /* Shared memory directory */
    const char *shmdir = ov_get_shmdir();
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("[shm: %s] ", shmdir);

    /* Blinking badge — visible when ctrl_mode is ON, READ ONLY when OFF */
    int ctrl_w = 0;
    if (lay->ctrl_mode)
    {
        /* Software blinking badge for "CONTROL" (fast 2.5Hz blink) */
        if ((lay->ctrl_blink % 4) < 2)
        {
            ov_buf_bg(OV_ANIM_PULSE_BG_MAX.r, OV_ANIM_PULSE_BG_MAX.g, OV_ANIM_PULSE_BG_MAX.b);
            ov_buf_fg(OV_ANIM_PULSE_FG_MAX.r, OV_ANIM_PULSE_FG_MAX.g, OV_ANIM_PULSE_FG_MAX.b);
        }
        else
        {
            ov_buf_bg(220, 40, 40);   /* vibrant red */
            ov_buf_fg(255, 255, 255); /* white text */
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
        ov_buf_bg(20, 180, 20);   /* deep green background */
        ov_buf_fg(220, 255, 220); /* light text */
        ov_buf_bold();
        ov_buf_printf(" [c] READ ONLY ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ctrl_w = 15; /* visual width of " [c] READ ONLY " */
    }

    ov_buf_printf(" ");
    int hover_w = 0;
    if (lay->mouse_hover)
    {
        /* Mouse hover active badge */
        ov_buf_bg(180, 180, 20);   /* deep yellow background */
        ov_buf_fg(20, 20, 20);     /* dark text */
        ov_buf_bold();
        ov_buf_printf(" [m] HOVER: ON ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        hover_w = 15; /* visual width of " [m] HOVER: ON " */
    }
    else
    {
        /* Mouse hover inactive badge */
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_bold();
        ov_buf_printf(" [m] HOVER: OFF ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        hover_w = 16; /* visual width of " [m] HOVER: OFF " */
    }

    ov_buf_printf(" ");
    int         filter_w = 0;
    ov_focus_t  fpanel   = ov_get_effective_filter_panel(lay);
    const char *pname    = (fpanel == OV_FOCUS_STREAMS) ? "STRM"
                           : (fpanel == OV_FOCUS_PROCS) ? "PROC"
                           : (fpanel == OV_FOCUS_FPS)   ? "FPS"
                                                        : "FILTER";
    const char *fpat     = (fpanel != OV_FOCUS_GRAPH) ? ov_get_panel_filter_pattern(lay, fpanel)
                                                      : ov_get_filter_pattern(lay);
    int is_act           = (fpanel != OV_FOCUS_GRAPH) ? ov_is_panel_filter_active(lay, fpanel)
                                                      : ov_is_filter_active(lay);

    if (is_act)
    {
        /* Software blinking badge for "FILTER ON" (fast 2.5Hz blink) */
        if ((lay->ctrl_blink % 4) < 2)
        {
            ov_buf_bg(255, 190, 0);   /* bright amber/gold */
            ov_buf_fg(20, 20, 20);    /* dark text */
        }
        else
        {
            ov_buf_bg(230, 80, 20);   /* vibrant red-orange */
            ov_buf_fg(255, 255, 255); /* white text */
        }
        ov_buf_bold();
        char fbadge[64];
        if (fpanel != OV_FOCUS_GRAPH)
        {
            snprintf(fbadge, sizeof(fbadge), " [f] %s: /%.10s/ ", pname, fpat);
        }
        else
        {
            snprintf(fbadge, sizeof(fbadge), " [f] FILTER ON: /%.12s/ ", fpat);
        }
        filter_w = (int) strlen(fbadge);
        ov_buf_printf("%s", fbadge);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
    }
    else if (fpat[0] != '\0')
    {
        /* Defined but paused filter badge: shows retained query */
        ov_theme_bg(OV_BG_PANEL_ALT);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_bold();
        char fbadge[64];
        if (fpanel != OV_FOCUS_GRAPH)
        {
            snprintf(fbadge, sizeof(fbadge), " [f] %s: OFF (/%.8s/) ", pname, fpat);
        }
        else
        {
            snprintf(fbadge, sizeof(fbadge), " [f] FILTER: OFF (/%.12s/) ", fpat);
        }
        filter_w = (int) strlen(fbadge);
        ov_buf_printf("%s", fbadge);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
    }
    else
    {
        /* Inactive / empty filter badge */
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_bold();
        char fbadge[64];
        if (fpanel != OV_FOCUS_GRAPH)
        {
            snprintf(fbadge, sizeof(fbadge), " [/] %s: ALL ", pname);
        }
        else
        {
            snprintf(fbadge, sizeof(fbadge), " [/] FILTER: OFF ");
        }
        filter_w = (int) strlen(fbadge);
        ov_buf_printf("%s", fbadge);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
    }

    int commit_w   = (int) strlen(MILK_GIT_COMMIT) + 3;
    int shmdir_w   = (int) strlen(shmdir) + 8;
    int chars_left = 17 + commit_w + shmdir_w + 1 + ctrl_w + 1 + hover_w + 1 +
                     filter_w; /* +1 for heartbeat */

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

    int pad = r.width - chars_left;
    if (pad > 0)
    {
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_hline(' ', pad);
    }

    ov_theme_bg(OV_BG_HEADER);
}

/**
 * ov_render_tabs - render dedicated tab selection bar and prominent help button.
 * @lay: layout state
 *
 * Renders on row 2 (lay->r_tabs.row). Left side displays function key view tabs
 * ([F2:DASH] .. [F7:LOOPS]), and right side displays prominent [h: HELP] button
 * with slow blink color when idle, and active pill styling when help is open.
 */
void ov_render_tabs(
    OV_LAYOUT *lay)
{
    OV_RECT r = lay->r_tabs;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);

    /* Render view tabs */
    int tabs_total_width = 0;
    int tab_widths[OV_VIEW_COUNT];
    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        tab_widths[v] = (int) strlen(ov_view_label((ov_view_t) v)) + 9;
        tabs_total_width += tab_widths[v];
    }

    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        if (v == (int) lay->view)
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf(" %s", OV_LCARS_LEFT);
            ov_theme_bg(OV_FG_TITLE);
            ov_theme_fg(OV_BG_TERMINAL);
            ov_buf_bold();
            ov_buf_printf(" F%d:%s ", v + 2, ov_view_label((ov_view_t) v));
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
            ov_buf_printf(" F%d:%s ", v + 2, ov_view_label((ov_view_t) v));
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("] ");
        }
    }

    /* Help button [h: HELP] - prominent with slow blink */
    int help_width = 11; /* visual width of " [h: HELP] " */
    int pad        = r.width - tabs_total_width - help_width;
    if (pad > 0)
    {
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_hline(' ', pad);
    }
    else
    {
        pad = 0;
    }

    /* Render prominent help button */
    if (lay->show_help)
    {
        /* Active state when help overlay is visible: light-blue solid pill */
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf(" %s", OV_LCARS_LEFT);
        ov_theme_bg(OV_FG_TITLE);
        ov_theme_fg(OV_BG_TERMINAL);
        ov_buf_bold();
        ov_buf_printf("h: HELP");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("%s ", OV_LCARS_RIGHT);
    }
    else
    {
        /* Slow blinking prominent amber badge (1s bright, 1s dim) */
        struct timespec now_ts;
        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_printf(" ");
        if ((now_ts.tv_sec % 2) == 0)
        {
            /* Bright prominent state: vibrant gold/amber bg, dark crisp text */
            ov_buf_bg(240, 175, 20);
            ov_buf_fg(20, 20, 25);
        }
        else
        {
            /* Dim prominent state themed to panel bg with warning text */
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_WARN);
        }
        ov_buf_bold();
        ov_buf_printf("[h: HELP]");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_printf(" ");
    }

    /* Pad trailing space if line not completely filled */
    int rendered_w = tabs_total_width + pad + help_width;
    if (rendered_w < r.width)
    {
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_hline(' ', r.width - rendered_w);
    }
    ov_theme_bg(OV_BG_HEADER);
}

static void ov_draw_tooltip(OV_LAYOUT *lay)
{
    if (!lay->mouse_hover || lay->hover_tooltip[0] == '\0')
    {
        return;
    }

    int len = (int) strlen(lay->hover_tooltip);
    if (len == 0)
    {
        return;
    }

    /* Try drawing above the cursor first */
    int tr = ov_mouse_row - 1;
    int tc = ov_mouse_col;

    /* Screen boundary clamping */
    if (tr < 0)
    {
        tr = ov_mouse_row + 1; /* flip below */
    }

    if (tc + len + 2 > lay->term_cols)
    {
        tc = lay->term_cols - len - 2;
    }
    if (tc < 0)
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

void ov_render_theme_popup(OV_LAYOUT *lay)
{
    if (!lay->theme_popup_active)
    {
        return;
    }

    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts);
    double elapsed = (now_ts.tv_sec - lay->theme_popup_ts.tv_sec) +
                     (now_ts.tv_nsec - lay->theme_popup_ts.tv_nsec) * 1e-9;
    if (elapsed >= 1.0)
    {
        lay->theme_popup_active = 0;
        return;
    }

    int nthemes = ov_theme_count();
    int pw      = 46;
    int ph      = nthemes + 2;

    int pr = lay->term_rows - ph;
    int pc = lay->term_cols - pw - 2;

    if (pr < 2)
    {
        pr = 2;
    }
    if (pc < 1)
    {
        pc = 1;
    }
    if (pr + ph > lay->term_rows)
    {
        ph = lay->term_rows - pr;
    }
    if (pc + pw > lay->term_cols)
    {
        pw = lay->term_cols - pc;
    }

    lay->r_theme_popup.row    = pr;
    lay->r_theme_popup.col    = pc;
    lay->r_theme_popup.height = ph;
    lay->r_theme_popup.width  = pw;

    /* Top border */
    ov_buf_pos(pr, pc);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_WARN);
    ov_buf_bold();
    ov_buf_printf("%s%s", OV_BOX_TL, OV_BOX_H);
    ov_theme_fg(OV_FG_TITLE);
    ov_buf_printf(" THEME SELECTOR (↑/↓ • ESC) ");
    ov_theme_fg(OV_FG_WARN);
    int top_rem = (pc + pw - 1) - ov__cursor_col;
    if (top_rem > 0)
    {
        ov_buf_hline_utf8(OV_BOX_H, top_rem);
    }
    ov_buf_printf("%s", OV_BOX_TR);
    ov_buf_reset_attr();

    /* Render theme items */
    for (int i = 0; i < nthemes && (pr + 1 + i) < (pr + ph - 1); i++)
    {
        int               row       = pr + 1 + i;
        int               is_sel    = (i == lay->theme_popup_sel);
        int               is_active = (i == ov_theme_get_active_index());
        const ov_theme_t *th        = ov_theme_get(i);

        ov_buf_pos(row, pc);

        /* Left border */
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("%s", OV_BOX_V);

        /* Row content */
        ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : OV_BG_PANEL;
        ov_theme_bg(row_bg);

        if (is_sel)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf(" ▶ ");
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("   ");
        }

        /* Swatch: 4 color preview blocks */
        ov_buf_bg(th->bg_terminal.r, th->bg_terminal.g, th->bg_terminal.b);
        ov_buf_fg(th->fg_title.r, th->fg_title.g, th->fg_title.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_stream.r, th->fg_stream.g, th->fg_stream.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_active.r, th->fg_active.g, th->fg_active.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_warn.r, th->fg_warn.g, th->fg_warn.b);
        ov_buf_printf("■ ");

        /* Restore row background */
        ov_theme_bg(row_bg);

        /* Theme name */
        if (is_sel)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_BRIGHT);
        }
        else
        {
            ov_theme_fg(OV_FG_TEXT);
        }
        ov_buf_printf("%-18.18s ", th->name);

        /* Active tag */
        if (is_active)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf("● active");
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("        ");
        }

        /* Pad row to right edge */
        render_pad_to_col(pc + pw - 1);

        /* Right border */
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("%s", OV_BOX_V);
        ov_buf_reset_attr();
    }

    /* Bottom border with auto-close countdown */
    int bot_row = pr + ph - 1;
    ov_buf_pos(bot_row, pc);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_WARN);
    ov_buf_bold();
    ov_buf_printf("%s%s", OV_BOX_BL, OV_BOX_H);

    double remain = 1.0 - elapsed;
    if (remain < 0.0)
    {
        remain = 0.0;
    }
    char hint[48];
    snprintf(hint, sizeof(hint), " auto-closes in %.1fs ", remain);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("%s", hint);

    ov_theme_fg(OV_FG_WARN);
    int bot_rem = (pc + pw - 1) - ov__cursor_col;
    if (bot_rem > 0)
    {
        ov_buf_hline_utf8(OV_BOX_H, bot_rem);
    }
    ov_buf_printf("%s", OV_BOX_BR);
    ov_buf_reset_attr();
}

void ov_render_frame(OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_buf_reset_size(lay->term_rows, lay->term_cols);

    /* Perform global hit-test to populate hover state */
    ov_hittest(lay, m, ov_mouse_row, ov_mouse_col);
    ov_hittest_resolve_globals(lay, m);

    /* Ensure there exists a valid selected parameter when in the PARAMS panel on F5 view */
    int cur_fidx = ov_get_selected_fps_idx(lay, m);
    if (lay->view == OV_VIEW_FPS && cur_fidx >= 0 && cur_fidx < m->nb_fps)
    {
        const OV_FPS   *fps = &m->fps[cur_fidx];
        fps_tree_item_t items[1024];
        int             nitems = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);

        if (nitems > 0)
        {
            if (lay->fps_param_sel < 0)
            {
                lay->fps_param_sel = 0;
            }
            else if (lay->fps_param_sel >= nitems)
            {
                lay->fps_param_sel = nitems - 1;
            }
        }
    }

    /* One-shot sort: only runs once when the user
     * presses S or s.  Order stays frozen until
     * the user explicitly presses S/s again. */
    if (lay->sort_pending)
    {
        OV_MODEL *mm = (OV_MODEL *) (uintptr_t) m;

        /* Calculate ancestry depths before sorting */
        int8_t depths[OV_MAX_NODES];
        for (int i = 0; i < OV_MAX_NODES; i++)
        {
            depths[i] = 127;
        }

        int        sel_node       = -1;
        ov_focus_t focus          = lay->freeze ? lay->freeze_focus : lay->focus;
        int        sel_stream_idx = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
        int        sel_proc_idx   = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
        int        sel_fps_idx    = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;

        char saved_sel_stream[80] = { 0 };
        char saved_sel_proc[80]   = { 0 };
        char saved_sel_fps[80]    = { 0 };

        {
            const char *names[OV_MAX_NODES];
            int         fidx[OV_MAX_NODES];
            const char *f_str = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
            const char *f_prc = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
            const char *f_fps = ov_get_active_filter_for(lay, OV_FOCUS_FPS);

            /* Streams */
            for (int i = 0; i < mm->nb_streams; i++)
            {
                names[i] = mm->streams[i].name;
            }
            int fn = ov_filter_build(f_str, names, mm->nb_streams, fidx, OV_MAX_NODES);
            if (lay->sel_stream >= 0 && lay->sel_stream < fn)
            {
                strncpy(saved_sel_stream, mm->streams[fidx[lay->sel_stream]].name, 79);
            }
            if (focus == OV_FOCUS_STREAMS && sel_stream_idx >= 0 && sel_stream_idx < fn)
            {
                sel_node = mm->streams[fidx[sel_stream_idx]].node_idx;
            }

            /* Procs */
            for (int i = 0; i < mm->nb_procs; i++)
            {
                names[i] = mm->procs[i].name;
            }
            fn = ov_filter_build(f_prc, names, mm->nb_procs, fidx, OV_MAX_NODES);
            if (lay->sel_proc >= 0 && lay->sel_proc < fn)
            {
                strncpy(saved_sel_proc, mm->procs[fidx[lay->sel_proc]].name, 79);
            }
            if (focus == OV_FOCUS_PROCS && sel_proc_idx >= 0 && sel_proc_idx < fn)
            {
                sel_node = mm->procs[fidx[sel_proc_idx]].node_idx;
            }

            /* FPS */
            for (int i = 0; i < mm->nb_fps; i++)
            {
                names[i] = mm->fps[i].name;
            }
            fn = ov_filter_build(f_fps, names, mm->nb_fps, fidx, OV_MAX_NODES);
            if (lay->sel_fps >= 0 && lay->sel_fps < fn)
            {
                strncpy(saved_sel_fps, mm->fps[fidx[lay->sel_fps]].name, 79);
            }
            if (focus == OV_FOCUS_FPS && sel_fps_idx >= 0 && sel_fps_idx < fn)
            {
                sel_node = mm->fps[fidx[sel_fps_idx]].node_idx;
            }
        }

        if (sel_node >= 0)
        {
            sg_mode_t smode = (focus == OV_FOCUS_FPS) ? SG_MODE_FPS : SG_MODE_FULL;
            sg_compute_node_depths(mm, sel_node, smode, depths);
        }
        ov_sort_set_depths(depths);

        ov_sort_streams(mm, lay->sort_key_stream, lay->sort_dir_stream);
        ov_sort_procs(mm, lay->sort_key_proc, lay->sort_dir_proc);
        ov_sort_fps(mm, lay->sort_key_fps, lay->sort_dir_fps);

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

        {
            const char *names[OV_MAX_NODES];
            int         fidx[OV_MAX_NODES];
            const char *f_str = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
            const char *f_prc = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
            const char *f_fps = ov_get_active_filter_for(lay, OV_FOCUS_FPS);

            if (saved_sel_stream[0] != '\0')
            {
                for (int i = 0; i < mm->nb_streams; i++)
                {
                    names[i] = mm->streams[i].name;
                }
                int fn = ov_filter_build(f_str, names, mm->nb_streams, fidx, OV_MAX_NODES);
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(saved_sel_stream, mm->streams[fidx[i]].name) == 0)
                    {
                        lay->sel_stream = i;
                        int page_h      = lay->r_streams.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_stream < lay->scroll_stream)
                            {
                                lay->scroll_stream = lay->sel_stream;
                            }
                            if (lay->sel_stream >= lay->scroll_stream + page_h)
                            {
                                lay->scroll_stream = lay->sel_stream - page_h + 1;
                            }
                        }
                        break;
                    }
                }
            }

            if (saved_sel_proc[0] != '\0')
            {
                for (int i = 0; i < mm->nb_procs; i++)
                {
                    names[i] = mm->procs[i].name;
                }
                int fn = ov_filter_build(f_prc, names, mm->nb_procs, fidx, OV_MAX_NODES);
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(saved_sel_proc, mm->procs[fidx[i]].name) == 0)
                    {
                        lay->sel_proc = i;
                        int page_h    = lay->r_procs.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_proc < lay->scroll_proc)
                            {
                                lay->scroll_proc = lay->sel_proc;
                            }
                            if (lay->sel_proc >= lay->scroll_proc + page_h)
                            {
                                lay->scroll_proc = lay->sel_proc - page_h + 1;
                            }
                        }
                        break;
                    }
                }
            }

            if (saved_sel_fps[0] != '\0')
            {
                for (int i = 0; i < mm->nb_fps; i++)
                {
                    names[i] = mm->fps[i].name;
                }
                int fn = ov_filter_build(f_fps, names, mm->nb_fps, fidx, OV_MAX_NODES);
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(saved_sel_fps, mm->fps[fidx[i]].name) == 0)
                    {
                        lay->sel_fps = i;
                        int page_h   = lay->r_fps.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_fps < lay->scroll_fps)
                            {
                                lay->scroll_fps = lay->sel_fps;
                            }
                            if (lay->sel_fps >= lay->scroll_fps + page_h)
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
        g_last_model      = m;
    }
    else
    {
        /* A new scan model arrived. Re-apply the saved order so items don't shuffle. */
        OV_MODEL *mm = (OV_MODEL *) (uintptr_t) m;
        ov_apply_rank_sort(mm);
        g_last_model = m;
    }

    /* Enforce active selection tracking:
     * If the selected item no longer exists in the filtered list
     * (e.g. removed by an external process), reset the selection to 0.
     * Otherwise, clamp bounds and update the tracked name. */
    {
        const char *names[OV_MAX_NODES];
        int         fidx[OV_MAX_NODES];
        const char *f_str = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
        const char *f_prc = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
        const char *f_fps = ov_get_active_filter_for(lay, OV_FOCUS_FPS);

        /* Streams */
        for (int i = 0; i < m->nb_streams; i++)
        {
            names[i] = m->streams[i].name;
        }
        int fn = ov_filter_build(f_str, names, m->nb_streams, fidx, OV_MAX_NODES);
        if (fn > 0)
        {
            if (lay->sel_name_stream[0] != '\0')
            {
                int still_exists = 0;
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(m->streams[fidx[i]].name, lay->sel_name_stream) == 0)
                    {
                        still_exists    = 1;
                        lay->sel_stream = i;
                        int page_h      = lay->r_streams.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_stream < lay->scroll_stream)
                            {
                                lay->scroll_stream = lay->sel_stream;
                            }
                            if (lay->sel_stream >= lay->scroll_stream + page_h)
                            {
                                lay->scroll_stream = lay->sel_stream - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if (!still_exists)
                {
                    lay->sel_stream = 0;
                }
            }
            if (lay->sel_stream >= fn)
            {
                lay->sel_stream = fn - 1;
            }
            if (lay->sel_stream < 0)
            {
                lay->sel_stream = 0;
            }
            strncpy(lay->sel_name_stream, m->streams[fidx[lay->sel_stream]].name, 79);
        }
        else
        {
            lay->sel_stream         = 0;
            lay->sel_name_stream[0] = '\0';
        }

        /* Procs */
        for (int i = 0; i < m->nb_procs; i++)
        {
            names[i] = m->procs[i].name;
        }
        fn = ov_filter_build(f_prc, names, m->nb_procs, fidx, OV_MAX_NODES);
        if (fn > 0)
        {
            if (lay->sel_name_proc[0] != '\0')
            {
                int still_exists = 0;
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(m->procs[fidx[i]].name, lay->sel_name_proc) == 0 &&
                        m->procs[fidx[i]].PID == lay->sel_pid_proc)
                    {
                        still_exists  = 1;
                        lay->sel_proc = i;
                        int page_h    = lay->r_procs.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_proc < lay->scroll_proc)
                            {
                                lay->scroll_proc = lay->sel_proc;
                            }
                            if (lay->sel_proc >= lay->scroll_proc + page_h)
                            {
                                lay->scroll_proc = lay->sel_proc - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if (!still_exists)
                {
                    lay->sel_proc = 0;
                }
            }
            if (lay->sel_proc >= fn)
            {
                lay->sel_proc = fn - 1;
            }
            if (lay->sel_proc < 0)
            {
                lay->sel_proc = 0;
            }
            strncpy(lay->sel_name_proc, m->procs[fidx[lay->sel_proc]].name, 79);
            lay->sel_pid_proc = m->procs[fidx[lay->sel_proc]].PID;
        }
        else
        {
            lay->sel_proc         = 0;
            lay->sel_name_proc[0] = '\0';
            lay->sel_pid_proc     = 0;
        }

        /* FPS */
        for (int i = 0; i < m->nb_fps; i++)
        {
            names[i] = m->fps[i].name;
        }
        fn = ov_filter_build(f_fps, names, m->nb_fps, fidx, OV_MAX_NODES);
        if (fn > 0)
        {
            if (lay->sel_name_fps[0] != '\0')
            {
                int still_exists = 0;
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(m->fps[fidx[i]].name, lay->sel_name_fps) == 0)
                    {
                        still_exists = 1;
                        lay->sel_fps = i;
                        int page_h   = lay->r_fps.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_fps < lay->scroll_fps)
                            {
                                lay->scroll_fps = lay->sel_fps;
                            }
                            if (lay->sel_fps >= lay->scroll_fps + page_h)
                            {
                                lay->scroll_fps = lay->sel_fps - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if (!still_exists)
                {
                    lay->sel_fps = 0;
                }
            }
            if (lay->sel_fps >= fn)
            {
                lay->sel_fps = fn - 1;
            }
            if (lay->sel_fps < 0)
            {
                lay->sel_fps = 0;
            }
            strncpy(lay->sel_name_fps, m->fps[fidx[lay->sel_fps]].name, 79);
        }
        else
        {
            lay->sel_fps         = 0;
            lay->sel_name_fps[0] = '\0';
        }
    }

    /* Always patch graph node .index fields to
     * reflect current array positions — needed
     * whether we just sorted or scan rebuilt the
     * model.  Each item's node_idx still points
     * to its graph node; update the reverse link. */
    {
        OV_MODEL *mm = (OV_MODEL *) (uintptr_t) m;
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
    ov_render_tabs(lay);

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
            int rendered = 0;
            if (lay->graph_tab_mode == 1)
            {
                ov_render_loops_panel(lay, m);
                rendered = 1;
            }
            else if (lay->graph_tab_mode == 2)
            {
                rendered = ov_render_detail_panel(lay, m);
            }
            else if (lay->graph_tab_mode == 3)
            {
                rendered = ov_render_resources_panel(lay, m);
            }

            if (!rendered)
            {
                ov_render_graph_panel(lay, m);
            }
            break;
        case OV_VIEW_GRAPH:
            ov_render_graph_panel(lay, m);
            break;
        case OV_VIEW_LOOPS:
            ov_render_loops_view(lay, m);
            break;
        case OV_VIEW_STREAMS:
            ov_render_streams_panel(lay, m, &rel);
            break;
        case OV_VIEW_PROCS:
            ov_render_procs_panel(lay, m, &rel);
            break;
        case OV_VIEW_FPS:
            ov_render_fps_param_info(lay, m);
            ov_render_fps_panel(lay, m, &rel);
            int cur_fsel = ov_get_selected_fps_idx(lay, m);
            if (cur_fsel >= 0 && cur_fsel < m->nb_fps &&
                m->fps[cur_fsel].nb_disp_params > 0)
            {
                ov_render_fps_params_panel(lay, m);
            }
            else
            {
                /* No params: draw empty right panel */
                ov_draw_panel_border(lay->r_fps_params.row, lay->r_fps_params.col,
                                     lay->r_fps_params.height, lay->r_fps_params.width, "PARAMS",
                                     OV_FG_DIM, 0, 0);
            }
            break;
        default:
            break;
        }
    }

    if (lay->show_help)
    {
        ov_render_help(lay, m);
    }

    if (!lay->show_help)
    {
        ov_render_cmdlog(lay);
    }
    ov_render_status(lay, m);

    /* Highlight movable edges if hovering */
    if (lay->mouse_hover && !lay->show_help)
    {
        ov_theme_fg(OV_FG_WARN);
        ov_theme_bg(OV_BG_TERMINAL);
        ov_buf_bold();

        if (lay->cmdlog_split_hover)
        {
            int cmdlog_top =
                (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
            if (cmdlog_top > 1)
            {
                ov_buf_pos(cmdlog_top - 1, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
            }
        }

        if (lay->view == OV_VIEW_DASHBOARD)
        {
            if (lay->dash_split_h_hover)
            {
                int r = lay->r_streams.row + lay->r_streams.height - 1;
                ov_buf_pos(r, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
                ov_buf_pos(r + 1, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
            }
            if (lay->dash_split_v_hover)
            {
                int c = lay->r_streams.width;
                for (int rr = lay->r_streams.row; rr < lay->r_fps.row + lay->r_fps.height; rr++)
                {
                    ov_buf_pos(rr, c);
                    ov_buf_printf("%s", OV_BOX_V_D);
                    ov_buf_pos(rr, c + 1);
                    ov_buf_printf("%s", OV_BOX_V_D);
                }
            }
        }
        else if (lay->view == OV_VIEW_FPS)
        {
            if (lay->fps_split_hover)
            {
                int c = lay->r_fps_list.width;
                for (int rr = lay->r_fps_list.row;
                     rr < lay->r_fps_list.row + lay->r_fps_list.height; rr++)
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
    ov_render_theme_popup(lay);
    ov_draw_tooltip(lay);

    ov_buf_flush_delta(lay->term_rows, lay->term_cols);
}
