/**
 * @file overview_render.c
 * @brief Panel rendering for milkCTRL
 *
 * Renders header, stream/proc/FPS panels, connection
 * graph, and status bar using buffered ANSI output.
 *
 * Flicker-free: uses cursor-home instead of screen-clear,
 * and carefully avoids double-drawing (erasing then writing)
 * any cell on the screen.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <regex.h>

#include "overview_defs.h"
#include "overview_ansi.h"
#include "overview_theme.h"
#include "overview_data.h"
#include "overview_layout.h"

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

static void render_pad_spaces(int chars_written, int panel_width);

/**
 * render_highlighted_name - print a string with regex match highlighting
 */
static void render_highlighted_name(
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
static int ov_filter_build(
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
 * Bitset helpers for up to OV_MAX_* items each.
 * We use uint64_t arrays as compact bitmaps.
 */
#define BITS_PER_WORD 64
#define OV_BSET_WORDS(n) (((n) + BITS_PER_WORD - 1) / BITS_PER_WORD)

#define OV_STREAM_WORDS OV_BSET_WORDS(OV_MAX_STREAMS)
#define OV_FPS_WORDS    OV_BSET_WORDS(OV_MAX_FPS)
#define OV_PROC_WORDS   OV_BSET_WORDS(OV_MAX_PROCS)

typedef struct
{
    uint64_t streams[OV_STREAM_WORDS];
    uint64_t fps[OV_FPS_WORDS];
    uint64_t procs[OV_PROC_WORDS];
    /* direction annotations for procs: bit set = writes, clear = reads */
    uint64_t proc_writes[OV_PROC_WORDS];
    /*
     * For each related FPS, a bitmask of stream_param_name[] indices
     * that matched the selected stream.  Bit k set means param index k
     * matched.  Zero means no specific params identified.
     * Supports up to OV_FPS_MAX_STREAM_PARAMS (24) params per FPS.
     */
    uint32_t fps_param_mask[OV_MAX_FPS];
} OV_RELATED;

static void bset(uint64_t *words, int idx)
{
    words[idx / BITS_PER_WORD] |= (UINT64_C(1) << (idx % BITS_PER_WORD));
}

static int bget(const uint64_t *words, int idx)
{
    return (words[idx / BITS_PER_WORD] >> (idx % BITS_PER_WORD)) & 1;
}

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

    /* Determine the graph node index of the selected item */
    int sel_node = -1;
    if (lay->focus == OV_FOCUS_STREAMS && lay->sel_stream >= 0
        && lay->sel_stream < m->nb_streams)
    {
        sel_node = m->streams[lay->sel_stream].node_idx;
    }
    else if (lay->focus == OV_FOCUS_FPS && lay->sel_fps >= 0
             && lay->sel_fps < m->nb_fps)
    {
        sel_node = m->fps[lay->sel_fps].node_idx;
    }
    else if (lay->focus == OV_FOCUS_PROCS && lay->sel_proc >= 0
             && lay->sel_proc < m->nb_procs)
    {
        sel_node = m->procs[lay->sel_proc].node_idx;
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
            if (lay->focus == OV_FOCUS_STREAMS
                && lay->sel_stream >= 0
                && lay->sel_stream < m->nb_streams)
            {
                const char *sname =
                    m->streams[lay->sel_stream].name;
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

static const char *render_dtype(uint8_t dt)
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

static void clear_row(
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
static void render_pad_spaces(int chars_written, int panel_width)
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
static void render_scroll_indicators(
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

    /* c1 visual length: 17 chars for " ● milkCTRL " */
    int chars_left = 17 + ctrl_w + c2 + c3 + c4 + c5;

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

void ov_render_streams_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    const OV_RELATED *rel)
{
    OV_RECT r = lay->r_streams;

    /* Build filtered index array */
    const char *names[OV_MAX_STREAMS];
    for (int i = 0; i < m->nb_streams; i++)
    {
        names[i] = m->streams[i].name;
    }
    int filt_idx[OV_MAX_STREAMS];
    int filt_n = ov_filter_build(
        lay->filter_stream, names,
        m->nb_streams, filt_idx, OV_MAX_STREAMS);

    int has_re = 0;
    regex_t re;
    if (lay->filter_stream[0] != '\0')
    {
        if (regcomp(&re, lay->filter_stream, REG_EXTENDED | REG_ICASE) == 0)
        {
            has_re = 1;
        }
    }

    /* Panel title with filter indicator */
    char title[80];
    if (lay->filter_stream[0] != '\0')
    {
        snprintf(title, sizeof(title),
                 "STREAMS /%s/", lay->filter_stream);
    }
    else
    {
        snprintf(title, sizeof(title), "STREAMS");
    }
    ov_draw_panel_border(
        r.row, r.col, r.height, r.width,
        title, OV_FG_STREAM,
        lay->focus == OV_FOCUS_STREAMS);

    int hrow = r.row + 1;
    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");
    ov_theme_fg(OV_FG_DIM);
    char htext[128];
    int hlen = snprintf(
        htext, sizeof(htext),
        "%-14s %3s %11s %6s",
        "NAME", "TYP", "SIZE", "Hz");
    ov_buf_printf("%s", htext);
    render_pad_spaces(1 + hlen, r.width);

    int max_rows = r.height - 3;
    int start = lay->scroll_stream;

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 1 + i;
        int fi = start + i;
        if (fi < filt_n)
        {
            int si = filt_idx[fi];
            const OV_STREAM *s = &m->streams[si];
            int is_sel =
                (fi == lay->sel_stream
                 && lay->focus == OV_FOCUS_STREAMS);
            int is_rel =
                (!is_sel
                 && lay->focus != OV_FOCUS_STREAMS
                 && rel != NULL
                 && bget(rel->streams, si));
            ov_rgb_t row_bg = is_sel
                ? OV_BG_SELECTED
                : is_rel ? OV_BG_RELATED
                         : OV_BG_PANEL;
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);
            ov_buf_printf(" ");

            int n1 = 15;
            render_highlighted_name(
                s->name, 14, &re, has_re,
                s->active ? OV_FG_STREAM : OV_FG_DIM,
                row_bg);
            ov_buf_printf(" ");

            ov_theme_fg(OV_FG_TEXT);
            int n2 = snprintf(
                NULL, 0, "%3s ",
                render_dtype(s->datatype));
            ov_buf_printf(
                "%3s ", render_dtype(s->datatype));

            char sizebuf[16];
            if (s->naxis == 1)
            {
                snprintf(sizebuf, sizeof(sizebuf),
                         "%u",
                         (unsigned) s->size[0]);
            }
            else if (s->naxis == 2)
            {
                snprintf(sizebuf, sizeof(sizebuf),
                         "%ux%u",
                         (unsigned) s->size[0],
                         (unsigned) s->size[1]);
            }
            else
            {
                snprintf(sizebuf, sizeof(sizebuf),
                         "%ux%ux%u",
                         (unsigned) s->size[0],
                         (unsigned) s->size[1],
                         (unsigned) s->size[2]);
            }

            int n3 = snprintf(
                NULL, 0, "%11s ", sizebuf);
            ov_buf_printf("%11s ", sizebuf);

            int n4 = 0;
            if (s->update_hz > 0.1)
            {
                ov_rgb_t hzc = ov_rgb_lerp(
                    OV_FG_DIM, OV_FG_ACTIVE,
                    (float)(s->update_hz / 5000.0));
                ov_theme_fg(hzc);
                n4 = snprintf(
                    NULL, 0, "%6.1f",
                    s->update_hz);
                ov_buf_printf(
                    "%6.1f", s->update_hz);
            }
            else
            {
                ov_theme_fg(OV_FG_DIM);
                n4 = snprintf(NULL, 0, "     -");
                ov_buf_printf("     -");
            }

            int n5 = 0;
            if (s->update_hz > 0.1)
            {
                ov_theme_fg(OV_FG_ACTIVE);
                ov_buf_printf(" ●");
                n5 = 2;
            }
            render_pad_spaces(
                1 + n1 + n2 + n3 + n4 + n5,
                r.width);
        }
        else
        {
            clear_row(
                row, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(
        r, lay->scroll_stream, max_rows,
        filt_n, OV_FG_STREAM);
    ov_buf_reset_attr();

    if (has_re)
    {
        regfree(&re);
    }
}

/**
 * render_trigmode_label - short label for trigger mode.
 */
static const char *render_trigmode_label(int mode)
{
    switch (mode)
    {
    case 0:  return "IMM";
    case 1:  return "CN0";
    case 2:  return "CN1";
    case 3:  return "SEM";
    case 4:  return "DLY";
    case 5:  return "SMP";
    case 6:  return "CN2";
    default: return " - ";
    }
}

void ov_render_procs_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    const OV_RELATED *rel)
{
    OV_RECT r = lay->r_procs;

    /* Build filtered index array */
    const char *names[OV_MAX_PROCS];
    for (int i = 0; i < m->nb_procs; i++)
    {
        names[i] = m->procs[i].name;
    }
    int filt_idx[OV_MAX_PROCS];
    int filt_n = ov_filter_build(
        lay->filter_proc, names,
        m->nb_procs, filt_idx, OV_MAX_PROCS);

    int has_re = 0;
    regex_t re;
    if (lay->filter_proc[0] != '\0')
    {
        if (regcomp(&re, lay->filter_proc, REG_EXTENDED | REG_ICASE) == 0)
        {
            has_re = 1;
        }
    }

    char title[80];
    if (lay->filter_proc[0] != '\0')
    {
        snprintf(title, sizeof(title),
                 "PROCESSES /%s/", lay->filter_proc);
    }
    else
    {
        snprintf(title, sizeof(title), "PROCESSES");
    }
    ov_draw_panel_border(
        r.row, r.col, r.height, r.width,
        title, OV_FG_PROC,
        lay->focus == OV_FOCUS_PROCS);

    int hrow = r.row + 1;
    int hs   = lay->hscroll_proc;

    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");
    ov_theme_fg(OV_FG_DIM);

    /* Full header text (wider than panel) */
    char htext[256];
    int hlen = snprintf(
        htext, sizeof(htext),
        "%-14s %6s %4s %6s %3s %-10s"
        " %7s %2s",
        "NAME", "PID", "STAT", "Hz",
        "TRG", "trig-strm",
        "exec", "");
    /* Apply hscroll: skip first hs chars */
    {
        int vis = hlen - hs;
        if (vis < 0)
        {
            vis = 0;
        }
        const char *start = htext + hs;
        if (hs >= hlen)
        {
            start = "";
            vis   = 0;
        }
        ov_buf_printf("%.*s", vis, start);
        render_pad_spaces(1 + vis, r.width);
    }

    int max_rows = r.height - 3;
    int start = lay->scroll_proc;

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 1 + i;
        int fi = start + i;
        if (fi < filt_n)
        {
            int pi = filt_idx[fi];
            const OV_PROC *p = &m->procs[pi];
            int is_sel = (fi == lay->sel_proc
                          && lay->focus == OV_FOCUS_PROCS);
            int is_rel = (!is_sel
                          && lay->focus != OV_FOCUS_PROCS
                          && rel != NULL
                          && bget(rel->procs, pi));
            int is_write = (is_rel && rel != NULL
                            && bget(rel->proc_writes, pi));
            ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED
                            : is_rel ? OV_BG_RELATED
                                     : OV_BG_PANEL;

            /* Build the full row text into a local
             * buffer so we can apply hscroll */
            char rbuf[256];
            int rlen = 0;

            /* Name */
            rlen += snprintf(
                rbuf + rlen,
                sizeof(rbuf) - (size_t) rlen,
                "%-14.14s ", p->name);

            /* PID */
            rlen += snprintf(
                rbuf + rlen,
                sizeof(rbuf) - (size_t) rlen,
                "%6d ", (int) p->PID);

            /* Status label */
            const char *sl;
            switch (p->loopstat)
            {
            case 0:  sl = "IDLE"; break;
            case 1:  sl = " RUN"; break;
            case 2:  sl = "PAUS"; break;
            case 3:  sl = "TERM"; break;
            case 4:  sl = "ERR!"; break;
            default: sl = " ?? "; break;
            }
            rlen += snprintf(
                rbuf + rlen,
                sizeof(rbuf) - (size_t) rlen,
                "%4s ", sl);

            /* Hz */
            if (p->loop_hz > 0.1)
            {
                rlen += snprintf(
                    rbuf + rlen,
                    sizeof(rbuf) - (size_t) rlen,
                    "%6.1f ", p->loop_hz);
            }
            else
            {
                rlen += snprintf(
                    rbuf + rlen,
                    sizeof(rbuf) - (size_t) rlen,
                    "     - ");
            }

            /* Trigger mode label */
            rlen += snprintf(
                rbuf + rlen,
                sizeof(rbuf) - (size_t) rlen,
                "%3s ", render_trigmode_label(
                    p->triggermode));

            /* Trigger stream name (truncated) */
            if (p->trigstreamname[0] != '\0'
                && p->triggermode > 0)
            {
                rlen += snprintf(
                    rbuf + rlen,
                    sizeof(rbuf) - (size_t) rlen,
                    "%-10.10s ",
                    p->trigstreamname);
            }
            else
            {
                rlen += snprintf(
                    rbuf + rlen,
                    sizeof(rbuf) - (size_t) rlen,
                    "%-10s ", "-");
            }

            /* Exec time (ms) */
            if (p->MeasureTiming
                && p->dtmedian_exec_ns > 0)
            {
                double exec_ms =
                    1.0e-6
                    * (double) p->dtmedian_exec_ns;
                rlen += snprintf(
                    rbuf + rlen,
                    sizeof(rbuf) - (size_t) rlen,
                    "%7.3f", exec_ms);
            }
            else
            {
                rlen += snprintf(
                    rbuf + rlen,
                    sizeof(rbuf) - (size_t) rlen,
                    "      -");
            }

            /* Direction arrow */
            if (is_rel)
            {
                const char *arr =
                    is_write ? " W" : " R";
                rlen += snprintf(
                    rbuf + rlen,
                    sizeof(rbuf) - (size_t) rlen,
                    "%s", arr);
            }

            /* Missed frame badge */
            if (p->triggermissed > 0)
            {
                rlen += snprintf(
                    rbuf + rlen,
                    sizeof(rbuf) - (size_t) rlen,
                    " M:%d", p->triggermissed);
            }

            /* Now render with hscroll and color */
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);
            ov_buf_printf(" ");

            /* Apply hscroll: show rbuf[hs..] */
            int vis = rlen - hs;
            if (vis < 0)
            {
                vis = 0;
            }

            /* Colorize fields individually.
             * For simplicity, render the scrolled
             * text as a single colored block.
             * Advanced per-field coloring can be
             * added later. */
            {
                /* Choose name color */
                ov_theme_fg(OV_FG_PROC);
                /* For the full-row approach, we print
                 * each segment manually to colorize */
            }

            /* Simpler approach: print full row, then
             * pad — use per-field color for key cols */
            /* Reset and reposition */
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);
            ov_buf_printf(" ");

            /* ---- per-field colored output ---- */
            int printed = 1;
            int avail = r.width - 2;

            /* Helper macro: skip hs chars, then
             * print at most avail-printed chars */
            #define PROC_FIELD(color, fmt, ...)        \
            do {                                       \
                char _fb[80];                          \
                int _fl = snprintf(                    \
                    _fb, sizeof(_fb), fmt,              \
                    __VA_ARGS__);                       \
                int _skip = 0;                         \
                if (hs > 0) {                          \
                    _skip = (hs < _fl) ? hs : _fl;     \
                    hs -= _skip;                       \
                }                                      \
                int _vis = _fl - _skip;                \
                int _max = avail - printed;             \
                if (_vis > _max) _vis = _max;          \
                if (_vis > 0) {                        \
                    ov_theme_fg(color);                 \
                    ov_buf_printf("%.*s",               \
                        _vis, _fb + _skip);            \
                    printed += _vis;                    \
                }                                      \
            } while(0)

            /* We need a mutable copy of hs for
             * the macro-based field skipping */
            int hs_rem = lay->hscroll_proc;

            /* Re-do per-field with colors */
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);
            ov_buf_printf(" ");
            printed = 1;

            /* Name */
            {
                char fb[80];
                int fl = snprintf(fb, sizeof(fb),
                    "%-14.14s ", p->name);
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    regmatch_t pm[1];
                    if (has_re && regexec(&re, p->name, 1, pm, 0) == 0)
                    {
                        int b_len = pm[0].rm_so;
                        if (b_len > 14) b_len = 14;
                        int m_len = pm[0].rm_eo - pm[0].rm_so;
                        if (b_len + m_len > 14) m_len = 14 - b_len;
                        
                        int seg_start[3] = {0, b_len, b_len + m_len};
                        int seg_len[3] = {b_len, m_len, fl - (b_len + m_len)};
                        int is_match[3] = {0, 1, 0};
                        
                        for (int s = 0; s < 3; s++) {
                            int s_start = seg_start[s];
                            int s_len = seg_len[s];
                            if (s_len <= 0) continue;
                            
                            int print_start = s_start;
                            if (print_start < skip) print_start = skip;
                            int print_end = s_start + s_len;
                            if (print_end > skip + vv) print_end = skip + vv;
                            
                            if (print_end > print_start) {
                                int len_to_print = print_end - print_start;
                                if (is_match[s]) {
                                    ov_buf_bold();
                                    ov_buf_fg(255, 255, 255);
                                } else {
                                    ov_theme_fg(OV_FG_PROC);
                                }
                                ov_buf_printf("%.*s", len_to_print, fb + print_start);
                                if (is_match[s]) {
                                    ov_buf_reset_attr();
                                    ov_theme_bg(row_bg);
                                }
                            }
                        }
                    }
                    else
                    {
                        ov_theme_fg(OV_FG_PROC);
                        ov_buf_printf(
                            "%.*s", vv, fb + skip);
                    }
                    printed += vv;
                }
            }
            /* PID */
            {
                char fb[80];
                int fl = snprintf(fb, sizeof(fb),
                    "%6d ", (int) p->PID);
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_TEXT);
                    ov_buf_printf(
                        "%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Status */
            {
                ov_rgb_t sc;
                switch (p->loopstat)
                {
                case 0:  sc = OV_FG_DIM;    break;
                case 1:  sc = OV_FG_ACTIVE;  break;
                case 2:  sc = OV_FG_WARN;    break;
                case 3:  sc = OV_FG_ERROR;   break;
                case 4:  sc = OV_FG_ERROR;   break;
                default: sc = OV_FG_DIM;     break;
                }
                char fb[80];
                int fl = snprintf(fb, sizeof(fb),
                    "%4s ", sl);
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    ov_theme_fg(sc);
                    ov_buf_printf(
                        "%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Hz */
            {
                char fb[80];
                int fl;
                ov_rgb_t hzc;
                if (p->loop_hz > 0.1)
                {
                    hzc = ov_rgb_lerp(
                        OV_FG_DIM, OV_FG_ACTIVE,
                        (float)(p->loop_hz
                                / 5000.0));
                    fl = snprintf(fb, sizeof(fb),
                        "%6.1f ", p->loop_hz);
                }
                else
                {
                    hzc = OV_FG_DIM;
                    fl = snprintf(fb, sizeof(fb),
                        "     - ");
                }
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    ov_theme_fg(hzc);
                    ov_buf_printf(
                        "%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Trigger mode */
            {
                char fb[80];
                int fl = snprintf(fb, sizeof(fb),
                    "%3s ",
                    render_trigmode_label(
                        p->triggermode));
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_CONN);
                    ov_buf_printf(
                        "%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Trigger stream */
            {
                char fb[80];
                int fl;
                if (p->trigstreamname[0] != '\0'
                    && p->triggermode > 0)
                {
                    fl = snprintf(fb, sizeof(fb),
                        "%-10.10s ",
                        p->trigstreamname);
                }
                else
                {
                    fl = snprintf(fb, sizeof(fb),
                        "%-10s ", "-");
                }
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_STREAM);
                    ov_buf_printf(
                        "%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Exec time */
            {
                char fb[80];
                int fl;
                ov_rgb_t ec = OV_FG_DIM;
                if (p->MeasureTiming
                    && p->dtmedian_exec_ns > 0)
                {
                    double exec_ms =
                        1.0e-6
                        * (double) p->dtmedian_exec_ns;
                    fl = snprintf(fb, sizeof(fb),
                        "%7.3f", exec_ms);
                    /* Color: green < 1ms, yellow < 10ms,
                     * red > 10ms */
                    if (exec_ms < 1.0)
                    {
                        ec = OV_FG_ACTIVE;
                    }
                    else if (exec_ms < 10.0)
                    {
                        ec = OV_FG_WARN;
                    }
                    else
                    {
                        ec = OV_FG_ERROR;
                    }
                }
                else
                {
                    fl = snprintf(fb, sizeof(fb),
                        "      -");
                }
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    ov_theme_fg(ec);
                    ov_buf_printf(
                        "%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Direction arrow */
            if (is_rel)
            {
                char fb[80];
                const char *arr =
                    is_write ? " ▶" : " ◀";
                int fl = snprintf(fb, sizeof(fb),
                    "%s", arr);
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_CONN);
                    ov_buf_printf(
                        "%.*s", vv, fb + skip);
                    printed += vv;
                }
            }
            /* Missed frame badge */
            if (p->triggermissed > 0)
            {
                char fb[80];
                int fl = snprintf(fb, sizeof(fb),
                    " M:%d", p->triggermissed);
                int skip = 0;
                if (hs_rem > 0)
                {
                    skip = (hs_rem < fl)
                         ? hs_rem : fl;
                    hs_rem -= skip;
                }
                int vv = fl - skip;
                int mx = avail - printed;
                if (vv > mx) { vv = mx; }
                if (vv > 0)
                {
                    ov_theme_fg(OV_FG_WARN);
                    ov_buf_printf(
                        "%.*s", vv, fb + skip);
                    printed += vv;
                }
            }

            #undef PROC_FIELD

            /* Pad remainder */
            {
                int rem = avail - printed;
                if (rem > 0)
                {
                    ov_buf_hline(' ', rem);
                }
            }
        }
        else
        {
            clear_row(
                row, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(
        r, lay->scroll_proc, max_rows,
        filt_n, OV_FG_PROC);
    ov_buf_reset_attr();

    if (has_re)
    {
        regfree(&re);
    }
}

void ov_render_fps_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    const OV_RELATED *rel)
{
    OV_RECT r = lay->r_fps;

    /* Build filtered index array */
    const char *names[OV_MAX_FPS];
    for (int i = 0; i < m->nb_fps; i++)
    {
        names[i] = m->fps[i].name;
    }
    int fidx[OV_MAX_FPS];
    int filt_n = ov_filter_build(
        lay->filter_fps, names,
        m->nb_fps, fidx, OV_MAX_FPS);

    int has_re = 0;
    regex_t re;
    if (lay->filter_fps[0] != '\0')
    {
        if (regcomp(&re, lay->filter_fps, REG_EXTENDED | REG_ICASE) == 0)
        {
            has_re = 1;
        }
    }

    char title[80];
    if (lay->filter_fps[0] != '\0')
    {
        snprintf(title, sizeof(title),
                 "FPS /%s/", lay->filter_fps);
    }
    else
    {
        snprintf(title, sizeof(title), "FPS");
    }
    ov_draw_panel_border(
        r.row, r.col, r.height, r.width,
        title, OV_FG_FPS,
        lay->focus == OV_FOCUS_FPS);

    int hrow = r.row + 1;
    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");
    ov_theme_fg(OV_FG_DIM);
    char htext[128];
    int hlen = snprintf(htext, sizeof(htext), "%-18s %1s %1s %3s", "NAME", "C", "R", "STR");
    ov_buf_printf("%s", htext);
    render_pad_spaces(1 + hlen, r.width);

    int max_rows = r.height - 3;
    int start = lay->scroll_fps;

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 1 + i;
        int ffi = start + i;
        if (ffi < filt_n)
        {
            int fi = fidx[ffi];
            const OV_FPS *f = &m->fps[fi];
            int is_sel = (ffi == lay->sel_fps
                          && lay->focus
                             == OV_FOCUS_FPS);
            int is_rel = (!is_sel
                && lay->focus != OV_FOCUS_FPS
                && rel != NULL
                && bget(rel->fps, fi));
            ov_rgb_t row_bg = is_sel
                ? OV_BG_SELECTED
                : is_rel ? OV_BG_RELATED
                         : OV_BG_PANEL;

            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);
            ov_buf_printf(" ");

            int n1 = 19;
            render_highlighted_name(
                f->name, 18, &re, has_re,
                OV_FG_FPS, row_bg);
            ov_buf_printf(" ");

            ov_theme_fg(f->conf_alive ? OV_FG_ACTIVE : OV_FG_DIM);
            int n2 = snprintf(NULL, 0, "%s ", f->conf_alive ? "C" : "-");
            ov_buf_printf("%s ", f->conf_alive ? "C" : "-");

            ov_theme_fg(f->run_alive ? OV_FG_ACTIVE : OV_FG_DIM);
            int n3 = snprintf(NULL, 0, "%s ", f->run_alive ? "R" : "-");
            ov_buf_printf("%s ", f->run_alive ? "R" : "-");

            ov_theme_fg(OV_FG_TEXT);
            int n4 = snprintf(NULL, 0, "%3d", f->nb_stream_params);
            ov_buf_printf("%3d", f->nb_stream_params);

            /* When cross-highlighted by a stream selection, iterate all
             * stream params of this FPS that match the selected stream
             * (fp_param_mask has one bit per matching param index). */
            int n5 = 0;
            if (is_rel && rel != NULL
                && lay->focus == OV_FOCUS_STREAMS)
            {
                uint32_t mask = rel->fps_param_mask[fi];
                for (int sp = 0;
                     mask != 0 && sp < f->nb_stream_params;
                     sp++, mask >>= 1)
                {
                    if (!(mask & 1))
                    {
                        continue;
                    }

                    const char *kname = f->stream_param_name[sp];

                    /* Special label for the trigger-stream parameter */
                    if (strcmp(kname, "procinfo.triggersname") == 0)
                    {
                        ov_buf_bg(120, 80, 10);  /* amber bg */
                        ov_buf_fg(255, 210, 80); /* gold text */
                        ov_buf_bold();
                        int w = snprintf(NULL, 0, " [TRIG]");
                        ov_buf_printf(" [TRIG]");
                        ov_buf_reset_attr();
                        ov_theme_bg(is_rel ? OV_BG_RELATED : OV_BG_PANEL);
                        n5 += w;
                    }
                    else
                    {
                        ov_theme_fg(OV_FG_CONN);
                        int w = snprintf(NULL, 0, " :%s", kname);
                        ov_buf_printf(" :%s", kname);
                        n5 += w;
                    }
                } /* for sp */
            } /* if is_rel stream */

            render_pad_spaces(1 + n1 + n2 + n3 + n4 + n5, r.width);
        }
        else
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(
        r, lay->scroll_fps, max_rows,
        filt_n, OV_FG_FPS);
    ov_buf_reset_attr();

    if (has_re)
    {
        regfree(&re);
    }
}

/* =========================================================
 * Detail pane — replaces CONNECTIONS when item selected
 * ========================================================= */

/**
 * ov_render_detail_panel - show detail for selected item.
 * @lay: layout (holds selection + graph rect)
 * @m:   data model
 *
 * Renders into the graph panel rectangle. Returns 1 if
 * detail was drawn, 0 if nothing to show (caller should
 * fall back to graph panel).
 */
static int ov_render_detail_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    OV_RECT r = lay->r_graph;
    int max_rows = r.height - 2;
    int row = r.row + 1;

    /* ---- Stream detail ---- */
    if (lay->focus == OV_FOCUS_STREAMS
        && lay->sel_stream >= 0
        && lay->sel_stream < m->nb_streams)
    {
        const OV_STREAM *s =
            &m->streams[lay->sel_stream];

        ov_draw_panel_border(
            r.row, r.col, r.height, r.width,
            "STREAM DETAIL", OV_FG_STREAM, 0);

        int ri = 0;

        /* Name */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s", s->name);
            ov_buf_printf(" %s", s->name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Type + Size */
        if (ri < max_rows)
        {
            char szb[48];
            if (s->naxis == 1)
            {
                snprintf(szb, sizeof(szb),
                    "%u", (unsigned) s->size[0]);
            }
            else if (s->naxis == 2)
            {
                snprintf(szb, sizeof(szb),
                    "%ux%u",
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
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " Type: %s  Size: %s"
                "  Elements: %lu",
                render_dtype(s->datatype), szb,
                (unsigned long) s->nelement);
            ov_buf_printf(
                " Type: %s  Size: %s"
                "  Elements: %lu",
                render_dtype(s->datatype), szb,
                (unsigned long) s->nelement);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Counters */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            int n = snprintf(NULL, 0,
                " cnt0: %lu  Hz: %.1f",
                (unsigned long) s->cnt0,
                s->update_hz);
            ov_buf_printf(
                " cnt0: %lu  Hz: %.1f",
                (unsigned long) s->cnt0,
                s->update_hz);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* PIDs */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " Creator: %d  Owner: %d"
                "  Inode: %lu",
                (int) s->creatorPID,
                (int) s->ownerPID,
                (unsigned long) s->inode);
            ov_buf_printf(
                " Creator: %d  Owner: %d"
                "  Inode: %lu",
                (int) s->creatorPID,
                (int) s->ownerPID,
                (unsigned long) s->inode);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Proctrace entries */
        if (ri < max_rows && s->nb_proctrace > 0)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " Process trace (%d):",
                s->nb_proctrace);
            ov_buf_printf(
                " Process trace (%d):",
                s->nb_proctrace);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;

            for (int t = 0;
                 t < s->nb_proctrace
                 && ri < max_rows; t++)
            {
                /* Find proc name by PID */
                const char *pname = "???";
                for (int pp = 0;
                     pp < m->nb_procs; pp++)
                {
                    if (m->procs[pp].PID
                        == s->proctrace_pid[t])
                    {
                        pname = m->procs[pp].name;
                        break;
                    }
                }
                ov_buf_pos(
                    row + ri, r.col + 1);
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_PROC);
                int n2 = snprintf(NULL, 0,
                    "  PID %d (%s)"
                    "  mode:%s",
                    (int) s->proctrace_pid[t],
                    pname,
                    render_trigmode_label(
                        s->proctrace_trigmode[t]));
                ov_buf_printf(
                    "  PID %d (%s)"
                    "  mode:%s",
                    (int) s->proctrace_pid[t],
                    pname,
                    render_trigmode_label(
                        s->proctrace_trigmode[t]));
                render_pad_spaces(
                    n2, r.width);
                ri++;
            }
        }
        /* Clear remaining rows */
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return 1;
    } /* STREAM detail */

    /* ---- Process detail ---- */
    if (lay->focus == OV_FOCUS_PROCS
        && lay->sel_proc >= 0
        && lay->sel_proc < m->nb_procs)
    {
        const OV_PROC *p =
            &m->procs[lay->sel_proc];

        ov_draw_panel_border(
            r.row, r.col, r.height, r.width,
            "PROCESS DETAIL", OV_FG_PROC, 0);

        int ri = 0;

        /* Name + PID */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s  (PID %d)",
                p->name, (int) p->PID);
            ov_buf_printf(" %s  (PID %d)",
                p->name, (int) p->PID);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Status + loop info */
        if (ri < max_rows)
        {
            const char *sl;
            switch (p->loopstat)
            {
            case 0:  sl = "IDLE"; break;
            case 1:  sl = "RUNNING"; break;
            case 2:  sl = "PAUSED"; break;
            case 3:  sl = "TERMINATING"; break;
            case 4:  sl = "ERROR"; break;
            default: sl = "UNKNOWN"; break;
            }
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            int n = snprintf(NULL, 0,
                " Status: %s  Loops: %ld"
                "  Hz: %.1f",
                sl, (long) p->loopcnt,
                p->loop_hz);
            ov_buf_printf(
                " Status: %s  Loops: %ld"
                "  Hz: %.1f",
                sl, (long) p->loopcnt,
                p->loop_hz);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Trigger info */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_CONN);
            int n = snprintf(NULL, 0,
                " Trigger: %s  stream: %s"
                "  sem: %d",
                render_trigmode_label(
                    p->triggermode),
                (p->trigstreamname[0] != '\0')
                    ? p->trigstreamname : "-",
                p->triggersem);
            ov_buf_printf(
                " Trigger: %s  stream: %s"
                "  sem: %d",
                render_trigmode_label(
                    p->triggermode),
                (p->trigstreamname[0] != '\0')
                    ? p->trigstreamname : "-",
                p->triggersem);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Missed frames */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            if (p->triggermissed > 0)
            {
                ov_theme_fg(OV_FG_WARN);
            }
            else
            {
                ov_theme_fg(OV_FG_DIM);
            }
            int n = snprintf(NULL, 0,
                " Missed: %d (cumul: %lu)",
                p->triggermissed,
                (unsigned long)
                    p->triggermissed_cumul);
            ov_buf_printf(
                " Missed: %d (cumul: %lu)",
                p->triggermissed,
                (unsigned long)
                    p->triggermissed_cumul);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Exec time + overhead */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            if (p->MeasureTiming
                && p->dtmedian_exec_ns > 0)
            {
                double exec_ms = 1.0e-6
                    * (double) p->dtmedian_exec_ns;
                double overhead = 0.0;
                if (p->dtmedian_iter_ns > 0)
                {
                    overhead = 100.0
                        * (double) p->dtmedian_exec_ns
                        / (double) p->dtmedian_iter_ns;
                }
                ov_theme_fg(OV_FG_TEXT);
                int n = snprintf(NULL, 0,
                    " Exec: %.3f ms  Load: %.1f%%"
                    "  RT: %d",
                    exec_ms, overhead,
                    p->rt_priority);
                ov_buf_printf(
                    " Exec: %.3f ms  Load: %.1f%%"
                    "  RT: %d",
                    exec_ms, overhead,
                    p->rt_priority);
                render_pad_spaces(n, r.width);
            }
            else
            {
                ov_theme_fg(OV_FG_DIM);
                int n = snprintf(NULL, 0,
                    " Timing: disabled  RT: %d",
                    p->rt_priority);
                ov_buf_printf(
                    " Timing: disabled  RT: %d",
                    p->rt_priority);
                render_pad_spaces(n, r.width);
            }
            ri++;
        }
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return 1;
    } /* PROCESS detail */

    /* ---- FPS detail ---- */
    if (lay->focus == OV_FOCUS_FPS
        && lay->sel_fps >= 0
        && lay->sel_fps < m->nb_fps)
    {
        const OV_FPS *f =
            &m->fps[lay->sel_fps];

        ov_draw_panel_border(
            r.row, r.col, r.height, r.width,
            "FPS DETAIL", OV_FG_FPS, 0);

        int ri = 0;

        /* Name */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s", f->name);
            ov_buf_printf(" %s", f->name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Description */
        if (ri < max_rows
            && f->description[0] != '\0')
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            int n = snprintf(NULL, 0,
                " %s", f->description);
            ov_buf_printf(" %s", f->description);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Conf / Run status */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " Conf: %s (PID %d)"
                "  Run: %s (PID %d)",
                f->conf_alive ? "ALIVE" : "dead",
                (int) f->confpid,
                f->run_alive ? "ALIVE" : "dead",
                (int) f->runpid);
            ov_buf_printf(
                " Conf: %s (PID %d)"
                "  Run: %s (PID %d)",
                f->conf_alive ? "ALIVE" : "dead",
                (int) f->confpid,
                f->run_alive ? "ALIVE" : "dead",
                (int) f->runpid);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Stream params */
        if (ri < max_rows
            && f->nb_stream_params > 0)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " Stream params (%d):",
                f->nb_stream_params);
            ov_buf_printf(
                " Stream params (%d):",
                f->nb_stream_params);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;

            for (int sp = 0;
                 sp < f->nb_stream_params
                 && ri < max_rows; sp++)
            {
                ov_buf_pos(
                    row + ri, r.col + 1);
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_DIM);
                ov_buf_printf("  ");
                ov_theme_fg(OV_FG_CONN);
                ov_buf_printf(
                    "%-24.24s",
                    f->stream_param_name[sp]);
                ov_theme_fg(OV_FG_STREAM);
                int n2 = snprintf(NULL, 0,
                    " = %s",
                    f->stream_param_value[sp]);
                ov_buf_printf(
                    " = %s",
                    f->stream_param_value[sp]);
                render_pad_spaces(
                    2 + 24 + n2, r.width);
                ri++;
            }
        }
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return 1;
    } /* FPS detail */

    return 0; /* no detail to show */
}

void ov_render_graph_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    OV_RECT r = lay->r_graph;
    ov_draw_panel_border(r.row, r.col, r.height, r.width, "CONNECTIONS", OV_FG_CONN, lay->focus == OV_FOCUS_GRAPH);

    int max_rows = r.height - 2;
    int row = r.row + 1;

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
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf(" ");

        ov_rgb_t sc;
        switch (src->type) {
        case OV_NODE_STREAM: sc = OV_FG_STREAM; break;
        case OV_NODE_FPS:    sc = OV_FG_FPS;    break;
        case OV_NODE_PROC:   sc = OV_FG_PROC;   break;
        }

        int is_sel = (ei == lay->sel_graph && lay->focus == OV_FOCUS_GRAPH);
        ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : OV_BG_PANEL;
        ov_theme_bg(row_bg);

        ov_theme_fg(sc);
        int n1 = snprintf(NULL, 0, "%-12.12s", src->name);
        ov_buf_printf("%-12.12s", src->name);

        ov_theme_fg(OV_FG_CONN);
        int n2 = snprintf(NULL, 0, " %s%s ", OV_BOX_H, OV_TRI_R);
        ov_buf_printf(" %s%s ", OV_BOX_H, OV_TRI_R);
        int vis_n2 = 4; 

        ov_rgb_t tc;
        switch (tgt->type) {
        case OV_NODE_STREAM: tc = OV_FG_STREAM; break;
        case OV_NODE_FPS:    tc = OV_FG_FPS;    break;
        case OV_NODE_PROC:   tc = OV_FG_PROC;   break;
        }
        ov_theme_fg(tc);
        int n3 = snprintf(NULL, 0, "%-12.12s", tgt->name);
        ov_buf_printf("%-12.12s", tgt->name);

        ov_theme_fg(OV_FG_DIM);
        int n4 = snprintf(NULL, 0, " [%.6s]", e->label);
        ov_buf_printf(" [%.6s]", e->label);

        render_pad_spaces(1 + n1 + vis_n2 + n3 + n4, r.width);
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
        case 1:  sort_label = " [sort:Hz]"; break;
        case 2:  sort_label = " [sort:typ]"; break;
        default: sort_label = " [sort:name]"; break;
        }
        break;
    case OV_FOCUS_PROCS:
        switch (lay->sort_key_proc)
        {
        case 1:  sort_label = " [sort:PID]"; break;
        case 2:  sort_label = " [sort:Hz]"; break;
        case 3:  sort_label = " [sort:stat]"; break;
        default: sort_label = " [sort:name]"; break;
        }
        break;
    case OV_FOCUS_FPS:
        switch (lay->sort_key_fps)
        {
        case 1:  sort_label = " [sort:alive]"; break;
        default: sort_label = " [sort:name]"; break;
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
        { "  F2-F6 / ^Left/^Right  Switch views",   0 },
        { "  TAB      Cycle panel focus",            0 },
        { "  UP/DOWN  Navigate list",                0 },
        { "  Left/Right  Horizontal scroll",         0 },
        { "  PgUp/Dn  Scroll page",                  0 },
        { "  +/-      Adjust scan rate",             0 },
        { "  h        Toggle this help",             0 },
        { "  D        Toggle detail pane",           0 },
        { "  p        Freeze/Pause display",         0 },
        { "  S        Sort by Hz/activity",            0 },
        { "  s        Sort alphabetical",              0 },
        { "  /        Filter (regex), ESC=clear",      0 },
        { "  q / x   Exit",                          0 },
        { "",                                        0 },
        { "Control mode  (c to toggle)",             0 },
        { "  FPS panel:",                            0 },
        { "    r  toggle run process",               0 },
        { "    s  toggle conf process",              0 },
        { "  STREAMS panel:",                        0 },
        { "    d  delete stream",                    0 },
        { "  PROCS panel:",                          0 },
        { "    k  send SIGTERM",                     0 },
        { "",                                        0 },
        { "Colors:  ",                               1 },
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

        if (i == 0 || strcmp(lines[i].text, "Control mode  (c to toggle)") == 0)
        {
            /* Section headers in bright */
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
        ov_sort_streams(mm, lay->sort_key_stream);
        ov_sort_procs(mm, lay->sort_key_proc);
        ov_sort_fps(mm, lay->sort_key_fps);
        
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
    ov_buf_append("\033[?2026h", 8);
    ov_buf_append("\033[H", 3);

    ov_render_header(lay, m);

    /* To prevent flickering on terminals that do not support synchronized updates,
     * we skip rendering the background panels when the help overlay is active.
     * The existing background is preserved on the terminal's screen. */
    if (!lay->show_help)
    {
        switch (lay->view)
        {
        case OV_VIEW_DASHBOARD:
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
    ov_buf_append("\033[?2026l", 8);
    ov_buf_flush();
}
