// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_insp.c
 * @brief   F3 stream inspector: a panel beside (or under) the stream list with one page per tab. Pages write
 *          lines into a buffer (ov_insp_line, ov_insp_kv); the frame draws the part that fits, scrolled.
 */

#include "overview_render_internal.h"

#include <stdarg.h>
#include <time.h>

#define INSP_MAX_LINES 256

typedef struct
{
    ov_rgb_t fg;
    int      bold;
    char     label[24]; /* drawn dim before the text; "" = none */
    char     text[232];
} INSP_LINE;

static INSP_LINE s_line[INSP_MAX_LINES];
static int       s_nline;

/* Tab labels as last drawn: screen column and width, for clicks */
static struct
{
    int col, w;
} s_tab[OV_INSP_TAB_COUNT];
static int s_tab_row;

static const char *const s_tab_name[OV_INSP_TAB_COUNT] = { "INFO", "SEMS", "TRACE", "KEYS" };

/** ov_insp_tab_name - label of inspector page @t. */
const char *ov_insp_tab_name(int t)
{
    return (t >= 0 && t < OV_INSP_TAB_COUNT) ? s_tab_name[t] : "";
}

/** ov_insp_tab_at - inspector page whose label is at screen (@mr, @mc) as last drawn, or -1. */
int ov_insp_tab_at(int mr, int mc)
{
    for (int t = 0; mr == s_tab_row && t < OV_INSP_TAB_COUNT; t++)
    {
        if (s_tab[t].w > 0 && mc >= s_tab[t].col && mc < s_tab[t].col + s_tab[t].w)
        {
            return t;
        }
    }
    return -1;
}

/** ov_insp_line - add a page line in @fg (printf-style). */
void ov_insp_line(ov_rgb_t fg, const char *fmt, ...)
{
    if (s_nline >= INSP_MAX_LINES)
    {
        return;
    }
    INSP_LINE *l = &s_line[s_nline++];
    va_list    ap;
    l->fg       = fg;
    l->bold     = 0;
    l->label[0] = '\0';
    va_start(ap, fmt);
    vsnprintf(l->text, sizeof(l->text), fmt, ap);
    va_end(ap);
}

/** ov_insp_kv - add a "label  value" line, the label dim, the value in @fg. */
void ov_insp_kv(const char *label, ov_rgb_t fg, const char *fmt, ...)
{
    if (s_nline >= INSP_MAX_LINES)
    {
        return;
    }
    INSP_LINE *l = &s_line[s_nline++];
    va_list    ap;
    l->fg   = fg;
    l->bold = 0;
    snprintf(l->label, sizeof(l->label), "%-12s", label);
    va_start(ap, fmt);
    vsnprintf(l->text, sizeof(l->text), fmt, ap);
    va_end(ap);
}

/** fmt_time - wall-clock time @ts and its age ("2026-10-20 09:12:03 (5m ago)"); "-" when unset. */
static const char *fmt_time(char *buf, size_t n, struct timespec ts)
{
    if (ts.tv_sec <= 0)
    {
        return "-";
    }
    struct tm tm;
    char      when[32];
    time_t    t   = ts.tv_sec;
    long      age = (long) (time(NULL) - t);
    localtime_r(&t, &tm);
    strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &tm);
    if (age < 0)
    {
        snprintf(buf, n, "%s", when);
    }
    else
    {
        snprintf(buf, n, "%s (%ld%s ago)", when,
                 (age < 120)    ? age
                 : (age < 7200) ? age / 60
                                : age / 3600,
                 (age < 120)    ? "s"
                 : (age < 7200) ? "m"
                                : "h");
    }
    return buf;
}

/** pid_name - "PID name" of a process (name when it is a processinfo process), "-" for none. */
static const char *pid_name(const OV_MODEL *m, pid_t pid, char *buf, size_t n)
{
    if (pid <= 0)
    {
        return "-";
    }
    int pi = ov_find_proc_by_pid(m, pid);
    snprintf(buf, n, "%d %s", (int) pid, (pi >= 0) ? m->procs[pi].name : "");
    return buf;
}

/* INFO: the stream's metadata */
static void page_info(const OV_MODEL *m, const OV_STREAM *s, const OV_STREAM_DETAIL *d)
{
    char b1[64], loc[16];
    snprintf(loc, sizeof(loc), (d->location < 0) ? "CPU" : "GPU %d", (int) d->location);
    ov_insp_kv("Type", OV_FG_TEXT, "%s  %s  (%" PRIu64 " elements)", render_dtype(s->datatype),
               s->size_str, (uint64_t) s->nelement);
    ov_insp_kv("Memory", OV_FG_TEXT, "%sB  %s", ov_fmt_si(b1, sizeof(b1), (double) d->memsize),
               loc);
    ov_insp_kv("Version", OV_FG_TEXT, "%s%s", d->version, d->shared ? "  shared" : "  not shared");
    ov_insp_kv("Created", OV_FG_TEXT, "%s", fmt_time(b1, sizeof(b1), d->creationtime));
    ov_insp_kv("Written", OV_FG_TEXT, "%s", fmt_time(b1, sizeof(b1), d->writetime));
    ov_insp_kv("Accessed", OV_FG_TEXT, "%s", fmt_time(b1, sizeof(b1), d->lastaccesstime));
    ov_insp_kv("Creator", ov_pid_color(s->creatorPID), "%s",
               pid_name(m, s->creatorPID, b1, sizeof(b1)));
    ov_insp_kv("Owner", ov_pid_color(s->ownerPID), "%s", pid_name(m, s->ownerPID, b1, sizeof(b1)));
    ov_insp_kv("Counters", s->cnt_active ? OV_FG_ACTIVE : OV_FG_TEXT,
               "cnt0 %" PRIu64 "  cnt1 %" PRIu64 "  cnt2 %" PRIu64, d->cnt0, d->cnt1, d->cnt2);
    ov_insp_kv("Rate", OV_FG_TEXT, "%.1f Hz  (last scan %.1f Hz)%s", s->update_hz,
               s->update_hz_inst, d->write ? "  being written" : "");
    ov_insp_kv("Inode", OV_FG_DIM, "%" PRIu64 "%s%s", (uint64_t) s->inode,
               s->is_link ? "  link to " : "", s->is_link ? s->link_target : "");
    ov_insp_kv("Semaphores", OV_FG_TEXT, "%d", d->nb_sem);
    ov_insp_kv("Keywords", OV_FG_TEXT, "%d set, %d slots", d->nb_kw_listed, d->nb_kw);
    if (d->cb_size > 0)
    {
        ov_insp_kv("Circ buffer", OV_FG_TEXT, "%u slices, at %u, cycle %" PRIu64, d->cb_size,
                   d->cb_index, d->cb_cycle);
    }
}

/* SEMS: every semaphore with its value and the last reader and writer */
static void page_sems(const OV_MODEL *m, const OV_STREAM_DETAIL *d)
{
    char rb[96], wb[96];
    ov_insp_line(OV_FG_DIM, " %2s %6s  %-26s %s", "#", "VALUE", "READER", "WRITER");
    for (int i = 0; i < d->nb_sem; i++)
    {
        pid_name(m, d->sem_read[i], rb, sizeof(rb));
        pid_name(m, d->sem_write[i], wb, sizeof(wb));
        ov_insp_line((d->sem_read_alive[i] || d->sem_write_alive[i]) ? OV_FG_TEXT : OV_FG_DIM,
                     " %2d %6d  %-24.24s%s %s%s", i, d->semval[i], rb,
                     (d->sem_read[i] > 0 && !d->sem_read_alive[i]) ? "\xe2\x80\xa0" : " ",
                     wb, /* † gone */
                     (d->sem_write[i] > 0 && !d->sem_write_alive[i]) ? "\xe2\x80\xa0" : "");
    }
    if (d->nb_sem == 0)
    {
        ov_insp_line(OV_FG_DIM, " No semaphores");
    }
    else
    {
        ov_insp_line(OV_FG_DIM, " \xe2\x80\xa0 process gone");
    }
}

/** draw_tabs - the page labels on @row, the current one lit; records where each is. */
static void draw_tabs(const OV_LAYOUT *lay, OV_RECT r, int row)
{
    int col   = r.col + 2;
    s_tab_row = row;
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_printf(" ");
    for (int t = 0; t < OV_INSP_TAB_COUNT; t++)
    {
        int cur = (t == lay->strm_insp_tab);
        ov_theme_bg(cur ? OV_FG_STREAM : OV_BG_PANEL);
        ov_theme_fg(cur ? OV_BG_TERMINAL : OV_FG_DIM);
        if (cur)
        {
            ov_buf_bold();
        }
        ov_buf_printf(" %s ", s_tab_name[t]);
        ov_buf_reset_attr();
        s_tab[t].col = col;
        s_tab[t].w   = (int) strlen(s_tab_name[t]) + 2;
        col += s_tab[t].w + 1;
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf(" ");
    }
    render_pad_spaces(col - r.col - 1, r.width);
}

/**
 * ov_render_stream_inspector - the inspector panel of the F3 view (lay->r_strm_insp) for the stream under the
 * list cursor, on the page lay->strm_insp_tab, scrolled by lay->strm_insp_scroll (clamped here).
 */
void ov_render_stream_inspector(OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r = lay->r_strm_insp;
    if (r.width <= 0 || r.height < 5)
    {
        return;
    }
    int  si = (m != NULL) ? ov_get_selected_stream_idx(lay, m) : -1;
    char title[STRINGMAXLEN_IMAGE_NAME + 16];
    snprintf(title, sizeof(title), "STREAM %s", (si >= 0) ? m->streams[si].name : "");
    ov_draw_panel_border(r.row, r.col, r.height, r.width, title, OV_FG_STREAM, lay->strm_insp_focus,
                         0);
    draw_tabs(lay, r, r.row + 1);

    s_nline = 0;
    if (si < 0)
    {
        ov_insp_line(OV_FG_DIM, " No stream selected");
    }
    else if (m->streams[si].err || m->streams[si].pending)
    {
        ov_insp_line(m->streams[si].err ? OV_FG_ERROR : OV_FG_DIM, " %s",
                     m->streams[si].err ? ov_stream_err_label(m->streams[si].err)
                                        : "listed, not mapped yet");
    }
    else if (!m->insp.valid || strcmp(m->insp.name, m->streams[si].name) != 0)
    {
        ov_insp_line(OV_FG_DIM, " Reading %s \xe2\x80\xa6",
                     m->streams[si].name); /* next scan fills it */
    }
    else if (lay->strm_insp_tab == OV_INSP_SEMS)
    {
        page_sems(m, &m->insp);
    }
    else if (lay->strm_insp_tab == OV_INSP_TRACE)
    {
        ov_insp_page_trace(m, &m->streams[si]);
    }
    else if (lay->strm_insp_tab == OV_INSP_KEYS)
    {
        ov_insp_page_keys(&m->insp);
    }
    else
    {
        page_info(m, &m->streams[si], &m->insp);
    }

    int body              = r.height - 3; /* border, tab row */
    int maxs              = (s_nline > body) ? s_nline - body : 0;
    int sc                = lay->strm_insp_scroll;
    lay->strm_insp_scroll = (sc > maxs) ? maxs : (sc < 0) ? 0 : sc;
    for (int i = 0; i < body; i++)
    {
        int              li = lay->strm_insp_scroll + i;
        const INSP_LINE *l  = (li < s_nline) ? &s_line[li] : NULL;
        ov_buf_pos(r.row + 2 + i, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        int n = 0;
        if (l != NULL && l->label[0] != '\0')
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" %s", l->label);
            n += 1 + (int) strlen(l->label);
        }
        if (l != NULL)
        {
            int cols = 0, room = r.width - 2 - n;
            int bytes = ov_utf8_fit(l->text, (int) strlen(l->text), (room > 0) ? room : 0, &cols);
            ov_theme_fg(l->fg);
            ov_buf_printf("%.*s", bytes, l->text);
            n += cols;
        }
        render_pad_spaces(n, r.width);
    }
    render_scroll_indicators(r, lay->strm_insp_scroll, body, s_nline, OV_FG_STREAM);
}
