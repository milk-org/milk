// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_insp_pages.c
 * @brief   F3 stream inspector pages TRACE (the process chain that wrote the stream, with timing) and KEYS
 *          (the stream's keywords).
 */

#include "overview_render_internal.h"

#include <time.h>

/* Trigger status of a trace hop */
static const char *trig_status(int st)
{
    return (st == PROCESSINFO_TRIGGERSTATUS_WAITING)    ? "waiting"
           : (st == PROCESSINFO_TRIGGERSTATUS_RECEIVED) ? "received"
           : (st == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) ? "timed out"
                                                        : "-";
}

/** ov_insp_page_trace - each hop of @s's process trace: writer, trigger stream, mode, status, timing. */
void ov_insp_page_trace(const OV_MODEL *m, const OV_STREAM *s)
{
    char            b1[16], b2[16], b3[16];
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    int64_t now_ns = (int64_t) now.tv_sec * 1000000000LL + now.tv_nsec;
    if (s->nb_proctrace == 0)
    {
        ov_insp_line(OV_FG_DIM,
                     " No process trace: no process has written this stream through processinfo");
        return;
    }
    ov_insp_line(OV_FG_TITLE, " %d hop%s, end to end %s", s->nb_proctrace,
                 (s->nb_proctrace > 1) ? "s" : "",
                 ov_fmt_dt_ns(b1, sizeof(b1), ov_trace_end_to_end_ns(m, s)));
    for (int t = 0; t < s->nb_proctrace; t++)
    {
        const OV_TRACE_HOP *h  = &ov_stream_hops(m, s)[t];
        int                 pi = ov_find_proc_by_pid(m, h->pid);
        int                 ti = ov_find_stream_by_inode(m, h->inode);
        OV_HOP              hop;
        char                lag[24] = "-";
        ov_trace_hop(m, s, t, &hop);
        if (hop.lag >= 0)
        {
            snprintf(lag, sizeof(lag), "%lld", (long long) hop.lag);
        }
        ov_insp_line(ov_pid_color(h->pid), " %d. %d %s", t, (int) h->pid,
                     (pi >= 0) ? m->procs[pi].name : "");
        if (ti >= 0)
        {
            ov_insp_kv("   trigger", OV_FG_STREAM, "%s  %s, %s", m->streams[ti].name,
                       processinfo_triggermode_name(h->trigmode, 1), trig_status(h->status));
        }
        else
        {
            ov_insp_kv("   trigger", OV_FG_DIM, "inode %" PRIu64 "  %s, %s", (uint64_t) h->inode,
                       processinfo_triggermode_name(h->trigmode, 1), trig_status(h->status));
        }
        ov_insp_kv("   started", OV_FG_TEXT, "%s ago  compute %s  hand-off %s  lag %s",
                   ov_fmt_dt_ns(b1, sizeof(b1), (h->tstart_ns > 0) ? now_ns - h->tstart_ns : -1),
                   ov_fmt_dt_ns(b2, sizeof(b2), hop.compute_ns),
                   ov_fmt_dt_ns(b3, sizeof(b3), hop.handoff_ns), lag);
    }
}

/** ov_insp_page_keys - the keywords of the inspected stream: name, type, value, comment. */
void ov_insp_page_keys(const OV_STREAM_DETAIL *d)
{
    if (d->nb_kw_listed == 0)
    {
        ov_insp_line(OV_FG_DIM, " No keywords set (%d slots)", d->nb_kw);
        return;
    }
    ov_insp_line(OV_FG_DIM, " %-16s %s %-20s %s", "NAME", "T", "VALUE", "COMMENT");
    for (int i = 0; i < d->nb_kw_listed; i++)
    {
        ov_insp_line(OV_FG_TEXT, " %-16s %c %-20s %s", d->kw[i].name, d->kw[i].type, d->kw[i].value,
                     d->kw[i].comment);
    }
    if (d->nb_kw_listed == OV_INSP_MAX_KW && d->nb_kw > OV_INSP_MAX_KW)
    {
        ov_insp_line(OV_FG_DIM, " (first %d shown)", OV_INSP_MAX_KW);
    }
}
