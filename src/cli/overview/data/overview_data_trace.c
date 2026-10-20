// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_trace.c
 * @brief   Timing of a stream's trigger chain from its process trace (proctrace).
 *
 * Hop 0 is the process that wrote the stream; hop t+1 wrote the stream that triggered hop t. Times are
 * CLOCK_MILK (realtime) nanoseconds written by processinfo_update_output_stream.
 */

#include "overview_data_internal.h"

/**
 * ov_trace_hop - timing of hop @t of stream @s: compute time (trigger to write; only when its process
 * measures timing), hand-off (upstream write to this trigger) and lag (frames the trigger stream moved on
 * since the write). Unknown values are -1. Return: 0, or -1 for a hop @s does not have.
 */
int ov_trace_hop(const OV_MODEL *m, const OV_STREAM *s, int t, OV_HOP *h)
{
    h->compute_ns = h->handoff_ns = h->lag = -1;
    if (t < 0 || t >= s->nb_proctrace)
    {
        return -1;
    }
    int64_t t0 = s->proctrace_tstart_ns[t], t1 = s->proctrace_tupdate_ns[t];
    int     pi = ov_find_proc_by_pid(m, s->proctrace_pid[t]);
    if (t0 > 0 && t1 >= t0 && pi >= 0 && m->procs[pi].MeasureTiming)
    {
        h->compute_ns = t1 - t0;
    }
    if (t + 1 < s->nb_proctrace && t0 > 0 && s->proctrace_tupdate_ns[t + 1] > 0 &&
        t0 >= s->proctrace_tupdate_ns[t + 1])
    {
        h->handoff_ns = t0 - s->proctrace_tupdate_ns[t + 1];
    }
    int si = ov_find_stream_by_inode(m, s->proctrace_inode[t]);
    if (si >= 0 && s->proctrace_tupdate_ns[t] > 0 && m->streams[si].cnt0 >= s->proctrace_cnt0[t])
    {
        h->lag = (int64_t) (m->streams[si].cnt0 - s->proctrace_cnt0[t]);
    }
    return 0;
}

/** ov_trace_end_to_end_ns - from the first traced trigger to the write of @s, or -1 when unknown. */
int64_t ov_trace_end_to_end_ns(const OV_STREAM *s)
{
    int     n  = s->nb_proctrace;
    int64_t t0 = (n > 0) ? s->proctrace_tstart_ns[n - 1] : 0;
    int64_t t1 = (n > 0) ? s->proctrace_tupdate_ns[0] : 0;
    return (t0 > 0 && t1 >= t0) ? t1 - t0 : -1;
}
