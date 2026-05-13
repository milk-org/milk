/**
 * @file overview_data.c
 * @brief Scan and graph-building for milkCTRL
 *
 * Implements the three scanners (streams, FPS, processes)
 * and the cross-referencing graph builder.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include "overview_defs.h"
#include "overview_data.h"

#include "ImageStreamIO/ImageStreamIO.h"

/* fps_shmdirname.h for function_parameter_struct_shmdirname */
#include "fps_shmdirname.h"



/* STRINGMAXLEN_FPS_DIRNAME defined in fps_types.h */
#define OV_SHMDIR_MAXLEN STRINGMAXLEN_FPS_DIRNAME



#include "overview_data_internal.h"

/* =========================================================
 * Full scan
 * ========================================================= */

void ov_model_full_scan(OV_MODEL *model)
{
    static struct timespec s_last_scan = {0, 0};
    static int s_has_prev_scan = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Reset per-tick PID liveness cache */
    pid_cache_reset();

    /* Compute time delta for rate estimation
     * (used by scache_rate_update inside scan).
     * Use a static timestamp so triple-buffered
     * slots all share the same previous time. */
    if (s_has_prev_scan)
    {
        struct timespec dt_ts = ov_timespec_diff(
                                    s_last_scan, t0);
        s_scan_dt_sec =
            (double) dt_ts.tv_sec
            + (double) dt_ts.tv_nsec * 1.0e-9;
    }
    else
    {
        s_scan_dt_sec = 0.0;
    }

    /* Clear dynamic parts */
    model->nb_streams = 0;
    model->nb_fps     = 0;
    model->nb_procs   = 0;
    model->nb_nodes   = 0;
    model->nb_edges   = 0;

    ov_scan_streams(model);
    ov_scan_fps(model);
    ov_scan_procs(model);

    ov_build_graph(model);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    struct timespec dt = ov_timespec_diff(t0, t1);
    model->scan_time_ms =
        (double) dt.tv_sec * 1000.0
        + (double) dt.tv_nsec * 1.0e-6;
    s_last_scan = t0;
    s_has_prev_scan = 1;
    model->last_scan_time = t0;
    model->scan_count++;
}


/* =========================================================
 * Snapshot export
 * ========================================================= */

void ov_model_export_snapshot(const OV_MODEL *m)
{
    if (m == NULL)
    {
        return;
    }

    time_t now = time(NULL);
    struct tm *tm_ptr = localtime(&now);
    char fname[128];
    strftime(fname, sizeof(fname),
        "/tmp/milkCTRL_snapshot_%Y%m%d_%H%M%S.txt",
        tm_ptr);

    FILE *fp = fopen(fname, "w");
    if (fp == NULL)
    {
        return;
    }

    char tstr[64];
    strftime(tstr, sizeof(tstr),
        "%Y-%m-%d %H:%M:%S", tm_ptr);
    fprintf(fp,
        "# milkCTRL snapshot — %s\n"
        "# streams: %d  procs: %d  fps: %d"
        "  edges: %d\n\n",
        tstr, m->nb_streams, m->nb_procs,
        m->nb_fps, m->nb_edges);

    /* Streams */
    fprintf(fp, "=== STREAMS (%d) ===\n", m->nb_streams);
    fprintf(fp,
        "%-20s %4s %12s %8s %10s %7s %10s\n",
        "NAME", "TYP", "SIZE",
        "Hz", "INODE", "OWNER", "COUNT");
    for (int i = 0; i < m->nb_streams; i++)
    {
        const OV_STREAM *s = &m->streams[i];
        fprintf(fp,
            "%-20s %4d %12s %8.1f %10" PRIu64 " %7d %10" PRIu64 "\n",
            s->name, s->datatype, s->size_str,
            s->update_hz, (uint64_t) s->inode,
            (int) s->ownerPID,
            (uint64_t) s->cnt0);
    }

    /* Processes */
    fprintf(fp,
        "\n=== PROCESSINFO (%d) ===\n", m->nb_procs);
    fprintf(fp,
        "%-20s %7s %6s %8s %10s %s\n",
        "NAME", "PID", "STAT",
        "Hz", "MEM(KB)", "TRIGGER");
    for (int i = 0; i < m->nb_procs; i++)
    {
        const OV_PROC *p = &m->procs[i];
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
        fprintf(fp,
            "%-20s %7d %6s %8.1f %10" PRId64 " %s\n",
            p->name, (int) p->PID, sl,
            p->loop_hz, (int64_t) p->mem_rss_kb,
            p->trigstreamname[0]
                ? p->trigstreamname : "-");
    }

    /* FPS */
    fprintf(fp, "\n=== FPS (%d) ===\n", m->nb_fps);
    fprintf(fp,
        "%-24s %4s %4s %10s %s\n",
        "NAME", "CONF", "RUN", "MEM(KB)",
        "DESCRIPTION");
    for (int i = 0; i < m->nb_fps; i++)
    {
        const OV_FPS *f = &m->fps[i];
        fprintf(fp,
            "%-24s %4s %4s %10" PRId64 " %s\n",
            f->name,
            f->conf_alive ? "Y" : "-",
            f->run_alive  ? "Y" : "-",
            (int64_t) f->mem_rss_kb,
            f->description);
    }

    /* Edges */
    fprintf(fp,
        "\n=== EDGES (%d) ===\n", m->nb_edges);
    for (int i = 0; i < m->nb_edges; i++)
    {
        const OV_EDGE *e = &m->edges[i];
        if (e->src_node >= 0
            && e->src_node < m->nb_nodes
            && e->tgt_node >= 0
            && e->tgt_node < m->nb_nodes)
        {
            fprintf(fp, "  %s -> %s  [%s]\n",
                m->nodes[e->src_node].name,
                m->nodes[e->tgt_node].name,
                e->label);
        }
    }

    fclose(fp);
}
