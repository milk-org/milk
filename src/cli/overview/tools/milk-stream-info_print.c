// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-stream-info_print.c
 * @brief Detailed report formatter and connection printer for streams.
 */

#include "milk-stream-info.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * dtype_name - Get human-readable datatype name
 * @dt: ImageStreamIO datatype code
 *
 * Return: Constant string name of the datatype.
 */
static const char *dtype_name(
    uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "UINT8";
    case _DATATYPE_INT8:
        return "INT8";
    case _DATATYPE_UINT16:
        return "UINT16";
    case _DATATYPE_INT16:
        return "INT16";
    case _DATATYPE_UINT32:
        return "UINT32";
    case _DATATYPE_INT32:
        return "INT32";
    case _DATATYPE_UINT64:
        return "UINT64";
    case _DATATYPE_INT64:
        return "INT64";
    case _DATATYPE_FLOAT:
        return "FLOAT";
    case _DATATYPE_DOUBLE:
        return "DOUBLE";
    case _DATATYPE_COMPLEX_FLOAT:
        return "COMPLEX_FLOAT";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "COMPLEX_DOUBLE";
    default:
        return "UNKNOWN";
    }
}

/**
 * dtype_bytes - Get element byte size for datatype
 * @dt: ImageStreamIO datatype code
 *
 * Return: Size in bytes of a single element.
 */
static unsigned int dtype_bytes(
    uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
    case _DATATYPE_INT8:
        return 1;
    case _DATATYPE_UINT16:
    case _DATATYPE_INT16:
        return 2;
    case _DATATYPE_UINT32:
    case _DATATYPE_INT32:
    case _DATATYPE_FLOAT:
        return 4;
    case _DATATYPE_UINT64:
    case _DATATYPE_INT64:
    case _DATATYPE_DOUBLE:
    case _DATATYPE_COMPLEX_FLOAT:
        return 8;
    case _DATATYPE_COMPLEX_DOUBLE:
        return 16;
    default:
        return 0;
    }
}

/**
 * pid_status_str - Format process PID status with ANSI color tags
 * @pid: Process ID to query
 *
 * Return: Color-formatted string (ALIVE, ZOMBIE, DEAD, or N/A).
 */
static const char *pid_status_str(
    pid_t pid)
{
    if (pid <= 0)
    {
        return C_DIM "N/A" C_RST;
    }
    ov_pid_status_t st = pid_get_status(pid);
    switch (st)
    {
    case OV_PID_ALIVE:
        return C_ALIVE "ALIVE" C_RST;
    case OV_PID_ZOMBIE:
        return C_WARN "ZOMBIE" C_RST;
    default:
        return C_DEAD "DEAD" C_RST;
    }
}

/**
 * proc_name_by_pid - Lookup process name from model by PID
 * @m:   Pointer to data model
 * @pid: Process ID to find
 *
 * Return: Process name string or NULL if not found.
 */
static const char *proc_name_by_pid(
    const OV_MODEL *m,
    pid_t           pid)
{
    int pi = ov_find_proc_by_pid(m, pid);
    if (pi >= 0)
    {
        return m->procs[pi].name;
    }
    return NULL;
}

/**
 * print_stream_info - Print formatted metadata report for a shared memory stream
 * @m:  Pointer to data model
 * @si: Stream index in @m->streams
 */
void print_stream_info(
    const OV_MODEL *m,
    int             si)
{
    const OV_STREAM *s = &m->streams[si];

    /* ---- Header ---- */
    printf(C_TITLE "========================================"
                   "================\n" C_RST);
    printf(C_LABEL " %-20s" C_RST ": " C_NAME "%s" C_RST "\n", "Stream Name", s->name);
    printf(C_LABEL " %-20s" C_RST ": " C_VAL "%s (%d)" C_RST "\n", "Data Type",
           dtype_name(s->datatype), s->datatype);

    /* Dimensions */
    {
        char dimstr[64];
        if (s->naxis == 1)
        {
            snprintf(dimstr, sizeof(dimstr), "1D  %u", (unsigned) s->size[0]);
        }
        else if (s->naxis == 2)
        {
            snprintf(dimstr, sizeof(dimstr), "2D  %u x %u", (unsigned) s->size[0],
                     (unsigned) s->size[1]);
        }
        else
        {
            snprintf(dimstr, sizeof(dimstr), "3D  %u x %u x %u", (unsigned) s->size[0],
                     (unsigned) s->size[1], (unsigned) s->size[2]);
        }
        printf(C_LABEL " %-20s" C_RST ": " C_VAL "%s" C_RST "\n", "Dimensions", dimstr);
    }

    printf(C_LABEL " %-20s" C_RST ": " C_VAL "%" PRIu64 C_RST "\n", "Elements",
           (uint64_t) s->nelement);

    {
        uint64_t bytes = (uint64_t) s->nelement * dtype_bytes(s->datatype);
        printf(C_LABEL " %-20s" C_RST ": " C_VAL "%" PRIu64 " bytes" C_RST "\n", "Memory", bytes);
    }

    printf(C_LABEL " %-20s" C_RST ": " C_VAL "%" PRIu64 C_RST "\n", "Inode", (uint64_t) s->inode);
    printf(C_TITLE "========================================"
                   "================\n" C_RST);

    /* ---- Ownership ---- */
    printf("\n" C_HDR " Ownership" C_RST "\n");
    {
        const char *cname = proc_name_by_pid(m, s->creatorPID);
        printf("   %-18s: " C_PROC "%d" C_RST, "Creator PID", (int) s->creatorPID);
        if (cname)
        {
            printf(" (%s)", cname);
        }
        printf(" [%s]\n", pid_status_str(s->creatorPID));
    }
    {
        const char *oname = proc_name_by_pid(m, s->ownerPID);
        printf("   %-18s: " C_PROC "%d" C_RST, "Owner PID", (int) s->ownerPID);
        if (oname)
        {
            printf(" (%s)", oname);
        }
        printf(" [%s]\n", pid_status_str(s->ownerPID));
    }

    /* ---- Counters ---- */
    printf("\n" C_HDR " Counters" C_RST "\n");
    printf("   %-18s: " C_VAL "%" PRIu64 C_RST "\n", "cnt0", (uint64_t) s->cnt0);
    if (s->update_hz > 0.01)
    {
        printf("   %-18s: " C_VAL "%.1f Hz" C_RST "\n", "Update rate", s->update_hz);
    }

    /* ---- Semaphores ---- */
    printf("\n" C_HDR " Semaphores" C_RST " (%d active)\n", s->nb_sem);
    if (s->nb_sem > 0)
    {
        printf("   ");
        for (int i = 0; i < s->nb_sem; i++)
        {
            printf("[%d]=%d  ", i, s->semval[i]);
        }
        printf("\n");
    }

    /* ---- Connections (from graph) ---- */
    printf("\n" C_HDR " Connections" C_RST " (from system graph)\n");

    int sni = s->node_idx;
    if (sni < 0)
    {
        printf("   " C_DIM "(stream not in graph)" C_RST "\n");
    }
    else
    {
        int found_any = 0;

        /* Written by (PROC → stream) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].tgt_node != sni)
            {
                continue;
            }
            if (m->edges[e].type != OV_EDGE_PROC_WRITES_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].src_node;
            if (ni < 0 || ni >= m->nb_nodes || m->nodes[ni].type != OV_NODE_PROC)
            {
                continue;
            }
            int pi = m->nodes[ni].index;
            printf("   %-18s: " C_PROC "%s" C_RST " (PID %d)\n", "Written by",
                   m->procs[pi].name, (int) m->procs[pi].PID);
            found_any = 1;
        }

        /* Triggers (stream → PROC) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].src_node != sni)
            {
                continue;
            }
            if (m->edges[e].type != OV_EDGE_STREAM_TRIGGERS_PROC &&
                m->edges[e].type != OV_EDGE_PROC_TRIGGER_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].tgt_node;
            if (ni < 0 || ni >= m->nb_nodes || m->nodes[ni].type != OV_NODE_PROC)
            {
                continue;
            }
            int pi = m->nodes[ni].index;
            printf("   %-18s: " C_PROC "%s" C_RST " (PID %d)\n", "Triggers",
                   m->procs[pi].name, (int) m->procs[pi].PID);
            found_any = 1;
        }

        /* Read by (stream → PROC via sem) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].src_node != sni)
            {
                continue;
            }
            if (m->edges[e].type != OV_EDGE_STREAM_READ_BY_PROC)
            {
                continue;
            }
            int ni = m->edges[e].tgt_node;
            if (ni < 0 || ni >= m->nb_nodes || m->nodes[ni].type != OV_NODE_PROC)
            {
                continue;
            }
            int pi = m->nodes[ni].index;
            printf("   %-18s: " C_PROC "%s" C_RST " (PID %d)\n", "Read by (sem)",
                   m->procs[pi].name, (int) m->procs[pi].PID);
            found_any = 1;
        }

        /* FPS input to (stream → FPS) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].src_node != sni)
            {
                continue;
            }
            if (m->edges[e].type != OV_EDGE_FPS_INPUT_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].tgt_node;
            if (ni < 0 || ni >= m->nb_nodes || m->nodes[ni].type != OV_NODE_FPS)
            {
                continue;
            }
            int fi = m->nodes[ni].index;
            printf("   %-18s: " C_FPS "%s" C_RST "\n", "FPS input to", m->fps[fi].name);
            found_any = 1;
        }

        /* FPS output of (FPS → stream) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].tgt_node != sni)
            {
                continue;
            }
            if (m->edges[e].type != OV_EDGE_FPS_OUTPUT_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].src_node;
            if (ni < 0 || ni >= m->nb_nodes || m->nodes[ni].type != OV_NODE_FPS)
            {
                continue;
            }
            int fi = m->nodes[ni].index;
            printf("   %-18s: " C_FPS "%s" C_RST "\n", "FPS output of", m->fps[fi].name);
            found_any = 1;
        }

        if (!found_any)
        {
            printf("   " C_DIM "(no connections found)" C_RST "\n");
        }
    }

    /* ---- Process trace ---- */
    if (s->nb_proctrace > 0)
    {
        printf("\n" C_HDR " Process Trace" C_RST " (STREAM_PROC_TRACE)\n");
        for (int t = 0; t < s->nb_proctrace; t++)
        {
            const char *pn = proc_name_by_pid(m, s->proctrace_pid[t]);
            printf("   [%d] PID=%-6d  trig_inode=%-8" PRIu64 "  mode=%d",
                   t, (int) s->proctrace_pid[t], (uint64_t) s->proctrace_inode[t],
                   s->proctrace_trigmode[t]);
            if (pn)
            {
                printf("  (%s)", pn);
            }
            printf("\n");
        }
    }

    printf("\n");
}
