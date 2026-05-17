/**
 * @file milk-stream-info.c
 * @brief Print detailed info for a single SHM stream
 *
 * Displays stream metadata (type, dimensions, counters,
 * semaphores) and cross-referenced connections (written
 * by, triggers, read by, FPS linkage) using the
 * OV_MODEL graph.
 *
 * No CLIcore dependency. Links: ImageStreamIO +
 * milkprocessinfo + milkfps + m + rt + pthread.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>

#include "overview_defs.h"
#include "overview_data.h"
#include "milk_help.h"
#include <inttypes.h>

/* Required by overview_defs.h (extern) */
volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;

/* One-line description */
#define SI_ONELINE \
    "print detailed info and connections " \
    "for a shared-memory stream"

#define SI_DESC_LONG \
    "Scan the ImageStreamIO shared-memory area and the FPS\n" \
    "registry to build a connection graph, then print a rich\n" \
    "diagnostic view for the specified stream: type, dimensions,\n" \
    "memory footprint, counters, semaphores, and connections\n" \
    "(written by, triggers, read by, FPS linkage)."

/* Replace local ANSI macros with milk_help.h equivalents */
#define C_RST    MH_RST
#define C_BOLD   MH_BOLD
#define C_DIM    MH_DIM
#define C_TITLE  MH_TITLE
#define C_HDR    MH_HDR
#define C_LABEL  MH_DFLT
#define C_NAME   MH_CMD
#define C_PROC   MH_NOTE
#define C_FPS    MH_NOTE
#define C_VAL    MH_BOLD
#define C_ALIVE  "\033[1;32m"
#define C_DEAD   MH_ERR
#define C_WARN   MH_ERR
#define C_SEP    MH_DFLT

/* =========================================================
 * Datatype name helper
 * ========================================================= */

static const char *dtype_name(uint8_t dt)
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

static unsigned int dtype_bytes(uint8_t dt)
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

/* =========================================================
 * PID status string
 * ========================================================= */

static const char *pid_status_str(pid_t pid)
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

/* =========================================================
 * Find process name from model by PID
 * ========================================================= */

static const char *proc_name_by_pid(
    const OV_MODEL *m,
    pid_t pid)
{
    int pi = ov_find_proc_by_pid(m, pid);
    if (pi >= 0)
    {
        return m->procs[pi].name;
    }
    return NULL;
}

/* =========================================================
 * Print the stream info
 * ========================================================= */

static void print_stream_info(
    const OV_MODEL *m,
    int si)
{
    const OV_STREAM *s = &m->streams[si];

    /* ---- Header ---- */
    printf(C_TITLE
           "========================================"
           "================\n" C_RST);
    printf(C_LABEL " %-20s" C_RST ": "
           C_NAME "%s" C_RST "\n",
           "Stream Name", s->name);
    printf(C_LABEL " %-20s" C_RST ": "
           C_VAL "%s (%d)" C_RST "\n",
           "Data Type",
           dtype_name(s->datatype),
           s->datatype);

    /* Dimensions */
    {
        char dimstr[64];
        if (s->naxis == 1)
        {
            snprintf(dimstr, sizeof(dimstr),
                     "1D  %u",
                     (unsigned) s->size[0]);
        }
        else if (s->naxis == 2)
        {
            snprintf(dimstr, sizeof(dimstr),
                     "2D  %u x %u",
                     (unsigned) s->size[0],
                     (unsigned) s->size[1]);
        }
        else
        {
            snprintf(dimstr, sizeof(dimstr),
                     "3D  %u x %u x %u",
                     (unsigned) s->size[0],
                     (unsigned) s->size[1],
                     (unsigned) s->size[2]);
        }
        printf(C_LABEL " %-20s" C_RST ": "
               C_VAL "%s" C_RST "\n",
               "Dimensions", dimstr);
    }

    printf(C_LABEL " %-20s" C_RST ": "
           C_VAL "%" PRIu64 C_RST "\n",
           "Elements",
           (uint64_t) s->nelement);

    {
        uint64_t bytes =
            (uint64_t) s->nelement
            * dtype_bytes(s->datatype);
        printf(C_LABEL " %-20s" C_RST ": "
               C_VAL "%" PRIu64 " bytes" C_RST "\n",
               "Memory", bytes);
    }

    printf(C_LABEL " %-20s" C_RST ": "
           C_VAL "%" PRIu64 C_RST "\n",
           "Inode",
           (uint64_t) s->inode);
    printf(C_TITLE
           "========================================"
           "================\n" C_RST);

    /* ---- Ownership ---- */
    printf("\n" C_HDR " Ownership" C_RST "\n");
    {
        const char *cname =
            proc_name_by_pid(m, s->creatorPID);
        printf("   %-18s: " C_PROC "%d" C_RST,
               "Creator PID",
               (int) s->creatorPID);
        if (cname)
        {
            printf(" (%s)", cname);
        }
        printf(" [%s]\n",
               pid_status_str(s->creatorPID));
    }
    {
        const char *oname =
            proc_name_by_pid(m, s->ownerPID);
        printf("   %-18s: " C_PROC "%d" C_RST,
               "Owner PID",
               (int) s->ownerPID);
        if (oname)
        {
            printf(" (%s)", oname);
        }
        printf(" [%s]\n",
               pid_status_str(s->ownerPID));
    }

    /* ---- Counters ---- */
    printf("\n" C_HDR " Counters" C_RST "\n");
    printf("   %-18s: " C_VAL "%" PRIu64 C_RST "\n",
           "cnt0",
           (uint64_t) s->cnt0);
    if (s->update_hz > 0.01)
    {
        printf("   %-18s: " C_VAL "%.1f Hz"
               C_RST "\n",
               "Update rate", s->update_hz);
    }

    /* ---- Semaphores ---- */
    printf("\n" C_HDR " Semaphores" C_RST
           " (%d active)\n", s->nb_sem);
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
    printf("\n" C_HDR " Connections"
           C_RST " (from system graph)\n");

    int sni = s->node_idx;
    if (sni < 0)
    {
        printf("   " C_DIM
               "(stream not in graph)"
               C_RST "\n");
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
            if (m->edges[e].type
                != OV_EDGE_PROC_WRITES_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].src_node;
            if (ni < 0
                || ni >= m->nb_nodes
                || m->nodes[ni].type
                    != OV_NODE_PROC)
            {
                continue;
            }
            int pi = m->nodes[ni].index;
            printf("   %-18s: " C_PROC "%s"
                   C_RST " (PID %d)\n",
                   "Written by",
                   m->procs[pi].name,
                   (int) m->procs[pi].PID);
            found_any = 1;
        }

        /* Triggers (stream → PROC) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].src_node != sni)
            {
                continue;
            }
            if (m->edges[e].type
                != OV_EDGE_STREAM_TRIGGERS_PROC
                && m->edges[e].type
                != OV_EDGE_PROC_TRIGGER_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].tgt_node;
            if (ni < 0
                || ni >= m->nb_nodes
                || m->nodes[ni].type
                    != OV_NODE_PROC)
            {
                continue;
            }
            int pi = m->nodes[ni].index;
            printf("   %-18s: " C_PROC "%s"
                   C_RST " (PID %d)\n",
                   "Triggers",
                   m->procs[pi].name,
                   (int) m->procs[pi].PID);
            found_any = 1;
        }

        /* Read by (stream → PROC via sem) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].src_node != sni)
            {
                continue;
            }
            if (m->edges[e].type
                != OV_EDGE_STREAM_READ_BY_PROC)
            {
                continue;
            }
            int ni = m->edges[e].tgt_node;
            if (ni < 0
                || ni >= m->nb_nodes
                || m->nodes[ni].type
                    != OV_NODE_PROC)
            {
                continue;
            }
            int pi = m->nodes[ni].index;
            printf("   %-18s: " C_PROC "%s"
                   C_RST " (PID %d)\n",
                   "Read by (sem)",
                   m->procs[pi].name,
                   (int) m->procs[pi].PID);
            found_any = 1;
        }

        /* FPS input to (stream → FPS) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].src_node != sni)
            {
                continue;
            }
            if (m->edges[e].type
                != OV_EDGE_FPS_INPUT_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].tgt_node;
            if (ni < 0
                || ni >= m->nb_nodes
                || m->nodes[ni].type
                    != OV_NODE_FPS)
            {
                continue;
            }
            int fi = m->nodes[ni].index;
            printf("   %-18s: " C_FPS "%s"
                   C_RST "\n",
                   "FPS input to",
                   m->fps[fi].name);
            found_any = 1;
        }

        /* FPS output of (FPS → stream) */
        for (int e = 0; e < m->nb_edges; e++)
        {
            if (m->edges[e].tgt_node != sni)
            {
                continue;
            }
            if (m->edges[e].type
                != OV_EDGE_FPS_OUTPUT_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].src_node;
            if (ni < 0
                || ni >= m->nb_nodes
                || m->nodes[ni].type
                    != OV_NODE_FPS)
            {
                continue;
            }
            int fi = m->nodes[ni].index;
            printf("   %-18s: " C_FPS "%s"
                   C_RST "\n",
                   "FPS output of",
                   m->fps[fi].name);
            found_any = 1;
        }

        if (!found_any)
        {
            printf("   " C_DIM
                   "(no connections found)"
                   C_RST "\n");
        }
    }

    /* ---- Process trace ---- */
    if (s->nb_proctrace > 0)
    {
        printf("\n" C_HDR " Process Trace"
               C_RST " (STREAM_PROC_TRACE)\n");
        for (int t = 0;
             t < s->nb_proctrace; t++)
        {
            const char *pn =
                proc_name_by_pid(
                    m, s->proctrace_pid[t]);
            printf("   [%d] PID=%-6d"
                   "  trig_inode=%-8" PRIu64
                   "  mode=%d",
                   t,
                   (int) s->proctrace_pid[t],
                   (uint64_t)
                       s->proctrace_inode[t],
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

/* =========================================================
 * Help
 * ========================================================= */

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, SI_ONELINE, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<stream_name>%s\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", SI_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-stream-info%s %sdm00disp%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = {
        "milk-stream-list:list active shared memory streams",
        "milk-stream-rm:remove shared memory streams",
        "milk-procinfo-info:inspect processinfo memory contents"
    };
    milk_help_see_also(see_also, 3, mh_color);
}

/* =========================================================
 * main
 * ========================================================= */

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                SI_ONELINE, SI_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
        return 0;
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    static struct option long_opts[] = {
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(
                argc, argv, "h",
                long_opts, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h': break; /* handled above */
        default:
            printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    if (optind >= argc)
    {
        printf("\n\033[1;31mERROR\033[0m stream name required\n\n");
        print_help(argv[0], 1);
        return 1;
    }

    const char *stream_name = argv[optind];

    /* Build the system model */
    OV_MODEL model;
    memset(&model, 0, sizeof(model));
    ov_model_full_scan(&model);

    /* Find the requested stream */
    int si = ov_find_stream_by_name(
                 &model, stream_name);
    if (si < 0)
    {
        PRINT_ERROR(
            "stream '%s' not found in shared memory",
            stream_name);
        ov_scan_cache_cleanup();
        return 1;
    }

    print_stream_info(&model, si);

    ov_scan_cache_cleanup();
    return 0;
}
