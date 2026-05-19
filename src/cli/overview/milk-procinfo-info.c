/**
 * @file milk-procinfo-info.c
 * @brief Print detailed info for a single processinfo
 *
 * Displays process status, timing statistics, trigger
 * configuration, and cross-referenced connections
 * (writes, reads, FPS linkage) using the OV_MODEL graph.
 *
 * No CLIcore dependency. Links: ImageStreamIO +
 * milkprocessinfo + milkfps + m + rt + pthread.
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>

#include "overview_defs.h"
#include "overview_data.h"
#include "milk_help.h"

/* Required by overview_defs.h (extern) */
volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;

/* One-line description */
#define PI_ONELINE \
    "print detailed info and connections " \
    "for a processinfo entry"

#define PI_DESC_LONG \
    "Scan the processinfo shared-memory list and the FPS registry\n" \
    "to build a connection graph, then print a rich diagnostic view\n" \
    "for the specified process: status, loop timing, trigger\n" \
    "configuration, streams written/read, and linked FPS entries.\n" \
    "Process can be specified by name or by --pid PID."

/* Replace local ANSI macros with milk_help.h equivalents */
#define C_RST    MH_RST
#define C_BOLD   MH_BOLD
#define C_DIM    MH_DIM
#define C_TITLE  MH_TITLE
#define C_HDR    MH_HDR
#define C_LABEL  MH_DFLT
#define C_NAME   MH_CMD
#define C_STREAM MH_DFLT
#define C_FPS    MH_NOTE
#define C_VAL    MH_BOLD
#define C_ALIVE  "\033[1;32m"
#define C_DEAD   MH_ERR
#define C_WARN   MH_ERR
#define C_RUN    "\033[1;32m"
#define C_STOP   MH_DIM

/* =========================================================
 * Loop status / CTRL val strings
 * ========================================================= */

static const char *loopstat_str(int stat)
{
    switch(stat)
    {
    case 0: return C_STOP "IDLE" C_RST;
    case 1: return C_RUN "ACTIVE" C_RST;
    case 2: return C_WARN "ERROR" C_RST;
    case 3: return C_WARN "CRASHED" C_RST;
    default: return C_DIM "UNKNOWN" C_RST;
    }
}

/**
 * @brief Map control value to a display string.
 */
static const char *ctrlval_str(int val)
{
    switch(val)
    {
    case 0: return C_STOP "STOP" C_RST;
    case 1: return C_RUN "RUN" C_RST;
    case 2: return C_WARN "PAUSE" C_RST;
    default: return C_DIM "UNKNOWN" C_RST;
    }
}

/**
 * @brief Map trigger mode to a display string.
 */
static const char *trigmode_str(int mode)
{
    switch(mode)
    {
    case 0: return "IMMEDIATE";
    case 1: return "SEMAPHORE";
    case 2: return "DELAY";
    case 3: return "TIMER";
    default: return "UNKNOWN";
    }
}

/* =========================================================
 * Print the processinfo
 * ========================================================= */

static void print_proc_info(
    const OV_MODEL *m,
    int            pi)
{
    const OV_PROC *p = &m->procs[pi];

    /* ---- Header ---- */
    printf(C_TITLE "========================================" "================\n" C_RST);
    printf(C_LABEL " %-20s" C_RST ": " C_NAME "%s" C_RST "\n", "Process Name", p->name);
    printf(C_LABEL " %-20s" C_RST ": "
           C_VAL "%d" C_RST " [%s]\n",
           "PID", (int) p->PID,
           (pid_get_status(p->PID) == OV_PID_ALIVE) ? C_ALIVE "ALIVE" C_RST : C_DEAD "DEAD" C_RST);
    printf(C_TITLE "========================================" "================\n" C_RST);

    /* ---- Status ---- */
    printf("\n" C_HDR " Status" C_RST "\n");
    printf("   %-18s: %s\n", "Loop stat", loopstat_str(p->loopstat));
    printf("   %-18s: %s\n", "CTRL val", ctrlval_str(p->CTRLval));
    printf("   %-18s: " C_VAL "%" PRId64 "" C_RST "\n", "Loop count", (int64_t) p->loopcnt);
    if(p->rt_priority > 0)
    {
        printf("   %-18s: " C_VAL "%d" C_RST "\n", "RT priority", p->rt_priority);
    }
    if(p->cpu_used > 0.01f)
    {
        printf("   %-18s: " C_VAL "%.1f%%" C_RST "\n", "CPU usage", p->cpu_used);
    }
    if(p->mem_rss_kb > 0)
    {
        printf("   %-18s: " C_VAL "%" PRId64 " KB"
               C_RST "\n", "RSS memory", (int64_t) p->mem_rss_kb);
    }

    /* ---- Timing ---- */
    printf("\n" C_HDR " Timing" C_RST "\n");
    if(p->dtmedian_iter_ns > 0)
    {
        double iter_us = (double) p->dtmedian_iter_ns / 1e3;
        printf("   %-18s: " C_VAL "%.1f µs"
               C_RST "  (%.1f Hz)\n", "Iter (median)", iter_us, p->loop_hz);
    }
    else
    {
        printf("   %-18s: " C_DIM "N/A" C_RST "\n", "Iter (median)");
    }
    if(p->dtmedian_exec_ns > 0)
    {
        double exec_us = (double) p->dtmedian_exec_ns / 1e3;
        double duty = 0.0;
        if(p->dtmedian_iter_ns > 0)
        {
            duty = 100.0 * (double) p->dtmedian_exec_ns / (double) p->dtmedian_iter_ns;
        }
        printf("   %-18s: " C_VAL "%.1f µs"
               C_RST "  (duty: %.1f%%)\n", "Exec (median)", exec_us, duty);
    }
    else
    {
        printf("   %-18s: " C_DIM "N/A" C_RST "\n", "Exec (median)");
    }
    printf("   %-18s: %s\n",
           "Measure timing", p->MeasureTiming ? C_ALIVE "ON" C_RST : C_DIM "OFF" C_RST);

    /* ---- Trigger ---- */
    printf("\n" C_HDR " Trigger" C_RST "\n");
    printf("   %-18s: " C_VAL "%s (%d)"
           C_RST "\n", "Mode", trigmode_str(p->triggermode), p->triggermode);
    if(p->trigstreamname[0] != '\0')
    {
        printf("   %-18s: " C_STREAM "%s" C_RST "\n", "Stream", p->trigstreamname);
    }
    if(p->triggermode == 1)
    {
        printf("   %-18s: " C_VAL "%d" C_RST "\n", "Semaphore", p->triggersem);
    }
    printf("   %-18s: " C_VAL "%d" C_RST
           " (cumul: %" PRIu64 ")\n",
           "Missed frames", p->triggermissed, (uint64_t) p->triggermissed_cumul);

    /* ---- Connections (from graph) ---- */
    printf("\n" C_HDR " Connections" C_RST " (from system graph)\n");

    int pni = p->node_idx;
    if(pni < 0)
    {
        printf("   " C_DIM "(process not in graph)" C_RST "\n");
    }
    else
    {
        int found_any = 0;

        /* Triggered by (stream → proc) */
        for(int e = 0; e < m->nb_edges; e++)
        {
            if(m->edges[e].tgt_node != pni)
            {
                continue;
            }
            if(m->edges[e].type
                    != OV_EDGE_STREAM_TRIGGERS_PROC
                    && m->edges[e].type
                    != OV_EDGE_PROC_TRIGGER_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].src_node;
            if(ni < 0
                    || ni >= m->nb_nodes
                    || m->nodes[ni].type
                    != OV_NODE_STREAM)
            {
                continue;
            }
            int si = m->nodes[ni].index;
            printf("   %-18s: " C_STREAM "%s" C_RST "\n", "Triggered by", m->streams[si].name);
            found_any = 1;
        }

        /* Writes (proc → stream) */
        for(int e = 0; e < m->nb_edges; e++)
        {
            if(m->edges[e].src_node != pni)
            {
                continue;
            }
            if(m->edges[e].type
                    != OV_EDGE_PROC_WRITES_STREAM)
            {
                continue;
            }
            int ni = m->edges[e].tgt_node;
            if(ni < 0
                    || ni >= m->nb_nodes
                    || m->nodes[ni].type
                    != OV_NODE_STREAM)
            {
                continue;
            }
            int si = m->nodes[ni].index;
            printf("   %-18s: " C_STREAM "%s" C_RST "\n", "Writes", m->streams[si].name);
            found_any = 1;
        }

        /* Reads (stream → proc via sem) */
        for(int e = 0; e < m->nb_edges; e++)
        {
            if(m->edges[e].tgt_node != pni)
            {
                continue;
            }
            if(m->edges[e].type
                    != OV_EDGE_STREAM_READ_BY_PROC)
            {
                continue;
            }
            int ni = m->edges[e].src_node;
            if(ni < 0
                    || ni >= m->nb_nodes
                    || m->nodes[ni].type
                    != OV_NODE_STREAM)
            {
                continue;
            }
            int si = m->nodes[ni].index;
            printf("   %-18s: " C_STREAM "%s" C_RST "\n", "Reads (sem)", m->streams[si].name);
            found_any = 1;
        }

        /* Managed by FPS (FPS → proc) */
        for(int e = 0; e < m->nb_edges; e++)
        {
            if(m->edges[e].tgt_node != pni)
            {
                continue;
            }
            if(m->edges[e].type
                    != OV_EDGE_FPS_RUNS_PROC)
            {
                continue;
            }
            int ni = m->edges[e].src_node;
            if(ni < 0
                    || ni >= m->nb_nodes
                    || m->nodes[ni].type
                    != OV_NODE_FPS)
            {
                continue;
            }
            int fi = m->nodes[ni].index;
            printf("   %-18s: " C_FPS "%s" C_RST "\n", "Managed by FPS", m->fps[fi].name);
            found_any = 1;
        }

        if(!found_any)
        {
            printf("   " C_DIM "(no connections found)" C_RST "\n");
        }
    }

    printf("\n");
}

/* =========================================================
 * Help
 * ========================================================= */

static void print_help(
    const char *progname,
    int        mh_color)
{
    milk_help_banner(progname, PI_ONELINE, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%s<process_name>%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", PI_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "--pid PID",
           mh_color ? MH_RST : "", "Look up process by PID instead of name");
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
    printf("  %s$ milk-procinfo-info%s %smyproc%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-procinfo-info%s %s--pid%s %s1234%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] =
    {
        "milk-procinfo-list:list active processinfo instances",
        "milk-procinfo-rm:remove a processinfo instance",
        "milk-stream-info:inspect stream metadata and data"
    };
    milk_help_see_also(see_also, 3, mh_color);
}

/* =========================================================
 * Find proc by name in model
 * ========================================================= */

static int find_proc_by_name(
    const OV_MODEL *m,
    const char     *name)
{
    for(int i = 0; i < m->nb_procs; i++)
    {
        if(m->procs[i].valid
                && strcmp(m->procs[i].name,
                          name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/* =========================================================
 * main
 * ========================================================= */

int main(
    int argc,
    char *argv[])
{
    int action = milk_help_init(argc, argv, PI_ONELINE, PI_DESC_LONG);
    if(action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if(action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    pid_t target_pid = 0;

    static struct option long_opts[] =
    {
        {"help", no_argument, 0, 'h'},
        {"pid",  required_argument, 0, 'p'},
        {0, 0, 0, 0}
    };

    int opt;
    while((opt = getopt_long(
                     argc, argv, "hp:",
                     long_opts, NULL)) != -1)
    {
        switch(opt)
        {
        case 'h':
            break; /* handled above */
        case 'p': target_pid = (pid_t) atoi(optarg);
            break;
        default: printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    const char *proc_name = NULL;
    if(optind < argc)
    {
        proc_name = argv[optind];
    }

    if(target_pid <= 0 && proc_name == NULL)
    {
        printf("\n\033[1;31mERROR\033[0m process name or --pid required\n\n");
        print_help(argv[0], 1);
        return 1;
    }

    /* Build the system model */
    OV_MODEL model;
    memset(&model, 0, sizeof(model));
    ov_model_full_scan(&model);

    /* Find the requested process */
    int pi = -1;
    if(target_pid > 0)
    {
        pi = ov_find_proc_by_pid(&model, target_pid);
    }
    else
    {
        pi = find_proc_by_name(&model, proc_name);
    }

    if(pi < 0)
    {
        if(target_pid > 0)
        {
            PRINT_ERROR("process with PID %d not found", (int) target_pid);
        }
        else
        {
            PRINT_ERROR("process '%s' not found", proc_name);
        }
        ov_scan_cache_cleanup();
        return 1;
    }

    print_proc_info(&model, pi);

    ov_scan_cache_cleanup();
    return 0;
}
