/**
 * @file milk-fps-info.c
 * @brief Print content and connections of an FPS entry
 *
 * Shows FPS header, parameter table, and optionally
 * cross-referenced connections (run process, input/output
 * streams) using the OV_MODEL graph.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <signal.h>

#include "fps.h"
#include "fps_globals.h"
#include "fps_scan.h"
#include "fps_printparameter_valuestring.h"

#ifdef FPS_INFO_CONNECTIONS
#include "overview_data.h"
/* Required by overview_defs.h (extern) */
volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;
#endif

/* =========================================================
 * ANSI helpers for connection display
 * ========================================================= */

#define CI_RST    "\033[0m"
#define CI_HDR    "\033[1;35m"
#define CI_LABEL  "\033[1;34m"
#define CI_STREAM "\033[1;36m"
#define CI_PROC   "\033[1;33m"
#define CI_DIM    "\033[2m"

/* =========================================================
 * Connection display (requires OV_MODEL)
 * ========================================================= */

#ifdef FPS_INFO_CONNECTIONS
static void print_connections(const char *fpsname)
{
    OV_MODEL model;
    memset(&model, 0, sizeof(model));
    ov_model_full_scan(&model);

    /* Find our FPS in the model */
    int fi = -1;
    for (int i = 0; i < model.nb_fps; i++)
    {
        if (model.fps[i].valid
            && strcmp(model.fps[i].name,
                      fpsname) == 0)
        {
            fi = i;
            break;
        }
    }

    if (fi < 0)
    {
        printf("\n" CI_HDR " Connections" CI_RST
               CI_DIM
               " (FPS not found in graph)"
               CI_RST "\n\n");
        ov_scan_cache_cleanup();
        return;
    }

    int fni = model.fps[fi].node_idx;
    if (fni < 0)
    {
        printf("\n" CI_HDR " Connections" CI_RST
               CI_DIM
               " (FPS not in graph)"
               CI_RST "\n\n");
        ov_scan_cache_cleanup();
        return;
    }

    printf("\n" CI_HDR " Connections" CI_RST
           " (from system graph)\n");

    int found_any = 0;

    /* FPS → process (runs) */
    for (int e = 0; e < model.nb_edges; e++)
    {
        if (model.edges[e].src_node != fni)
        {
            continue;
        }
        if (model.edges[e].type
            != OV_EDGE_FPS_RUNS_PROC)
        {
            continue;
        }
        int ni = model.edges[e].tgt_node;
        if (ni < 0
            || ni >= model.nb_nodes
            || model.nodes[ni].type
                != OV_NODE_PROC)
        {
            continue;
        }
        int pi = model.nodes[ni].index;
        printf("   %-18s: " CI_PROC "%s"
               CI_RST " (PID %d)\n",
               "Runs process",
               model.procs[pi].name,
               (int) model.procs[pi].PID);
        found_any = 1;
    }

    /* stream → FPS (input) */
    for (int e = 0; e < model.nb_edges; e++)
    {
        if (model.edges[e].tgt_node != fni)
        {
            continue;
        }
        if (model.edges[e].type
            != OV_EDGE_FPS_INPUT_STREAM)
        {
            continue;
        }
        int ni = model.edges[e].src_node;
        if (ni < 0
            || ni >= model.nb_nodes
            || model.nodes[ni].type
                != OV_NODE_STREAM)
        {
            continue;
        }
        int si = model.nodes[ni].index;
        printf("   %-18s: " CI_STREAM "%s"
               CI_RST "\n",
               "Input stream",
               model.streams[si].name);
        found_any = 1;
    }

    /* FPS → stream (output) */
    for (int e = 0; e < model.nb_edges; e++)
    {
        if (model.edges[e].src_node != fni)
        {
            continue;
        }
        if (model.edges[e].type
            != OV_EDGE_FPS_OUTPUT_STREAM)
        {
            continue;
        }
        int ni = model.edges[e].tgt_node;
        if (ni < 0
            || ni >= model.nb_nodes
            || model.nodes[ni].type
                != OV_NODE_STREAM)
        {
            continue;
        }
        int si = model.nodes[ni].index;
        printf("   %-18s: " CI_STREAM "%s"
               CI_RST "\n",
               "Output stream",
               model.streams[si].name);
        found_any = 1;
    }

    if (!found_any)
    {
        printf("   " CI_DIM
               "(no connections found)"
               CI_RST "\n");
    }

    printf("\n");
    ov_scan_cache_cleanup();
}
#endif /* FPS_INFO_CONNECTIONS */

/* =========================================================
 * Help
 * ========================================================= */

static void print_help(const char *progname)
{
    printf("Usage: %s [options] <fpsname>\n",
           progname);
    printf("Print content of a Function "
           "Parameter Structure (FPS).\n");
    printf("\nOptions:\n");
    printf("  -v, --verbose        "
           "Verbose mode\n");
    printf("  -i, --info           "
           "Show stream info on "
           "separate line\n");
#ifdef FPS_INFO_CONNECTIONS
    printf("  -c, --connections    "
           "Show graph connections\n");
#endif
    printf("  -h, --help           "
           "Show this help message\n");
    printf("  -h1, --help-oneline  "
           "Print one-line description "
           "and exit\n");
}

/* =========================================================
 * main
 * ========================================================= */

int main(int argc, char *argv[])
{
    /* Handle -h1/--help-oneline */
    if (argc >= 2
        && (strcmp(argv[1], "-h1") == 0
            || strcmp(argv[1],
                      "--help-oneline") == 0))
    {
        printf("print content of a Function "
               "Parameter Structure (FPS)\n");
        return 0;
    }

    int verbose = 0;
    int show_info = 0;
    int show_connections = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose",      no_argument, 0, 'v'},
        {"info",         no_argument, 0, 'i'},
        {"connections",  no_argument, 0, 'c'},
        {"help",         no_argument, 0, 'h'},
        {"help-oneline", no_argument, 0, '1'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(
                argc, argv, "vich1",
                long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'v':
            verbose = 1;
            break;
        case 'i':
            show_info = 1;
            break;
        case 'c':
            show_connections = 1;
            break;
        case 'h':
            print_help(argv[0]);
            return 0;
        case '1':
            printf("print content of a "
                   "Function Parameter "
                   "Structure (FPS)\n");
            return 0;
        default:
            print_help(argv[0]);
            return 1;
        }
    }

    if (optind >= argc)
    {
        fprintf(stderr,
                "Error: missing FPS name.\n");
        print_help(argv[0]);
        return 1;
    }

    const char *fpsname = argv[optind];

    FPS fps;
    fps.SMfd = -1;

    if (fps_connect(fpsname, &fps, 0) == -1)
    {
        fprintf(stderr,
                "Error: cannot connect "
                "to FPS '%s'.\n", fpsname);
        return 1;
    }

    function_parameter_print_info(
        &fps, verbose, show_info);

    fps_disconnect(&fps);

#ifdef FPS_INFO_CONNECTIONS
    if (show_connections)
    {
        print_connections(fpsname);
    }
#else
    if (show_connections)
    {
        fprintf(stderr,
                "Warning: --connections not "
                "available in this build\n");
    }
#endif

    return 0;
}
