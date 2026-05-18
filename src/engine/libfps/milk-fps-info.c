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

#include "milk_help.h"

#define CI_RST    MH_RST
#define CI_HDR    MH_HDR
#define CI_LABEL  MH_ARG
#define CI_STREAM MH_TITLE
#define CI_PROC   MH_NOTE
#define CI_DIM    MH_DIM

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
    for(int ii = 0; ii < model.nb_fps; ii++)
    {
        if(model.fps[ii].valid
                && strcmp(model.fps[ii].name,
                          fpsname) == 0)
        {
            fi = ii;
            break;
        }
    }

    if(fi < 0)
    {
        printf("\n" CI_HDR " Connections" CI_RST
               CI_DIM
               " (FPS not found in graph)"
               CI_RST "\n\n");
        ov_scan_cache_cleanup();
        return;
    }

    int fni = model.fps[fi].node_idx;
    if(fni < 0)
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

    /* FPS -> process (runs) */
    for(int ee = 0; ee < model.nb_edges; ee++)
    {
        if(model.edges[ee].src_node != fni)
        {
            continue;
        }
        if(model.edges[ee].type
                != OV_EDGE_FPS_RUNS_PROC)
        {
            continue;
        }
        int ni = model.edges[ee].tgt_node;
        if(ni < 0
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

    /* stream -> FPS (input) */
    for(int ee = 0; ee < model.nb_edges; ee++)
    {
        if(model.edges[ee].tgt_node != fni)
        {
            continue;
        }
        if(model.edges[ee].type
                != OV_EDGE_FPS_INPUT_STREAM)
        {
            continue;
        }
        int ni = model.edges[ee].src_node;
        if(ni < 0
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

    /* FPS -> stream (output) */
    for(int ee = 0; ee < model.nb_edges; ee++)
    {
        if(model.edges[ee].src_node != fni)
        {
            continue;
        }
        if(model.edges[ee].type
                != OV_EDGE_FPS_OUTPUT_STREAM)
        {
            continue;
        }
        int ni = model.edges[ee].tgt_node;
        if(ni < 0
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

    if(!found_any)
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

#define FI_DESC \
    "print content of a Function Parameter Structure (FPS)"

#define FI_DESC_LONG \
    "Display all parameters and current values for an FPS instance.\n" \
    "Output columns: parameter key, type, current value, flags.\n" \
    "With -v, also shows limits, defaults, and FPS header metadata.\n" \
    "With -c, cross-references stream and process connections\n" \
    "from the live system graph built by milk-CTRL/overview."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, FI_DESC, mh_color);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<fpsname>%s\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FI_DESC_LONG);

    milk_help_section("Arguments", mh_color);
    printf("  %s%-14s%s %s\n\n",
           mh_color ? MH_ARG : "", "<fpsname>",
           mh_color ? MH_RST : "", "Name of the FPS to inspect");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-v, --verbose",
           mh_color ? MH_RST : "", "Verbose (limits, defaults, header)");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-i, --info",
           mh_color ? MH_RST : "", "Show stream info on separate line");
#ifdef FPS_INFO_CONNECTIONS
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-c, --connections",
           mh_color ? MH_RST : "", "Show stream/process graph connections");
#endif
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "Print one-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Print verbose description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");

    milk_help_section("Examples", mh_color);
    printf("  %s$ %smilk-fps-info%s %smyfps00%s\n",
           mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ %smilk-fps-info%s -v %smyfps00%s\n\n",
           mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    const char *see_also[] =
    {
        "milk-fps-list:list active FPS instances",
        "milk-fps-set:set an FPS parameter value",
        "milk-fps-track:monitor FPS parameters",
        "milk-fpsCTRL:launch the FPS dashboard TUI"
    };
    milk_help_see_also(see_also, 4, mh_color);
}

/* =========================================================
 * main
 * ========================================================= */

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                FI_DESC, FI_DESC_LONG);
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

    int verbose = 0;
    int show_info = 0;
    int show_connections = 0;
    int opt;

    static struct option long_options[] =
    {
        {"verbose",      no_argument, 0, 'v'},
        {"info",         no_argument, 0, 'i'},
        {"connections",  no_argument, 0, 'c'},
        {"help",         no_argument, 0, 'h'},
        {"help-oneline", no_argument, 0, '1'},
        {0, 0, 0, 0}
    };

    while((opt = getopt_long(
                     argc, argv, "vich1",
                     long_options, NULL)) != -1)
    {
        switch(opt)
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
        case '1':
            /* Handled above by milk_help_init() */
            break;
        default:
            printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    if(optind >= argc)
    {
        printf("\n\033[1;31mERROR\033[0m missing FPS name.\n\n");
        print_help(argv[0], 1);
        return 1;
    }

    const char *fpsname = argv[optind];

    FPS fps;
    fps.SMfd = -1;

    if(fps_connect(fpsname, &fps, 0) == -1)
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
    if(show_connections)
    {
        print_connections(fpsname);
    }
#else
    if(show_connections)
    {
        fprintf(stderr,
                "Warning: --connections not "
                "available in this build\n");
    }
#endif

    return 0;
}
