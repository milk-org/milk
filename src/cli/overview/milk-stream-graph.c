/**
 * @file milk-stream-graph.c
 * @brief Standalone stream dependency graph tool
 *
 * Computes and displays ancestor/descendant lineage
 * for a given shared-memory stream.  Supports trigger,
 * input, and full traversal modes with machine-readable,
 * pretty TrueColor, and interactive output.
 *
 * No CLIcore dependency.  Links: ImageStreamIO +
 * milkprocessinfo + milkfps + m + rt + pthread.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <termios.h>
#include <poll.h>
#include <time.h>

#include "overview_defs.h"
#include "overview_data.h"
#include "stream_graph.h"
#include "milk_help.h"

/* =========================================================
 * Version / description
 * ========================================================= */

#define SG_VERSION "1.0.0"
#define SG_ONELINE \
    "stream dependency graph with loop detection"

#define SG_DESC_LONG \
    "Compute and display ancestor/descendant lineage for a stream.\n" \
    "Supports trigger, input, and full traversal modes with\n" \
    "text, TrueColor ANSI, JSON, and interactive output formats."

/* =========================================================
 * ANSI escape helpers (TrueColor)
 * ========================================================= */

#define SGC_RESET   "\033[0m"
#define SGC_BOLD    "\033[1m"
#define SGC_DIM     "\033[2m"
#define SGC_BLINK   "\033[5m"

/* TrueColor foreground */
#define SGC_FG(r,g,b) \
    "\033[38;2;" #r ";" #g ";" #b "m"

#define SGC_STREAM  SGC_FG(100,200,255)
#define SGC_PROC    SGC_FG(120,220,120)
#define SGC_FPS     SGC_FG(240,200,80)
#define SGC_LOOP    SGC_FG(255,80,80)
#define SGC_DEPTH   SGC_FG(160,160,160)
#define SGC_HEADER  SGC_FG(200,180,255)
#define SGC_TEXT    SGC_FG(200,200,200)
#define SGC_ARROW   SGC_FG(140,140,180)

/* Box-drawing chars */
#define SG_TREE_V   "│"
#define SG_TREE_BR  "├"
#define SG_TREE_END "└"
#define SG_TREE_H   "─"

/* =========================================================
 * Output mode
 * ========================================================= */

typedef enum
{
    OUT_TEXT   = 0,
    OUT_PRETTY = 1,
    OUT_JSON   = 2,
} sg_output_t;

/* =========================================================
 * Signal handler
 * ========================================================= */

static volatile sig_atomic_t sg_quit = 0;

static void sg_sighandler(int sig)
{
    (void) sig;
    sg_quit = 1;
}

/* =========================================================
 * Terminal raw mode (for interactive)
 * ========================================================= */

static struct termios sg_orig_termios;
static int sg_raw_active = 0;

static void sg_raw_enter(void)
{
    if (sg_raw_active)
    {
        return;
    }
    tcgetattr(STDIN_FILENO, &sg_orig_termios);
    struct termios raw = sg_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    /* Hide cursor */
    printf("\033[?25l");
    fflush(stdout);
    sg_raw_active = 1;
}

static void sg_raw_exit(void)
{
    if (!sg_raw_active)
    {
        return;
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH,
              &sg_orig_termios);
    /* Show cursor, reset attrs */
    printf("\033[?25h" SGC_RESET "\n");
    fflush(stdout);
    sg_raw_active = 0;
}

/* =========================================================
 * Usage
 * ========================================================= */

static void print_usage(const char *prog, int mh_color)
{
    milk_help_banner(prog, SG_ONELINE, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<stream>%s\n\n",
           mh_color ? MH_CMD : "", prog, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", SG_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-m, --mode MODE",
           mh_color ? MH_RST : "", "Traversal mode: trigger|input|full (default: trigger)");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-p, --pretty",
           mh_color ? MH_RST : "", "Force TrueColor ANSI output");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-t, --text",
           mh_color ? MH_RST : "", "Plain text machine-readable output");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-j, --json",
           mh_color ? MH_RST : "", "JSON machine-readable output");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-i, --interactive",
           mh_color ? MH_RST : "", "Interactive navigation");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-d DIR",
           mh_color ? MH_RST : "", "Override SHM directory");
    printf("  %s%-25s%s %s (default: %d)\n",
           mh_color ? MH_OPT : "", "--depth N",
           mh_color ? MH_RST : "", "Max traversal depth", SG_MAX_DEPTH);
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
    milk_help_section("Interactive keys", mh_color);
    printf("  UP/DOWN   Navigate list\n");
    printf("  ENTER     Re-root on selected stream\n");
    printf("  r         Rescan graph\n");
    printf("  t/i/f     Switch mode (trigger/input/full)\n");
    printf("  g         Go to stream by name\n");
    printf("  q         Quit\n\n");
    const char *see_also[] = {
        "milk-stream-info:inspect stream metadata and data",
        "milk-stream-list:list active shared memory streams"
    };
    milk_help_see_also(see_also, 2, mh_color);
}

/* =========================================================
 * Parse mode string
 * ========================================================= */

static sg_mode_t parse_mode(const char *s)
{
    if (strcmp(s, "input") == 0)
    {
        return SG_MODE_INPUT;
    }
    if (strcmp(s, "full") == 0)
    {
        return SG_MODE_FULL;
    }
    return SG_MODE_TRIGGER;
}

/* =========================================================
 * Scan + build model
 * ========================================================= */

static void sg_scan_model(OV_MODEL *model)
{
    memset(model, 0, sizeof(*model));
    ov_scan_streams(model);
    ov_scan_fps(model);
    ov_scan_procs(model);
    ov_build_graph(model);
}

/* =========================================================
 * Text output (machine-readable)
 * ========================================================= */

static void sg_print_text(
    const OV_MODEL *m,
    const char     *stream_name,
    sg_mode_t       mode,
    const SG_LINEAGE *lin)
{
    printf("# milk-stream-graph v%s\n", SG_VERSION);
    printf("# stream: %s\n", stream_name);
    printf("# mode: %s\n", sg_mode_label(mode));
    printf("# has_loop: %s\n",
           lin->has_loop ? "yes" : "no");

    for (int i = 0; i < lin->nb_ancestors; i++)
    {
        const SG_LINEAGE_ENTRY *e =
            &lin->ancestors[i];
        const char *sn =
            m->streams[e->stream_idx].name;
        printf("ANCESTOR  depth=%d  stream=%s"
               "  via=%s",
               e->depth, sn, e->via_name);
        if (e->is_loop)
        {
            printf("  LOOP");
        }
        printf("\n");
    }

    for (int i = 0; i < lin->nb_descendants; i++)
    {
        const SG_LINEAGE_ENTRY *e =
            &lin->descendants[i];
        const char *sn =
            m->streams[e->stream_idx].name;
        printf("DESCENDANT  depth=%d  stream=%s"
               "  via=%s",
               e->depth, sn, e->via_name);
        if (e->is_loop)
        {
            printf("  LOOP");
        }
        printf("\n");
    }

    if (lin->has_loop && lin->cycle_len > 0)
    {
        printf("CYCLE ");
        for (int i = 0; i < lin->cycle_len; i++)
        {
            if (i > 0)
            {
                printf(" -> ");
            }
            printf("%s",
                   m->streams[
                       lin->cycle_path[i]].name);
        }
        printf("\n");
    }
}

/* =========================================================
 * JSON output
 * ========================================================= */

static void sg_print_json(
    const OV_MODEL *m,
    const char     *stream_name,
    sg_mode_t       mode,
    const SG_LINEAGE *lin)
{
    printf("{\n");
    printf("  \"stream\": \"%s\",\n", stream_name);
    printf("  \"mode\": \"%s\",\n",
           sg_mode_label(mode));
    printf("  \"has_loop\": %s,\n",
           lin->has_loop ? "true" : "false");

    /* ancestors */
    printf("  \"ancestors\": [");
    for (int i = 0; i < lin->nb_ancestors; i++)
    {
        const SG_LINEAGE_ENTRY *e =
            &lin->ancestors[i];
        if (i > 0)
        {
            printf(",");
        }
        printf("\n    {\"stream\": \"%s\","
               " \"depth\": %d,"
               " \"via\": \"%s\","
               " \"is_loop\": %s}",
               m->streams[e->stream_idx].name,
               e->depth, e->via_name,
               e->is_loop ? "true" : "false");
    }
    printf("\n  ],\n");

    /* descendants */
    printf("  \"descendants\": [");
    for (int i = 0; i < lin->nb_descendants; i++)
    {
        const SG_LINEAGE_ENTRY *e =
            &lin->descendants[i];
        if (i > 0)
        {
            printf(",");
        }
        printf("\n    {\"stream\": \"%s\","
               " \"depth\": %d,"
               " \"via\": \"%s\","
               " \"is_loop\": %s}",
               m->streams[e->stream_idx].name,
               e->depth, e->via_name,
               e->is_loop ? "true" : "false");
    }
    printf("\n  ],\n");

    /* cycle path */
    printf("  \"cycle\": [");
    if (lin->has_loop)
    {
        for (int i = 0;
             i < lin->cycle_len; i++)
        {
            if (i > 0)
            {
                printf(", ");
            }
            printf("\"%s\"",
                   m->streams[
                       lin->cycle_path[i]].name);
        }
    }
    printf("]\n");
    printf("}\n");
}

/* =========================================================
 * Pretty output (TrueColor ANSI)
 * ========================================================= */

static void sg_print_pretty(
    const OV_MODEL *m,
    const char     *stream_name,
    sg_mode_t       mode,
    const SG_LINEAGE *lin)
{
    printf(SGC_BOLD SGC_HEADER
           "Stream Graph" SGC_RESET
           SGC_TEXT "  stream: "
           SGC_BOLD SGC_STREAM "%s" SGC_RESET
           SGC_TEXT "  mode: "
           SGC_FPS "%s" SGC_RESET "\n\n",
           stream_name, sg_mode_label(mode));

    if (lin->nb_ancestors == 0 && lin->nb_descendants == 0)
    {
        printf(SGC_DIM "  No lineage found\n\n" SGC_RESET);
    }
    else
    {
        /* Ancestors */
        for (int i = lin->nb_ancestors - 1; i >= 0; i--)
        {
            const SG_LINEAGE_ENTRY *e = &lin->ancestors[i];
            const char *sn = m->streams[e->stream_idx].name;

            printf(" " SGC_DEPTH "-%-2d" SGC_RESET " " SGC_STREAM "%s" SGC_RESET, e->depth, sn);
            if (e->is_loop)
            {
                printf(" " SGC_BLINK SGC_LOOP "[LOOP]" SGC_RESET);
            }
            printf("\n");

            if (e->via_name[0] != '\0')
            {
                printf("      " SGC_ARROW "│" SGC_RESET "\n");
                printf("      " SGC_ARROW "▼ " SGC_PROC "[%s]" SGC_RESET "\n", e->via_name);
            }
        }

        /* Root */
        printf(" " SGC_DEPTH " 0 " SGC_RESET " " SGC_BOLD SGC_STREAM "%s" SGC_RESET "\n", stream_name);

        /* Descendants */
        for (int i = 0; i < lin->nb_descendants; i++)
        {
            const SG_LINEAGE_ENTRY *e = &lin->descendants[i];
            const char *sn = m->streams[e->stream_idx].name;

            if (e->via_name[0] != '\0')
            {
                printf("      " SGC_ARROW "│" SGC_RESET "\n");
                printf("      " SGC_ARROW "▼ " SGC_PROC "[%s]" SGC_RESET "\n", e->via_name);
            }

            printf(" " SGC_DEPTH "+%-2d" SGC_RESET " " SGC_STREAM "%s" SGC_RESET, e->depth, sn);
            if (e->is_loop)
            {
                printf(" " SGC_BLINK SGC_LOOP "[LOOP]" SGC_RESET);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* Cycle info */
    if (lin->has_loop && lin->cycle_len > 0)
    {
        printf(SGC_BOLD SGC_LOOP
               " Cycle detected:" SGC_RESET " ");
        for (int i = 0;
             i < lin->cycle_len; i++)
        {
            if (i > 0)
            {
                printf(SGC_ARROW " -> "
                       SGC_RESET);
            }
            printf(SGC_STREAM "%s" SGC_RESET,
                   m->streams[
                       lin->cycle_path[i]].name);
        }
        printf("\n\n");
    }
}

/* =========================================================
 * Interactive mode
 * ========================================================= */

static void sg_interactive(
    OV_MODEL  *model,
    const char *initial_stream,
    sg_mode_t   mode)
{
    sg_raw_enter();

    struct sigaction sa;
    sa.sa_handler = sg_sighandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    char current_stream[STRINGMAXLEN_IMAGE_NAME];
    strncpy(current_stream, initial_stream,
            sizeof(current_stream) - 1);
    current_stream[
        sizeof(current_stream) - 1] = '\0';

    int sel = 0;
    int need_scan = 1;
    SG_LINEAGE lin;
    int total_items = 0;

    while (!sg_quit)
    {
        if (need_scan)
        {
            sg_scan_model(model);
            need_scan = 0;
        }

        int si = ov_find_stream_by_name(
                     model, current_stream);

        memset(&lin, 0, sizeof(lin));
        if (si >= 0)
        {
            sg_compute_lineage(
                model, si, mode, &lin);
        }

        total_items = lin.nb_ancestors
                      + lin.nb_descendants + 1;
        if (sel >= total_items && total_items > 0)
        {
            sel = total_items - 1;
        }
        if (sel < 0)
        {
            sel = 0;
        }

        /* Render */
        printf("\033[2J\033[H");
        printf(SGC_BOLD SGC_HEADER
               " milk-stream-graph"
               SGC_RESET SGC_TEXT
               "  stream: " SGC_BOLD SGC_STREAM
               "%s" SGC_RESET
               SGC_TEXT "  mode: " SGC_FPS "%s"
               SGC_RESET "\n",
               current_stream,
               sg_mode_label(mode));
        printf(SGC_DIM
               " q:quit r:rescan t/i/f:mode"
               " ENTER:re-root g:goto"
               SGC_RESET "\n\n");

        if (si < 0)
        {
            printf(SGC_LOOP
                   "  Stream '%s' not found\n"
                   SGC_RESET, current_stream);
        }
        else
        {
            int row = 0;

            for (int i = lin.nb_ancestors - 1; i >= 0; i--)
            {
                const SG_LINEAGE_ENTRY *e = &lin.ancestors[i];
                const char *sn = model->streams[e->stream_idx].name;

                if (row == sel) printf("\033[7m");
                printf(" " SGC_DEPTH "-%-2d" SGC_RESET " " SGC_STREAM "%s" SGC_RESET, e->depth, sn);
                if (e->is_loop) printf(" " SGC_LOOP "[LOOP]" SGC_RESET);
                if (row == sel) printf("\033[27m");
                printf("\n");
                row++;

                if (e->via_name[0] != '\0')
                {
                    printf("      " SGC_ARROW "│" SGC_RESET "\n");
                    printf("      " SGC_ARROW "▼ " SGC_PROC "[%s]" SGC_RESET "\n", e->via_name);
                }
            }

            if (row == sel) printf("\033[7m");
            printf(" " SGC_DEPTH " 0 " SGC_RESET " " SGC_BOLD SGC_STREAM "%s" SGC_RESET, current_stream);
            if (row == sel) printf("\033[27m");
            printf("\n");
            row++;

            for (int i = 0; i < lin.nb_descendants; i++)
            {
                const SG_LINEAGE_ENTRY *e = &lin.descendants[i];
                const char *sn = model->streams[e->stream_idx].name;

                if (e->via_name[0] != '\0')
                {
                    printf("      " SGC_ARROW "│" SGC_RESET "\n");
                    printf("      " SGC_ARROW "▼ " SGC_PROC "[%s]" SGC_RESET "\n", e->via_name);
                }

                if (row == sel) printf("\033[7m");
                printf(" " SGC_DEPTH "+%-2d" SGC_RESET " " SGC_STREAM "%s" SGC_RESET, e->depth, sn);
                if (e->is_loop) printf(" " SGC_LOOP "[LOOP]" SGC_RESET);
                if (row == sel) printf("\033[27m");
                printf("\n");
                row++;
            }

            /* Cycle */
            if (lin.has_loop
                && lin.cycle_len > 0)
            {
                printf("\n" SGC_BOLD SGC_LOOP
                       " Cycle:" SGC_RESET " ");
                for (int c = 0;
                     c < lin.cycle_len; c++)
                {
                    if (c > 0)
                    {
                        printf(SGC_ARROW " -> "
                               SGC_RESET);
                    }
                    printf(SGC_STREAM "%s"
                           SGC_RESET,
                           model->streams[
                               lin.cycle_path[c]]
                               .name);
                }
                printf("\n");
            }
        }
        fflush(stdout);

        /* Wait for input */
        struct pollfd pfd;
        pfd.fd     = STDIN_FILENO;
        pfd.events = POLLIN;

        if (poll(&pfd, 1, 200) <= 0)
        {
            continue;
        }

        char buf[8];
        int n = (int) read(
            STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0)
        {
            continue;
        }

        if (buf[0] == 'q')
        {
            break;
        }
        else if (buf[0] == 'r')
        {
            need_scan = 1;
        }
        else if (buf[0] == 't')
        {
            mode = SG_MODE_TRIGGER;
        }
        else if (buf[0] == 'i')
        {
            mode = SG_MODE_INPUT;
        }
        else if (buf[0] == 'f')
        {
            mode = SG_MODE_FULL;
        }
        else if (buf[0] == '\n'
                 || buf[0] == '\r')
        {
            /* Re-root on selected stream */
            if (total_items > 0)
            {
                int idx = -1;
                if (sel < lin.nb_ancestors)
                {
                    idx = lin.ancestors[lin.nb_ancestors - 1 - sel].stream_idx;
                }
                else if (sel == lin.nb_ancestors)
                {
                    /* selected root */
                }
                else
                {
                    idx = lin.descendants[sel - lin.nb_ancestors - 1].stream_idx;
                }
                
                if (idx >= 0)
                {
                    strncpy(current_stream,
                            model->streams[idx].name,
                            sizeof(current_stream)
                            - 1);
                    sel = 0;
                    need_scan = 1;
                }
            }
        }
        else if (buf[0] == 'g')
        {
            /* Go to stream by name */
            sg_raw_exit();
            printf("Enter stream name: ");
            fflush(stdout);
            char name[STRINGMAXLEN_IMAGE_NAME];
            if (fgets(name, sizeof(name),
                      stdin) != NULL)
            {
                /* Trim newline */
                char *nl = strchr(name, '\n');
                if (nl)
                {
                    *nl = '\0';
                }
                if (name[0] != '\0')
                {
                    strncpy(current_stream, name,
                            sizeof(current_stream)
                            - 1);
                    sel = 0;
                    need_scan = 1;
                }
            }
            sg_raw_enter();
        }
        else if (n >= 3
                 && buf[0] == '\033'
                 && buf[1] == '[')
        {
            /* Arrow keys */
            if (buf[2] == 'A') /* UP */
            {
                if (sel > 0)
                {
                    sel--;
                }
            }
            else if (buf[2] == 'B') /* DOWN */
            {
                if (sel < total_items - 1)
                {
                    sel++;
                }
            }
        }
    } /* main loop */

    sg_raw_exit();
}

/* =========================================================
 * main
 * ========================================================= */

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                SG_ONELINE, SG_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
        return 0;
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_usage(argv[0], mh_color);
        return 0;
    }

    sg_mode_t   mode   = SG_MODE_TRIGGER;
    sg_output_t output = isatty(STDOUT_FILENO) ? OUT_PRETTY : OUT_TEXT;
    int         interactive = 0;
    const char *stream_name = NULL;

    /* Parse arguments */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0
            || strcmp(argv[i], "--help") == 0)
        {
            break; /* handled above */
        }
        if ((strcmp(argv[i], "-m") == 0
             || strcmp(argv[i], "--mode") == 0)
            && i + 1 < argc)
        {
            mode = parse_mode(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-p") == 0
            || strcmp(argv[i], "--pretty") == 0)
        {
            output = OUT_PRETTY;
            continue;
        }
        if (strcmp(argv[i], "-t") == 0
            || strcmp(argv[i], "--text") == 0)
        {
            output = OUT_TEXT;
            continue;
        }
        if (strcmp(argv[i], "-j") == 0
            || strcmp(argv[i], "--json") == 0)
        {
            output = OUT_JSON;
            continue;
        }
        if (strcmp(argv[i], "-i") == 0
            || strcmp(argv[i],
                      "--interactive") == 0)
        {
            interactive = 1;
            continue;
        }
        if (strcmp(argv[i], "-d") == 0
            && i + 1 < argc)
        {
            setenv("MILK_SHM_DIR",
                   argv[++i], 1);
            continue;
        }
        if (strcmp(argv[i], "--depth") == 0
            && i + 1 < argc)
        {
            /* depth is compile-time constant;
             * accepted for compat, ignored */
            i++;
            continue;
        }
        /* positional: stream name */
        if (argv[i][0] != '-')
        {
            stream_name = argv[i];
            continue;
        }
        printf("\n\033[1;31mERROR\033[0m: Invalid option: %s\n\n", argv[i]);
        print_usage(argv[0], mh_color);
        return 1;
    }

    if (stream_name == NULL)
    {
        printf("\n\033[1;31mERROR\033[0m: stream name required.\n\n");
        print_usage(argv[0], mh_color);
        return 1;
    }

    /* Scan system */
    OV_MODEL *model = calloc(1, sizeof(OV_MODEL));
    if (model == NULL)
    {
        PRINT_ERROR("memory allocation failed");
        return 1;
    }

    if (interactive)
    {
        sg_interactive(model, stream_name, mode);
        free(model);
        return 0;
    }

    /* One-shot mode */
    sg_scan_model(model);

    int si = ov_find_stream_by_name(
                 model, stream_name);
    if (si < 0)
    {
        PRINT_ERROR("stream '%s' not found",
                    stream_name);
        free(model);
        return 1;
    }

    SG_LINEAGE lin;
    sg_compute_lineage(model, si, mode, &lin);

    switch (output)
    {
    case OUT_TEXT:
        sg_print_text(
            model, stream_name, mode, &lin);
        break;
    case OUT_PRETTY:
        sg_print_pretty(
            model, stream_name, mode, &lin);
        break;
    case OUT_JSON:
        sg_print_json(
            model, stream_name, mode, &lin);
        break;
    }

    free(model);
    return 0;
}
