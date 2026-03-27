/**
 * @file milk-seq.c
 * @brief Standalone milk-seq sequencer daemon
 *
 * Runs the robust standalone sequencer engine out-of-process.
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include "fpsseq.h"
#include "fps.h"
#include "fps_scan.h"

static int keep_running = 1;

static void sigterm_handler(int signum)
{
    (void)signum;
    keep_running = 0;
}

static void print_help()
{
    printf("milk-seq - Standalone FPS Sequencer Engine\n\n");
    printf("Usage: milk-seq -n <name> [options]\n\n");
    printf("Options:\n");
    printf("  -n <name>        Sequencer name (required)\n");
    printf("  -f <script.seq>  Sequence file to load on startup\n");
    printf("  --headless       Run silently without TUI (default)\n");
    printf("  --fifo <path>    Custom FIFO path (default: /tmp/milkseq.<name>.fifo)\n");
    printf("  --timeout <sec>  Exit after idle for <sec> seconds\n");
    printf("  -h, --help       Show this brief help\n\n");
    printf("For extensive documentation, including scripting syntax and\n");
    printf("architecture overview, run `milk-seq-help`.\n");
}


int main(int argc, char **argv)
{
    char seq_name[FPSSEQ_NAME_MAX] = {0};
    char script_file[FPSSEQ_SCRIPT_PATH_MAX] = {0};
    char custom_fifo[FPSSEQ_FIFO_PATH_MAX] = {0};
    int timeout_sec = 0;

    // Parse arguments
    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            strncpy(seq_name, argv[i+1], FPSSEQ_NAME_MAX - 1);
            i += 2;
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            strncpy(script_file, argv[i+1], FPSSEQ_SCRIPT_PATH_MAX - 1);
            i += 2;
        } else if (strcmp(argv[i], "--fifo") == 0 && i + 1 < argc) {
            strncpy(custom_fifo, argv[i+1], FPSSEQ_FIFO_PATH_MAX - 1);
            i += 2;
        } else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
            timeout_sec = atoi(argv[i+1]);
            i += 2;
        } else if (strcmp(argv[i], "--headless") == 0) {
            // currently implied
            i++;
        } else {
            fprintf(stderr, "Unknown flag: %s\n", argv[i]);
            print_help();
            return 1;
        }
    }

    if (seq_name[0] == '\0') {
        fprintf(stderr, "Error: Sequencer name (-n) is required.\n");
        return 1;
    }

    // Set up signals
    struct sigaction action;
    memset(&action, 0, sizeof(struct sigaction));
    action.sa_handler = sigterm_handler;
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);

    // Initialize milk process info (required for fps library tracking)
    // we name ourselves milk-seq.<name>
    char procname[64];
    snprintf(procname, sizeof(procname), "milk-seq.%s", seq_name);
    processinfo_setup(procname, "milk-seq", "FPS Sequencer", "main", __FILE__, __LINE__);

    // Create sequencer state
    MILKSEQ_STATE *state = milkseq_create(seq_name);
    if (!state) {
        fprintf(stderr, "Failed to create sequencer state '%s'\n", seq_name);
        return 1;
    }

    if (custom_fifo[0] != '\0') {
        strncpy(state->fifo_path, custom_fifo, sizeof(state->fifo_path) - 1);
        // Destroy default fifo and make custom
        char default_fifo[FPSSEQ_FIFO_PATH_MAX];
        snprintf(default_fifo, sizeof(default_fifo), "/tmp/milkseq.%s.fifo", seq_name);
        if (strcmp(default_fifo, custom_fifo) != 0) {
            unlink(default_fifo);
        }
        mkfifo(state->fifo_path, 0666);
    }

    int fifo_fd = open(state->fifo_path, O_RDONLY | O_NONBLOCK);
    if (fifo_fd == -1) {
        fprintf(stderr, "Failed to open FIFO %s\n", state->fifo_path);
        milkseq_destroy(seq_name);
        return 1;
    }

    // We need to keep track of the local fps list to feed to the scheduler
    FUNCTION_PARAMETER_STRUCT fps[NB_FPS_MAX];
    memset(fps, 0, sizeof(FUNCTION_PARAMETER_STRUCT) * NB_FPS_MAX);
    KEYWORD_TREE_NODE *keywnode = calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));
    FPSCTRL_PROCESS_VARS fpsCTRLvar = {0};

    // Scan initial FPS tree
    int NBkwn = 0;
    int fpsindex = 0;
    long pindex = 0;
    functionparameter_scan_fps(1, "_ALL", fps, keywnode, &NBkwn, &fpsindex, &pindex, 0);

    // Load initial script if provided
    if (script_file[0] != '\0') {
        strncpy(state->script_path, script_file, sizeof(state->script_path) - 1);
        if (milkseq_load_script(state, script_file, fps, keywnode) != 0) {
            fprintf(stderr, "Failed to load script: %s\n", script_file);
        }
    }

    printf("milk-seq started: %s\n", seq_name);
    state->status = MILKSEQ_STATUS_RUNNING;

    long iter = 0;
    time_t last_idle_time = time(NULL);

    // Main Run Loop
    while (keep_running) {
        // Read FIFO
        int cmds_read = milkseq_fifo_read(state, fifo_fd);

        // Periodically rescan the FPS tree (like fpsCTRL does)
        if (iter % 100 == 0) {
            functionparameter_scan_fps(1, "_ALL", fps, keywnode, &NBkwn, &fpsindex, &pindex, 0);
        }

        // Run scheduler step
        int launched = milkseq_scheduler_step(state, fps, keywnode, &fpsCTRLvar);

        if (cmds_read > 0 || launched > 0 || state->NBtasks_active > 0) {
            last_idle_time = time(NULL);
        } else if (timeout_sec > 0 && (time(NULL) - last_idle_time) > timeout_sec) {
            printf("Idle timeout reached. Exiting.\n");
            break;
        }

        if (fpsCTRLvar.exitloop) {
            printf("Exit command received. Shutting down.\n");
            break;
        }

        usleep(10000); // 10ms tick
        iter++;
    }

    state->status = MILKSEQ_STATUS_STOPPING;
    printf("Shutting down milk-seq...\n");

    close(fifo_fd);
    milkseq_destroy(seq_name);

    free(keywnode);
    return 0;
}
