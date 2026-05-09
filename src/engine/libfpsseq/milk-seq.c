/**
 * @file milk-seq.c
 * @brief Standalone milk-seq sequencer daemon
 *
 * Runs the robust standalone sequencer engine
 * out-of-process. Supports --daemon for POSIX
 * daemonization (double-fork, setsid, PID file,
 * log file redirect).
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

/**
 * @brief Resolve SHM directory for PID/log files
 *
 * Checks $MILK_SHM_DIR, then /milk/shm,
 * then falls back to /tmp.
 */
static const char *get_shm_dir(void)
{
    const char *dir = getenv("MILK_SHM_DIR");
    if (dir && dir[0] != '\0') {
        return dir;
    }
    struct stat st;
    if (stat("/milk/shm", &st) == 0
        && S_ISDIR(st.st_mode))
    {
        return "/milk/shm";
    }
    return "/tmp";
}

/**
 * @brief POSIX double-fork daemonization
 *
 * Detaches from terminal, creates new session,
 * writes PID file, redirects stdout/stderr to
 * log file. Returns 0 in the daemon child,
 * or -1 on error. The original process exits.
 */
static int daemonize(
    const char *pidpath,
    const char *logpath)
{
    /* First fork — parent exits */
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid > 0) {
        _exit(0); /* parent exits */
    }

    /* New session leader */
    if (setsid() < 0) {
        return -1;
    }

    /* Second fork — prevent re-acquiring tty */
    pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid > 0) {
        _exit(0);
    }

    /* Redirect stdout/stderr to log file (or /dev/null on failure) */
    int logfd = open(
        logpath,
        O_WRONLY | O_CREAT | O_APPEND,
        0644);
    if (logfd < 0) {
        /* Fall back to /dev/null to avoid inheriting the terminal */
        logfd = open("/dev/null", O_WRONLY);
        if (logfd < 0) {
            return -1;
        }
    }
    if (dup2(logfd, STDOUT_FILENO) < 0 ||
        dup2(logfd, STDERR_FILENO) < 0)
    {
        close(logfd);
        return -1;
    }
    close(logfd);

    /* Redirect stdin from /dev/null */
    int nullfd = open("/dev/null", O_RDONLY);
    if (nullfd < 0) {
        return -1;
    }
    if (dup2(nullfd, STDIN_FILENO) < 0) {
        close(nullfd);
        return -1;
    }
    close(nullfd);
    /* Write PID file atomically and exclusively */
    {
        int pidfd = open(pidpath, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (pidfd < 0) {
            /* Fail daemonization if PID file cannot be created */
            return -1;
        }
        if (dprintf(pidfd, "%d\n", (int)getpid()) < 0) {
            close(pidfd);
            return -1;
        }
        close(pidfd);
    }

    return 0;
}

static void print_help()
{
    printf("milk-seq - Standalone FPS Sequencer Engine\n\n");
    printf("Usage: milk-seq -n <name> [options]\n\n");
    printf("Options:\n");
    printf("  -n <name>        Sequencer name (required)\n");
    printf("  -f <script.seq>  Sequence file to load on startup\n");
    printf("  --headless       Run silently without TUI (default)\n");
    printf("  --daemon         Daemonize (fork, detach from terminal)\n");
    printf("  --fifo <path>    Custom FIFO path (default: /tmp/milkseq.<name>.fifo)\n");
    printf("  --timeout <sec>  Exit after idle for <sec> seconds\n");
    printf("  -h, --help       Show this brief help\n\n");
    printf("When --daemon is used, PID file and log are written to\n");
    printf("$MILK_SHM_DIR (default /milk/shm):\n");
    printf("  PID: $MILK_SHM_DIR/milkseq.<name>.pid\n");
    printf("  Log: $MILK_SHM_DIR/milkseq.<name>.log\n\n");
    printf("For extensive documentation, run `milk-seq-help`.\n");
}


int main(int argc, char **argv)
{
    char seq_name[FPSSEQ_NAME_MAX] = {0};
    char script_file[FPSSEQ_SCRIPT_PATH_MAX] = {0};
    char custom_fifo[FPSSEQ_FIFO_PATH_MAX] = {0};
    int timeout_sec = 0;
    int do_daemon = 0;
    char pidpath[256] = {0};
    char logpath[256] = {0};

    // Parse arguments
    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-h1") == 0 ||
            strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("FPS sequencer daemon\n");
            return 0;
        } else if (strcmp(argv[i], "-h") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
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
        } else if (strcmp(argv[i], "--daemon") == 0) {
            do_daemon = 1;
            i++;
        } else {
            fprintf(stderr, "Unknown flag: %s\n", argv[i]);
            print_help();
            return 1;
        }
    }

    if (seq_name[0] == '\0') {
        fprintf(stderr,
                "Error: Sequencer name (-n) is "
                "required.\n");
        return 1;
    }

    /* Build PID/log file paths */
    {
        const char *shmdir = get_shm_dir();
        snprintf(pidpath, sizeof(pidpath),
                 "%s/milkseq.%s.pid",
                 shmdir, seq_name);
        snprintf(logpath, sizeof(logpath),
                 "%s/milkseq.%s.log",
                 shmdir, seq_name);
    }

    /* Daemonize if requested */
    if (do_daemon) {
        if (daemonize(pidpath, logpath) < 0) {
            fprintf(stderr,
                    "Failed to daemonize\n");
            return 1;
        }
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
    FPS fps[NB_FPS_MAX];
    memset(fps, 0, sizeof(FPS) * NB_FPS_MAX);
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

    /* Clean up PID file */
    if (do_daemon && pidpath[0] != '\0') {
        unlink(pidpath);
    }

    free(keywnode);
    return 0;
}
