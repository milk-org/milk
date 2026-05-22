/**
 * @file milk-seq.c
 * @brief Standalone milk-seq sequencer daemon
 *
 * Runs the robust standalone sequencer engine
 * out-of-process. Supports --daemon for POSIX
 * daemonization (double-fork, setsid, PID file,
 * log file redirect).
 */

#include <signal.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "fpsseq.h"
#include "fps_scan.h"

#define SEQ_ONELINE "FPS sequencer daemon"
#define SEQ_DESC_LONG                                                \
    "Standalone FPS sequencer engine that manages task scheduling\n" \
    "and execution for milk FPS pipelines.\n"                        \
    "Reads commands from a FIFO, executes tasks in order, and\n"     \
    "optionally daemonizes (double-fork, PID file, log redirect)."

static int keep_running = 1;

/**
 * @brief Handles SIGTERM by terminating the main execution loop.
 */
static void sigterm_handler(int signum)
{
    (void) signum;
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
    if (dir && dir[0] != '\0')
    {
        return dir;
    }
    struct stat st;
    if (stat("/milk/shm", &st) == 0 && S_ISDIR(st.st_mode))
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
static int daemonize(const char *pidpath, const char *logpath)
{
    /* First fork -- parent exits */
    pid_t pid = fork();
    if (pid < 0)
    {
        return -1;
    }
    if (pid > 0)
    {
        _exit(0); /* parent exits */
    }

    /* New session leader */
    if (setsid() < 0)
    {
        return -1;
    }

    /* Second fork -- prevent re-acquiring tty */
    pid = fork();
    if (pid < 0)
    {
        return -1;
    }
    if (pid > 0)
    {
        _exit(0);
    }

    /* Redirect stdout/stderr to log file (or /dev/null on failure) */
    int logfd = open(logpath, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (logfd < 0)
    {
        /* Fall back to /dev/null to avoid inheriting the terminal */
        logfd = open("/dev/null", O_WRONLY);
        if (logfd < 0)
        {
            return -1;
        }
    }
    if (dup2(logfd, STDOUT_FILENO) < 0 || dup2(logfd, STDERR_FILENO) < 0)
    {
        close(logfd);
        return -1;
    }
    close(logfd);

    /* Redirect stdin from /dev/null */
    int nullfd = open("/dev/null", O_RDONLY);
    if (nullfd < 0)
    {
        return -1;
    }
    if (dup2(nullfd, STDIN_FILENO) < 0)
    {
        close(nullfd);
        return -1;
    }
    close(nullfd);
    /* Write PID file atomically and exclusively */
    {
        int pidfd = open(pidpath, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (pidfd < 0)
        {
            /* Fail daemonization if PID file cannot be created */
            return -1;
        }
        if (dprintf(pidfd, "%d\n", (int) getpid()) < 0)
        {
            close(pidfd);
            return -1;
        }
        close(pidfd);
    }

    return 0;
}

/**
 * @brief Print help message for milk-seq.
 */
static void print_help(const char *prog, int mh_color)
{
    milk_help_banner(prog, SEQ_ONELINE, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s-n <name>%s [%soptions%s]\n\n", mh_color ? MH_CMD : "", prog,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", SEQ_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "-n <name>", mh_color ? MH_RST : "",
           "Sequencer name (required)");
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "-f <script.seq>", mh_color ? MH_RST : "",
           "Sequence file to load on startup");
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "--headless", mh_color ? MH_RST : "",
           "Run silently without TUI (default)");
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "--daemon", mh_color ? MH_RST : "",
           "Daemonize (fork, detach from terminal)");
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "--fifo <path>", mh_color ? MH_RST : "",
           "Custom FIFO path (default: /tmp/milkseq.<name>.fifo)");
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "--timeout <sec>", mh_color ? MH_RST : "",
           "Exit after idle for <sec> seconds");
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-28s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-28s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Daemon paths", mh_color);
    printf("  PID: $MILK_SHM_DIR/milkseq.<name>.pid\n");
    printf("  Log: $MILK_SHM_DIR/milkseq.<name>.log\n\n");
}


int main(int argc, char **argv)
{
    int help_action = milk_help_init(argc, argv, SEQ_ONELINE, SEQ_DESC_LONG);
    if (help_action == MH_ACTION_H1 || help_action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (help_action == MH_ACTION_HELP);
    if (help_action == MH_ACTION_HELP || help_action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    char seq_name[FPSSEQ_NAME_MAX]           = { 0 };
    char script_file[FPSSEQ_SCRIPT_PATH_MAX] = { 0 };
    char custom_fifo[FPSSEQ_FIFO_PATH_MAX]   = { 0 };
    int  timeout_sec                         = 0;
    int  do_daemon                           = 0;
    char pidpath[256]                        = { 0 };
    char logpath[256]                        = { 0 };

    // Parse arguments
    for (int i = 1; i < argc;)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            break; /* handled above */
        }
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            strncpy(seq_name, argv[i + 1], FPSSEQ_NAME_MAX - 1);
            i += 2;
        }
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc)
        {
            strncpy(script_file, argv[i + 1], FPSSEQ_SCRIPT_PATH_MAX - 1);
            i += 2;
        }
        else if (strcmp(argv[i], "--fifo") == 0 && i + 1 < argc)
        {
            strncpy(custom_fifo, argv[i + 1], FPSSEQ_FIFO_PATH_MAX - 1);
            i += 2;
        }
        else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc)
        {
            timeout_sec = atoi(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "--headless") == 0)
        {
            // currently implied
            i++;
        }
        else if (strcmp(argv[i], "--daemon") == 0)
        {
            do_daemon = 1;
            i++;
        }
        else
        {
            printf("\n\033[1;31mERROR\033[0m invalid option: %s\n\n", argv[i]);
            print_help(argv[0], 1);
            return 1;
        }
    }

    if (seq_name[0] == '\0')
    {
        printf("\n\033[1;31mERROR\033[0m sequencer name (-n) is required\n\n");
        print_help(argv[0], 1);
        return 1;
    }

    /* Build PID/log file paths */
    {
        const char *shmdir = get_shm_dir();
        snprintf(pidpath, sizeof(pidpath), "%s/milkseq.%s.pid", shmdir, seq_name);
        snprintf(logpath, sizeof(logpath), "%s/milkseq.%s.log", shmdir, seq_name);
    }

    /* Daemonize if requested */
    if (do_daemon)
    {
        if (daemonize(pidpath, logpath) < 0)
        {
            PRINT_ERROR("Failed to daemonize");
            return 1;
        }
    }

    {
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
        if (!state)
        {
            PRINT_ERROR("Failed to create sequencer state '%s'", seq_name);
            return 1;
        }

        if (custom_fifo[0] != '\0')
        {
            strncpy(state->fifo_path, custom_fifo, sizeof(state->fifo_path) - 1);
            // Destroy default fifo and make custom
            char default_fifo[FPSSEQ_FIFO_PATH_MAX];
            snprintf(default_fifo, sizeof(default_fifo), "/tmp/milkseq.%s.fifo", seq_name);
            if (strcmp(default_fifo, custom_fifo) != 0)
            {
                unlink(default_fifo);
            }
            mkfifo(state->fifo_path, 0666);
        }

        int fifo_fd = open(state->fifo_path, O_RDONLY | O_NONBLOCK);
        if (fifo_fd == -1)
        {
            PRINT_ERROR("Failed to open FIFO %s", state->fifo_path);
            milkseq_destroy(seq_name);
            return 1;
        }

        // We need to keep track of the local fps list to feed to the scheduler
        FPS fps[NB_FPS_MAX];
        memset(fps, 0, sizeof(FPS) * NB_FPS_MAX);
        KEYWORD_TREE_NODE   *keywnode   = calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));
        FPSCTRL_PROCESS_VARS fpsCTRLvar = { 0 };

        // Scan initial FPS tree
        int  NBkwn    = 0;
        int  fpsindex = 0;
        long pindex   = 0;
        functionparameter_scan_fps(1, "_ALL", fps, keywnode, &NBkwn, &fpsindex, &pindex, 0);

        // Load initial script if provided
        if (script_file[0] != '\0')
        {
            strncpy(state->script_path, script_file, sizeof(state->script_path) - 1);
            if (milkseq_load_script(state, script_file, fps, keywnode) != 0)
            {
                PRINT_ERROR("Failed to load script: %s", script_file);
            }
        }

        printf("milk-seq started: %s\n", seq_name);
        state->status = MILKSEQ_STATUS_RUNNING;

        long   iter           = 0;
        time_t last_idle_time = time(NULL);

        // Main Run Loop
        while (keep_running)
        {
            // Read FIFO
            int cmds_read = milkseq_fifo_read(state, fifo_fd);

            // Periodically rescan the FPS tree (like fpsCTRL does)
            if (iter % 100 == 0)
            {
                functionparameter_scan_fps(1, "_ALL", fps, keywnode, &NBkwn, &fpsindex, &pindex, 0);
            }

            // Run scheduler step
            int launched = milkseq_scheduler_step(state, fps, keywnode, &fpsCTRLvar);

            if (cmds_read > 0 || launched > 0 || state->NBtasks_active > 0)
            {
                last_idle_time = time(NULL);
            }
            else if (timeout_sec > 0 && (time(NULL) - last_idle_time) > timeout_sec)
            {
                printf("Idle timeout reached. Exiting.\n");
                break;
            }

            if (fpsCTRLvar.exitloop)
            {
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
        if (do_daemon && pidpath[0] != '\0')
        {
            unlink(pidpath);
        }

        free(keywnode);
        return 0;
    }
}
