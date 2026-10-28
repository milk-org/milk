// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore.c
 * @brief Main Command Line Interface (CLI) implementation
 *
 * This file contains the initialization and main loop logic for the milk-cli.
 * It is responsible for setting up the environment, loading persistent state
 * (history, aliases, configurations), initializing core modules, and
 * entering the main Read-Eval-Print Loop (REPL) in `runCLI()`.
 */

/*
 * Exit code
 * 	- 0: no error
 * 	- 1: error (non-specific)
 * 	- 2: error loading libraries
 * 	- 3: missing file required to proceed
 * 	- 4: system call error
 */
#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <malloc.h>
#include <stddef.h> // offsetof()
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <sys/resource.h> // getrlimit
#include <termios.h>
#include <unistd.h>

#include <sys/time.h>

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <ncurses.h>
#include <poll.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sched.h>
#include <signal.h>

#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 1000000
#endif

#ifdef _OPENACC
#    include <openacc.h>
#endif

#ifdef USE_CFITSIO
#    include <fitsio.h>
#endif


#include "CLIcore.h"
#include "CLIcore_script.h"

//#include "initmodules.h"

#include "ImageStreamIO/ImageStreamIO.h"

#include "COREMOD_arith/COREMOD_arith.h"
#ifdef USE_CFITSIO
#    include "COREMOD_iofits/COREMOD_iofits.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

#include "CLIcore_UI_execute.h"
#include "CLIcore_checkargs.h"
#include "CLIcore_datainit.h"
#include "CLIcore_help.h"
#include "CLIcore_memory.h"
#include "CLIcore_modules.h"
#include "CLIcore_setSHMdir.h"
#include "CLIcore_signals.h"
#include "cli_calc_parser.h"
#include "milk_bash_engine.h"
#include "milk_catalog.h"
#include "milk_server.h"
#include "milk_srvlog.h"
#include "milkscript.h"
#include "treesitter/cli_treesitter.h"

#include "milk_path.h"
#include "milk_rt.h"

/*-----------------------------------------
*       Globals exported to all modules
*/

DATA __attribute__((used)) data;

pid_t CLIPID;

int C_ERRNO;

int Verbose    = 0;
int Listimfile = 0;


char CLIstartupfilename[STRINGMAXLEN_CLISTARTUPFILENAME] = ""; // set by -s

static int  single_command_flag = 0;
static char single_command_string[STRINGMAXLEN_CLICMDLINE];

static char server_sockpath[512] = "";
static int  server_owner_fd      = -1;
static int  server_ready_fd      = -1;
static long server_owner_pid     = 0;  /* --owner-pid or --creator: the owner the session records */
static long server_idle_s        = 0;  /* --idle-timeout: end after this long without a command */
static char cli_session[64]      = ""; /* --session NAME: the named session milk-cli's bash joins */
static int  catalog_flag         = 0;  /* --catalog: write the install-time command catalog */
static int  script_tool = 0; /* --format (1), --format-in-place (2), --outline (3) <file> */
static char script_tool_file[1024];

// fifo input
static fd_set cli_fdin_set;

/*-----------------------------------------
*       Forward References
*/
int user_function();
/**
 * @brief atexit handler: performs CLI cleanup on exit.
 */
void fnExit1(void);
/**
 * @brief Initialize CLI command subsystem.
 *
 * Sets up readline, signal handlers, and module
 * loading infrastructure.
 */
void runCLI_cmd_init();
/**
 * @brief Free CLI resources on shutdown.
 *
 * Releases command tables, module handles, and
 * process state.
 */
static void runCLI_free();

static volatile sig_atomic_t sigwinch_received = 0;

static int command_line_process_options(int argc, char **argv);

/// CLI commands
int exitCLI();

/* Forward declarations for FIFO helpers */
void    cli_fifo_close(void);
int     cli_fifo_open(const char *path);
errno_t cli_fifo(void);


static int exitCLI_done = 0;

/**
 * exitCLI - Perform graceful shutdown of the interactive CLI session
 *
 * Cleans up the terminal scroll region, closes and unlinks any active command
 * FIFO, removes temporary image list files, persists the command history to
 * disk, and clears the main REPL loop flag so the process can exit cleanly.
 * Guarded by an internal flag to ensure idempotency if called multiple times.
 *
 * Return: RETURN_SUCCESS
 */
errno_t exitCLI()
{
    if (exitCLI_done)
    {
        return RETURN_SUCCESS;
    }
    exitCLI_done = 1;

    /* Clean up hint area FIRST, before any output
     * that would scroll within the restricted
     * scroll region and desync row positions. */
    CLI_cleanup_scroll_region();

    if (data.fifoON == 1 || data.fifofd >= 0)
    {
        cli_fifo_close();
    }

    if (Listimfile == 1)
    {
        EXECUTE_SYSTEM_COMMAND_NOCHECK("rm imlist.txt");
    }

    if (dcquiet == 0)
    {
        printf("Closing PID %ld (prompt process)\n", (long) getpid());
    }

    cli_history_save();
    data.CLIloopON = 0; // stop CLI loop

    return RETURN_SUCCESS;
}

/**
 * function_parameter_structure_load__cli - CLI wrapper to load an FPS from shared memory
 *
 * Validates that the first CLI command argument is a string naming a Function
 * Parameter Structure (FPS) and loads it into the current session via
 * function_parameter_structure_load().
 *
 * Return: CLICMD_SUCCESS on valid argument, CLICMD_INVALID_ARG otherwise
 */
errno_t function_parameter_structure_load__cli()
{
    DEBUG_TRACEPOINT("calling CLI_checkarg");
    if (CLI_checkarg(1, CLIARG_STR) == 0)
    {
        function_parameter_structure_load(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

/**
 * @brief Open (or create) a command FIFO
 *
 * If path is NULL, auto-generates a default path
 * based on process name and PID.
 *
 * @param path  FIFO path, or NULL for auto
 * @return 0 on success, -1 on error
 */
int cli_fifo_open(const char *path)
{
    /* Close any existing FIFO first */
    if (data.fifoON == 1)
    {
        cli_fifo_close();
    }

    /* Set the path */
    if (path != NULL && path[0] != '\0')
    {
        snprintf(data.fifoname, STRINGMAXLEN_FULLFILENAME, "%s", path);
    }
    else
    {
        WRITE_FULLFILENAME(data.fifoname, "%s/.%s.fifo.%07d", dcshmdir, data.processname, getpid());
    }

    /* Create the FIFO if it doesn't exist */
    struct stat sb;
    if (stat(data.fifoname, &sb) != 0)
    {
        if (mkfifo(data.fifoname, 0666) != 0)
        {
            printf("\033[31mfifo: cannot create"
                   " '%s': %s\033[0m\n",
                   data.fifoname, strerror(errno));
            return -1;
        }
    }

    data.fifofd = open(data.fifoname, O_RDWR | O_NONBLOCK);
    if (data.fifofd == -1)
    {
        PRINT_ERROR("open: %s", strerror(errno));
        printf("File name : %s\n", data.fifoname);
        return -1;
    }

    data.fifoON = 1;

    if (dcquiet == 0)
    {
        printf("\033[36m[fifo]\033[0m "
               "opened: %s (fd=%d)\n",
               data.fifoname, data.fifofd);
    }
    return 0;
}

/**
 * @brief Close the current command FIFO
 */
void cli_fifo_close(void)
{
    if (data.fifofd >= 0)
    {
        close(data.fifofd);
        data.fifofd = -1;
    }
    if (data.fifoON == 1 && data.fifoname[0] != '\0')
    {
        unlink(data.fifoname);
    }
    data.fifoON      = 0;
    data.fifoname[0] = '\0';
}

/**
 * CLI_startup - Initialize the interactive CLI overlay and runtime defaults
 *
 * Records the CLI process ID, prints OpenMP/OpenACC runtime configuration when
 * not in quiet mode, initializes the core runtime subsystem via
 * CLI_runtime_init(), and sets default interactive UI preferences
 * (autocompletion, history suggestions, argument hints, fuzzy matching, syntax
 * diagnostics, auto-indentation, and delimiter matching).
 *
 * Return: RETURN_SUCCESS
 */
errno_t CLI_startup()
{
    DEBUG_TRACE_FSTART();

    // get PID and write it to shell env variable MILK_CLI_PID
    CLIPID = getpid();
    if (dcquiet == 0)
    {
        printf("        CLI PID = %d\n", (int) CLIPID);

        EXECUTE_SYSTEM_COMMAND_NOCHECK("echo -n \"        \"; cat /proc/%d/status | grep "
                                       "Cpus_allowed_list",
                                       CLIPID);
    }

    if (Verbose)
    {
        fprintf(stdout, "%s: compiled %s %s\n", __FILE__, __DATE__, __TIME__);
    }

#ifdef _OPENMP
    if (dcquiet == 0)
    {
        printf("        Running with openMP %d, max threads = %d  "
               "(OMP_NUM_THREADS)\n",
               _OPENMP, omp_get_max_threads());
    }
#else
    if (dcquiet == 0)
    {
        printf("        Compiled without openMP\n");
    }
#endif

#ifdef _OPENACC
    int openACC_devtype = acc_get_device_type();
    if (dcquiet == 0)
    {
        printf("        Running with openACC version %d.  %d device(s), type "
               "%d\n",
               _OPENACC, acc_get_num_devices(openACC_devtype), openACC_devtype);
    }
#endif

    CLI_runtime_init();

    data.autocomplete         = 1; // autocomplete preview ON by default
    data.autocomplete_history = 1; // history suggestions ON
    data.autocomplete_arghint = 1; // argument hint line ON
    data.autocomplete_fuzzy   = 1; // fuzzy matching ON
    data.syntax_diagnostics   = 1; // syntax diagnostics ON
    data.auto_indent          = 4; // auto-indentation spaces (4 spaces)
    data.show_match           = 1; // structural delimiter & keyword matching ON

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * sighandler - Record terminal window size change signals (SIGWINCH)
 * @sig: Signal number received (unused)
 *
 * Sets an atomic flag when SIGWINCH arrives while readline is idle so the main
 * select() loop can query the updated terminal dimensions and resize readline.
 *
 * Return: void
 */
static void sighandler(int sig)
{
    (void) sig;
    sigwinch_received = 1;
}

/**
 * readline_lazy_init - initialize readline on first call.
 * @prompt:  prompt string to install with the callback handler
 * @flag:    pointer to the initialized flag; set to 1 on init
 *
 * Idempotent: does nothing if *flag is already 1.
 * Extracted from runCLI() to eliminate a copy-pasted block.
 */
static void readline_lazy_init(const char *prompt, int *flag)
{
    if (*flag != 0)
    {
        return;
    }
    *flag = 1;
#ifdef USE_READLINE
    DEBUG_TRACEPOINT("initialize readline");
    rl_attempted_completion_function   = CLI_completion;
    rl_completion_display_matches_hook = cli_menu_display_matches;
    rl_getc_function                   = cli_custom_getc;

    /* Isolate readline initialization from external ~/.inputrc */
    {
        char *saved_inputrc = getenv("INPUTRC");
        char *saved_copy    = (saved_inputrc != NULL) ? strdup(saved_inputrc) : NULL;

        setenv("INPUTRC", "/dev/null", 1);
        rl_initialize();

        if (saved_copy != NULL)
        {
            setenv("INPUTRC", saved_copy, 1);
            free(saved_copy);
        }
        else
        {
            unsetenv("INPUTRC");
        }
    }
    {
        struct sigaction sa_winch;
        sa_winch.sa_handler = sighandler;
        sigemptyset(&sa_winch.sa_mask);
        sa_winch.sa_flags = SA_RESTART;
        sigaction(SIGWINCH, &sa_winch, NULL);
    }
    CLI_setup_hint_area();
    cli_set_active_prompt(prompt);
    rl_callback_handler_install(prompt, (rl_vcpfunc_t *) &rl_cb_linehandler);
    CLI_configure_readline();
#else
    (void) prompt;
#endif
}


/**
 * handle_fifo_input - read one line from the FIFO and execute it.
 * @prompt:  CLI prompt string (printed after execution for feedback)
 *
 * Reads bytes one at a time from data.fifofd until a newline is
 * received, then calls CLI_execute_line() on the accumulated buffer.
 * Returns 1 if a line was consumed, 0 if the FIFO fd was not set.
 *
 * Extracted from runCLI() to reduce depth-8 nesting.
 */
static int handle_fifo_input(const char *prompt)
{
    if (!(data.fifoON == 1 && data.fifofd >= 0 && FD_ISSET(data.fifofd, &cli_fdin_set)))
    {
        return 0;
    }

    ssize_t bytes;
    size_t  total_bytes = 0;
    char    buf0[1];
    char    buf1[STRINGMAXLEN_CLICMDLINE];

    for (;;)
    {
        bytes = read(data.fifofd, buf0, 1);
        if (bytes > 0)
        {
            if (total_bytes < sizeof(buf1) - 1)
            {
                buf1[total_bytes++] = buf0[0];
            }
        }
        else if (bytes == 0)
        {
            /* EOF: writer closed the FIFO. Re-open to avoid 100% CPU select() loop */
            close(data.fifofd);
            data.fifofd = open(data.fifoname, O_RDWR | O_NONBLOCK);
            break;
        }
        else
        {
            if (errno == EWOULDBLOCK || errno == EAGAIN)
            {
                break;
            }
            else
            {
                PRINT_ERROR("read: %s", strerror(errno));
                close(data.fifofd);
                data.fifofd = open(data.fifoname, O_RDWR | O_NONBLOCK);
                return -1; /* signal error */
            }
        }

        if (buf0[0] == '\n')
        {
            if (total_bytes > 0 && buf1[total_bytes - 1] == '\n')
            {
                buf1[total_bytes - 1] = '\0';
            }
            else
            {
                buf1[total_bytes] = '\0';
            }

            /* a multi-line construct (here-document, if/for, open quote) waits for the rest */
            static char  *acc     = NULL;
            static size_t acc_len = 0;
            size_t        ll      = strlen(buf1);
            char         *na      = (char *) realloc(acc, acc_len + ll + 2);
            if (na == NULL)
            {
                break;
            }
            acc = na;
            memcpy(acc + acc_len, buf1, ll);
            acc_len += ll;
            acc[acc_len++] = '\n';
            acc[acc_len]   = '\0';
            if (milk_shell_incomplete(acc))
            {
                break;
            }
            acc[--acc_len] = '\0';

            strncpy(data.CLIcmdline, acc, STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

            printf("\033[36m[fifo]\033[0m \u2190 \"%s\"\n", data.CLIcmdline);

            struct timespec ft0, ft1;
            clock_gettime(CLOCK_MONOTONIC, &ft0);

            cli_history_log_prompt(data.CLIcmdline);
            {
                int exit_req = 0;
                int rc       = milk_bash_engine_exec(acc, &exit_req);
                if (exit_req || (dcerrorexit && rc != 0))
                {
                    data.CLIloopON = 0;
                }
            }
            free(acc);
            acc     = NULL;
            acc_len = 0;

            clock_gettime(CLOCK_MONOTONIC, &ft1);
            {
                double fe = (double) (ft1.tv_sec - ft0.tv_sec) +
                            1.0e-9 * (double) (ft1.tv_nsec - ft0.tv_nsec);
                printf("\033[36m[fifo]\033[0m \u2713 (%.3fs)\n", fe);
            }

            printf("%s", prompt);
            fflush(stdout);
            break;
        }
    }
    return 1;
}

/**
 * print_non_tty_prompt - Emit the CLI prompt without readline escape delimiters
 * @prompt: Prompt string containing optional `\001` / `\002` readline markers
 *
 * Used when stdin is a pipe or file (`!isatty(STDIN_FILENO)`) and quiet mode is
 * off so external line-oriented drivers (such as `milk.cliwrap.CLI`) still
 * observe the expected prompt without initializing readline.
 *
 * Return: void
 */
static void print_non_tty_prompt(const char *prompt)
{
    if (prompt == NULL)
    {
        return;
    }
    for (const char *p = prompt; *p != '\0'; p++)
    {
        if (*p != '\001' && *p != '\002')
        {
            putchar((unsigned char) *p);
        }
    }
    fflush(stdout);
}

/** batch_tail_bash() - 1 if @batch leaves the engine in bash mode, with @tail at the start of its last
 * bash-mode run: only there can an unfinished construct hold the batch back (a bare `@calc` line
 * toggles, `exit`/`quit` leave calc mode, as the engine does). */
static int batch_tail_bash(const char *batch, const char **tail)
{
    int mode = cli_calc_mode;
    *tail    = batch;
    for (const char *p = batch; *p != '\0';)
    {
        const char *nl   = strchr(p, '\n');
        const char *next = (nl != NULL) ? nl + 1 : p + strlen(p);
        const char *a    = p;
        const char *b    = (nl != NULL) ? nl : next;
        while (a < b && isspace((unsigned char) *a))
        {
            a++;
        }
        while (b > a && isspace((unsigned char) b[-1]))
        {
            b--;
        }
        size_t n = (size_t) (b - a);
        if (n == 5 && strncmp(a, "@calc", 5) == 0)
        {
            mode = !mode;
            if (!mode)
            {
                *tail = next;
            }
        }
        else if (mode && n == 4 && (strncmp(a, "exit", 4) == 0 || strncmp(a, "quit", 4) == 0))
        {
            mode  = 0;
            *tail = next;
        }
        p = next;
    }
    return !mode;
}

/**
 * handle_non_tty_stdin - Read and batch-execute piped or redirected stdin lines
 * @promptstring: Base prompt string passed to `runCLI()`
 * @prompt:       Active prompt buffer updated after command execution
 *
 * Drains all currently readable bytes from `STDIN_FILENO` in 64 KiB chunks and
 * dispatches complete lines (or the entire remaining buffer at EOF) to
 * `milk_bash_engine_exec()` in a single batch, bypassing per-character readline
 * callbacks and per-line IPC round-trips.
 *
 * Return: void
 */
static void handle_non_tty_stdin(char *promptstring, char *prompt)
{
    static char  *acc     = NULL;
    static size_t acc_len = 0;
    static size_t acc_cap = 0;

    struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };
    char          chunk[65536];
    int           got_eof = 0;

    for (;;)
    {
        ssize_t r = read(STDIN_FILENO, chunk, sizeof(chunk));
        if (r > 0)
        {
            if (acc_len + (size_t) r + 1 > acc_cap)
            {
                size_t ncap = (acc_cap == 0) ? 65536 : acc_cap * 2;
                while (ncap < acc_len + (size_t) r + 1)
                {
                    ncap *= 2;
                }
                char *na = (char *) realloc(acc, ncap);
                if (na == NULL)
                {
                    break;
                }
                acc     = na;
                acc_cap = ncap;
            }
            memcpy(acc + acc_len, chunk, (size_t) r);
            acc_len += (size_t) r;
            acc[acc_len] = '\0';

            /* keep draining while input is ready, up to 1 MiB: a producer faster than this loop
             * (`yes cmd | milk-cli`) must still see its lines run and not grow the buffer forever */
            if (acc_len < (1u << 20) && poll(&pfd, 1, 0) > 0 &&
                (pfd.revents & (POLLIN | POLLHUP | POLLERR)))
            {
                continue;
            }
            break;
        }
        else if (r == 0)
        {
            got_eof = 1;
            break;
        }
        else
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }
    }

    if (acc_len == 0)
    {
        if (got_eof)
        {
            free(acc);
            acc            = NULL;
            acc_cap        = 0;
            data.CLIloopON = 0;
        }
        return;
    }

    size_t consume_len = acc_len;
    size_t batch_len   = acc_len;
    if (!got_eof)
    {
        const char *last_nl = (const char *) memrchr(acc, '\n', acc_len);
        if (last_nl == NULL)
        {
            return;
        }
        batch_len   = (size_t) (last_nl - acc);
        consume_len = batch_len + 1;
    }
    else if (batch_len > 0 && acc[batch_len - 1] == '\n')
    {
        batch_len--;
    }

    char *batch = (char *) malloc(batch_len + 1);
    if (batch == NULL)
    {
        return;
    }
    memcpy(batch, acc, batch_len);
    batch[batch_len] = '\0';

    const char *tail;
    if (!got_eof && batch_tail_bash(batch, &tail) && milk_shell_incomplete(tail))
    {
        free(batch);
        return;
    }

    size_t rem = acc_len - consume_len;
    if (rem > 0)
    {
        memmove(acc, acc + consume_len, rem);
    }
    acc_len      = rem;
    acc[acc_len] = '\0';

    if (dcquiet == 0)
    {
        putchar('\n');
        fflush(stdout);
    }

    if (batch[0] != '\0')
    {
        if (data.echo_input)
        {
            printf("\033[32m[echo]\033[0m \u2190 \"%s\"\n", batch);
        }
        cli_history_log_prompt_lines(batch); /* every line, as the tty path logs each prompt */
        int exit_req = 0;
        int rc       = 0;
        cli_fault_isolation_arm();
        if (sigsetjmp(*cli_get_repl_env(), 1) == 0)
        {
            rc = milk_bash_engine_exec(batch, &exit_req);
        }
        else
        {
            dcsigINT  = 0;
            dcsigSEGV = 0;
            dcsigBUS  = 0;
            dcsigABRT = 0;
            milk_bash_engine_abort();
        }
        cli_fault_isolation_disarm();
        if (exit_req || (dcerrorexit && rc != 0))
        {
            data.CLIloopON = 0;
        }
    }
    free(batch);

    if (got_eof)
    {
        free(acc);
        acc            = NULL;
        acc_len        = 0;
        acc_cap        = 0;
        data.CLIloopON = 0;
    }
    else if (data.CLIloopON == 1 && dcquiet == 0 && data.fifoON == 0)
    {
        runCLI_prompt(promptstring, prompt);
        print_non_tty_prompt(prompt);
    }
}


/**
 * @brief Main entry point for the interactive CLI read-eval-print loop (REPL)
 *
 * This function is the heart of the milk command-line interface. It performs:
 * 1. Environment initialization (signals, paths, shared memory).
 * 2. Loading of user config, aliases, and history.
 * 3. Loading of core and dynamically requested modules.
 * 4. The main REPL: reading input (from stdin, readline, or FIFO),
 *    parsing it, and executing commands via `CLI_execute_line()`.
 *
 * @param argc          Number of command-line arguments passed to the host process
 * @param argv          Array of command-line arguments
 * @param promptstring  String to display at the CLI prompt
 * @return              errno_t Status code (0 on normal exit)
 */

errno_t runCLI(int argc, char *argv[], char *promptstring)
{
    DEBUG_TRACE_FSTART();


    int fdmax;
    int n;

    int initstartup = 0; /// becomes 1 after startup

    int blockCLIinput = 0;
    /* select() timeout of the input loop: input, the bash engine and signals wake it at once, so
     * this only bounds how late a dynamically opened fifo is picked up */
    int            cliwaitus = 50000;
    struct timeval tv;

    strncpy(data.processname, argv[0], STRINGMAXLEN_PROCESSNAME - 1);

    // Set CLI prompt
    char prompt[STRINGMAXLEN_CLIPROMPT];
    runCLI_prompt(promptstring, prompt);

    // Call shared script engine init
    {
        int ms_status = milkscript_init(argc, argv);
        if (ms_status != 0)
        {
            PRINT_ERROR("ERROR: milkscript_init() failed with code %d", ms_status);
            DEBUG_TRACE_FEXIT();
            return ms_status;
        }
    }

    // CLI interactive overlay (signals, autocomplete)
    CLI_startup();

    // Initialize tree-sitter parser/query
    cli_ts_init();

    // Enable syntax highlighting by default based on terminal capabilities
    // (level 2 = 256-color tree-sitter, level 1 = 16-color legacy)
    data.syntax_highlight = (cli_ts_detect_color_level() >= 2) ? 2 : 1;

    // Disable command timing by default
    data.print_cmd_timing = 0;

    // Load persistent command history
    cli_history_load();

    // Initialize structured history log
    cli_history_log_init();

    // (SHM directory setup is handled by milkscript_init)

    DEBUG_TRACEPOINT("CLI start");

    // initialize fifo to process name
    DEBUG_TRACEPOINT("set default fifo name");
    WRITE_FULLFILENAME(data.fifoname, "%s/.%s.fifo.%07d", dcshmdir, data.processname, getpid());

    DEBUG_TRACEPOINT("Get command-line options");
    if (command_line_process_options(argc, argv) != RETURN_SUCCESS)
    {
        DEBUG_TRACE_FEXIT();
        return EXIT_FAILURE;
    }
    if (server_sockpath[0] != '\0')
    {
        // a server's output is a file: fully buffered, its last lines before a crash would be lost
        setvbuf(stdout, NULL, _IOLBF, 0);
    }

    if (script_tool != 0) /* script tools: no session */
    {
        int trc = (script_tool == 3) ? cli_outline_file(script_tool_file)
                                     : cli_format_file(script_tool_file, script_tool == 2, 4);
        DEBUG_TRACE_FEXIT();
        return (trc == 0) ? RETURN_SUCCESS : RETURN_FAILURE;
    }

    dcprogstatus = 1;
    if (!catalog_flag && server_sockpath[0] == '\0')
    {
        printf("\n");
    }

    // (modules, the milk module and the help commands are registered by milkscript_init)

    DEBUG_TRACEPOINT("Initialize data control block");
    CLI_data_init();

    if (dcdebug > 0)
    {
        printf("DEBUG: %s: start\n", __func__);
    }

    runCLI_cmd_init();

    if (catalog_flag)
    {
        int cat_rc = milk_catalog_build(dcinstalldir);
        cli_ts_cleanup();
        DEBUG_TRACE_FEXIT();
        return (cat_rc == 0) ? RETURN_SUCCESS : RETURN_FAILURE;
    }

    if (server_sockpath[0] != '\0' && cli_session[0] != '\0')
    {
        fprintf(stderr, "milk-cli: --session joins a session, --server runs one: not both\n");
        return RETURN_FAILURE;
    }
    if (cli_session[0] != '\0') // its bash joins session NAME, started if it does not run
    {
        char    sock[256];
        char    self[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
        if (n > 0)
        {
            self[n] = '\0';
        }
        int jrc = milk_srvlog_session_join(cli_session, (n > 0) ? self : NULL, sock, sizeof(sock));
        if (jrc != 0)
        {
            exit(jrc);
        }
        milk_bash_engine_set_session(sock);
    }

    if (server_sockpath[0] != '\0')
    {
        int srv_rc = milk_server_run(server_sockpath, server_owner_fd, server_ready_fd,
                                     (pid_t) server_owner_pid,
                                     (server_idle_s > 0) ? (unsigned) server_idle_s : 0U);
        cli_ts_cleanup();
        DEBUG_TRACE_FEXIT();
        return srv_rc;
    }

    // ~/.milkrc: interactive sessions only, once every module and command is registered
    if (!single_command_flag && CLIstartupfilename[0] == '\0' && isatty(fileno(stdin)))
    {
        cli_milkrc_load();
    }

    // fifo
    fdmax = fileno(stdin);
    if (data.fifoON == 1)
    {
        if (cli_fifo_open(data.fifoname) != 0)
        {
            DEBUG_TRACE_FEXIT();
            return EXIT_FAILURE;
        }
        if (data.fifofd > fdmax)
        {
            fdmax = data.fifofd;
        }
    }

    C_ERRNO = 0; // initialize C error variable to 0 (no error)

    data.CLIloopON = 1; // start CLI loop

    int realine_initialized    = 0;
    int non_tty_prompt_printed = 0;
    int stdin_is_tty           = isatty(fileno(stdin));

    while (data.CLIloopON == 1)
    {
        FILE *fp;

        DEBUG_TRACEPOINT("Start CLI loop");

        data.CMDexecuted        = 0;
        data.CLIexecuteCMDready = 0;

        if ((fp = fopen("STOPCLI", "r")) != NULL)
        {
            fprintf(stdout, "STOPCLI FILE FOUND. Exiting...\n");
            fclose(fp);
            exit(3);
        }

        if (Listimfile == 1)
        {
            fp = fopen("imlist.txt", "w");
            list_image_ID_ofp_simple(fp);
            fclose(fp);
        }

        // Keep the number of image addresses available
        //  NB_IMAGES_BUFFER above the number of used images
        //
        //  Keep the number of variables addresses available
        //  NB_VARIABLES_BUFFER above the number of used variables

        if (memory_re_alloc() != RETURN_SUCCESS)
        {
            fprintf(stderr,
                    "%c[%d;%dm ERROR [ FILE: %s   FUNCTION: %s   LINE: "
                    "%d ]  %c[%d;m\n",
                    (char) 27, 1, 31, __FILE__, __func__, __LINE__, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm Memory re-allocation failed  %c[%d;m\n", (char) 27, 1, 31,
                    (char) 27, 0);
            exit(EXIT_FAILURE);
        }

        // Run the -s startup file (fed through the fifo when fifo input is on)
        if (initstartup == 0)
        {
            if (single_command_flag)
            {
                int exit_req   = 0;
                int rc         = milk_bash_engine_exec(single_command_string, &exit_req);
                dcexitcode     = rc;
                data.CLIloopON = 0;
                break;
            }
            else if (CLIstartupfilename[0] == '\0')
            {
                // no startup file
            }
            else if (data.fifoON == 1)
            {
                char qfile[2 * STRINGMAXLEN_CLISTARTUPFILENAME + 8];
                char qfifo[4 * STRINGMAXLEN_FULLFILENAME + 8];
                if (milkpath_shell_quote(qfile, sizeof(qfile), CLIstartupfilename) == 0 &&
                    milkpath_shell_quote(qfifo, sizeof(qfifo), data.fifoname) == 0)
                {
                    EXECUTE_SYSTEM_COMMAND_NOCHECK("cat %s > %s 2> /dev/null", qfile, qfifo);
                }

                if (dcquiet == 0)
                {
                    printf("[%s -> %s]\n", CLIstartupfilename, data.fifoname);
                    printf("IMPORTING FILE %s ... \n", CLIstartupfilename);
                }
            }
            else
            {
                // Native OS execution (typically from shebang -s)
                FILE *fp = fopen(CLIstartupfilename, "r");
                if (fp != NULL)
                {
                    fclose(fp);
                    milk_bash_engine_exec_file(CLIstartupfilename, NULL);
                }
                else
                {
                    fprintf(stderr, "milk-cli: cannot open '%s': %s\n", CLIstartupfilename,
                            strerror(errno));
                    dcexitcode = 1;
                }
                data.CLIloopON = 0; // -s runs the file and exits
                break;
            }
        }
        initstartup = 1;

        if (single_command_flag == 0 && data.fifoON == 0 && stdin_is_tty)
        {
            cli_set_interactive_mode(1);
        }

        DEBUG_TRACEPOINT("Get user input fifo=%d",
                         data.fifoON); //===============================
        tv.tv_sec  = 0;
        tv.tv_usec = cliwaitus;

        FD_ZERO(
            &cli_fdin_set); // Initializes the file descriptor set cli_fdin_set to have zero bits for all file descriptors.
        if (data.fifoON == 1 && data.fifofd >= 0)
        {
            FD_SET(data.fifofd, &cli_fdin_set);
        }

        FD_SET(
            fileno(stdin),
            &cli_fdin_set); // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.
        if (milk_bash_engine_listen_fd() >= 0)
        {
            FD_SET(milk_bash_engine_listen_fd(), &cli_fdin_set); // background jobs calling milk
        }

        if (data.fifoON == 0)
        {
            if (stdin_is_tty)
            {
                readline_lazy_init(prompt, &realine_initialized);
            }
            else if (dcquiet == 0 && !non_tty_prompt_printed)
            {
                runCLI_prompt(promptstring, prompt);
                print_non_tty_prompt(prompt);
                non_tty_prompt_printed = 1;
            }
        }

        DEBUG_TRACEPOINT("loop entry");
        while ((data.CLIexecuteCMDready == 0) && (data.CLIloopON == 1))
        {
            // Special interrupt clause if CLI mode (not FIFO) AND stdin has been closed.
            if (dcsigINT == 1)
            {
                // stop CLI input loop
                data.CLIloopON = 0;
            }
            if (milk_bash_engine_ended())
            {
                fprintf(stderr, "\nmilk-cli: this session was ended (@session end)\n");
                data.CLIloopON = 0;
                continue;
            }

            if (sigwinch_received)
            {
                sigwinch_received = 0;
#ifdef USE_READLINE
                {
                    struct winsize ws;
                    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) >= 0)
                    {
                        rl_set_screen_size(ws.ws_row, ws.ws_col);
                    }
                }
#endif
            }

            /* Recompute fdmax in case fifo
             * was opened dynamically */
            fdmax = fileno(stdin);
            if (data.fifoON == 1 && data.fifofd > fdmax)
            {
                fdmax = data.fifofd;
            }
            if (milk_bash_engine_listen_fd() > fdmax)
            {
                fdmax = milk_bash_engine_listen_fd();
            }
            n = select(fdmax + 1, &cli_fdin_set, NULL, NULL, &tv);

            if (n == 0) // nothing received, need to re-init and go back to select call
            {
                tv.tv_sec  = 0;
                tv.tv_usec = cliwaitus;

                FD_ZERO(&cli_fdin_set);
                if (data.fifoON == 1 && data.fifofd >= 0)
                {
                    FD_SET(data.fifofd, &cli_fdin_set);
                }
                FD_SET(fileno(stdin), &cli_fdin_set);
                if (milk_bash_engine_listen_fd() >= 0)
                {
                    FD_SET(milk_bash_engine_listen_fd(), &cli_fdin_set);
                }
                continue;
            }
            if (n > 0 && milk_bash_engine_listen_fd() >= 0 &&
                FD_ISSET(milk_bash_engine_listen_fd(), &cli_fdin_set))
            {
                milk_bash_engine_serve();
                if (!FD_ISSET(fileno(stdin), &cli_fdin_set) &&
                    !(data.fifoON == 1 && data.fifofd >= 0 && FD_ISSET(data.fifofd, &cli_fdin_set)))
                {
                    FD_ZERO(&cli_fdin_set);
                    if (data.fifoON == 1 && data.fifofd >= 0)
                    {
                        FD_SET(data.fifofd, &cli_fdin_set);
                    }
                    FD_SET(fileno(stdin), &cli_fdin_set);
                    FD_SET(milk_bash_engine_listen_fd(), &cli_fdin_set);
                    continue;
                }
            }
            if (n == -1)
            {
                if (errno == EINTR) // no command received
                {
                    continue;
                }
                else
                {
                    PRINT_ERROR("select: %s", strerror(errno));
                    DEBUG_TRACE_FEXIT();
                    return EXIT_FAILURE;
                }
            }
            DEBUG_TRACEPOINT(" ");

            blockCLIinput = 0;

            DEBUG_TRACEPOINT("fifo ON");
            {
                int fifo_ret = handle_fifo_input(prompt);
                if (fifo_ret == -1)
                {
                    DEBUG_TRACE_FEXIT();
                    return EXIT_FAILURE;
                }
                if (fifo_ret == 1)
                {
                    blockCLIinput = 1;
                }
            }

            if (blockCLIinput == 0) /* fifo cleared */
            {
                DEBUG_TRACEPOINT("fifo cleared");
                if (stdin_is_tty)
                {
                    readline_lazy_init(prompt, &realine_initialized);
                }
            }

            //printf("fifo cleared, accepting user input through CLI\n");

            if (blockCLIinput == 0)
            {
                // revert to default mode
                if (FD_ISSET(fileno(stdin), &cli_fdin_set))
                {
                    if (!stdin_is_tty)
                    {
                        handle_non_tty_stdin(promptstring, prompt);
                        data.CLIexecuteCMDready = 1;
                    }
                    else
                    {
#ifdef USE_READLINE
                        DEBUG_TRACEPOINT("readline callback");
                        rl_callback_read_char();
#else
                        // Fallback: no readline
                        if (fgets(data.CLIcmdline, sizeof(data.CLIcmdline), stdin))
                        {
                            data.CLIcmdline[strcspn(data.CLIcmdline, "\n")] = 0; // strip newline
                            cli_history_log_prompt(data.CLIcmdline);
                            {
                                int exit_req = 0;
                                int rc       = 0;
                                cli_fault_isolation_arm();
                                if (sigsetjmp(*cli_get_repl_env(), 1) == 0)
                                {
                                    rc = milk_bash_engine_exec(data.CLIcmdline, &exit_req);
                                }
                                else
                                {
                                    dcsigINT  = 0;
                                    dcsigSEGV = 0;
                                    dcsigBUS  = 0;
                                    dcsigABRT = 0;
                                    milk_bash_engine_abort();
                                }
                                cli_fault_isolation_disarm();
                                if (exit_req || (dcerrorexit && rc != 0))
                                {
                                    data.CLIloopON = 0;
                                }
                            }
                        }
                        else
                        {
                            data.CLIloopON = 0;
                        }
#endif
                    }
                }
            }
        }
    }

    exitCLI();

#ifdef USE_READLINE
    if (realine_initialized)
    {
        CLI_cleanup_scroll_region();
        rl_callback_handler_remove();
    }
#endif

    cli_set_interactive_mode(0);

    milk_bash_engine_shutdown();

    cli_ts_cleanup();

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * runCLI_free - Release dynamically allocated CLI runtime tables and RNG state
 *
 * Frees the image table, variable table, FPS array, and random number generator
 * state when dynamic data allocation is used.
 *
 * Return: void
 */
static void __attribute__((unused)) runCLI_free()
{
#ifndef DATA_STATIC_ALLOC
    // Free
    DEBUG_TRACEPOINT("free dcimg");
    free(dcimg);

    DEBUG_TRACEPOINT("free dcvar");
    free(dcvar);

    DEBUG_TRACEPOINT("free data.fps");
    if (dcfpsarr == NULL)
    {
        printf("NULL pointer\n");
    }
    else
    {
        free(dcfpsarr);
    }

#endif
    //  free(data.cmd);
    DEBUG_TRACEPOINT("free dcrndgen");
    milk_rng_free();
}

/**
 * user_function - Placeholder user callback that prints a two-dash progress marker
 *
 * Legacy hook retained for compatibility with external callers expecting a
 * default user_function() symbol.
 *
 * Return: Always 0
 */
int user_function()
{
    printf("-");
    fflush(stdout);
    printf("-");
    fflush(stdout);

    return (0);
}

/**
 * @brief atexit handler: performs CLI cleanup on exit.
 */
void fnExit1(void)
{
    //
}


/**
 * command_line_process_options - Parse CLI command-line flags and configure session state
 * @argc: Number of command-line arguments passed to the process
 * @argv: Array of command-line argument strings
 *
 * Processes short and long getopt options (`-h`, `-v`, `-i`, `-o`, `-e`, `-q`,
 * `-Z`, `-A`, `-d`, `-n`, `-p`, `-f`, `-F`, `-c`, `-s`, and long-only server,
 * catalog, and script-tool flags). Sets the process name (defaulting to
 * `<timestamp>p<pid>` when `-n` is omitted) and updates the OS thread title via
 * prctl(PR_SET_NAME).
 *
 * Return: RETURN_SUCCESS on valid options, RETURN_FAILURE on unrecognized option
 */
static int command_line_process_options(int argc, char **argv)
{
    int  option_index = 0;
    char command[STRINGMAXLEN_COMMAND];

    static struct option long_options[] = { /* These options set a flag. */
                                            { "verbose", no_argument, &Verbose, 1 },
                                            { "listimf", no_argument, &Listimfile, 1 },
                                            /* These options don't set a flag.
        We distinguish them by their indices. */
                                            { "help", no_argument, 0, 'h' },
                                            { "version", no_argument, 0, 'v' },
                                            { "info", no_argument, 0, 'i' },
                                            { "overwrite", no_argument, 0, 'o' },
                                            { "errorexit", no_argument, 0, 'e' },
                                            { "idle", no_argument, 0, 'Z' },
                                            { "autocomplete", no_argument, 0, 'A' },
                                            { "no-autocomplete", no_argument, 0, 0x100 },
                                            { "no-history-suggest", no_argument, 0, 0x101 },
                                            { "no-arg-hints", no_argument, 0, 0x102 },
                                            { "no-fuzzy", no_argument, 0, 0x103 },
                                            { "server", required_argument, 0, 0x104 },
                                            { "owner-fd", required_argument, 0, 0x105 },
                                            { "ready-fd", required_argument, 0, 0x106 },
                                            { "catalog", no_argument, 0, 0x107 },
                                            { "format", required_argument, 0, 0x109 },
                                            { "format-in-place", required_argument, 0, 0x10a },
                                            { "outline", required_argument, 0, 0x10b },
                                            { "owner-pid", required_argument, 0, 0x108 },
                                            { "creator", required_argument, 0, 0x10c },
                                            { "idle-timeout", required_argument, 0, 0x10d },
                                            { "session", required_argument, 0, 0x10e },
                                            { "quiet", no_argument, 0, 'q' },
                                            { "fifoflag", no_argument, 0, 'f' },
                                            { "command", required_argument, 0, 'c' },
                                            { "debug", required_argument, 0, 'd' },
                                            { "pname", required_argument, 0, 'n' },
                                            { "priority", required_argument, 0, 'p' },
                                            { "fifoname", required_argument, 0, 'F' },
                                            { "startup", required_argument, 0, 's' },
                                            { 0, 0, 0, 0 }
    };

    data.fifoON          = 0; // default
    data.processnameflag = 0; // default

    while (1)
    {
        int c;

        c = getopt_long(argc, argv, "hvic:d:oen:p:fF:s:Aq", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
        {
            break;
        }

        switch (c)
        {
        case 0:
            /* If this option set a flag, do nothing else now. */
            if (long_options[option_index].flag != 0)
            {
                break;
            }
            printf("option %s", long_options[option_index].name);
            if (optarg)
            {
                printf(" with arg %s", optarg);
            }
            printf("\n");
            break;

        case 'h':
            help();
            exit(EXIT_SUCCESS);
            break;

        case 'v':
            printf("%s   %s\n", dcpkgname, dcpkgver);
            exit(EXIT_SUCCESS);
            break;

        case 'i':
            printInfo();
            exit(EXIT_SUCCESS);
            break;

        case 'o':
            puts("CAUTION - WILL OVERWRITE EXISTING FITS FILES\n");
            dcoverwrite = 1;
            break;

        case 'e':
            if (dcquiet == 0)
            {
                printf("Exit on error ON\n");
            }
            dcerrorexit = 1;
            break;

        case 'q':
            dcquiet = 1;
            break;

        case 'Z':
            printf("Idle mode: only runs process when X is idle (pid "
                   "%ld)\n",
                   (long) getpid());
            snprintf(command, STRINGMAXLEN_COMMAND, "runidle %ld > /dev/null &\n", (long) getpid());
            if (system(command) != 0)
            {
                PRINT_ERROR("system() returns non-zero value");
            }
            break;

        case 'A':
            if (dcquiet == 0)
            {
                printf("Autocomplete preview ON\n");
            }
            data.autocomplete = 1;
            break;

        case 0x100:
            data.autocomplete = 0;
            break;

        case 0x101:
            data.autocomplete_history = 0;
            break;

        case 0x102:
            data.autocomplete_arghint = 0;
            break;

        case 0x103:
            data.autocomplete_fuzzy = 0;
            break;

        case 0x104:
            if (optarg != NULL)
            {
                snprintf(server_sockpath, sizeof(server_sockpath), "%s", optarg);
                dcquiet = 1;
            }
            break;

        case 0x105:
            if (optarg != NULL)
            {
                server_owner_fd = atoi(optarg);
                fcntl(server_owner_fd, F_SETFD, FD_CLOEXEC); // not for what the server runs
            }
            break;

        case 0x108: // exit when that process exits (pidfd polled like the owner pipe)
            if (optarg != NULL)
            {
                server_owner_pid = atol(optarg);
                int pfd          = (int) syscall(SYS_pidfd_open, (pid_t) atol(optarg), 0);
                if (pfd < 0 && errno == ESRCH)
                {
                    exit(0);
                }
                if (pfd >= 0)
                {
                    fcntl(pfd, F_SETFD, FD_CLOEXEC);
                    server_owner_fd = pfd;
                }
            }
            break;

        case 0x10c: // the shell that created a named session: recorded, not tied
            server_owner_pid = (optarg != NULL) ? atol(optarg) : 0;
            break;

        case 0x10d:
            server_idle_s = (optarg != NULL) ? atol(optarg) : 0;
            break;

        case 0x10e:
            snprintf(cli_session, sizeof(cli_session), "%s", (optarg != NULL) ? optarg : "");
            break;

        case 0x106:
            if (optarg != NULL)
            {
                server_ready_fd = atoi(optarg);
                fcntl(server_ready_fd, F_SETFD, FD_CLOEXEC);
            }
            break;

        case 0x107:
            catalog_flag = 1;
            dcquiet      = 1;
            break;

        case 0x109:
        case 0x10a:
        case 0x10b:
            script_tool = c - 0x108;
            snprintf(script_tool_file, sizeof(script_tool_file), "%s", optarg);
            break;


        case 'd':
            printf("debug level : '%s'\n", optarg);
            dcdebug = atoi(optarg);
            printf("Debug = %d\n", dcdebug);
            break;

        case 'n':
            if (optarg != NULL && optarg[0] != '\0')
            {
                if (dcquiet == 0)
                {
                    printf("process name '%s'\n", optarg);
                }
                strncpy(data.processname, optarg, STRINGMAXLEN_PROCESSNAME - 1);
                data.processname[STRINGMAXLEN_PROCESSNAME - 1] = '\0';
                data.processnameflag                           = 1; // this process has been named

                // extract first word before '.'
                // it can be used to name processinfo and function parameter structure for process
                char tmpstring[STRINGMAXLEN_PROCESSNAME];
                strncpy(tmpstring, data.processname, STRINGMAXLEN_PROCESSNAME - 1);
                tmpstring[STRINGMAXLEN_PROCESSNAME - 1] = '\0';
                char *firstword                         = strtok(tmpstring, ".");
                if (firstword == NULL || firstword[0] == '\0')
                {
                    firstword = data.processname;
                }
                strncpy(data.processname0, firstword, STRINGMAXLEN_PROCESSNAME - 1);
                data.processname0[STRINGMAXLEN_PROCESSNAME - 1] = '\0';
                prctl(PR_SET_NAME, optarg, 0, 0, 0);
            }
            break;

        case 'p':
            milkrt_RTPrio(atoi(optarg));
            break;

        case 'f':
            if (dcquiet == 0)
            {
                printf("fifo input ON\n");
            }
            data.fifoON = 1;
            break;

        case 'F':
            printf("using input fifo '%s'\n", optarg);
            data.fifoON = 1;
            snprintf(data.fifoname, STRINGMAXLEN_FULLFILENAME, "%s", optarg);
            printf("FIFO NAME = %s\n", data.fifoname);
            break;

        case 'c':
            strncpy(single_command_string, optarg, STRINGMAXLEN_CLICMDLINE - 1);
            single_command_string[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
            single_command_flag                                = 1;
            break;

        case 's':
            strncpy(CLIstartupfilename, optarg, STRINGMAXLEN_CLISTARTUPFILENAME - 1);
            CLIstartupfilename[STRINGMAXLEN_CLISTARTUPFILENAME - 1] = '\0';
            if (dcquiet == 0)
            {
                printf("Startup file : %s\n", CLIstartupfilename);
            }
            break;

        case '?':
            /* getopt_long already printed an error message. */
            return RETURN_FAILURE;

        default:
            fprintf(stderr, "Unrecognized or invalid option.\n");
            return RETURN_FAILURE;
        }
    }

    /* If no -n option was given, build a default session name:
     * "<unix_timestamp>p<pid>"  e.g. "1740801234p12345"
     */
    if (data.processnameflag == 0)
    {
        time_t ts   = time(NULL);
        int    slen = snprintf(data.processname, STRINGMAXLEN_PROCESSNAME, "%ldp%ld", (long) ts,
                               (long) CLIPID);
        if (slen < 1 || slen >= STRINGMAXLEN_PROCESSNAME)
        {
            PRINT_ERROR("snprintf error building default processname");
        }
        strncpy(data.processname0, data.processname, STRINGMAXLEN_PROCESSNAME - 1);
        data.processname0[STRINGMAXLEN_PROCESSNAME - 1] = '\0';
        data.processnameflag                            = 1;
        prctl(PR_SET_NAME, data.processname, 0, 0, 0);
    }

    return RETURN_SUCCESS;
}
