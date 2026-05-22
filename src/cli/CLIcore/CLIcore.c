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
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/prctl.h>
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
#include "../libmilkscript/milkscript.h"
#include "treesitter/cli_treesitter.h"

/*-----------------------------------------
*       Globals exported to all modules
*/

DATA __attribute__((used)) data;

pid_t CLIPID;

int C_ERRNO;

int Verbose    = 0;
int Listimfile = 0;


char CLIstartupfilename[STRINGMAXLEN_CLISTARTUPFILENAME] = "CLIstartup.txt";

static int  single_command_flag = 0;
static char single_command_string[STRINGMAXLEN_CLICMDLINE];

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

/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name CLIcore functions */

/// CLI functions

errno_t exitCLI()
{
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
    //    exit(0);
    cli_history_save();
    data.CLIloopON = 0; // stop CLI loop

    return RETURN_SUCCESS;
}

errno_t load_so__cli()
{
    load_sharedobj(data.cmdargtoken[1].val.string);
    return CLICMD_SUCCESS;
}

errno_t load_module__cli()
{
    if (data.cmdargtoken[1].type == 3)
    {
        load_module_shared(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t CLIcore__load_module_as__cli()
{
    DEBUG_TRACEPOINT("calling CLI_checkarg");
    if (0 + CLI_checkarg(1, CLIARG_STR) + CLI_checkarg(2, CLIARG_STR) == 0)
    {
        strncpy(data.moduleshortname, data.cmdargtoken[2].val.string,
                STRINGMAXLEN_MODULE_SHORTNAME - 1);
        load_module_shared(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t set_processinfoON()
{
    dcprocinfo = 1;

    return RETURN_SUCCESS;
}

errno_t set_processinfoOFF()
{
    dcprocinfo = 0;

    return RETURN_SUCCESS;
}

errno_t set_default_precision_single()
{
    dcprecision = 0;

    return RETURN_SUCCESS;
}

errno_t set_default_precision_double()
{
    dcprecision = 1;

    return RETURN_SUCCESS;
}

errno_t milk_usleep__cli()
{
    if (data.cmdargtoken[1].type == 2)
    {
        usleep(data.cmdargtoken[1].val.numl);
        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
    }
}


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


void fnExit_fifoclose()
{
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

errno_t CLI_startup()
{
    DEBUG_TRACE_FSTART();

    if (dcquiet == 1)
    {
        // ImageStreamIO_set_verbosity(0);
    }

    // get PID and write it to shell env variable MILK_CLI_PID
    CLIPID = getpid();
    if (dcquiet == 0)
    {
        printf("        CLI PID = %d\n", (int) CLIPID);

        EXECUTE_SYSTEM_COMMAND_NOCHECK("echo -n \"        \"; cat /proc/%d/status | grep "
                                       "Cpus_allowed_list",
                                       CLIPID);
    }

    //	printf("    _SC_CLK_TCK = %d\n", sysconf(_SC_CLK_TCK));

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

    // to take advantage of kernel priority:
    // owner=root mode=4755

    getresuid(&dcruid, &dceuid, &dcsuid);
    //This sets it to the privileges of the normal user
    if (seteuid(dcruid) != 0)
    {
        PRINT_ERROR("seteuid error");
    }

    // Initialize random-number generator
    // Pure-C xorshift64* (replaces GSL)
    milk_rng_init((uint64_t) time(NULL));

    // warm up
    //for(i=0; i<10; i++)
    //    v1 = gsl_rng_uniform (dcrndgen);

    dcprogstatus = 0;

    // Initialize installdir
    char *installdir_env = getenv("MILK_INSTALLDIR");
    if (installdir_env != NULL)
    {
        strncpy(dcinstalldir, installdir_env, STRINGMAXLEN_DIRNAME - 1);
    }
    else
    {
#ifdef INSTALLDIR
        strncpy(dcinstalldir, INSTALLDIR, STRINGMAXLEN_DIRNAME - 1);
#else
        strncpy(dcinstalldir, "/usr/local/milk", STRINGMAXLEN_DIRNAME - 1);
#endif
    }

    // Initialize sourcedir
    char *sourcedir_env = getenv("MILK_SOURCEDIR");
    if (sourcedir_env != NULL)
    {
        strncpy(dcsourcedir, sourcedir_env, STRINGMAXLEN_DIRNAME - 1);
    }
    else
    {
#ifdef SOURCEDIR
        strncpy(dcsourcedir, SOURCEDIR, STRINGMAXLEN_DIRNAME - 1);
#else
        strncpy(dcsourcedir, "", STRINGMAXLEN_DIRNAME - 1);
#endif
    }


    dcdebug     = 0;
    dcoverwrite = 0;
    dcprecision = 0; // float is default precision
    dcshareddft = 0; // do not allocate shared memory for images
    snprintf(dcsavedir, STRINGMAXLEN_DIRNAME, ".");

    data.CLIlogON             = 0; // log every command
    data.fifoON               = 0;
    data.fifofd               = -1;
    dcprocinfo                = 1; // process info for intensive processes
    dcprocinfoact             = 0; // toggles to 1 when process is logged
    data.autocomplete         = 1; // autocomplete preview ON by default
    data.autocomplete_history = 1; // history suggestions ON
    data.autocomplete_arghint = 1; // argument hint line ON
    data.autocomplete_fuzzy   = 1; // fuzzy matching ON

    // signal handling

    dcsigact.sa_handler = sig_handler;
    sigemptyset(&dcsigact.sa_mask);
    dcsigact.sa_flags = 0;

    // Request the kernel for a sigint if parent dies
    // This is useful if stdin is a pipe from the parent process
    // and the parent dies suddenly. This confuses libreadline.
    prctl(PR_SET_PDEATHSIG, SIGINT);

    dcsigUSR1 = 0;
    dcsigUSR2 = 0;
    dcsigTERM = 0;
    dcsigINT  = 0;
    dcsigBUS  = 0;
    dcsigSEGV = 0;
    dcsigABRT = 0;
    dcsigHUP  = 0;
    dcsigPIPE = 0;

    if (sigaction(SIGUSR1, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGUSR1\n");
    }
    if (sigaction(SIGUSR2, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGUSR2\n");
    }

    set_signal_catch();

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// RegisterCLIcmd duplicate removed


/* Handle SIGWINCH and window size changes when readline is not active and
   reading a character. */
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
    rl_attempted_completion_function = CLI_completion;
    rl_initialize();
    {
        struct sigaction sa_winch;
        sa_winch.sa_handler = sighandler;
        sigemptyset(&sa_winch.sa_mask);
        sa_winch.sa_flags = SA_RESTART;
        sigaction(SIGWINCH, &sa_winch, NULL);
    }
    CLI_setup_hint_area();
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
    char    buf1[1024];

    for (;;)
    {
        bytes = read(data.fifofd, buf0, 1);
        if (bytes > 0)
        {
            buf1[total_bytes] = buf0[0];
            total_bytes += (size_t) bytes;
        }
        else
        {
            if (errno == EWOULDBLOCK)
            {
                break;
            }
            else
            {
                PRINT_ERROR("read: %s", strerror(errno));
                return -1; /* signal error */
            }
        }

        if (buf0[0] == '\n')
        {
            buf1[total_bytes - 1] = '\0';
            strncpy(data.CLIcmdline, buf1, STRINGMAXLEN_CLICMDLINE - 1);

            printf("\033[36m[fifo]\033[0m \u2190 \"%s\"\n", data.CLIcmdline);

            struct timespec ft0, ft1;
            clock_gettime(CLOCK_MONOTONIC, &ft0);

            cli_history_log_prompt(data.CLIcmdline);
            CLI_execute_line();

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

    int            blockCLIinput = 0;
    int            cliwaitus     = 100;
    struct timeval tv; // sleep 100 us after reading FIFO

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

    // Load persistent command aliases
    cli_alias_load();

    // Initialize tree-sitter parser/query
    cli_ts_init();

    // Enable syntax highlighting by default based on terminal capabilities
    // (level 2 = 256-color tree-sitter, level 1 = 16-color legacy)
    data.syntax_highlight = (cli_ts_detect_color_level() >= 2) ? 2 : 1;

    // Disable command timing by default
    data.print_cmd_timing = 0;

    // Load startup script (~/.milkrc)
    cli_milkrc_load();

    // Load persistent command history
    cli_history_load();

    // Initialize structured history log
    cli_history_log_init();

    // Load persistent bookmarks
    cli_bookmark_load();

    // (SHM directory setup is handled by milkscript_init)

    DEBUG_TRACEPOINT("CLI start");

    // initialize fifo to process name
    DEBUG_TRACEPOINT("set default fifo name");
    WRITE_FULLFILENAME(data.fifoname, "%s/.%s.fifo.%07d", dcshmdir, data.processname, getpid());

    DEBUG_TRACEPOINT("Get command-line options");
    command_line_process_options(argc, argv);

    dcprogstatus = 1;
    printf("\n");

    // (Module loading is now handled centrally by milkscript_init)

    // load other libs specified by environment variable MILKCLI_ADD_LIBS
    {
        char *CLI_ADD_LIBS = getenv("MILKCLI_ADD_LIBS");
        if (CLI_ADD_LIBS != NULL)
        {
            if (dcquiet == 0)
            {
                printf("        MILKCLI_ADD_LIBS '%s'\n", CLI_ADD_LIBS);
            }

            char *libname = strtok(CLI_ADD_LIBS, " ,;");
            while (libname != NULL)
            {
                DEBUG_TRACEPOINT("--- CLI Adding library: %s", libname);
                load_module_shared(libname);
                libname = strtok(NULL, " ,;");
            }
            printf("\n");
        }
        else
        {
            if (dcquiet == 0)
            {
                printf("        MILKCLI_ADD_LIBS not set -> no additional module loaded\n");
            }
        }
    }

    DEBUG_TRACEPOINT("Initialize data control block");
    CLI_data_init();

    if (dcdebug > 0)
    {
        printf("DEBUG: %s: start\n", __func__);
    }

    runCLI_cmd_init();

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

    int realine_initialized = 0;

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

        compute_image_memory();
        compute_nb_image();

        // If fifo is on and file CLIstatup.txt exists, load it
        if (initstartup == 0)
        {
            if (single_command_flag)
            {
                strncpy(data.CLIcmdline, single_command_string, STRINGMAXLEN_CLICMDLINE - 1);
                CLI_execute_line();
                data.CLIloopON = 0;
                break;
            }
            else if (data.fifoON == 1)
            {
                EXECUTE_SYSTEM_COMMAND_NOCHECK("file %s",
                                               CLIstartupfilename); //TEST
                EXECUTE_SYSTEM_COMMAND_NOCHECK("cat %s",
                                               CLIstartupfilename); //TEST
                EXECUTE_SYSTEM_COMMAND_NOCHECK("cat %s > %s 2> /dev/null", CLIstartupfilename,
                                               data.fifoname);

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
                    strncpy(data.cmdargtoken[0].val.string, "source",
                            STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                    strncpy(data.cmdargtoken[1].val.string, CLIstartupfilename,
                            STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                    data.cmdNBarg = 2;
                    cli_source();

                    // Exit the interactive loop, as shebang scripts should terminate upon completion
                    if (strcmp(CLIstartupfilename, "CLIstartup.txt") != 0)
                    {
                        data.CLIloopON = 0;
                        break;
                    }
                }
            }
        }
        initstartup = 1;

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

        if (data.fifoON == 0)
        {
            readline_lazy_init(prompt, &realine_initialized);
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

            {
                // CLI loop delay to keep CPU load light
                struct timespec nsts;
                nsts.tv_sec  = 0;
                nsts.tv_nsec = 3000000; // 3 ms delay
                nanosleep(&nsts, NULL);
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
                continue;
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
                readline_lazy_init(prompt, &realine_initialized);
            }

            //printf("fifo cleared, accepting user input through CLI\n");

            if (blockCLIinput == 0)
            {
                // revert to default mode
                if (FD_ISSET(fileno(stdin), &cli_fdin_set))
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
                        CLI_execute_line();
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

#ifdef USE_READLINE
    CLI_cleanup_scroll_region();
    rl_callback_handler_remove();
#endif

    cli_trap_run_exit();

    cli_ts_cleanup();

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/*
 * runCLI_cmd_init() and its static helper callbacks
 * have been extracted to CLIcore_cmd_registry.c.
 * The function prototype is forward-declared above.
 */

/* Body moved to CLIcore_cmd_registry.c */

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


static int command_line_process_options(int argc, char **argv)
{
    int                option_index = 0;
    struct sched_param schedpar;
    char               command[STRINGMAXLEN_COMMAND];

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

        c = getopt_long(argc, argv, "hvic:d:oen:p:fF:s:A", long_options, &option_index);

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


        case 'd':
            printf("debug level : '%s'\n", optarg);
            dcdebug = atoi(optarg);
            printf("Debug = %d\n", dcdebug);
            break;

        case 'n':
            if (dcquiet == 0)
            {
                printf("process name '%s'\n", optarg);
            }
            strncpy(data.processname, optarg, STRINGMAXLEN_PROCESSNAME - 1);
            data.processnameflag = 1; // this process has been named

            // extract first word before '.'
            // it can be used to name processinfo and function parameter structure for process
            char tmpstring[200];
            strncpy(tmpstring, data.processname, STRINGMAXLEN_PROCESSNAME - 1);
            char *firstword;
            firstword = strtok(tmpstring, ".");
            strncpy(data.processname0, firstword, STRINGMAXLEN_PROCESSNAME - 1);
            prctl(PR_SET_NAME, optarg, 0, 0, 0);
            break;

        case 'p':
            schedpar.sched_priority = atoi(optarg);
            printf("RUNNING WITH RT PRIORITY = %d\n", schedpar.sched_priority);

            if (seteuid(dceuid) != 0) //This goes up to maximum privileges
            {
                PRINT_ERROR("seteuid() returns non-zero value");
            }
            sched_setscheduler(0, SCHED_FIFO,
                               &schedpar); //other option is SCHED_RR, might be faster

            if (seteuid(dcruid) != 0) //Go back to normal privileges
            {
                PRINT_ERROR("seteuid() returns non-zero value");
            }
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
            single_command_flag = 1;
            break;

        case 's':
            strncpy(CLIstartupfilename, optarg, STRINGMAXLEN_CLISTARTUPFILENAME - 1);
            if (dcquiet == 0)
            {
                printf("Startup file : %s\n", CLIstartupfilename);
            }
            break;

        case '?':
            /* getopt_long already printed an error message. */
            break;

        default:
            abort();
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
        data.processnameflag = 1;
        prctl(PR_SET_NAME, data.processname, 0, 0, 0);
    }

    return RETURN_SUCCESS;
}
