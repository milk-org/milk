/**
 * @file CLIcore.c
 * @brief main C file
 *
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
#define _GNU_SOURCE
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
#include <readline/history.h>
#include <readline/readline.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
#endif

#ifdef _OPENACC
#include <openacc.h>
#endif

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif


#include "CLIcore.h"
#include "CLIcore_script.h"
#include "streamCTRL/streamCTRL_TUI.h"

//#include "initmodules.h"

#include "ImageStreamIO/ImageStreamIO.h"

#include "COREMOD_arith/COREMOD_arith.h"
#ifdef USE_CFITSIO
#include "COREMOD_iofits/COREMOD_iofits.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

#include "CLIcore_UI.h"
#include "CLIcore_checkargs.h"
#include "CLIcore_datainit.h"
#include "CLIcore_help.h"
#include "CLIcore_memory.h"
#include "CLIcore_modules.h"
#include "CLIcore_setSHMdir.h"
#include "CLIcore_signals.h"

/*-----------------------------------------
*       Globals exported to all modules
*/

DATA __attribute__((used)) data;

pid_t CLIPID;

int C_ERRNO;

int Verbose    = 0;
int Listimfile = 0;


char CLIstartupfilename[STRINGMAXLEN_CLISTARTUPFILENAME] = "CLIstartup.txt";

// fifo input
static int    fifofd;
static fd_set cli_fdin_set;

/*-----------------------------------------
*       Forward References
*/
int         user_function();
void        fnExit1(void);
void        runCLI_cmd_init();
static void runCLI_free();

static volatile sig_atomic_t sigwinch_received = 0;

static int command_line_process_options(int argc, char **argv);

/// CLI commands
static int exitCLI();

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

    if(data.fifoON == 1)
    {
        EXECUTE_SYSTEM_COMMAND("rm %s", data.fifoname);
    }

    if(Listimfile == 1)
    {
        EXECUTE_SYSTEM_COMMAND("rm imlist.txt");
    }

    if(dcquiet == 0)
    {
        printf("Closing PID %ld (prompt process)\n", (long) getpid());
    }
    //    exit(0);
    cli_history_save();
    data.CLIloopON = 0; // stop CLI loop

    return RETURN_SUCCESS;
}

static errno_t load_so__cli()
{
    load_sharedobj(data.cmdargtoken[1].val.string);
    return CLICMD_SUCCESS;
}

static errno_t load_module__cli()
{

    if(data.cmdargtoken[1].type == 3)
    {
        load_module_shared(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t CLIcore__load_module_as__cli()
{
    DEBUG_TRACEPOINT("calling CLI_checkarg");
    if(0 + CLI_checkarg(1, CLIARG_STR) + CLI_checkarg(2, CLIARG_STR) == 0)
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
    if(data.cmdargtoken[1].type == 2)
    {
        usleep(data.cmdargtoken[1].val.numl);
        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
    }
}

#ifdef USE_NCURSES
errno_t functionparameter_CTRLscreen__cli()
{
    DEBUG_TRACEPOINT("calling CLI_checkarg");
    if((CLI_checkarg(1, CLIARG_INT64) == 0) &&
            (CLI_checkarg(2, CLIARG_STR) == 0) &&
            (CLI_checkarg(3, CLIARG_STR) == 0))
    {
        functionparameter_CTRLscreen((uint32_t) data.cmdargtoken[1].val.numl,
                                     data.cmdargtoken[2].val.string,
                                     data.cmdargtoken[3].val.string,
                                     0.0);
        return RETURN_SUCCESS;
    }
    else
    {
        printf("Wrong args (%d)\n", data.cmdargtoken[1].type);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}
#endif

errno_t function_parameter_structure_load__cli()
{
    DEBUG_TRACEPOINT("calling CLI_checkarg");
    if(CLI_checkarg(1, CLIARG_STR) == 0)
    {
        function_parameter_structure_load(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

#ifdef USE_NCURSES
errno_t processinfo_CTRLscreen__cli()
{
    return (processinfo_CTRLscreen());
}

errno_t streamCTRL_CTRLscreen__cli()
{
    return (streamCTRL_CTRLscreen());
}
#endif

void fnExit_fifoclose()
{
    //	printf("Running atexit function fnExit_fifoclose\n");
    //	if ( data.fifoON == 1)
    //	{
    //		if (fifofd != -1) {
    //			close(fifofd);
    //		}
    //	}

    //	FD_ZERO(&cli_fdin_set);  // Initializes the file descriptor set cli_fdin_set to have zero bits for all file descriptors.
    //       if(data.fifoON==1)
    //           FD_SET(fifofd, &cli_fdin_set);  // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.
    //    FD_SET(fileno(stdin), &cli_fdin_set);  // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.

    // reset terminal properties
    //	system("tset");
}

errno_t CLI_startup()
{
    DEBUG_TRACE_FSTART();

    if(dcquiet == 1)
    {
        // ImageStreamIO_set_verbosity(0);
    }

    // get PID and write it to shell env variable MILK_CLI_PID
    CLIPID = getpid();
    if(dcquiet == 0)
    {
        printf("        CLI PID = %d\n", (int) CLIPID);

        EXECUTE_SYSTEM_COMMAND(
            "echo -n \"        \"; cat /proc/%d/status | grep "
            "Cpus_allowed_list",
            CLIPID);
    }

    //	printf("    _SC_CLK_TCK = %d\n", sysconf(_SC_CLK_TCK));

    if(Verbose)
    {
        fprintf(stdout, "%s: compiled %s %s\n", __FILE__, __DATE__, __TIME__);
    }

#ifdef _OPENMP
    if(dcquiet == 0)
    {
        printf(
            "        Running with openMP %d, max threads = %d  "
            "(OMP_NUM_THREADS)\n",
            _OPENMP,
            omp_get_max_threads());
    }
#else
    if(dcquiet == 0)
    {
        printf("        Compiled without openMP\n");
    }
#endif

#ifdef _OPENACC
    int openACC_devtype = acc_get_device_type();
    if(dcquiet == 0)
    {
        printf(
            "        Running with openACC version %d.  %d device(s), type "
            "%d\n",
            _OPENACC,
            acc_get_num_devices(openACC_devtype),
            openACC_devtype);
    }
#endif

    // to take advantage of kernel priority:
    // owner=root mode=4755

    getresuid(&dcruid, &dceuid, &dcsuid);
    //This sets it to the privileges of the normal user
    if(seteuid(dcruid) != 0)
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
    if(installdir_env != NULL)
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
    if(sourcedir_env != NULL)
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


    dcdebug         = 0;
    dcoverwrite     = 0;
    dcprecision     = 0;  // float is default precision
    dcshareddft    = 0;  // do not allocate shared memory for images
    snprintf(dcsavedir, STRINGMAXLEN_DIRNAME, ".");

    data.CLIlogON          = 0; // log every command
    data.fifoON            = 0;
    dcprocinfo       = 1; // process info for intensive processes
    dcprocinfoact = 0; // toggles to 1 when process is logged
    data.autocomplete      = 1; // autocomplete preview ON by default
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

    if(sigaction(SIGUSR1, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGUSR1\n");
    }
    if(sigaction(SIGUSR2, &dcsigact, NULL) == -1)
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
 * @brief Command Line Interface (CLI) main\n
 *
 * Uses readline to read user input\n
 * parsing done with bison and flex
 */

errno_t runCLI(int argc, char *argv[], char *promptstring)
{
    DEBUG_TRACE_FSTART();

    int fdmax;
    int n;

    ssize_t bytes;
    size_t  total_bytes;
    char    buf0[1];
    char    buf1[1024];

    int initstartup = 0; /// becomes 1 after startup

    int            blockCLIinput = 0;
    int            cliwaitus     = 100;
    struct timeval tv; // sleep 100 us after reading FIFO


    strncpy(data.processname, argv[0], STRINGMAXLEN_PROCESSNAME - 1);

    // Set CLI prompt
    char prompt[STRINGMAXLEN_CLIPROMPT];
    runCLI_prompt(promptstring, prompt);

    // CLI initialize
    CLI_startup();

    // Load persistent command aliases
    cli_alias_load();

    // Enable syntax highlighting by default
    data.syntax_highlight = 1;
    
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

    // set shared memory directory
    setSHMdir();

    DEBUG_TRACEPOINT("CLI start");

    // initialize fifo to process name
    DEBUG_TRACEPOINT("set default fifo name");
    WRITE_FULLFILENAME(data.fifoname,
                       "%s/.%s.fifo.%07d",
                       dcshmdir,
                       data.processname,
                       getpid());

    DEBUG_TRACEPOINT("Get command-line options");
    command_line_process_options(argc, argv);

    dcprogstatus = 1;
    printf("\n");

    // Pre-load milkfpsCLI so its constructor registers
    // fps_generic_CLIfunction_ptr and fps_fill_farg_examples_ptr
    // before any V2 module commands are used.
    {
        load_sharedobj("libmilkfpsCLI.so");
    }

    // Explicitly reference core module constructors to ensure linker doesn't drop them
    // (these are single-run safe due to their internal INITSTATUS mechanism)
    extern void libinit_COREMOD_memory(void);
#ifdef USE_CFITSIO
    extern void libinit_COREMOD_iofits(void);
#endif
    extern void libinit_COREMOD_arith(void);
    extern void libinit_COREMOD_tools(void);
    libinit_COREMOD_memory();
#ifdef USE_CFITSIO
    libinit_COREMOD_iofits();
#endif
    libinit_COREMOD_arith();
    libinit_COREMOD_tools();

    // uncomment following two lines to auto-load all modules
    //DEBUG_TRACEPOINT("LOAD MODULES (shared objects)");
    load_module_shared_local();

    // load other libs specified by environment variable MILKCLI_ADD_LIBS
    char *CLI_ADD_LIBS = getenv("MILKCLI_ADD_LIBS");
    if(CLI_ADD_LIBS != NULL)
    {
        if(dcquiet == 0)
        {
            printf("        MILKCLI_ADD_LIBS '%s'\n", CLI_ADD_LIBS);
        }

        char *libname;
        libname = strtok(CLI_ADD_LIBS, " ,;");

        while(libname != NULL)
        {
            DEBUG_TRACEPOINT("--- CLI Adding library: %s", libname);
            // load_sharedobj(libname);
            load_module_shared(libname);
            libname = strtok(NULL, " ,;");
        }
        printf("\n");
    }
    else
    {
        if(dcquiet == 0)
        {
            printf(
                "        MILKCLI_ADD_LIBS not set -> no additional "
                "module loaded\n");
        }
    }

    DEBUG_TRACEPOINT("Initialize data control block");
    CLI_data_init();

    if(dcdebug > 0)
    {
        printf("DEBUG: %s: start\n", __func__);
    }

    runCLI_cmd_init();

    // fifo
    fdmax = fileno(stdin);
    if(data.fifoON == 1)
    {
        if(dcquiet == 0)
        {
            printf("Creating fifo %s\n", data.fifoname);
        }
        mkfifo(data.fifoname, 0666);
        fifofd = open(data.fifoname, O_RDWR | O_NONBLOCK);
        if(fifofd == -1)
        {
            perror("open");
            printf("File name : %s\n", data.fifoname);
            DEBUG_TRACE_FEXIT();
            return EXIT_FAILURE;
        }
        if(fifofd > fdmax)
        {
            fdmax = fifofd;
        }
    }

    C_ERRNO = 0; // initialize C error variable to 0 (no error)

    data.CLIloopON = 1; // start CLI loop

    int realine_initialized = 0;

    while(data.CLIloopON == 1)
    {
        FILE *fp;

        DEBUG_TRACEPOINT("Start CLI loop");

        data.CMDexecuted = 0;
        data.CLIexecuteCMDready = 0;

        if((fp = fopen("STOPCLI", "r")) != NULL)
        {
            fprintf(stdout, "STOPCLI FILE FOUND. Exiting...\n");
            fclose(fp);
            exit(3);
        }

        if(Listimfile == 1)
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

        if(memory_re_alloc() != RETURN_SUCCESS)
        {
            fprintf(stderr,
                    "%c[%d;%dm ERROR [ FILE: %s   FUNCTION: %s   LINE: "
                    "%d ]  %c[%d;m\n",
                    (char) 27,
                    1,
                    31,
                    __FILE__,
                    __func__,
                    __LINE__,
                    (char) 27,
                    0);
            fprintf(stderr,
                    "%c[%d;%dm Memory re-allocation failed  %c[%d;m\n",
                    (char) 27,
                    1,
                    31,
                    (char) 27,
                    0);
            exit(EXIT_FAILURE);
        }

        compute_image_memory();
        compute_nb_image();

        // If fifo is on and file CLIstatup.txt exists, load it
        if(initstartup == 0)
        {
            if(data.fifoON == 1)
            {
                EXECUTE_SYSTEM_COMMAND("file %s",
                                       CLIstartupfilename); //TEST
                EXECUTE_SYSTEM_COMMAND("cat %s",
                                       CLIstartupfilename); //TEST
                EXECUTE_SYSTEM_COMMAND("cat %s > %s 2> /dev/null",
                                       CLIstartupfilename,
                                       data.fifoname);

                if(dcquiet == 0)
                {
                    printf("[%s -> %s]\n", CLIstartupfilename, data.fifoname);
                    printf("IMPORTING FILE %s ... \n", CLIstartupfilename);
                }
            }
            else
            {
                // Native OS execution (typically from shebang -s)
                FILE *fp = fopen(CLIstartupfilename, "r");
                if(fp != NULL)
                {
                    fclose(fp);
                    strncpy(data.cmdargtoken[0].val.string, "source", STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                    strncpy(data.cmdargtoken[1].val.string, CLIstartupfilename, STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                    data.cmdNBarg = 2;
                    cli_source();
                    
                    // Exit the interactive loop, as shebang scripts should terminate upon completion
                    if(strcmp(CLIstartupfilename, "CLIstartup.txt") != 0)
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
        if(data.fifoON == 1)
        {
            FD_SET(
                fifofd,
                &cli_fdin_set); // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.
        }

        FD_SET(
            fileno(stdin),
            &cli_fdin_set); // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.

        if(data.fifoON == 0)
        {
            if(realine_initialized == 0)
            {
                realine_initialized = 1;
#ifdef USE_READLINE
                // initialize readline
                DEBUG_TRACEPOINT("initialize readline");
                // Tell readline to use custom completion function
                rl_attempted_completion_function = CLI_completion;
                rl_initialize();

                /* Handle window size changes when readline is not active and reading
                     characters. */
                signal(SIGWINCH, sighandler);
                CLI_setup_hint_area();
                rl_callback_handler_install(
                    prompt,
                    (rl_vcpfunc_t *) &rl_cb_linehandler);
                CLI_configure_readline();
#endif
            }
        }

        DEBUG_TRACEPOINT("loop entry");
        while((data.CLIexecuteCMDready == 0) && (data.CLIloopON == 1))
        {
            // Special interrupt clause if CLI mode (not FIFO) AND stdin has been closed.
            if(dcsigINT == 1)
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

            if(sigwinch_received)
            {
                sigwinch_received = 0;
#ifdef USE_READLINE
                {
                    struct winsize ws;
                    if(ioctl(STDOUT_FILENO,
                             TIOCGWINSZ, &ws) >= 0)
                    {
                        rl_set_screen_size(
                            ws.ws_row, ws.ws_col);
                    }
                }
#endif
            }

            n = select(fdmax + 1, &cli_fdin_set, NULL, NULL, &tv);

            if(n ==
                    0) // nothing received, need to re-init and go back to select call
            {
                tv.tv_sec  = 0;
                tv.tv_usec = cliwaitus;

                FD_ZERO(
                    &cli_fdin_set); // Initializes the file descriptor set cli_fdin_set to have zero bits for all file descriptors.
                if(data.fifoON == 1)
                {
                    FD_SET(
                        fifofd,
                        &cli_fdin_set); // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.
                }
                FD_SET(
                    fileno(stdin),
                    &cli_fdin_set); // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.
                continue;
            }
            if(n == -1)
            {
                if(errno == EINTR)  // no command received
                {
                    continue;
                }
                else
                {
                    perror("select");
                    DEBUG_TRACE_FEXIT();
                    return EXIT_FAILURE;
                }
            }
            DEBUG_TRACEPOINT(" ");

            blockCLIinput = 0;

            if(data.fifoON == 1)
            {
                DEBUG_TRACEPOINT("fifo ON");
                if(FD_ISSET(fifofd, &cli_fdin_set))
                {
                    total_bytes = 0;
                    for(;;)
                    {
                        bytes = read(fifofd, buf0, 1);
                        if(bytes > 0)
                        {
                            buf1[total_bytes] = buf0[0];
                            total_bytes += (size_t) bytes;
                        }
                        else
                        {
                            if(errno == EWOULDBLOCK)
                            {
                                break;
                            }
                            else
                            {
                                perror("read");
                                DEBUG_TRACE_FEXIT();
                                return EXIT_FAILURE;
                            }
                        }
                        if(buf0[0] == '\n')
                        {
                            buf1[total_bytes - 1] = '\0';
                            strncpy(data.CLIcmdline,
                                buf1,
                                STRINGMAXLEN_CLICMDLINE - 1);

                            DEBUG_TRACEPOINT(
                                "CLI executing line: "
                                "%s",
                                data.CLIcmdline);
                            if(dcdebug > 0)
                            {
                                printf("DEBUG: %s: execute line, fifo mode\n", __func__);
                            }
                            CLI_execute_line();
                            DEBUG_TRACEPOINT("CLI line executed");

                            printf("%s", prompt);
                            fflush(stdout);
                            break;
                        }
                    }
                    blockCLIinput =
                        1; // keep blocking input while fifo is not empty
                }
            }

            if(blockCLIinput == 0)  // fifo has been cleared
            {
                DEBUG_TRACEPOINT("fifo cleared");
                if(realine_initialized == 0)
                {
                    realine_initialized = 1;
#ifdef USE_READLINE
                    // initialize readline
                    DEBUG_TRACEPOINT("initialize readline");
                    // Tell readline to use custom completion function
                    rl_attempted_completion_function = CLI_completion;
                    rl_initialize();

                    /* Handle window size changes when readline is not active and reading
                         characters. */
                    signal(SIGWINCH, sighandler);
                    CLI_setup_hint_area();
                    rl_callback_handler_install(
                        prompt,
                        (rl_vcpfunc_t *) &rl_cb_linehandler);
                    CLI_configure_readline();
#endif
                }
            }

            //printf("fifo cleared, accepting user input through CLI\n");

            if(blockCLIinput == 0)
            {
                // revert to default mode
                if(FD_ISSET(fileno(stdin), &cli_fdin_set))
                {
#ifdef USE_READLINE
                    DEBUG_TRACEPOINT("readline callback");
                    rl_callback_read_char();
#else
                    // Fallback for no readline: read from stdin directly
                    if (fgets(data.CLIcmdline, sizeof(data.CLIcmdline), stdin)) {
                        data.CLIcmdline[strcspn(data.CLIcmdline, "\n")] = 0; // strip newline
                        CLI_execute_line();
                    } else {
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

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * print_session_name - Print the CLI session name.
 *
 * Prints the process name set by the -n option.
 * If no name was given, prints "(none)".
 */
static errno_t print_session_name()
{
    printf("%s\n", data.processname);
    return RETURN_SUCCESS;
}

void runCLI_cmd_init()
{
    // ensure that commands below belong to root/MAIN module
    data.moduleindex = -1;

    RegisterCLIcommand("exit",
                       __FILE__,
                       exitCLI,
                       "exit program (same as quit command)",
                       "no argument",
                       "exit",
                       "exitCLI");

    RegisterCLIcommand("quit",
                       __FILE__,
                       exitCLI,
                       "exit program (same as exit command)",
                       "no argument",
                       "quit",
                       "exitCLI");

    RegisterCLIcommand("exitCLI",
                       __FILE__,
                       exitCLI,
                       "exit program (same as quit command)",
                       "no argument",
                       "exitCLI",
                       "exitCLI");

    RegisterCLIcommand("name",
                       __FILE__,
                       print_session_name,
                       "print CLI session name (set by milk -n)",
                       "no argument",
                       "name",
                       "print_session_name()");

    RegisterCLIcommand("help",
                       __FILE__,
                       help,
                       "show help",
                       "no argument",
                       "help",
                       "int help()");

    RegisterCLIcommand("fhelp",
                       __FILE__,
                       cli_fhelp,
                       "interactive fuzzy help search",
                       "no argument",
                       "fhelp",
                       "int cli_fhelp()");

    RegisterCLIcommand("?",
                       __FILE__,
                       help,
                       "show help",
                       "no argument",
                       "?",
                       "int help()");

    RegisterCLIcommand("helprl",
                       __FILE__,
                       help,
                       "show readline help",
                       "no argument",
                       "helprl",
                       "int help()");

    RegisterCLIcommand("cmd?",
                       __FILE__,
                       help_cmd,
                       "list/help command(s)",
                       "<command name>(optional)",
                       "cmd?",
                       "int help_cmd()");

    RegisterCLIcommand("cmdinfo?",
                       __FILE__,
                       cmdinfosearch,
                       "search for string/regex in command info",
                       "<search expression>",
                       "cmdinfo? image",
                       "int cmdinfosearch()");

    RegisterCLIcommand("m?",
                       __FILE__,
                       help_module,
                       "list/help module(s)",
                       "<module name>(optional)",
                       "m? COREMOD_memory",
                       "errno_t list_commands_module()");

    RegisterCLIcommand("soload",
                       __FILE__,
                       load_so__cli,
                       "load shared object",
                       "<shared object name>",
                       "soload mysharedobj.so",
                       "int load_sharedobj(char *libname)");

    RegisterCLIcommand("mload",
                       __FILE__,
                       load_module__cli,
                       "load module from shared object",
                       "<module name>",
                       "mload mymodule",
                       "errno_t load_module_shared(char *modulename)");

    RegisterCLIcommand("mloadas",
                       __FILE__,
                       CLIcore__load_module_as__cli,
                       "load module from shared object, use short name binding",
                       "<module name> <shortname>",
                       "mloadas mymodule mymod",
                       "errno_t load_module_shared(char *modulename)");

    RegisterCLIcommand("ci",
                       __FILE__,
                       printInfo,
                       "Print version, settings, info and exit",
                       "no argument",
                       "ci",
                       "int printInfo()");

    RegisterCLIcommand("dpsingle",
                       __FILE__,
                       set_default_precision_single,
                       "Set default precision to single",
                       "no argument",
                       "dpsingle",
                       "dcprecision = 0");

    RegisterCLIcommand("dpdouble",
                       __FILE__,
                       set_default_precision_double,
                       "Set default precision to double",
                       "no argument",
                       "dpdouple",
                       "dcprecision = 1");

    // process info

    RegisterCLIcommand("setprocinfoON",
                       __FILE__,
                       set_processinfoON,
                       "Set processes info ON",
                       "no argument",
                       "setprocinfoON",
                       "set_processinfoON()");

    RegisterCLIcommand("setprocinfoOFF",
                       __FILE__,
                       set_processinfoOFF,
                       "Set processes info OFF",
                       "no argument",
                       "setprocinfoOFF",
                       "set_processinfoOFF()");

#ifdef USE_NCURSES
    RegisterCLIcommand("procCTRL",
                       __FILE__,
                       processinfo_CTRLscreen__cli,
                       "processes control screen",
                       "no argument",
                       "procCTRL",
                       "processinfo_CTRLscreen()");

    // stream ctrl

    RegisterCLIcommand("streamCTRL",
                       __FILE__,
                       streamCTRL_CTRLscreen__cli,
                       "stream control screen",
                       "no argument",
                       "streamCTRL",
                       "streamCTRL_CTRLscreen()");
#endif

    // FPS
    RegisterCLIcommand("fpsload",
                       __FILE__,
                       function_parameter_structure_load__cli,
                       "Load function parameter struct (FPS)",
                       "<fpsname>",
                       "fpsload imanalyze",
                       "long function_parameter_structure_load(char *fpsname)");

#ifdef USE_NCURSES
    RegisterCLIcommand(
        "fpsCTRL",
        __FILE__,
        functionparameter_CTRLscreen__cli,
        "function parameters control screen",
        "no arg",
        "fpsCTRL fpsname",
        "int_fast8_t functionparameter_CTRLscreen(char *fpsname)");
#endif

    RegisterCLIcommand("usleep",
                       __FILE__,
                       milk_usleep__cli,
                       "usleep",
                       "<us>",
                       "usleep 1000",
                       "usleep(long tus)");

    RegisterCLIcommand("cd",
                       __FILE__,
                       cli_cd,
                       "change current directory",
                       "<dir>",
                       "cd /tmp",
                       "cli_cd()");

    RegisterCLIcommand("pwd",
                       __FILE__,
                       cli_pwd,
                       "print current directory",
                       "no argument",
                       "pwd",
                       "cli_pwd()");

    RegisterCLIcommand("alias",
                       __FILE__,
                       cli_alias_add,
                       "create/update command alias",
                       "<name> <command...>",
                       "alias ld mem.listim",
                       "cli_alias_add()");

    RegisterCLIcommand("unalias",
                       __FILE__,
                       cli_alias_remove,
                       "remove command alias",
                       "<name>",
                       "unalias ld",
                       "cli_alias_remove()");

    RegisterCLIcommand("aliases",
                       __FILE__,
                       cli_alias_list,
                       "list all command aliases",
                       "no argument",
                       "aliases",
                       "cli_alias_list()");

    RegisterCLIcommand("watch",
                       __FILE__,
                       cli_watch,
                       "repeat command at interval",
                       "<interval_ms> <command...>",
                       "watch 1000 mem.listim",
                       "cli_watch()");

    RegisterCLIcommand("time",
                       __FILE__,
                       cli_time,
                       "measure command execution time",
                       "<command...>",
                       "time mem.listim",
                       "cli_time()");

    RegisterCLIcommand("cmdstats",
                       __FILE__,
                       cli_cmdstats,
                       "show command usage statistics",
                       "no argument",
                       "cmdstats",
                       "cli_cmdstats()");

    RegisterCLIcommand(
        "cli.timing",
        __FILE__,
        cli_timing_toggle,
        "toggle display of command execution timing",
        "[on|off]",
        "cli.timing on",
        "cli_timing_toggle()");

#ifdef USE_READLINE
    RegisterCLIcommand(
        "synhl",
        __FILE__,
        cli_syntax_highlight_toggle,
        "toggle syntax highlighting",
        "[on|off]",
        "synhl off",
        "cli_syntax_highlight_toggle()");
#endif

    RegisterCLIcommand("source",
                       __FILE__,
                       cli_source,
                       "execute a milk script file",
                       "<filename>",
                       "source myscript.milk",
                       "cli_source()");

    RegisterCLIcommand(
        "savescript",
        __FILE__,
        cli_savescript,
        "save variables and functions "
        "to a script file",
        "<filename>",
        "savescript state.milk",
        "cli_savescript()");

    RegisterCLIcommand(
        "savehistory",
        __FILE__,
        cli_savehistory,
        "save command history to a file",
        "<filename>",
        "savehistory cmds.milk",
        "cli_savehistory()");

    RegisterCLIcommand(
        "setprompt",
        __FILE__,
        cli_setprompt,
        "set custom prompt format",
        "[<format>]",
        "setprompt \"%u@%h %d > \"",
        "cli_setprompt()");

    RegisterCLIcommand(
        "bookmark",
        __FILE__,
        cli_bookmark,
        "manage command bookmarks",
        "save|run|list|rm <name> [cmd]",
        "bookmark save myjob \"cmd1 ; cmd2\"",
        "cli_bookmark()");

    RegisterCLIcommand(
        "sessionlog",
        __FILE__,
        cli_sessionlog,
        "enable session command logging",
        "[on|off|<filename>]",
        "sessionlog on",
        "cli_sessionlog()");

    RegisterCLIcommand(
        "history",
        __FILE__,
        cli_history_show,
        "show recent command history",
        "[<N>]",
        "history 50",
        "cli_history_show()");

    RegisterCLIcommand(
        "searchhist",
        __FILE__,
        cli_searchhist,
        "search history for pattern",
        "<pattern>",
        "searchhist listim",
        "cli_searchhist()");

    RegisterCLIcommand(
        "fhist",
        __FILE__,
        cli_fhist,
        "interactive fuzzy search history",
        "",
        "fhist",
        "cli_fhist()");

    RegisterCLIcommand(
        "ghistory",
        __FILE__,
        cli_ghistory,
        "global history (all sessions)",
        "[N] [-s <session_id>]",
        "ghistory 50",
        "cli_ghistory()");

    RegisterCLIcommand(
        "lhistory",
        __FILE__,
        cli_lhistory,
        "local history (current session)",
        "[N]",
        "lhistory",
        "cli_lhistory()");

    RegisterCLIcommand(
        "fparam",
        __FILE__,
        cli_fparam,
        "interactive FPS parameter editor",
        "<fpsname>",
        "fparam cnt2push",
        "cli_fparam()");

    RegisterCLIcommand(
        "echo",
        __FILE__,
        cli_cmd_echo,
        "print arguments",
        "[-n] <args...>",
        "echo hello world",
        "cli_cmd_echo()");

    RegisterCLIcommand(
        "unset",
        __FILE__,
        cli_cmd_unset,
        "remove a CLI variable",
        "<varname>",
        "unset myvar",
        "cli_cmd_unset()");

    RegisterCLIcommand(
        "vars",
        __FILE__,
        cli_cmd_vars,
        "list all CLI variables",
        "",
        "vars",
        "cli_cmd_vars()");

    RegisterCLIcommand(
        "fpsset",
        __FILE__,
        cli_cmd_fpsset,
        "set FPS parameter value",
        "<fpsname.param> <value>",
        "fpsset loopctrl.gain 0.3",
        "cli_cmd_fpsset()");

    //  init_modules();

    if(dcquiet == 0)
    {
        printf("        Loaded %ld modules, %u commands\n",
               data.NBmodule,
               data.NBcmd);
        printf("        \n");
    }
}

static void runCLI_free()
{
#ifndef DATA_STATIC_ALLOC
    // Free
    DEBUG_TRACEPOINT("free dcimg");
    free(dcimg);

    DEBUG_TRACEPOINT("free dcvar");
    free(dcvar);

    DEBUG_TRACEPOINT("free data.fps");
    if(dcfpsarr == NULL)
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

void fnExit1(void)
{
    //
}


static int command_line_process_options(int argc, char **argv)
{
    int                option_index = 0;
    struct sched_param schedpar;
    char               command[STRINGMAXLEN_COMMAND];

    static struct option long_options[] =
    {
        /* These options set a flag. */
        {"verbose", no_argument, &Verbose, 1},
        {"listimf", no_argument, &Listimfile, 1},
        /* These options don't set a flag.
        We distinguish them by their indices. */
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"info", no_argument, 0, 'i'},
        {"overwrite", no_argument, 0, 'o'},
        {"errorexit", no_argument, 0, 'e'},
        {"idle", no_argument, 0, 'Z'},
        {"autocomplete", no_argument, 0, 'A'},
        {"no-autocomplete", no_argument, 0, 0x100},
        {"no-history-suggest", no_argument, 0, 0x101},
        {"no-arg-hints", no_argument, 0, 0x102},
        {"no-fuzzy", no_argument, 0, 0x103},
        {"fifoflag", no_argument, 0, 'f'},
        {"debug", required_argument, 0, 'd'},
        {"mmon", required_argument, 0, 'm'},
        {"pname", required_argument, 0, 'n'},
        {"priority", required_argument, 0, 'p'},
        {"fifoname", required_argument, 0, 'F'},
        {"startup", required_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    data.fifoON          = 0; // default
    data.processnameflag = 0; // default

    while(1)
    {
        int c;

        c = getopt_long(argc,
                        argv,
                        "hvid:oeZm:n:p:fF:s:A",
                        long_options,
                        &option_index);

        /* Detect the end of the options. */
        if(c == -1)
        {
            break;
        }

        switch(c)
        {
        case 0:
            /* If this option set a flag, do nothing else now. */
            if(long_options[option_index].flag != 0)
            {
                break;
            }
            printf("option %s", long_options[option_index].name);
            if(optarg)
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
            if(dcquiet == 0)
            {
                printf("Exit on error ON\n");
            }
            dcerrorexit = 1;
            break;

        case 'Z':
            printf(
                "Idle mode: only runs process when X is idle (pid "
                "%ld)\n",
                (long) getpid());
            snprintf(command, STRINGMAXLEN_COMMAND, "runidle %ld > /dev/null &\n",
                     (long) getpid());
            if(system(command) != 0)
            {
                PRINT_ERROR("system() returns non-zero value");
            }
            break;

        case 'A':
            if(dcquiet == 0)
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

        case 'm':
            printf("Starting memory monitor on '%s'\n", optarg);
            //memory_monitor(optarg);
            break;

        case 'n':
            if(dcquiet == 0)
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

            if(seteuid(dceuid) != 0)  //This goes up to maximum privileges
            {
                PRINT_ERROR("seteuid() returns non-zero value");
            }
            sched_setscheduler(
                0,
                SCHED_FIFO,
                &schedpar); //other option is SCHED_RR, might be faster

            if(seteuid(dcruid) != 0)  //Go back to normal privileges
            {
                PRINT_ERROR("seteuid() returns non-zero value");
            }
            break;

        case 'f':
            if(dcquiet == 0)
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

        case 's':
            strncpy(CLIstartupfilename,
                optarg,
                STRINGMAXLEN_CLISTARTUPFILENAME - 1);
            if(dcquiet == 0)
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
    if(data.processnameflag == 0)
    {
        time_t ts = time(NULL);
        int slen = snprintf(data.processname,
                            STRINGMAXLEN_PROCESSNAME,
                            "%ldp%ld",
                            (long) ts,
                            (long) CLIPID);
        if(slen < 1 || slen >= STRINGMAXLEN_PROCESSNAME)
        {
            PRINT_ERROR("snprintf error building default processname");
        }
        strncpy(data.processname0,
                data.processname,
                STRINGMAXLEN_PROCESSNAME - 1);
        data.processnameflag = 1;
        prctl(PR_SET_NAME, data.processname, 0, 0, 0);
    }

    return RETURN_SUCCESS;
}

