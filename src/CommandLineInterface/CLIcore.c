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



#define _GNU_SOURCE


/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stddef.h> // offsetof()
#include <sys/resource.h> // getrlimit
#include <termios.h>



#include <sys/time.h>



#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <getopt.h>
#include <ncurses.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdbool.h>
#ifndef __MACH__
#include <sys/prctl.h>
#endif
#include <sched.h>
#include <signal.h>

#include <readline/readline.h>
#include <readline/history.h>


# ifdef _OPENMP
# include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
# endif

#ifdef _OPENACC
#include <openacc.h>
#endif



#include <gsl/gsl_rng.h> // for random numbers
#include <fitsio.h>



#include "CommandLineInterface/CLIcore.h"

//#include "initmodules.h"

#include "ImageStreamIO/ImageStreamIO.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"


#include "CommandLineInterface/CLIcore_UI.h"
#include "CommandLineInterface/CLIcore_checkargs.h"
#include "CommandLineInterface/CLIcore_datainit.h"
#include "CommandLineInterface/CLIcore_help.h"
#include "CommandLineInterface/CLIcore_memory.h"
#include "CommandLineInterface/CLIcore_modules.h"
#include "CommandLineInterface/CLIcore_setSHMdir.h"
#include "CommandLineInterface/CLIcore_signals.h"




/*-----------------------------------------
*       Globals exported to all modules
*/

DATA __attribute__((used)) data;

pid_t CLIPID;


int C_ERRNO;


int Verbose = 0;
int Listimfile = 0;


char CLIstartupfilename[200] = "CLIstartup.txt";

// fifo input
static int fifofd;
static fd_set cli_fdin_set;


/*-----------------------------------------
*       Forward References
*/
int user_function();
void fnExit1(void);
void runCLI_cmd_init();
static void runCLI_free();

static int sigwinch_received = 0;







static int command_line_process_options(
    int argc,
    char **argv
);


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

    if(data.fifoON == 1)
    {
        EXECUTE_SYSTEM_COMMAND("rm %s", data.fifoname);
    }


    if(Listimfile == 1)
    {
        EXECUTE_SYSTEM_COMMAND("rm imlist.txt");
    }

    if(data.quiet == 0)
    {
        printf("Closing PID %ld (prompt process)\n", (long) getpid());
    }
    //    exit(0);
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
    if(0
            + CLI_checkarg(1, CLIARG_STR)
            + CLI_checkarg(2, CLIARG_STR)
            == 0)
    {
        strcpy(data.moduleshortname, data.cmdargtoken[2].val.string);
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
    data.processinfo  = 1;

    return RETURN_SUCCESS;
}

errno_t set_processinfoOFF()
{
    data.processinfo  = 0;

    return RETURN_SUCCESS;
}



errno_t set_default_precision_single()
{
    data.precision  = 0;

    return RETURN_SUCCESS;
}




errno_t set_default_precision_double()
{
    data.precision  = 1;

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


errno_t functionparameter_CTRLscreen__cli()
{
    DEBUG_TRACEPOINT("calling CLI_checkarg");
    if(
        (CLI_checkarg(1, CLIARG_LONG) == 0) &&
        (CLI_checkarg(2, CLIARG_STR) == 0) &&
        (CLI_checkarg(3, CLIARG_STR) == 0)
    )
    {
        functionparameter_CTRLscreen((uint32_t) data.cmdargtoken[1].val.numl,
                                     data.cmdargtoken[2].val.string, data.cmdargtoken[3].val.string);
        return RETURN_SUCCESS;
    }
    else
    {
        printf("Wrong args (%d)\n", data.cmdargtoken[1].type);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}



errno_t function_parameter_structure_load__cli()
{
    DEBUG_TRACEPOINT("calling CLI_checkarg");
    if(CLI_checkarg(1, CLIARG_STR) == 0)
    {
        function_parameter_structure_load(
            data.cmdargtoken[1].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}





errno_t processinfo_CTRLscreen__cli()
{
    return(processinfo_CTRLscreen());
}


errno_t streamCTRL_CTRLscreen__cli()
{
    return(streamCTRL_CTRLscreen());
}
















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









static errno_t runCLI_initialize(
)
{
    // NOTE: change to function call to ImageStreamIO_typename
    /* TYPESIZE[_DATATYPE_UINT8]                  = SIZEOF_DATATYPE_UINT8;
     TYPESIZE[_DATATYPE_INT8]                   = SIZEOF_DATATYPE_INT8;
     TYPESIZE[_DATATYPE_UINT16]                 = SIZEOF_DATATYPE_UINT16;
     TYPESIZE[_DATATYPE_INT16]                  = SIZEOF_DATATYPE_INT16;
     TYPESIZE[_DATATYPE_UINT32]                 = SIZEOF_DATATYPE_UINT32;
     TYPESIZE[_DATATYPE_INT32]                  = SIZEOF_DATATYPE_INT32;
     TYPESIZE[_DATATYPE_UINT64]                 = SIZEOF_DATATYPE_UINT64;
     TYPESIZE[_DATATYPE_INT64]                  = SIZEOF_DATATYPE_INT64;
     TYPESIZE[_DATATYPE_FLOAT]                  = SIZEOF_DATATYPE_FLOAT;
     TYPESIZE[_DATATYPE_DOUBLE]                 = SIZEOF_DATATYPE_DOUBLE;
     TYPESIZE[_DATATYPE_COMPLEX_FLOAT]          = SIZEOF_DATATYPE_COMPLEX_FLOAT;
     TYPESIZE[_DATATYPE_COMPLEX_DOUBLE]         = SIZEOF_DATATYPE_COMPLEX_DOUBLE;*/
    //    TYPESIZE[_DATATYPE_EVENT_UI8_UI8_UI16_UI8] = SIZEOF_DATATYPE_EVENT_UI8_UI8_UI16_UI8;




    // get PID and write it to shell env variable MILK_CLI_PID
    CLIPID = getpid();
    if(data.quiet == 0)
    {
        printf("        CLI PID = %d\n", (int) CLIPID);

        EXECUTE_SYSTEM_COMMAND("echo -n \"        \"; cat /proc/%d/status | grep Cpus_allowed_list",
                               CLIPID);
    }

    //	printf("    _SC_CLK_TCK = %d\n", sysconf(_SC_CLK_TCK));



    if(Verbose)
    {
        fprintf(stdout, "%s: compiled %s %s\n", __FILE__, __DATE__, __TIME__);
    }

# ifdef _OPENMP
    if(data.quiet == 0)
    {
        printf("        Running with openMP, max threads = %d  (OMP_NUM_THREADS)\n",
               omp_get_max_threads());
    }
# else
    if(data.quiet == 0)
    {
        printf("        Compiled without openMP\n");
    }
# endif

# ifdef _OPENACC
    int openACC_devtype = acc_get_device_type();
    if(data.quiet == 0)
    {
        printf("        Running with openACC version %d.  %d device(s), type %d\n",
               _OPENACC, acc_get_num_devices(openACC_devtype), openACC_devtype);
    }
# endif





    // to take advantage of kernel priority:
    // owner=root mode=4755

#ifndef __MACH__
    getresuid(&data.ruid, &data.euid, &data.suid);
    //This sets it to the privileges of the normal user
    if(seteuid(data.ruid) != 0)
    {
        PRINT_ERROR("seteuid error");
    }
#endif



    // Initialize random-number generator
    //
    const gsl_rng_type *rndgenType;
    //rndgenType = gsl_rng_ranlxs2; // best algorithm but slow
    //rndgenType = gsl_rng_ranlxs0; // not quite as good, slower
    rndgenType  = gsl_rng_rand; // not as good but ~10x faster fast
    data.rndgen = gsl_rng_alloc(rndgenType);
    gsl_rng_set(data.rndgen, time(NULL));

    // warm up
    //for(i=0; i<10; i++)
    //    v1 = gsl_rng_uniform (data.rndgen);


    data.progStatus        = 0;

    data.Debug             = 0;
    data.overwrite         = 0;
    data.precision         = 0; // float is default precision
    data.SHARED_DFT        = 0; // do not allocate shared memory for images
    data.NBKEYWORD_DFT     = 50; // allocate memory for 10 keyword per image
    sprintf(data.SAVEDIR, ".");

    data.CLIlogON          = 0;     // log every command
    data.fifoON            = 0;
    data.processinfo       = 1;  // process info for intensive processes
    data.processinfoActive = 0; // toggles to 1 when process is logged






    // signal handling

    data.sigact.sa_handler = sig_handler;
    sigemptyset(&data.sigact.sa_mask);
    data.sigact.sa_flags = 0;

    data.signal_USR1 = 0;
    data.signal_USR2 = 0;
    data.signal_TERM = 0;
    data.signal_INT  = 0;
    data.signal_BUS  = 0;
    data.signal_SEGV = 0;
    data.signal_ABRT = 0;
    data.signal_HUP  = 0;
    data.signal_PIPE = 0;

    if(sigaction(SIGUSR1, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGUSR1\n");
    }
    if(sigaction(SIGUSR2, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGUSR2\n");
    }

    set_signal_catch();


    return RETURN_SUCCESS;
}











/* Handle SIGWINCH and window size changes when readline is not active and
   reading a character. */
static void
sighandler (int sig)
{
    rl_resize_terminal ();
    //printf("RESIZE detected %d %d\n", COLS, LINES);
    sigwinch_received = 1;
}


/**
 * @brief Command Line Interface (CLI) main\n
 *
 * Uses readline to read user input\n
 * parsing done with bison and flex
 */


errno_t runCLI(
    int   argc,
    char *argv[],
    char *promptstring
)
{
    int     fdmax;
    int     n;

    ssize_t bytes;
    size_t  total_bytes;
    char    buf0[1];
    char    buf1[1024];

    int     initstartup = 0; /// becomes 1 after startup

    int     blockCLIinput = 0;
    int     cliwaitus = 100;
    struct  timeval tv;   // sleep 100 us after reading FIFO




    strcpy(data.processname, argv[0]);


    // Set CLI prompt
    char prompt[200];
    runCLI_prompt(promptstring, prompt);



    // CLI initialize
    runCLI_initialize();




    // set shared memory directory
    setSHMdir();


    DEBUG_TRACEPOINT("CLI start");

    // initialize fifo to process name
    DEBUG_TRACEPOINT("set default fifo name");
    //sprintf(data.fifoname, "%s.fifo.%07d", data.processname, getpid());
    WRITE_FULLFILENAME(data.fifoname,
                       "%s/.%s.fifo.%07d",
                       data.shmdir,
                       data.processname,
                       getpid());

    DEBUG_TRACEPOINT("Get command-line options");
    command_line_process_options(argc, argv);





    data.progStatus = 1;
    printf("\n");




    // uncomment following two lines to auto-load all modules
    //DEBUG_TRACEPOINT("LOAD MODULES (shared objects)");
    //load_module_shared_ALL();

    // load other libs specified by environment variable MILKCLI_ADD_LIBS
    char *CLI_ADD_LIBS = getenv("MILKCLI_ADD_LIBS");
    if(CLI_ADD_LIBS != NULL)
    {
        if(data.quiet == 0)
        {
            printf("        MILKCLI_ADD_LIBS '%s'\n", CLI_ADD_LIBS);
        }

        char *libname;
        libname = strtok(CLI_ADD_LIBS, " ,;");

        while(libname != NULL)
        {
            DEBUG_TRACEPOINT("--- CLI Adding library: %s\n", libname);
            // load_sharedobj(libname);
            load_module_shared(libname);
            libname = strtok(NULL, " ,;");
        }
        printf("\n");
    }
    else
    {
        if(data.quiet == 0)
        {
            printf("        MILKCLI_ADD_LIBS not set -> no additional module loaded\n");
        }
    }




    DEBUG_TRACEPOINT("Initialize data control block");
    CLI_data_init();

    runCLI_cmd_init();







    // fifo
    fdmax = fileno(stdin);
    if(data.fifoON == 1)
    {
        if(data.quiet == 0)
        {
            printf("Creating fifo %s\n", data.fifoname);
        }
        mkfifo(data.fifoname, 0666);
        fifofd = open(data.fifoname, O_RDWR | O_NONBLOCK);
        if(fifofd == -1)
        {
            perror("open");
            printf("File name : %s\n", data.fifoname);
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
                    "%c[%d;%dm ERROR [ FILE: %s   FUNCTION: %s   LINE: %d ]  %c[%d;m\n",
                    (char) 27, 1, 31, __FILE__, __func__, __LINE__, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Memory re-allocation failed  %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            exit(EXIT_FAILURE);
        }

        compute_image_memory();
        compute_nb_image();


        // If fifo is on and file CLIstatup.txt exists, load it
        if(initstartup == 0)
        {
            if(data.fifoON == 1)
            {
                EXECUTE_SYSTEM_COMMAND("file %s", CLIstartupfilename); //TEST
                EXECUTE_SYSTEM_COMMAND("cat %s", CLIstartupfilename); //TEST
                EXECUTE_SYSTEM_COMMAND("cat %s > %s 2> /dev/null", CLIstartupfilename,
                                       data.fifoname);

                if(data.quiet == 0)
                {
                    printf("[%s -> %s]\n", CLIstartupfilename, data.fifoname);
                    printf("IMPORTING FILE %s ... \n", CLIstartupfilename);
                }
            }
        }
        initstartup = 1;



        DEBUG_TRACEPOINT("Get user input fifo=%d", data.fifoON); //===============================
        tv.tv_sec = 0;
        tv.tv_usec = cliwaitus;


        FD_ZERO(&cli_fdin_set);  // Initializes the file descriptor set cli_fdin_set to have zero bits for all file descriptors.
        if(data.fifoON == 1)
        {
            FD_SET(fifofd,
                   &cli_fdin_set);  // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.
        }
        FD_SET(fileno(stdin),
               &cli_fdin_set);  // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.


        if(data.fifoON == 0)
        {
            if(realine_initialized == 0)
            {
                realine_initialized = 1;
                // initialize readline
                DEBUG_TRACEPOINT("initialize readline");
                // Tell readline to use custom completion function
                rl_attempted_completion_function = CLI_completion;
                rl_initialize();

                /* Handle window size changes when readline is not active and reading
                     characters. */
                signal (SIGWINCH, sighandler);
                rl_callback_handler_install(prompt, (rl_vcpfunc_t *) &rl_cb_linehandler);
            }
        }


        while((data.CLIexecuteCMDready == 0) && (data.CLIloopON == 1))
        {
            DEBUG_TRACEPOINT("processing input command");
            //printf("CLI get user input %d  [%d]\n", __LINE__, data.CLIloopON );
            n = select(fdmax + 1, &cli_fdin_set, NULL, NULL, &tv);

            if(n == 0)   // nothing received, need to re-init and go back to select call
            {
                tv.tv_sec = 0;
                tv.tv_usec = cliwaitus;


                FD_ZERO(&cli_fdin_set);  // Initializes the file descriptor set cli_fdin_set to have zero bits for all file descriptors.
                if(data.fifoON == 1)
                {
                    FD_SET(fifofd,
                           &cli_fdin_set);    // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.
                }
                FD_SET(fileno(stdin),
                       &cli_fdin_set);  // Sets the bit for the file descriptor fifofd in the file descriptor set cli_fdin_set.
                continue;
            }
            if(n == -1)
            {
                if(errno == EINTR)   // no command received
                {
                    continue;
                }
                else
                {
                    perror("select");
                    return EXIT_FAILURE;
                }
            }
            DEBUG_TRACEPOINT(" ");

            blockCLIinput = 0;

            if(data.fifoON == 1)
            {
                if(FD_ISSET(fifofd, &cli_fdin_set))
                {
                    total_bytes = 0;
                    for(;;)
                    {
                        bytes = read(fifofd, buf0, 1);
                        if(bytes > 0)
                        {
                            buf1[total_bytes] = buf0[0];
                            total_bytes += (size_t)bytes;
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
                                return EXIT_FAILURE;
                            }
                        }
                        if(buf0[0] == '\n')
                        {
                            buf1[total_bytes - 1] = '\0';
                            strcpy(data.CLIcmdline, buf1);
                            DEBUG_TRACEPOINT("CLI executing line: %s", data.CLIcmdline);
                            CLI_execute_line();
                            DEBUG_TRACEPOINT("CLI line executed");
                            printf("%s", prompt);
                            fflush(stdout);
                            break;
                        }
                    }
                    blockCLIinput = 1; // keep blocking input while fifo is not empty
                }
            }

            if(blockCLIinput == 0) // fifo has been cleared
            {
                if(realine_initialized == 0)
                {
                    realine_initialized = 1;
                    // initialize readline
                    DEBUG_TRACEPOINT("initialize readline");
                    // Tell readline to use custom completion function
                    rl_attempted_completion_function = CLI_completion;
                    rl_initialize();

                    /* Handle window size changes when readline is not active and reading
                         characters. */
                    signal (SIGWINCH, sighandler);
                    rl_callback_handler_install(prompt, (rl_vcpfunc_t *) &rl_cb_linehandler);
                }
            }

            //printf("fifo cleared, accepting user input through CLI\n");


            if(blockCLIinput == 0)
            {   // revert to default mode
                if(FD_ISSET(fileno(stdin), &cli_fdin_set))
                {
                    DEBUG_TRACEPOINT("readline callback");
                    rl_callback_read_char();
                    DEBUG_TRACEPOINT(" ");
                }
            }
            DEBUG_TRACEPOINT(" ");

        }
        data.CLIexecuteCMDready = 0;
        DEBUG_TRACEPOINT(" ");

        //TEST data.CLIloopON = 0;
    }
    DEBUG_TRACEPOINT("exit from CLI loop");

    // clear all images and variables
    clearall();

    DEBUG_TRACEPOINT("images and variables cleared");

    runCLI_free();

    DEBUG_TRACEPOINT("memory freed");


#if ( RL_READLINE_VERSION > 0x602 )
    rl_clear_history();
#endif

    rl_callback_handler_remove();

    DEBUG_TRACEPOINT("exit from runCLI function");

    return RETURN_SUCCESS;
}













void runCLI_cmd_init()
{
    // ensure that commands below belong to root/MAIN module
    data.moduleindex = -1;


    RegisterCLIcommand(
        "exit",
        __FILE__,
        exitCLI,
        "exit program (same as quit command)",
        "no argument",
        "exit",
        "exitCLI");

    RegisterCLIcommand(
        "quit",
        __FILE__,
        exitCLI,
        "exit program (same as quit command)",
        "no argument",
        "quit",
        "exitCLI");

    RegisterCLIcommand(
        "exitCLI",
        __FILE__,
        exitCLI,
        "exit program (same as quit command)",
        "no argument",
        "exitCLI",
        "exitCLI");


    RegisterCLIcommand(
        "help",
        __FILE__,
        help,
        "show help",
        "no argument",
        "help",
        "int help()");

    RegisterCLIcommand(
        "?",
        __FILE__,
        help,
        "show help",
        "no argument",
        "?",
        "int help()");

    RegisterCLIcommand(
        "helprl",
        __FILE__,
        help,
        "show readline help",
        "no argument",
        "helprl",
        "int help()");


    RegisterCLIcommand(
        "cmd?",
        __FILE__,
        help_cmd,
        "list/help command(s)",
        "<command name>(optional)",
        "cmd?",
        "int help_cmd()");

    RegisterCLIcommand(
        "cmdinfo?",
        __FILE__,
        cmdinfosearch,
        "search for string/regex in command info",
        "<search expression>",
        "cmdinfo? image",
        "int cmdinfosearch()");

    RegisterCLIcommand(
        "m?",
        __FILE__,
        help_module,
        "list/help module(s)",
        "<module name>(optional)",
        "m? COREMOD_memory",
        "errno_t list_commands_module()");

    RegisterCLIcommand(
        "soload",
        __FILE__,
        load_so__cli,
        "load shared object",
        "<shared object name>",
        "soload mysharedobj.so",
        "int load_sharedobj(char *libname)");

    RegisterCLIcommand(
        "mload",
        __FILE__,
        load_module__cli,
        "load module from shared object",
        "<module name>",
        "mload mymodule",
        "errno_t load_module_shared(char *modulename)");

    RegisterCLIcommand(
        "mloadas",
        __FILE__,
        CLIcore__load_module_as__cli,
        "load module from shared object, use short name binding",
        "<module name> <shortname>",
        "mloadas mymodule mymod",
        "errno_t load_module_shared(char *modulename)");

    RegisterCLIcommand(
        "ci",
        __FILE__,
        printInfo,
        "Print version, settings, info and exit",
        "no argument",
        "ci",
        "int printInfo()");


    RegisterCLIcommand(
        "dpsingle",
        __FILE__,
        set_default_precision_single,
        "Set default precision to single",
        "no argument",
        "dpsingle",
        "data.precision = 0");

    RegisterCLIcommand(
        "dpdouble",
        __FILE__,
        set_default_precision_double,
        "Set default precision to double",
        "no argument",
        "dpdouple",
        "data.precision = 1");




    // process info


    RegisterCLIcommand(
        "setprocinfoON",
        __FILE__,
        set_processinfoON,
        "Set processes info ON",
        "no argument",
        "setprocinfoON",
        "set_processinfoON()");

    RegisterCLIcommand(
        "setprocinfoOFF",
        __FILE__,
        set_processinfoOFF,
        "Set processes info OFF",
        "no argument",
        "setprocinfoOFF",
        "set_processinfoOFF()");


    RegisterCLIcommand(
        "procCTRL",
        __FILE__,
        processinfo_CTRLscreen__cli,
        "processes control screen",
        "no argument",
        "procCTRL",
        "processinfo_CTRLscreen()");




    // stream ctrl

    RegisterCLIcommand(
        "streamCTRL",
        __FILE__,
        streamCTRL_CTRLscreen__cli,
        "stream control screen",
        "no argument",
        "streamCTRL",
        "streamCTRL_CTRLscreen()");



    // FPS
    RegisterCLIcommand(
        "fpsload",
        __FILE__,
        function_parameter_structure_load__cli,
        "Load function parameter struct (FPS)",
        "<fpsname>",
        "fpsload imanalyze",
        "long function_parameter_structure_load(char *fpsname)");

    RegisterCLIcommand(
        "fpsCTRL",
        __FILE__,
        functionparameter_CTRLscreen__cli,
        "function parameters control screen",
        "no arg",
        "fpsCTRL fpsname",
        "int_fast8_t functionparameter_CTRLscreen(char *fpsname)");









    RegisterCLIcommand(
        "usleep",
        __FILE__,
        milk_usleep__cli,
        "usleep",
        "<us>",
        "usleep 1000",
        "usleep(long tus)");



    //  init_modules();
    // printf("TEST   %s  %ld   data.image[4934].used = %d\n", __FILE__, __LINE__, data.image[4934].used);

    if(data.quiet == 0)
    {
        printf("        Loaded %ld modules, %u commands\n", data.NBmodule, data.NBcmd);
        printf("        \n");
    }
}




static void runCLI_free()
{
#ifndef DATA_STATIC_ALLOC
    // Free
    DEBUG_TRACEPOINT("free data.image");
    free(data.image);

    DEBUG_TRACEPOINT("free data.variable");
    free(data.variable);

    DEBUG_TRACEPOINT("free data.fps");
    if(data.fpsarray == NULL)
    {
        printf("NULL pointer\n");
    }
    else
    {
        free(data.fpsarray);
    }



#endif
    //  free(data.cmd);
    DEBUG_TRACEPOINT("free data.rndgen");
    gsl_rng_free(data.rndgen);
}









int user_function()
{
    printf("-");
    fflush(stdout);
    printf("-");
    fflush(stdout);

    return(0);
}


void fnExit1(void)
{
    //
}











static int command_line_process_options(
    int argc,
    char **argv
)
{
    int option_index = 0;
    struct sched_param schedpar;
    char command[200];


    static struct option long_options[] =
    {
        /* These options set a flag. */
        {"verbose", no_argument,       &Verbose, 1},
        {"listimf", no_argument,       &Listimfile, 1},
        /* These options don't set a flag.
        We distinguish them by their indices. */
        {"help",        no_argument,       0, 'h'},
        {"version",     no_argument,       0, 'v'},
        {"info",        no_argument,       0, 'i'},
        {"overwrite",   no_argument,       0, 'o'},
        {"idle",        no_argument,       0, 'e'},
        {"fifoflag",    no_argument,       0, 'f'},
        {"debug",       required_argument, 0, 'd'},
        {"mmon",        required_argument, 0, 'm'},
        {"pname",       required_argument, 0, 'n'},
        {"priority",    required_argument, 0, 'p'},
        {"fifoname",    required_argument, 0, 'F'},
        {"startup",     required_argument, 0, 's'},
        {0, 0, 0, 0}
    };



    data.fifoON = 0; // default
    data.processnameflag = 0; // default

    while(1)
    {
        int c;

        c = getopt_long(argc, argv, "hvidoe:m:n:p:fF:s:",
                        long_options, &option_index);

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
            printf("%s   %s\n",  data.package_name, data.package_version);
            exit(EXIT_SUCCESS);
            break;

        case 'i':
            printInfo();
            exit(EXIT_SUCCESS);
            break;

        case 'd':
            data.Debug = atoi(optarg);
            printf("Debug = %d\n", data.Debug);
            break;

        case 'o':
            puts("CAUTION - WILL OVERWRITE EXISTING FITS FILES\n");
            data.overwrite = 1;
            break;

        case 'e':
            printf("Idle mode: only runs process when X is idle (pid %ld)\n",
                   (long) getpid());
            sprintf(command, "runidle %ld > /dev/null &\n", (long) getpid());
            if(system(command) != 0)
            {
                PRINT_ERROR("system() returns non-zero value");
            }
            break;

        case 'm':
            printf("Starting memory monitor on '%s'\n", optarg);
            memory_monitor(optarg);
            break;

        case 'n':
            if(data.quiet == 0)
            {
                printf("process name '%s'\n", optarg);
            }
            strcpy(data.processname, optarg);
            data.processnameflag = 1; // this process has been named

            // extract first word before '.'
            // it can be used to name processinfo and function parameter structure for process
            char tmpstring[200];
            strcpy(tmpstring, data.processname);
            char *firstword;
            firstword = strtok(tmpstring, ".");
            strcpy(data.processname0, firstword);
            prctl(PR_SET_NAME, optarg, 0, 0, 0);
            break;

        case 'p':
            schedpar.sched_priority = atoi(optarg);
            printf("RUNNING WITH RT PRIORITY = %d\n", schedpar.sched_priority);
#ifndef __MACH__

            if(seteuid(data.euid) != 0) //This goes up to maximum privileges
            {
                PRINT_ERROR("seteuid() returns non-zero value");
            }
            sched_setscheduler(0, SCHED_FIFO,
                               &schedpar); //other option is SCHED_RR, might be faster

            if(seteuid(data.ruid) != 0) //Go back to normal privileges
            {
                PRINT_ERROR("seteuid() returns non-zero value");
            }
#endif
            break;

        case 'f':
            if(data.quiet == 0)
            {
                printf("fifo input ON\n");
            }
            data.fifoON = 1;
            break;

        case 'F':
            printf("using input fifo '%s'\n", optarg);
            data.fifoON = 1;
            sprintf(data.fifoname, "%s", optarg);
            printf("FIFO NAME = %s\n", data.fifoname);
            break;

        case 's':
            strcpy(CLIstartupfilename, optarg);
            if(data.quiet == 0)
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


    return RETURN_SUCCESS;

}
