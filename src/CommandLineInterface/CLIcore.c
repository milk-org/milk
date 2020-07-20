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
#include "CommandLineInterface/CLIcore.h"
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

//#include "initmodules.h"

#include "ImageStreamIO/ImageStreamIO.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"


#include "CommandLineInterface/CLIcore_UI.h"
#include "CommandLineInterface/CLIcore_help.h"
#include "CommandLineInterface/CLIcore_memory.h"
#include "CommandLineInterface/CLIcore_modules.h"
#include "CommandLineInterface/CLIcore_setSHMdir.h"

#include "CommandLineInterface/calc.h"
#include "CommandLineInterface/calc_bison.h"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */







/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */





extern void yy_scan_string(const char *);
extern int yylex_destroy(void);




/*-----------------------------------------
*       Globals exported to all modules
*/

pid_t CLIPID;


uint8_t TYPESIZE[32];

int C_ERRNO;





int Verbose = 0;
int Listimfile = 0;



//double CFITSVARRAY[SZ_CFITSVARRAY];
//long CFITSVARRAY_LONG[SZ_CFITSVARRAY];
//int ECHO;

char CLIstartupfilename[200] = "CLIstartup.txt";



// fifo input
static int fifofd;
static fd_set cli_fdin_set;




/*-----------------------------------------
*       Forward References
*/
int user_function();
void fnExit1(void);
static void runCLI_data_init();
static void runCLI_free();









static int command_line_process_options(int argc, char **argv);


/// CLI commands
static int exitCLI();








/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name CLIcore functions */




static void set_terminal_echo_on()
{
    // Terminal settings
    struct termios termInfo;
    if(tcgetattr(0, &termInfo) == -1)
    {
        perror("tcgetattr");
        exit(1);
    }
    termInfo.c_lflag |= ECHO;  /* turn on ECHO */
    tcsetattr(0, TCSADRAIN, &termInfo);
}

/// signal catching


errno_t set_signal_catch()
{
    // catch signals for clean exit
    if(sigaction(SIGTERM, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGTERM\n");
    }

    if(sigaction(SIGINT, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGINT\n");
    }

    if(sigaction(SIGABRT, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGABRT\n");
    }

    if(sigaction(SIGBUS, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGBUS\n");
    }

    if(sigaction(SIGSEGV, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGSEGV\n");
    }

    if(sigaction(SIGHUP, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGHUP\n");
    }

    if(sigaction(SIGPIPE, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGPIPE\n");
    }

    return RETURN_SUCCESS;
}



static void fprintf_stdout(FILE *f, char const *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
}








/**
 * @brief Write entry into debug log
 *
 *
 */
errno_t write_process_log()
{
    FILE *fplog;
    char fname[STRINGMAXLEN_FILENAME];
    pid_t thisPID;

    thisPID = getpid();
    WRITE_FILENAME(fname, "logreport.%05d.log", thisPID);

    struct tm *uttime;
    time_t tvsec0;


    fplog = fopen(fname, "a");
    if(fplog != NULL)
    {
        struct timespec tnow;
        //        time_t now;
        clock_gettime(CLOCK_REALTIME, &tnow);
        tvsec0 = tnow.tv_sec;
        uttime = gmtime(&tvsec0);
        fprintf(fplog, "%04d%02d%02dT%02d%02d%02d.%09ld ",
                1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday, uttime->tm_hour,
                uttime->tm_min,  uttime->tm_sec, tnow.tv_nsec);

        fprintf(fplog, "    File    : %s\n", data.testpoint_file);
        fprintf(fplog, "    Function: %s\n", data.testpoint_func);
        fprintf(fplog, "    Line    : %d\n", data.testpoint_line);
        fprintf(fplog, "    Message : %s\n", data.testpoint_msg);
        fprintf(fplog, "\n");

        fclose(fplog);
    }

    return RETURN_SUCCESS;
}



/**
 * @brief Write to disk a process report
 *
 * This function is typically called upon crash to help debugging
 *
 * errortypestring describes the type of error or reason to issue report
 *
 */
errno_t write_process_exit_report(
    const char *restrict errortypestring
)
{
    FILE *fpexit;
    char fname[STRINGMAXLEN_FILENAME];
    pid_t thisPID;
    long fd_counter = 0;

    thisPID = getpid();
    
    WRITE_FILENAME(fname, "exitreport-%s.%05d.log", errortypestring, thisPID);

    printf("EXIT CONDITION < %s >: See report in file %s\n", errortypestring,
           fname);
    printf("    File    : %s\n", data.testpoint_file);
    printf("    Function: %s\n", data.testpoint_func);
    printf("    Line    : %d\n", data.testpoint_line);
    printf("    Message : %s\n", data.testpoint_msg);
    fflush(stdout);

    struct tm *uttime;
    time_t tvsec0, tvsec1;


    fpexit = fopen(fname, "w");
    if(fpexit != NULL)
    {
        fprintf_stdout(fpexit, "PID : %d\n", thisPID);

        struct timespec tnow;
        //        time_t now;
        clock_gettime(CLOCK_REALTIME, &tnow);
        tvsec0 = tnow.tv_sec;
        uttime = gmtime(&tvsec0);
        fprintf_stdout(fpexit, "Time: %04d%02d%02dT%02d%02d%02d.%09ld\n\n",
                       1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday, uttime->tm_hour,
                       uttime->tm_min,  uttime->tm_sec, tnow.tv_nsec);

        fprintf_stdout(fpexit, "Last encountered test point\n");
        tvsec1 = data.testpoint_time.tv_sec;
        uttime = gmtime(&tvsec1);
        fprintf_stdout(fpexit, "    Time    : %04d%02d%02dT%02d%02d%02d.%09ld\n",
                       1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday, uttime->tm_hour,
                       uttime->tm_min,  uttime->tm_sec, data.testpoint_time.tv_nsec);

        double timediff = 1.0 * (tvsec0 - tvsec1) + 1.0e-9 * (tnow.tv_nsec -
                          data.testpoint_time.tv_nsec);
        fprintf_stdout(fpexit, "              %.9f sec ago\n", timediff);

        fprintf_stdout(fpexit, "    File    : %s\n", data.testpoint_file);
        fprintf_stdout(fpexit, "    Function: %s\n", data.testpoint_func);
        fprintf_stdout(fpexit, "    Line    : %d\n", data.testpoint_line);
        fprintf_stdout(fpexit, "    Message : %s\n", data.testpoint_msg);
        fprintf_stdout(fpexit, "\n");

        // Check open file descriptors
        struct rlimit rlimits;
        int max_fd_number;

        fprintf_stdout(fpexit, "File descriptors\n");
        getrlimit(RLIMIT_NOFILE, &rlimits);
        max_fd_number = getdtablesize();
        fprintf_stdout(fpexit, "    max_fd_number  : %d\n", max_fd_number);
        fprintf_stdout(fpexit, "    rlim_cur       : %lu\n", rlimits.rlim_cur);
        fprintf_stdout(fpexit, "    rlim_max       : %lu\n", rlimits.rlim_max);
        for(int i = 0; i <= max_fd_number; i++)
        {
            struct stat stats;

            fstat(i, &stats);
            if(errno != EBADF)
            {
                fd_counter++;
            }
        }
        fprintf_stdout(fpexit, "    Open files     : %ld\n", fd_counter);

        fclose(fpexit);
    }

    return RETURN_SUCCESS;
}



/**
 * @brief Signal handler
 *
 *
 */
void sig_handler(
    int signo
)
{
    switch(signo)
    {

        case SIGINT:
            printf("PID %d sig_handler received SIGINT\n", CLIPID);
            data.signal_INT = 1;
            break;

        case SIGTERM:
            printf("PID %d sig_handler received SIGTERM\n", CLIPID);
            data.signal_TERM = 1;
            set_terminal_echo_on();
            exit(EXIT_FAILURE);
            break;

        case SIGUSR1:
            printf("PID %d sig_handler received SIGUSR1\n", CLIPID);
            data.signal_USR1 = 1;
            break;

        case SIGUSR2:
            printf("PID %d sig_handler received SIGUSR2\n", CLIPID);
            data.signal_USR2 = 1;
            break;

        case SIGBUS: // exit program after SIGSEGV
            printf("PID %d sig_handler received SIGBUS \n", CLIPID);
            write_process_exit_report("SIGBUS");
            data.signal_BUS = 1;
            set_terminal_echo_on();
            exit(EXIT_FAILURE);
            break;

        case SIGABRT:
            printf("PID %d sig_handler received SIGABRT\n", CLIPID);
            write_process_exit_report("SIGABRT");
            data.signal_ABRT = 1;
            set_terminal_echo_on();
            exit(EXIT_FAILURE);
            break;

        case SIGSEGV: // exit program after SIGSEGV
            printf("PID %d sig_handler received SIGSEGV\n", CLIPID);
            write_process_exit_report("SIGSEGV");
            data.signal_SEGV = 1;
            set_terminal_echo_on();
            exit(EXIT_FAILURE);
            break;

        case SIGHUP:
            printf("PID %d sig_handler received SIGHUP\n", CLIPID);
            data.signal_HUP = 1;
            break;

        case SIGPIPE:
            printf("PID %d sig_handler received SIGPIPE\n", CLIPID);
            data.signal_PIPE = 1;
            break;
    }
}




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

	if(data.quiet == 0) {
		printf("Closing PID %ld (prompt process)\n", (long) getpid());
	}
    //    exit(0);
    data.CLIloopON = 0; // stop CLI loop

    return RETURN_SUCCESS;
}







static errno_t help()
{

    EXECUTE_SYSTEM_COMMAND("more %s/src/CommandLineInterface/doc/help.txt",
                           data.sourcedir);

    return RETURN_SUCCESS;
}


static errno_t helpreadline()
{

    EXECUTE_SYSTEM_COMMAND("more %s/src/CommandLineInterface/doc/helpreadline.md",
                           data.sourcedir);

    return RETURN_SUCCESS;
}


static errno_t help_cmd()
{

    if((data.cmdargtoken[1].type == 3) || (data.cmdargtoken[1].type == 4)
            || (data.cmdargtoken[1].type == 5))
    {
        help_command(data.cmdargtoken[1].val.string);
    }
    else
    {
        list_commands();
    }

    return RETURN_SUCCESS;
}



static errno_t help_module()
{

    if(data.cmdargtoken[1].type == 3)
    {
        list_commands_module(data.cmdargtoken[1].val.string);
    }
    else
    {
        long i;
        printf("\n");
        printf("%2s  %10s %32s %10s %7s    %20s %s\n", "#", "shortname", "Name", "Package", "Version", "last compiled", 
               "description");
        printf("--------------------------------------------------------------------------------------------------------------\n");
        for(i = 0; i < data.NBmodule; i++)
        {
            printf("%2ld %10s \033[1m%32s\033[0m %10s %2d.%02d.%02d    %11s %8s  %s\n", 
					i, data.module[i].shortname,
                   data.module[i].name,
                   data.module[i].package,
                   data.module[i].versionmajor, data.module[i].versionminor, data.module[i].versionpatch,
                   data.module[i].datestring, data.module[i].timestring,
                   data.module[i].info);
        }
        printf("-------------------------------------------------------------------------------------------------------\n");
        printf("\n");
    }

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
    if(
        (CLI_checkarg(1, CLIARG_LONG) == 0) &&
        (CLI_checkarg(2, CLIARG_STR ) == 0) &&
        (CLI_checkarg(3, CLIARG_STR ) == 0)
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
    TYPESIZE[_DATATYPE_UINT8]                  = SIZEOF_DATATYPE_UINT8;
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
    TYPESIZE[_DATATYPE_COMPLEX_DOUBLE]         = SIZEOF_DATATYPE_COMPLEX_DOUBLE;
    //    TYPESIZE[_DATATYPE_EVENT_UI8_UI8_UI16_UI8] = SIZEOF_DATATYPE_EVENT_UI8_UI8_UI16_UI8;




    // get PID and write it to shell env variable MILK_CLI_PID
    CLIPID = getpid();
    if(data.quiet == 0)
    {
        printf("    CLI PID = %d\n", (int) CLIPID);

        EXECUTE_SYSTEM_COMMAND("echo -n \"    \"; cat /proc/%d/status | grep Cpus_allowed_list",
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
        printf("    Running with openMP, max threads = %d  (OMP_NUM_THREADS)\n",
               omp_get_max_threads());
    }
# else
    if(data.quiet == 0)
    {
        printf("    Compiled without openMP\n");
    }
# endif

# ifdef _OPENACC
    int openACC_devtype = acc_get_device_type();
    if(data.quiet == 0)
    {
        printf("    Running with openACC version %d.  %d device(s), type %d\n",
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
    data.NBKEWORD_DFT      = 10; // allocate memory for 10 keyword per image
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



    // initialize fifo to process name
    // default fifo name
    sprintf(data.fifoname, "%s.fifo.%07d", data.processname, getpid());

    // Get command-line options
    command_line_process_options(argc, argv);


    DEBUG_TRACEPOINT("CLI start");




    // initialize readline
    // Tell readline to use custom completion function
    rl_attempted_completion_function = CLI_completion;
    rl_initialize();



    data.progStatus = 1;



    printf("\n");




    // LOAD MODULES (shared objects)
    load_module_shared_ALL();

    // load other libs specified by environment variable CLI_ADD_LIBS
    char *CLI_ADD_LIBS = getenv("CLI_ADD_LIBS");
    if(CLI_ADD_LIBS != NULL)
    {
        if(data.quiet == 0)
        {
            printf(" [ CLI_ADD_LIBS ] '%s'\n", CLI_ADD_LIBS);
        }

        char *libname;
        libname = strtok(CLI_ADD_LIBS, " ,;");

        while(libname != NULL)
        {
            printf("--- CLI Adding library: %s\n", libname);
            load_sharedobj(libname);
            libname = strtok(NULL, " ,;");
        }
        printf("\n");
    }
    else
    {
        if(data.quiet == 0)
        {
            printf(" [ CLI_ADD_LIBS ] not set\n");
        }
    }










    // Initialize data control block
    runCLI_data_init();


    // initialize readline
    rl_callback_handler_install(prompt, (rl_vcpfunc_t *) &rl_cb_linehandler);


    // fifo
    fdmax = fileno(stdin);
    if(data.fifoON == 1)
    {
		if(data.quiet == 0) {
        printf("Creating fifo %s\n", data.fifoname);
	}
        mkfifo(data.fifoname, 0666);
        fifofd = open(data.fifoname, O_RDWR | O_NONBLOCK);
        if(fifofd == -1)
        {
            perror("open");
            return EXIT_FAILURE;
        }
        if(fifofd > fdmax)
        {
            fdmax = fifofd;
        }
    }


    C_ERRNO = 0; // initialize C error variable to 0 (no error)



    data.CLIloopON = 1; // start CLI loop


    while(data.CLIloopON == 1)
    {
        FILE *fp;

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

        compute_image_memory(data);
        compute_nb_image(data);

        // If fifo is on and file CLIstatup.txt exists, load it
        if(initstartup == 0)
        {
            if(data.fifoON == 1)
            {
				EXECUTE_SYSTEM_COMMAND("file %s", CLIstartupfilename); //TEST
                EXECUTE_SYSTEM_COMMAND("cat %s", CLIstartupfilename); //TEST
                EXECUTE_SYSTEM_COMMAND("cat %s > %s 2> /dev/null", CLIstartupfilename,
                                       data.fifoname);
                                       
                if(data.quiet == 0) { 
                printf("[%s -> %s]\n", CLIstartupfilename, data.fifoname);
                printf("IMPORTING FILE %s ... \n", CLIstartupfilename);
				}
            }
		}
        initstartup = 1;


        // -------------------------------------------------------------
        //                 get user input
        // -------------------------------------------------------------
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



        while((data.CLIexecuteCMDready == 0) && (data.CLIloopON == 1))
        {
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
                            CLI_execute_line();
                            printf("%s", prompt);
                            fflush(stdout);
                            break;
                        }
                    }
                    blockCLIinput = 1; // keep blocking input while fifo is not empty
                }
            }

            if(blockCLIinput == 0)  // revert to default mode
                if(FD_ISSET(fileno(stdin), &cli_fdin_set))
                {
                    rl_callback_read_char();
                }
        }
        data.CLIexecuteCMDready = 0;


        //TEST data.CLIloopON = 0;
    }
    DEBUG_TRACEPOINT("exit from CLI loop");

    // clear all images and variables
    clearall();


    runCLI_free();


	#if ( RL_READLINE_VERSION > 0x602 )
	rl_clear_history();
	#endif

    rl_callback_handler_remove();

    DEBUG_TRACEPOINT("exit from runCLI function");

    return RETURN_SUCCESS;
}



























/*^-----------------------------------------------------------------------------
|  Initialization the "data" structure
|
|
|
|
+-----------------------------------------------------------------------------*/
void runCLI_data_init()
{

    long tmplong;
    //  int i;
    struct timeval t1;


    /* initialization of the data structure
     */
    data.NB_MAX_IMAGE    = STATIC_NB_MAX_IMAGE;
    data.NB_MAX_VARIABLE = STATIC_NB_MAX_VARIABLE;
    data.NB_MAX_FPS      = 100;
    data.INVRANDMAX      = 1.0 / RAND_MAX;

    // do not remove files when delete command on SHM
    data.rmSHMfile       = 0;

    // initialize modules
    data.NB_MAX_MODULE = DATA_NB_MAX_MODULE;
    //  data.module = (MODULE*) malloc(sizeof(MODULE)*data.NB_MAX_MODULE);


    // initialize commands
    data.NB_MAX_COMMAND = 5000;
    if(data.Debug > 0)
    {
        printf("Allocating cmd array : %ld\n", sizeof(CMD)*data.NB_MAX_COMMAND);
        fflush(stdout);
    }

    data.NB_MAX_COMMAND = DATA_NB_MAX_COMMAND;
    // data.cmd = (CMD*) malloc(sizeof(CMD)*data.NB_MAX_COMMAND);
    //  data.NBcmd = 0;

    data.cmdNBarg = 0;



    // Allocate data.image

#ifdef DATA_STATIC_ALLOC
    // image static allocation mode
    data.NB_MAX_IMAGE = STATIC_NB_MAX_IMAGE;
    printf("STATIC ALLOCATION mode: set data.NB_MAX_IMAGE      = %5ld\n",
           data.NB_MAX_IMAGE);
#else
    data.image           = (IMAGE *) calloc(data.NB_MAX_IMAGE, sizeof(IMAGE));
    if(data.image == NULL)
    {
        PRINT_ERROR("Allocation of data.image has failed - exiting program");
        exit(1);
    }
    if(data.Debug > 0)
    {
        printf("Allocation of data.image completed %p\n", data.image);
        fflush(stdout);
    }
#endif

    for(long i = 0; i < data.NB_MAX_IMAGE; i++)
    {
        data.image[i].used = 0;
    }




    // Allocate data.variable

#ifdef DATA_STATIC_ALLOC
    // variable static allocation mode
    data.NB_MAX_VARIABLE = STATIC_NB_MAX_VARIABLE;
    printf("STATIC ALLOCATION mode: set data.NB_MAX_VARIABLE   = %5ld\n",
           data.NB_MAX_VARIABLE);
#else
    data.variable = (VARIABLE *) calloc(data.NB_MAX_VARIABLE, sizeof(VARIABLE));
    if(data.variable == NULL)
    {
        PRINT_ERROR("Allocation of data.variable has failed - exiting program");
        exit(1);
    }

    data.image[0].used   = 0;
    data.image[0].shmfd  = -1;
    tmplong              = data.NB_MAX_VARIABLE;
    data.NB_MAX_VARIABLE = data.NB_MAX_VARIABLE + NB_VARIABLES_BUFFER_REALLOC ;


    data.variable = (VARIABLE *) realloc(data.variable,
                                         data.NB_MAX_VARIABLE * sizeof(VARIABLE));
    for(long i = tmplong; i < data.NB_MAX_VARIABLE; i++)
    {
        data.variable[i].used = 0;
        data.variable[i].type = 0; /** defaults to floating point type */
    }

    if(data.variable == NULL)
    {
        PRINT_ERROR("Reallocation of data.variable has failed - exiting program");
        exit(1);
    }
#endif





	// Allocate data.fps
	data.fps = malloc(sizeof(FUNCTION_PARAMETER_STRUCT) * data.NB_MAX_FPS);
    // Initialize file descriptors to -1
    //
    for(int fpsindex = 0; fpsindex < data.NB_MAX_FPS; fpsindex++)
    {
        data.fps[fpsindex].SMfd = -1;
    }




    create_variable_ID("_PI", 3.14159265358979323846264338328);
    create_variable_ID("_e", exp(1));
    create_variable_ID("_gamma", 0.5772156649);
    create_variable_ID("_c", 299792458.0);
    create_variable_ID("_h", 6.626075540e-34);
    create_variable_ID("_k", 1.38065812e-23);
    create_variable_ID("_pc", 3.0856776e16);
    create_variable_ID("_ly", 9.460730472e15);
    create_variable_ID("_AU", 1.4959787066e11);


    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);
    //	printf("RAND: %ld\n", t1.tv_usec * t1.tv_sec);
    //  srand(time(NULL));


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
        "readfps",
        __FILE__,
        function_parameter_structure_load__cli,
        "Read function parameter struct",
        "<fpsname>",
        "readfps imanalyze",
        "long function_parameter_structure_load(char *fpsname)");



	 RegisterCLIcommand(
        "fparamCTRL",
        __FILE__,
        functionparameter_CTRLscreen__cli,
        "function parameters control screen",
        "no arg",
        "fparamCTRL fpsname",
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
    free(data.image);
    free(data.variable);
    free(data.fps);
#endif
    //  free(data.cmd);
    gsl_rng_free(data.rndgen);
}












/*^-----------------------------------------------------------------------------
|
|
|
|
|
+-----------------------------------------------------------------------------*/
int user_function()
{
    printf("-");
    fflush(stdout);
    printf("-");
    fflush(stdout);

    return(0);
}

/*^-----------------------------------------------------------------------------
|
|
|
|
|
+-----------------------------------------------------------------------------*/
void fnExit1(void)
{
    //
}
















/*^-----------------------------------------------------------------------------
| static PF
| command_line  : parse unix command line options.
|
|   int argc    :
|   char **argv :
|
|   TO DO : allow option values. eg: debug=3
+-----------------------------------------------------------------------------*/
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
				if(data.quiet == 0) {
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
				if(data.quiet == 0) {
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
                if(data.quiet == 0) {
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











// check that input CLI argument matches required argument type

int CLI_checkarg0(int argnum, int argtype, int errmsg)
{
    int rval; // 0 if OK, 1 if not
    long IDv;

    rval = 2;

    switch(argtype)
    {

        case 1:  // should be floating point
            switch(data.cmdargtoken[argnum].type)
            {
                case 1:
                    rval = 0;
                    break;
                case 2: // convert long to float
                    if(data.Debug > 0)
                    {
                        printf("Converting arg %d to floating point number\n", argnum);
                    }
                    data.cmdargtoken[argnum].val.numf = (double) data.cmdargtoken[argnum].val.numl;
                    data.cmdargtoken[argnum].type = 1;
                    rval = 0;
                    break;
                case 3:
                    IDv = variable_ID(data.cmdargtoken[argnum].val.string);
                    if(IDv == -1)
                    {
                        if(errmsg == 1)
                        {
                            printf("arg %d is string (=\"%s\"), but should be integer\n", argnum,
                                   data.cmdargtoken[argnum].val.string);
                        }
                        rval = 1;
                    }
                    else
                    {
                        switch(data.variable[IDv].type)
                        {
                            case 0: // double
                                data.cmdargtoken[argnum].val.numf = data.variable[IDv].value.f;
                                data.cmdargtoken[argnum].type = 1;
                                rval = 0;
                                break;
                            case 1: // long
                                data.cmdargtoken[argnum].val.numf = 1.0 * data.variable[IDv].value.l;
                                data.cmdargtoken[argnum].type = 1;
                                rval = 0;
                                break;
                            default:
                                if(errmsg == 1)
                                {
                                    printf("arg %d is string (=\"%s\"), but should be integer\n", argnum,
                                           data.cmdargtoken[argnum].val.string);
                                }
                                rval = 1;
                                break;
                        }
                    }
                    break;
                case 4:
                    if(errmsg == 1)
                    {
                        printf("arg %d is image (=\"%s\"), but should be floating point number\n",
                               argnum, data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 5:
                    if(errmsg == 1)
                    {
                        printf("arg %d is command (=\"%s\"), but should be floating point number\n",
                               argnum, data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 6:
                    data.cmdargtoken[argnum].val.numf = atof(data.cmdargtoken[argnum].val.string);
                    data.cmdargtoken[argnum].type = 1;
                    rval = 0;
                    break;
            }
            break;

        case 2:  // should be integer
            switch(data.cmdargtoken[argnum].type)
            {
                case 1:
                    if(errmsg == 1)
                    {
                        printf("converting floating point arg %d to integer\n", argnum);
                    }
                    data.cmdargtoken[argnum].val.numl = (long)(data.cmdargtoken[argnum].val.numf +
                                                        0.5);
                    data.cmdargtoken[argnum].type = 2;
                    rval = 0;
                    break;
                case 2:
                    rval = 0;
                    break;
                case 3:
                    IDv = variable_ID(data.cmdargtoken[argnum].val.string);
                    if(IDv == -1)
                    {
                        if(errmsg == 1)
                        {
                            printf("arg %d is string (=\"%s\"), but should be integer\n", argnum,
                                   data.cmdargtoken[argnum].val.string);
                        }
                        rval = 1;
                    }
                    else
                    {
                        switch(data.variable[IDv].type)
                        {
                            case 0: // double
                                data.cmdargtoken[argnum].val.numl = (long)(data.variable[IDv].value.f);
                                data.cmdargtoken[argnum].type = 2;
                                rval = 0;
                                break;
                            case 1: // long
                                data.cmdargtoken[argnum].val.numl = data.variable[IDv].value.l;
                                data.cmdargtoken[argnum].type = 2;
                                rval = 0;
                                break;
                            default:
                                if(errmsg == 1)
                                {
                                    printf("arg %d is string (=\"%s\"), but should be integer\n", argnum,
                                           data.cmdargtoken[argnum].val.string);
                                }
                                rval = 1;
                                break;
                        }
                    }
                    break;
                case 4:
                    if(errmsg == 1)
                    {
                        printf("arg %d is image (=\"%s\"), but should be integer\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 5:
                    if(errmsg == 1)
                    {
                        printf("arg %d is command (=\"%s\"), but should be integer\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
            }
            break;

        case 3:  // should be string, but not image
            switch(data.cmdargtoken[argnum].type)
            {
                case 1:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be string\n", argnum);
                    }
                    rval = 1;
                    break;
                case 2:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be string\n", argnum);
                    }
                    rval = 1;
                    break;
                case 3:
                    rval = 0;
                    break;
                case 4:
                    if(errmsg == 1)
                    {
                        printf("arg %d is existing image (=\"%s\"), but should be string\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 5:
                    printf("arg %d is command (=\"%s\"), but should be string\n", argnum,
                           data.cmdargtoken[argnum].val.string);
                    rval = 1;
                    break;
                case 6:
                    rval = 0;
                    break;
            }
            break;

        case 4:  // should be existing image
            switch(data.cmdargtoken[argnum].type)
            {
                case 1:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 2:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 3:
                    if(errmsg == 1)
                    {
                        printf("arg %d is string, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 4:
                    rval = 0;
                    break;
                case 5:
                    if(errmsg == 1)
                    {
                        printf("arg %d is command (=\"%s\"), but should be image\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 6:
                    rval = 0;
                    break;
            }
            break;
        case 5: // should be string (image or not)
            switch(data.cmdargtoken[argnum].type)
            {
                case 1:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be string or image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 2:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be string or image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 3:
                    rval = 0;
                    break;
                case 4:
                    rval = 0;
                    break;
                case 5:
                    if(errmsg == 1)
                    {
                        printf("arg %d is command (=\"%s\"), but should be image\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 6:
                    rval = 0;
                    break;
            }
            break;

    }


    if(rval == 2)
    {
        if(errmsg == 1)
        {
            printf("arg %d: wrong arg type %d :  %d\n", argnum, argtype,
                   data.cmdargtoken[argnum].type);
        }
        rval = 1;
    }


    return rval;
}



// check that input CLI argument matches required argument type
int CLI_checkarg(int argnum, int argtype)
{
    int rval;

    rval = CLI_checkarg0(argnum, argtype, 1);
    return rval;
}

// check that input CLI argument matches required argument type - do not print error message
int CLI_checkarg_noerrmsg(int argnum, int argtype)
{
    int rval;

    rval = CLI_checkarg0(argnum, argtype, 0);
    return rval;
}






