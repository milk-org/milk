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
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <stddef.h> // offsetof()
#include <sys/resource.h> // getrlimit
#include <termios.h>

//#include <pthread_np.h>

#ifdef __MACH__
#include <mach/mach_time.h>
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
static int clock_gettime(int clk_id, struct mach_timespec *t)
{
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t time;
    time = mach_absolute_time();
    double nseconds = ((double)time * (double)timebase.numer) / ((
                          double)timebase.denom);
    double seconds = ((double)time * (double)timebase.numer) / ((
                         double)timebase.denom * 1e9);
    t->tv_sec = seconds;
    t->tv_nsec = nseconds;
    return 0;
}
#else
#include <sys/time.h>
#endif


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
#include "00CORE/00CORE.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"



#include "CommandLineInterface/calc.h"
#include "CommandLineInterface/calc_bison.h"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */



#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define KRES  "\033[0m"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */


int DLib_index;
void *DLib_handle[1000];


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


char line[500];
int rlquit = false;
int CLIexecuteCMDready = 0;

//double CFITSVARRAY[SZ_CFITSVARRAY];
//long CFITSVARRAY_LONG[SZ_CFITSVARRAY];
//int ECHO;

char calctmpimname[200];
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


// readline auto-complete

int CLImatchMode = 0;

static char **CLI_completion(const char *, int, int);
char *CLI_generator(const char *, int);
char *dupstr(char *);
void *xmalloc(int);





static errno_t memory_re_alloc();

static int command_line_process_options(int argc, char **argv);


/// CLI commands
static int exitCLI();
static int help();


static errno_t list_commands();
static errno_t list_commands_module(const char *restrict modulename);
static errno_t load_sharedobj(const char *restrict libname);
static errno_t load_module_shared(const char *restrict modulename);
static errno_t load_module_shared_ALL();
static errno_t help_command(const char *restrict cmdkey);









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
    char fname[200];
    pid_t thisPID;

    thisPID = getpid();
    sprintf(fname, "logreport.%05d.log", thisPID);

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
    char fname[200];
    pid_t thisPID;
    long fd_counter = 0;

    thisPID = getpid();
    sprintf(fname, "exitreport-%s.%05d.log", errortypestring, thisPID);

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
 * Signal handler
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
            printf("sig_handler received SIGINT\n");
            data.signal_INT = 1;
            break;

        case SIGTERM:
            printf("sig_handler received SIGTERM\n");
            data.signal_TERM = 1;
            set_terminal_echo_on();
            exit(EXIT_FAILURE);
            break;

        case SIGUSR1:
            printf("sig_handler received SIGUSR1\n");
            data.signal_USR1 = 1;
            break;

        case SIGUSR2:
            printf("sig_handler received SIGUSR2\n");
            data.signal_USR2 = 1;
            break;

        case SIGBUS: // exit program after SIGSEGV
            printf("sig_handler received SIGBUS \n");
            write_process_exit_report("SIGBUS");
            data.signal_BUS = 1;
            set_terminal_echo_on();
            exit(EXIT_FAILURE);
            break;

        case SIGABRT:
            printf("sig_handler received SIGABRT\n");
            write_process_exit_report("SIGABRT");
            data.signal_ABRT = 1;
            set_terminal_echo_on();
            exit(EXIT_FAILURE);
            break;

        case SIGSEGV: // exit program after SIGSEGV
            printf("sig_handler received SIGSEGV\n");
            write_process_exit_report("SIGSEGV");
            data.signal_SEGV = 1;
            set_terminal_echo_on();
            exit(EXIT_FAILURE);
            break;

        case SIGHUP:
            printf("sig_handler received SIGHUP\n");
            data.signal_HUP = 1;
            break;

        case SIGPIPE:
            printf("sig_handler received SIGPIPE\n");
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

    printf("Closing PID %ld (prompt process)\n", (long) getpid());
    //    exit(0);
    data.CLIloopON = 0; // stop CLI loop

    return RETURN_SUCCESS;
}



static errno_t printInfo()
{
    float f1;
    printf("\n");
    printf("  PID = %d\n", CLIPID);


    printf("--------------- GENERAL ----------------------\n");
    printf("%s  %s\n",  data.package_name, data.package_version);
    printf("IMAGESTRUCT_VERSION %s\n", IMAGESTRUCT_VERSION);
    printf("%s BUILT   %s %s\n", __FILE__, __DATE__, __TIME__);
    printf("\n");
    printf("--------------- SETTINGS ---------------------\n");
    printf("procinfo status = %d\n", data.processinfo);

    if(data.precision == 0)
    {
        printf("Default precision upon startup : float\n");
    }
    if(data.precision == 1)
    {
        printf("Default precision upon startup : double\n");
    }
    printf("sizeof(struct timespec)        = %4ld bit\n",
           sizeof(struct timespec) * 8);
    printf("sizeof(pid_t)                  = %4ld bit\n", sizeof(pid_t) * 8);
    printf("sizeof(short int)              = %4ld bit\n", sizeof(short int) * 8);
    printf("sizeof(int)                    = %4ld bit\n", sizeof(int) * 8);
    printf("sizeof(long)                   = %4ld bit\n", sizeof(long) * 8);
    printf("sizeof(long long)              = %4ld bit\n", sizeof(long long) * 8);
    printf("sizeof(int_fast8_t)            = %4ld bit\n", sizeof(int_fast8_t) * 8);
    printf("sizeof(int_fast16_t)           = %4ld bit\n", sizeof(int_fast16_t) * 8);
    printf("sizeof(int_fast32_t)           = %4ld bit\n", sizeof(int_fast32_t) * 8);
    printf("sizeof(int_fast64_t)           = %4ld bit\n", sizeof(int_fast64_t) * 8);
    printf("sizeof(uint_fast8_t)           = %4ld bit\n", sizeof(uint_fast8_t) * 8);
    printf("sizeof(uint_fast16_t)          = %4ld bit\n",
           sizeof(uint_fast16_t) * 8);
    printf("sizeof(uint_fast32_t)          = %4ld bit\n",
           sizeof(uint_fast32_t) * 8);
    printf("sizeof(uint_fast64_t)          = %4ld bit\n",
           sizeof(uint_fast64_t) * 8);
    printf("sizeof(IMAGE_KEYWORD)          = %4ld bit\n",
           sizeof(IMAGE_KEYWORD) * 8);

    size_t offsetval = 0;
    size_t offsetval0 = 0;

    printf("sizeof(IMAGE_METADATA)         = %4ld bit  = %4zu byte ------------------\n",
           sizeof(IMAGE_METADATA) * 8, sizeof(IMAGE_METADATA));

    offsetval = offsetof(IMAGE_METADATA, version);


    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, name);
    printf("   version                     offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, naxis);
    printf("   name                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, size);
    printf("   naxis                       offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, nelement);
    printf("   size                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, datatype);
    printf("   nelement                    offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, imagetype);
    printf("   datatype                    offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, creationtime);
    printf("   imagetype                   offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);


    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, lastaccesstime);
    printf("   creationtime                offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, atime);
    printf("   lastaccesstime              offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, writetime);
    printf("   atime                       offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, location);
    printf("   writetime                   offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, location);
    printf("   shared                      offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, status);
    printf("   location                    offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, flag);
    printf("   status                      offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, sem);
    printf("   flag                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, sem);
    printf("   logflag                     offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, cnt0);
    printf("   sem                         offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, cnt1);
    printf("   cnt0                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, cnt2);
    printf("   cnt1                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, write);
    printf("   cnt2                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, NBkw);
    printf("   write                       offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, cudaMemHandle);
    printf("   NBkw                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    printf("   cudaMemHandle               offset = %4zu bit  = %4zu byte\n",
           8 * offsetval0, offsetval0);



    printf("sizeof(IMAGE)                  offset = %4zu bit  = %4zu byte ------------------\n",
           sizeof(IMAGE) * 8, sizeof(IMAGE));
    printf("   name                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, name),                      offsetof(IMAGE, name));
    printf("   used                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, used),                      offsetof(IMAGE, used));
    printf("   shmfd                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, shmfd),                     offsetof(IMAGE, shmfd));
    printf("   memsize                     offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, memsize),                   offsetof(IMAGE, memsize));
    printf("   semlog                      offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, semlog),                    offsetof(IMAGE, semlog));
    printf("   md                          offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, md),                        offsetof(IMAGE, md));
    printf("   atimearray                  offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, atimearray),                offsetof(IMAGE, atimearray));
    printf("   writetimearray              offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, writetimearray),            offsetof(IMAGE,
                   writetimearray));
    printf("   flagarray                   offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, flagarray),                 offsetof(IMAGE, flagarray));
    printf("   cntarray                    offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, cntarray),                  offsetof(IMAGE, cntarray));
    printf("   array                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, array),                     offsetof(IMAGE, array));
    printf("   semptr                      offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, semptr),                    offsetof(IMAGE, semptr));
    printf("   kw                          offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, kw),                        offsetof(IMAGE, kw));

    printf("sizeof(IMAGE_KEYWORD)          offset = %4zu bit  = %4zu byte ------------------\n",
           sizeof(IMAGE_KEYWORD) * 8, sizeof(IMAGE_KEYWORD));
    printf("   name                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, name), offsetof(IMAGE_KEYWORD, name));
    printf("   type                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, type), offsetof(IMAGE_KEYWORD, type));
    printf("   value                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, value), offsetof(IMAGE_KEYWORD, value));
    printf("   comment                     offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, comment), offsetof(IMAGE_KEYWORD, comment));

    printf("\n");
    printf("--------------- LIBRARIES --------------------\n");
    printf("READLINE : version %x\n", RL_READLINE_VERSION);
# ifdef _OPENMP
    printf("OPENMP   : Compiled by an OpenMP-compliant implementation.\n");
# endif
    printf("CFITSIO  : version %f\n", fits_get_version(&f1));
    printf("\n");

    printf("--------------- DIRECTORIES ------------------\n");
    printf("CONFIGDIR = %s\n", data.configdir);
    printf("SOURCEDIR = %s\n", data.sourcedir);
    printf("\n");

    printf("--------------- MALLOC INFO ------------------\n");
    malloc_stats();

    printf("\n");

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
        printf("%6s  %36s  %10s  %s\n", "Index", "Module name", "Package",
               "Module description");
        printf("-------------------------------------------------------------------------------------------------------\n");
        for(i = 0; i < data.NBmodule; i++)
        {
            printf("%6ld  %36s  %10s  %s\n", i, data.module[i].name, data.module[i].package,
                   data.module[i].info);
        }
        printf("-------------------------------------------------------------------------------------------------------\n");
        printf("\n");
    }

    return RETURN_SUCCESS;
}



static errno_t load_so()
{
    load_sharedobj(data.cmdargtoken[1].val.string);
    return RETURN_SUCCESS;
}




static errno_t load_module()
{

    if(data.cmdargtoken[1].type == 3)
    {
        load_module_shared(data.cmdargtoken[1].val.string);
        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
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



errno_t cfits_usleep_cli()
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


errno_t functionparameter_CTRLscreen_cli()
{
    if((CLI_checkarg(1, 2) == 0) && (CLI_checkarg(2, 5) == 0)
            && (CLI_checkarg(3, 5) == 0))
    {
        functionparameter_CTRLscreen((uint32_t) data.cmdargtoken[1].val.numl,
                                     data.cmdargtoken[2].val.string, data.cmdargtoken[3].val.string);
        return RETURN_SUCCESS;
    }
    else
    {
        printf("Wrong args (%d)\n", data.cmdargtoken[1].type);
    }
    return RETURN_SUCCESS;
}



errno_t processinfo_CTRLscreen_cli()
{
    return(processinfo_CTRLscreen());
}

errno_t streamCTRL_CTRLscreen_cli()
{
    return(streamCTRL_CTRLscreen());
}





static errno_t CLI_execute_line()
{

    long    i;
    char   *cmdargstring;
    char    str[200];
    FILE   *fp;
    time_t  t;
    struct  tm *uttime;
    struct  timespec *thetime = (struct timespec *)malloc(sizeof(struct timespec));
    char    command[200];

    //
    // If line starts with !, use system()
    //
    if(line[0] == '!')
    {
        line[0] = ' ';
        if(system(line) != 0)
        {
            printERROR(__FILE__, __func__, __LINE__, "system call error");
            exit(4);
        }
        data.CMDexecuted = 1;
    }
    else if(line[0] == '#')
    {
        // do nothing... this is a comment
        data.CMDexecuted = 1;
    }
    else
    {
        // some initialization
        data.parseerror = 0;
        data.calctmp_imindex = 0;
        for(i = 0; i < NB_ARG_MAX; i++)
        {
            data.cmdargtoken[0].type = 0;
        }


        // log command if CLIlogON active
        if(data.CLIlogON == 1)
        {
            t = time(NULL);
            uttime = gmtime(&t);
            clock_gettime(CLOCK_REALTIME, thetime);

            sprintf(data.CLIlogname,
                    "%s/logdir/%04d%02d%02d/%04d%02d%02d_CLI-%s.log",
                    getenv("HOME"),
                    1900 + uttime->tm_year,
                    1 + uttime->tm_mon,
                    uttime->tm_mday,
                    1900 + uttime->tm_year,
                    1 + uttime->tm_mon,
                    uttime->tm_mday,
                    data.processname);

            fp = fopen(data.CLIlogname, "a");
            if(fp == NULL)
            {
                printf("ERROR: cannot log into file %s\n", data.CLIlogname);
                sprintf(command,
                        "mkdir -p %s/logdir/%04d%02d%02d\n",
                        getenv("HOME"),
                        1900 + uttime->tm_year,
                        1 + uttime->tm_mon,
                        uttime->tm_mday);

                if(system(command) != 0)
                {
                    printERROR(__FILE__, __func__, __LINE__, "system() returns non-zero value");
                }
            }
            else
            {
                fprintf(fp,
                        "%04d/%02d/%02d %02d:%02d:%02d.%09ld %10s %6ld %s\n",
                        1900 + uttime->tm_year,
                        1 + uttime->tm_mon,
                        uttime->tm_mday,
                        uttime->tm_hour,
                        uttime->tm_min,
                        uttime->tm_sec,
                        thetime->tv_nsec,
                        data.processname,
                        (long) getpid(),
                        line);
                fclose(fp);
            }
        }



        //
        data.cmdNBarg = 0;
        // extract first word
        cmdargstring = strtok(line, " ");

        while(cmdargstring != NULL)   // iterate on words
        {

            if((cmdargstring[0] == '\"')
                    && (cmdargstring[strlen(cmdargstring) - 1] == '\"'))
            {
                // if within quotes, store as string
                printf("Unprocessed string : ");
                unsigned int stri;
                for(stri = 0; stri < strlen(cmdargstring) - 2; stri++)
                {
                    cmdargstring[stri] = cmdargstring[stri + 1];
                }
                cmdargstring[stri] = '\0';
                printf("%s\n", cmdargstring);
                data.cmdargtoken[data.cmdNBarg].type = 6;
                sprintf(data.cmdargtoken[data.cmdNBarg].val.string, "%s", cmdargstring);
            }
            else     // otherwise, process it
            {
                sprintf(str, "%s\n", cmdargstring);
                yy_scan_string(str);
                data.calctmp_imindex = 0;
                yyparse();
                yylex_destroy();
            }
            cmdargstring = strtok(NULL, " ");
            data.cmdNBarg++;
        }
        data.cmdargtoken[data.cmdNBarg].type = 0;


        i = 0;
        if(data.Debug == 1)
        {
            while(data.cmdargtoken[i].type != 0)
            {
                printf("TOKEN %ld type : %d\n", i, data.cmdargtoken[i].type);
                if(data.cmdargtoken[i].type == 1)   // double
                {
                    printf("\t double : %g\n", data.cmdargtoken[i].val.numf);
                }
                if(data.cmdargtoken[i].type == 2)   // long
                {
                    printf("\t long   : %ld\n", data.cmdargtoken[i].val.numl);
                }
                if(data.cmdargtoken[i].type == 3)   // new variable/image
                {
                    printf("\t string : %s\n", data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type == 4)   // existing image
                {
                    printf("\t string : %s\n", data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type == 5)   // command
                {
                    printf("\t string : %s\n", data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type == 6)   // unprocessed string
                {
                    printf("\t string : %s\n", data.cmdargtoken[i].val.string);
                }
                i++;
            }
        }

        if(data.parseerror == 0)
        {
            if(data.cmdargtoken[0].type == 5)
            {
                if(data.Debug == 1)
                {
                    printf("EXECUTING COMMAND %ld (%s)\n", data.cmdindex,
                           data.cmd[data.cmdindex].key);
                }
                data.cmd[data.cmdindex].fp();
                data.CMDexecuted = 1;
            }
        }
        for(i = 0; i < data.calctmp_imindex; i++)
        {
            sprintf(calctmpimname, "_tmpcalc%ld", i);
            if(image_ID(calctmpimname) != -1)
            {
                if(data.Debug == 1)
                {
                    printf("Deleting %s\n", calctmpimname);
                }
                delete_image_ID(calctmpimname);
            }
        }


        if(!((data.cmdargtoken[0].type == 3) || (data.cmdargtoken[0].type == 6)))
        {
            data.CMDexecuted = 1;
        }


        add_history(line);

    }

    free(thetime);

    return RETURN_SUCCESS;
}









/**
 * @brief Readline callback
 *
 **/
void rl_cb_linehandler(char *linein)
{

    if(NULL == linein)
    {
        rlquit = true;
        return;
    }

    CLIexecuteCMDready = 1;
    strcpy(line, linein);
    CLI_execute_line();
    free(linein);
}




errno_t RegisterModule(
    const char *restrict FileName,
    const char *restrict PackageName,
    const char *restrict InfoString
)
{
    int OKmsg = 0;

    strcpy(data.module[data.NBmodule].name,    basename(FileName));
    strcpy(data.module[data.NBmodule].package, PackageName);
    strcpy(data.module[data.NBmodule].info,    InfoString);

    if(data.progStatus == 0)
    {
        OKmsg = 1;
        printf(".");
        //	printf("  %02ld  LOADING %10s  module %40s\n", data.NBmodule, PackageName, FileName);
        //	fflush(stdout);
    }

    if(data.progStatus == 1)
    {
        OKmsg = 1;
        printf("  %02ld  Found unloaded shared object in ./libs/ -> LOADING %10s  module %40s\n",
               data.NBmodule,
               PackageName,
               FileName);
        fflush(stdout);
    }

    if(OKmsg == 0)
    {
        printf("  %02ld  ERROR: module load requested outside of normal step -> LOADING %10s  module %40s\n",
               data.NBmodule,
               PackageName,
               FileName);
        fflush(stdout);
    }

    data.NBmodule++;


    return RETURN_SUCCESS;
}




uint_fast16_t RegisterCLIcommand(
    const char *restrict CLIkey,
    const char *restrict CLImodule,
    errno_t (*CLIfptr)(),
    const char *restrict CLIinfo,
    const char *restrict CLIsyntax,
    const char *restrict CLIexample,
    const char *restrict CLICcall
)
{
    //	printf("Registering command    %20s   [%5ld]\n", CLIkey, data.NBcmd);

    strcpy(data.cmd[data.NBcmd].key, CLIkey);
    strcpy(data.cmd[data.NBcmd].module, CLImodule);
    data.cmd[data.NBcmd].fp = CLIfptr;
    strcpy(data.cmd[data.NBcmd].info, CLIinfo);
    strcpy(data.cmd[data.NBcmd].syntax, CLIsyntax);
    strcpy(data.cmd[data.NBcmd].example, CLIexample);
    strcpy(data.cmd[data.NBcmd].Ccall, CLICcall);
    data.NBcmd++;

    return(data.NBcmd);
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
    char command[200];

    CLIPID = getpid();
    printf("    CLI PID = %d\n", (int) CLIPID);
    sprintf(command,
            "echo -n \"    \"; cat /proc/%d/status | grep Cpus_allowed_list", CLIPID);
    if(system(command) != 0)
    {
        printERROR(__FILE__, __func__, __LINE__, "system() returns non-zero value");
    }

    //	printf("    _SC_CLK_TCK = %d\n", sysconf(_SC_CLK_TCK));



    if(Verbose)
    {
        fprintf(stdout, "%s: compiled %s %s\n", __FILE__, __DATE__, __TIME__);
    }

# ifdef _OPENMP
    printf("    Running with openMP, max threads = %d  (OMP_NUM_THREADS)\n",
           omp_get_max_threads());
# else
    printf("    Compiled without openMP\n");
# endif

# ifdef _OPENACC
    int openACC_devtype = acc_get_device_type();
    printf("    Running with openACC version %d.  %d device(s), type %d\n",
           _OPENACC, acc_get_num_devices(openACC_devtype), openACC_devtype);
# endif





    // to take advantage of kernel priority:
    // owner=root mode=4755

#ifndef __MACH__
    getresuid(&data.ruid, &data.euid, &data.suid);
    //This sets it to the privileges of the normal user
    if(seteuid(data.ruid) != 0)
    {
        printERROR(__FILE__, __func__, __LINE__, "seteuid error");
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




static errno_t setSHMdir()
{
    // SHM directory to store shared memory
    //
    // If MILK_SHM_DIR environment variable exists, use it
    // If fails, print warning, use SHAREDMEMDIR defined in ImageStruct.h
    // If fails -> use /tmp
    //
    char shmdirname[200];
    int shmdirOK = 0; // toggles to 1 when directory is found
    DIR *tmpdir;

    // first, we try the env variable if it exists
    char *MILK_SHM_DIR = getenv("MILK_SHM_DIR");
    if(MILK_SHM_DIR != NULL)
    {
        printf(" [ MILK_SHM_DIR ] '%s'\n", MILK_SHM_DIR);
        sprintf(shmdirname, "%s", MILK_SHM_DIR);

        // does this direcory exist ?
        tmpdir = opendir(shmdirname);
        if(tmpdir)   // directory exits
        {
            shmdirOK = 1;
            closedir(tmpdir);
            printf("    Using SHM directory %s\n", shmdirname);
        }
        else
        {
            printf("%c[%d;%dm", (char) 27, 1, 31); // set color red
            printf("    ERROR: Directory %s : %s\n", shmdirname, strerror(errno));
            printf("%c[%d;m", (char) 27, 0); // unset color red
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        printf("%c[%d;%dm", (char) 27, 1, 31); // set color red
        printf("    WARNING: Environment variable MILK_SHM_DIR not specified -> falling back to default %s\n",
               SHAREDMEMDIR);
        printf("    BEWARE : Other milk users may be using the same SHM directory on this machine, and could see your milk session data and temporary files\n");
        printf("    BEWARE : Some scripts may rely on MILK_SHM_DIR to find/access shared memory and temporary files, and WILL not run.\n");
        printf("             Please set MILK_SHM_DIR and restart CLI to set up user-specific shared memory and temporary files\n");
        printf("             Example: Add \"export MILK_SHM_DIR=/milk/shm\" to .bashrc\n");
        printf("%c[%d;m", (char) 27, 0); // unset color red
    }

    // second, we try SHAREDMEMDIR default
    if(shmdirOK == 0)
    {
        tmpdir = opendir(SHAREDMEMDIR);
        if(tmpdir)   // directory exits
        {
            sprintf(shmdirname, "%s", SHAREDMEMDIR);
            shmdirOK = 1;
            closedir(tmpdir);
            printf("    Using SHM directory %s\n", shmdirname);
        }
        else
        {
            printf("    Directory %s : %s\n", SHAREDMEMDIR, strerror(errno));
        }
    }

    // if all above fails, set to /tmp
    if(shmdirOK == 0)
    {
        tmpdir = opendir("/tmp");
        if(!tmpdir)
        {
            printf("    ERROR: Directory %s : %s\n", shmdirname, strerror(errno));
            exit(EXIT_FAILURE);
        }
        else
        {
            sprintf(shmdirname, "/tmp");
            shmdirOK = 1;
            printf("    Using SHM directory %s\n", shmdirname);

            printf("    NOTE: Consider creating tmpfs directory and setting env var MILK_SHM_DIR for improved performance :\n");
            printf("        $ echo \"tmpfs %s tmpfs rw,nosuid,nodev\" | sudo tee -a /etc/fstab\n",
                   SHAREDMEMDIR);
            printf("        $ sudo mkdir -p %s\n", SHAREDMEMDIR);
            printf("        $ sudo mount %s\n", SHAREDMEMDIR);
        }
    }

    sprintf(data.shmdir, "%s", shmdirname);


    // change / to . and write to shmsemdirname
    unsigned int stri;
    for(stri = 0; stri < strlen(shmdirname); stri++)
        if(shmdirname[stri] == '/')   // replace '/' by '.'
        {
            shmdirname[stri] = '.';
        }

    sprintf(data.shmsemdirname, "%s", shmdirname);
    printf("    semaphore naming : /dev/shm/sem.%s.<sname>_sem<xx>\n",
           data.shmsemdirname);

    return RETURN_SUCCESS;
}




static errno_t runCLI_prompt(
    char *promptstring,
    char *prompt
)
{

    if(strlen(promptstring) > 0)
    {
        if(data.processnameflag == 0)
        {
            sprintf(prompt, "%c[%d;%dm%s >%c[%dm ", 0x1B, 1, 36, promptstring, 0x1B, 0);
        }
        else
        {
            sprintf(prompt, "%c[%d;%dm%s-%s >%c[%dm ", 0x1B, 1, 36, promptstring,
                    data.processname, 0x1B, 0);
        }
    }
    else
    {
        sprintf(prompt, "%c[%d;%dm%s >%c[%dm ", 0x1B, 1, 36, data.processname, 0x1B, 0);
    }

    return RETURN_SUCCESS;
}





/**
 * @brief Command Line Interface (CLI) main\n
 *
 * uses readline to read user input\n
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


    // Get command-line options
    command_line_process_options(argc, argv);


    DEBUG_TRACEPOINT("CLI start");





    // initialize readline
    // Tell readline to use custom completion function
    rl_attempted_completion_function = CLI_completion;
    rl_initialize();


    // initialize fifo to process name
    if(data.fifoON == 1)
    {
        sprintf(data.fifoname, "%s.fifo", data.processname);
        printf("fifo name : %s\n", data.fifoname);
    }






    //    sprintf(DocDir,"%s", DOCDIR);
    //   sprintf(SrcDir,"%s", SOURCEDIR);
    //  sprintf(BuildFile,"%s", __FILE__);
    //  sprintf(BuildDate,"%s", __DATE__);
    //  sprintf(BuildTime,"%s", __TIME__);




    data.progStatus = 1;



    printf("\n");




    // LOAD MODULES (shared objects)
    load_module_shared_ALL();

    // load other libs specified by environment variable CLI_ADD_LIBS
    char *CLI_ADD_LIBS = getenv("CLI_ADD_LIBS");
    if(CLI_ADD_LIBS != NULL)
    {
        printf(" [ CLI_ADD_LIBS ] '%s'\n", CLI_ADD_LIBS);

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
        printf(" [ CLI_ADD_LIBS ] not set\n");
    }










    // Initialize data control block
    runCLI_data_init();


    // initialize readline
    rl_callback_handler_install(prompt, (rl_vcpfunc_t *) &rl_cb_linehandler);


    // fifo
    fdmax = fileno(stdin);
    if(data.fifoON == 1)
    {
        printf("Creating fifo %s\n", data.fifoname);
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



    //    int atexitfifoclose = 0;
    //		if( atexitfifoclose == 0)
    //			{
    //				printf("Registering exit function fnExit_fifoclose\n");
    //				atexit( fnExit_fifoclose );
    //				atexitfifoclose = 1;
    //			}




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
            if(data.fifoON == 1)
            {
                EXECUTE_SYSTEM_COMMAND("cat %s > %s 2> /dev/null", CLIstartupfilename,
                                       data.fifoname);
                printf("IMPORTING FILE %s ... ", CLIstartupfilename);
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



        while(CLIexecuteCMDready == 0)
        {
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
                            strcpy(line, buf1);
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
        CLIexecuteCMDready = 0;

        if(data.CMDexecuted == 0)
        {
            printf("Command not found, or command with no effect\n");
        }

        //TEST data.CLIloopON = 0;
    }
    DEBUG_TRACEPOINT("exit from CLI loop");

    // clear all images and variables
    clearall();


    runCLI_free();

    rl_clear_history();
    rl_callback_handler_remove();


    DEBUG_TRACEPOINT("exit from runCLI function");

    return RETURN_SUCCESS;
}









// READLINE CUSTOM COMPLETION



static char **CLI_completion(
    const char *text,
    int start,
    int __attribute__((unused)) end
)
{
    char **matches;

    matches = (char **)NULL;

    if(start == 0)
    {
        CLImatchMode = 0;    // try to match with command first
    }
    else
    {
        CLImatchMode = 1;    // do not try to match with command
    }

    matches = rl_completion_matches((char *)text, &CLI_generator);

    //    else
    //  rl_bind_key('\t',rl_abort);

    return (matches);

}



char *CLI_generator(
    const char *text,
    int         state
)
{
    static unsigned int list_index;
    static unsigned int list_index1;
    static unsigned int len;
    char      *name;

    if(!state)
    {
        list_index = 0;
        list_index1 = 0;
        len = strlen(text);
    }

    if(CLImatchMode == 0)
        while(list_index < data.NBcmd)
        {
            name = data.cmd[list_index].key;
            list_index++;
            if(strncmp(name, text, len) == 0)
            {
                return (dupstr(name));
            }
        }

    while(list_index1 < data.NB_MAX_IMAGE)
    {
        int iok;
        iok = data.image[list_index1].used;
        if(iok == 1)
        {
            name = data.image[list_index1].name;
            //	  printf("  name %d = %s %s\n", list_index1, data.image[list_index1].name, name);
        }
        list_index1++;
        if(iok == 1)
        {
            if(strncmp(name, text, len) == 0)
            {
                return (dupstr(name));
            }
        }
    }
    return ((char *)NULL);

}



char *dupstr(char *s)
{
    char *r;

    r = (char *) xmalloc((strlen(s) + 1));
    strcpy(r, s);
    return (r);
}



void *xmalloc(int size)
{
    void *buf;

    buf = malloc(size);
    if(!buf)
    {
        fprintf(stderr, "Error: Out of memory. Exiting.'n");
        exit(1);
    }

    return buf;
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
    data.quiet           = 1;
    data.NB_MAX_IMAGE    = STATIC_NB_MAX_IMAGE;
    data.NB_MAX_VARIABLE = STATIC_NB_MAX_VARIABLE;
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
        printERROR(__FILE__, __func__, __LINE__,
                   "Allocation of data.image has failed - exiting program");
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
        printERROR(__FILE__, __func__, __LINE__,
                   "Allocation of data.variable has failed - exiting program");
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
        printERROR(__FILE__, __func__, __LINE__,
                   "Reallocation of data.variable has failed - exiting program");
        exit(1);
    }
#endif



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



    strcpy(data.cmd[data.NBcmd].key, "exit");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = exitCLI;
    strcpy(data.cmd[data.NBcmd].info, "exit program (same as quit command)");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "exit");
    strcpy(data.cmd[data.NBcmd].Ccall, "exitCLT");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "quit");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = exitCLI;
    strcpy(data.cmd[data.NBcmd].info, "quit program (same as exit command)");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "quit");
    strcpy(data.cmd[data.NBcmd].Ccall, "exitCLI");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "exitCLI");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = exitCLI;
    strcpy(data.cmd[data.NBcmd].info, "quit program (same as exit command)");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "quit");
    strcpy(data.cmd[data.NBcmd].Ccall, "exitCLI");
    data.NBcmd++;





    strcpy(data.cmd[data.NBcmd].key, "help");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = help;
    strcpy(data.cmd[data.NBcmd].info, "print help");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "help");
    strcpy(data.cmd[data.NBcmd].Ccall, "int help()");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "?");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = help;
    strcpy(data.cmd[data.NBcmd].info, "print help");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "help");
    strcpy(data.cmd[data.NBcmd].Ccall, "int help()");
    data.NBcmd++;


    strcpy(data.cmd[data.NBcmd].key, "helprl");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = helpreadline;
    strcpy(data.cmd[data.NBcmd].info, "print readline help");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "helprl");
    strcpy(data.cmd[data.NBcmd].Ccall, "int helpreadline()");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "cmd?");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = help_cmd;
    strcpy(data.cmd[data.NBcmd].info, "list commands");
    strcpy(data.cmd[data.NBcmd].syntax, "command name (optional)");
    strcpy(data.cmd[data.NBcmd].example, "cmd?");
    strcpy(data.cmd[data.NBcmd].Ccall, "int help_cmd()");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "m?");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = help_module;
    strcpy(data.cmd[data.NBcmd].info, "list commands in a module");
    strcpy(data.cmd[data.NBcmd].syntax, "module name");
    strcpy(data.cmd[data.NBcmd].example, "m? COREMOD_memory.c");
    strcpy(data.cmd[data.NBcmd].Ccall, "errno_t list_commands_module()");
    data.NBcmd++;


    strcpy(data.cmd[data.NBcmd].key, "soload");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = load_so;
    strcpy(data.cmd[data.NBcmd].info, "load shared object");
    strcpy(data.cmd[data.NBcmd].syntax, "shared object name");
    strcpy(data.cmd[data.NBcmd].example, "soload mysharedobj.so");
    strcpy(data.cmd[data.NBcmd].Ccall, "int load_sharedobj(char *libname)");
    data.NBcmd++;


    strcpy(data.cmd[data.NBcmd].key, "mload");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = load_module;
    strcpy(data.cmd[data.NBcmd].info, "load module from shared object");
    strcpy(data.cmd[data.NBcmd].syntax, "module name");
    strcpy(data.cmd[data.NBcmd].example, "mload mymodule");
    strcpy(data.cmd[data.NBcmd].Ccall,
           "errno_t load_module_shared(char *modulename)");
    data.NBcmd++;


    strcpy(data.cmd[data.NBcmd].key, "ci");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = printInfo;
    strcpy(data.cmd[data.NBcmd].info, "Print version, settings, info and exit");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "ci");
    strcpy(data.cmd[data.NBcmd].Ccall, "int printInfo()");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "dpsingle");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = set_default_precision_single;
    strcpy(data.cmd[data.NBcmd].info, "Set default precision to single");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "dpsingle");
    strcpy(data.cmd[data.NBcmd].Ccall, "data.precision = 0");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "dpdouble");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = set_default_precision_double;
    strcpy(data.cmd[data.NBcmd].info, "Set default precision to doube");
    strcpy(data.cmd[data.NBcmd].syntax, "no argument");
    strcpy(data.cmd[data.NBcmd].example, "dpdouble");
    strcpy(data.cmd[data.NBcmd].Ccall, "data.precision = 1");
    data.NBcmd++;



    // process info

    strcpy(data.cmd[data.NBcmd].key, "setprocinfoON");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = set_processinfoON;
    strcpy(data.cmd[data.NBcmd].info, "set processes info ON");
    strcpy(data.cmd[data.NBcmd].syntax, "no arg");
    strcpy(data.cmd[data.NBcmd].example, "setprocinfoON");
    strcpy(data.cmd[data.NBcmd].Ccall, "set_processinfoON()");
    data.NBcmd++;


    strcpy(data.cmd[data.NBcmd].key, "setprocinfoOFF");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = set_processinfoOFF;
    strcpy(data.cmd[data.NBcmd].info, "set processes info OFF");
    strcpy(data.cmd[data.NBcmd].syntax, "no arg");
    strcpy(data.cmd[data.NBcmd].example, "setprocinfoOFF");
    strcpy(data.cmd[data.NBcmd].Ccall, "set_processinfoOFF()");
    data.NBcmd++;



    strcpy(data.cmd[data.NBcmd].key, "procCTRL");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = processinfo_CTRLscreen_cli;
    strcpy(data.cmd[data.NBcmd].info, "processes control screen");
    strcpy(data.cmd[data.NBcmd].syntax, "no arg");
    strcpy(data.cmd[data.NBcmd].example, "procCTRL");
    strcpy(data.cmd[data.NBcmd].Ccall, "processinfo_CTRLscreen()");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "streamCTRL");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = streamCTRL_CTRLscreen_cli;
    strcpy(data.cmd[data.NBcmd].info, "stream control screen");
    strcpy(data.cmd[data.NBcmd].syntax, "no arg");
    strcpy(data.cmd[data.NBcmd].example, "streamCTRL");
    strcpy(data.cmd[data.NBcmd].Ccall, "streamCTRL_CTRLscreen()");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "fparamCTRL");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = functionparameter_CTRLscreen_cli;
    strcpy(data.cmd[data.NBcmd].info, "function parameters control screen");
    strcpy(data.cmd[data.NBcmd].syntax, "function parameter structure name");
    strcpy(data.cmd[data.NBcmd].example, "fparamCTRL fpsname");
    strcpy(data.cmd[data.NBcmd].Ccall,
           "int_fast8_t functionparameter_CTRLscreen(char *fpsname)");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "usleep");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = cfits_usleep_cli;
    strcpy(data.cmd[data.NBcmd].info, "usleep");
    strcpy(data.cmd[data.NBcmd].syntax, "<us>");
    strcpy(data.cmd[data.NBcmd].example, "usleep 1000");
    strcpy(data.cmd[data.NBcmd].Ccall, "usleep(long tus)");
    data.NBcmd++;


    //  init_modules();
    // printf("TEST   %s  %ld   data.image[4934].used = %d\n", __FILE__, __LINE__, data.image[4934].used);

    printf("        %ld modules, %ld commands\n", data.NBmodule, data.NBcmd);
    printf("        \n");
}




static void runCLI_free()
{
#ifndef DATA_STATIC_ALLOC
    // Free
    free(data.image);
    free(data.variable);
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
|
|  memory_re_alloc    : keep the number of images addresses available
| 		 NB_IMAGES_BUFFER above the number of used images
|
|                keep the number of variables addresses available
|                NB_VARIABLES_BUFFER above the number of used variables
|
| NOTE:  this should probably be renamed and put in the module/memory/memory.c
|
+-----------------------------------------------------------------------------*/
static errno_t memory_re_alloc()
{
    /* keeps the number of images addresses available
     *  NB_IMAGES_BUFFER above the number of used images
     */


#ifdef DATA_STATIC_ALLOC
    // image static allocation mode
#else
    int current_NBimage = compute_nb_image(data);

    //	printf("DYNAMIC ALLOC. Current = %d, buffer = %d, max = %d\n", current_NBimage, NB_IMAGES_BUFFER, data.NB_MAX_IMAGE);///

    if((current_NBimage + NB_IMAGES_BUFFER) > data.NB_MAX_IMAGE)
    {
        long tmplong;
        IMAGE *ptrtmp;

        //   if(data.Debug>0)
        //    {
        printf("%p IMAGE STRUCT SIZE = %ld\n", data.image, (long) sizeof(IMAGE));
        printf("REALLOCATING IMAGE DATA BUFFER: %ld -> %ld\n", data.NB_MAX_IMAGE,
               data.NB_MAX_IMAGE + NB_IMAGES_BUFFER_REALLOC);
        fflush(stdout);
        //    }
        tmplong = data.NB_MAX_IMAGE;
        data.NB_MAX_IMAGE = data.NB_MAX_IMAGE + NB_IMAGES_BUFFER_REALLOC;
        ptrtmp = (IMAGE *) realloc(data.image, sizeof(IMAGE) * data.NB_MAX_IMAGE);
        if(data.Debug > 0)
        {
            printf("NEW POINTER = %p\n", ptrtmp);
            fflush(stdout);
        }
        data.image = ptrtmp;
        if(data.image == NULL)
        {
            printERROR(__FILE__, __func__, __LINE__,
                       "Reallocation of data.image has failed - exiting program");
            return -1;      //  exit(0);
        }
        if(data.Debug > 0)
        {
            printf("REALLOCATION DONE\n");
            fflush(stdout);
        }

        imageID i;
        for(i = tmplong; i < data.NB_MAX_IMAGE; i++)
        {
            data.image[i].used    = 0;
            data.image[i].shmfd   = -1;
            data.image[i].memsize = 0;
            data.image[i].semptr  = NULL;
            data.image[i].semlog  = NULL;
        }
    }
#endif


    /* keeps the number of variables addresses available
     *  NB_VARIABLES_BUFFER above the number of used variables
     */


#ifdef DATA_STATIC_ALLOC
    // variable static allocation mode
#else
    if((compute_nb_variable(data) + NB_VARIABLES_BUFFER) > data.NB_MAX_VARIABLE)
    {
        long tmplong;

        if(data.Debug > 0)
        {
            printf("REALLOCATING VARIABLE DATA BUFFER\n");
            fflush(stdout);
        }
        tmplong = data.NB_MAX_VARIABLE;
        data.NB_MAX_VARIABLE = data.NB_MAX_VARIABLE + NB_VARIABLES_BUFFER_REALLOC;
        data.variable = (VARIABLE *) realloc(data.variable,
                                             sizeof(VARIABLE) * data.NB_MAX_VARIABLE);
        if(data.variable == NULL)
        {
            printERROR(__FILE__, __func__, __LINE__,
                       "Reallocation of data.variable has failed - exiting program");
            return -1;   // exit(0);
        }

        int i;
        for(i = tmplong; i < data.NB_MAX_VARIABLE; i++)
        {
            data.variable[i].used = 0;
            data.variable[i].type = -1;
        }
    }
#endif


    return RETURN_SUCCESS;
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
static int command_line_process_options(int argc, char **argv)
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
        {"debug",       required_argument, 0, 'd'},
        {"mmon",        required_argument, 0, 'm'},
        {"pname",       required_argument, 0, 'n'},
        {"priority",    required_argument, 0, 'p'},
        {"fifo",        required_argument, 0, 'f'},
        {"startup",     required_argument, 0, 's'},
        {0, 0, 0, 0}
    };



    data.fifoON = 0; // default
    data.processnameflag = 0; // default

    while(1)
    {
        int c;

        c = getopt_long(argc, argv, "hvidoe:m:n:p:f:s:",
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
                    printERROR(__FILE__, __func__, __LINE__, "system() returns non-zero value");
                }
                break;

            case 'm':
                printf("Starting memory monitor on '%s'\n", optarg);
                memory_monitor(optarg);
                break;

            case 'n':
                printf("process name '%s'\n", optarg);
                strcpy(data.processname, optarg);
                data.processnameflag = 1; // this process has been named

                // extract first word before '.'
                // it can be used to name processinfo and function parameter structure for process
                char tmpstring[200];
                strcpy(tmpstring, data.processname);
                char *firstword;
                firstword = strtok(tmpstring, ".");
                strcpy(data.processname0, firstword);

                //            memcpy((void *)argv[0], optarg, strlen(optarg));
                strcpy(argv[0], optarg);
#ifdef __linux__
                prctl(PR_SET_NAME, optarg, 0, 0, 0);
#elif defined(HAVE_PTHREAD_SETNAME_NP) && defined(OS_IS_DARWIN)
                pthread_setname_np(optarg);
#endif
                //            prctl(PR_SET_NAME, optarg, 0, 0, 0);
                break;

            case 'p':
                schedpar.sched_priority = atoi(optarg);
                printf("RUNNING WITH RT PRIORITY = %d\n", schedpar.sched_priority);
#ifndef __MACH__

                if(seteuid(data.euid) != 0) //This goes up to maximum privileges
                {
                    printERROR(__FILE__, __func__, __LINE__, "seteuid() returns non-zero value");
                }
                sched_setscheduler(0, SCHED_FIFO,
                                   &schedpar); //other option is SCHED_RR, might be faster

                if(seteuid(data.ruid) != 0) //Go back to normal privileges
                {
                    printERROR(__FILE__, __func__, __LINE__, "seteuid() returns non-zero value");
                }
#endif
                break;

            case 'f':
                printf("using input fifo '%s'\n", optarg);
                data.fifoON = 1;
                sprintf(data.fifoname, "%s.%05d", optarg, CLIPID);
                //            strcpy(data.fifoname, optarg);
                printf("FIFO NAME = %s\n", data.fifoname);
                break;

            case 's':
                strcpy(CLIstartupfilename, optarg);
                printf("Startup file : %s\n", CLIstartupfilename);
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



/*^-----------------------------------------------------------------------------
|
|
| HELP ROUTINES
|
|
+-----------------------------------------------------------------------------*/



static errno_t list_commands()
{
    char cmdinfoshort[38];

    printf("----------- LIST OF COMMANDS ---------\n");
    for(unsigned int i = 0; i < data.NBcmd; i++)
    {
        strncpy(cmdinfoshort, data.cmd[i].info, 38);
        printf("   %-16s %-20s %-40s %-30s\n", data.cmd[i].key, data.cmd[i].module,
               cmdinfoshort, data.cmd[i].example);
    }

    return RETURN_SUCCESS;
}



static errno_t list_commands_module(
    const char *restrict modulename
)
{
    int mOK = 0;
    char cmdinfoshort[38];

    if(strlen(modulename) > 0)
    {
        for(unsigned int i = 0; i < data.NBcmd; i++)
        {
            char cmpstring[200];
            sprintf(cmpstring, "%s", basename(data.cmd[i].module));

            if(strcmp(modulename, cmpstring) == 0)
            {
                if(mOK == 0)
                {
                    printf("---- MODULE %s: LIST OF COMMANDS ---------\n", modulename);
                }



                strncpy(cmdinfoshort, data.cmd[i].info, 38);
                printf("   %-16s %-20s %-40s %-30s\n", data.cmd[i].key, cmpstring, cmdinfoshort,
                       data.cmd[i].example);
                mOK = 1;
            }
        }
    }

    if(mOK == 0)
    {
        if(strlen(modulename) > 0)
        {
            printf("---- MODULE %s DOES NOT EXIST OR DOES NOT HAVE COMMANDS ---------\n",
                   modulename);
        }

        for(unsigned int i = 0; i < data.NBcmd; i++)
        {
            char cmpstring[200];
            sprintf(cmpstring, "%s", basename(data.cmd[i].module));

            if(strncmp(modulename, cmpstring, strlen(modulename)) == 0)
            {
                if(mOK == 0)
                {
                    printf("---- MODULES %s* commands  ---------\n", modulename);
                }
                strncpy(cmdinfoshort, data.cmd[i].info, 38);
                printf("   %-16s %-20s %-40s %-30s\n", data.cmd[i].key,
                       basename(data.cmd[i].module), cmdinfoshort, data.cmd[i].example);
                mOK = 1;
            }
        }
    }

    return RETURN_SUCCESS;
}





static errno_t load_sharedobj(
    const char *restrict libname
)
{
    printf("[%5d] Loading shared object \"%s\"\n", DLib_index, libname);

    DLib_handle[DLib_index] = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if(!DLib_handle[DLib_index])
    {
        fprintf(stderr, "%s\n", dlerror());
        //exit(EXIT_FAILURE);
    }
    else
    {
        dlerror();
        // increment number of libs dynamically loaded
        DLib_index ++;
    }

    return RETURN_SUCCESS;
}




static errno_t load_module_shared(
    const char *restrict modulename
)
{
    int STRINGMAXLEN_LIBRARYNAME = 200;
    char libname[STRINGMAXLEN_LIBRARYNAME];
    char modulenameLC[STRINGMAXLEN_LIBRARYNAME];
    //    char c;
    //    int n;
    //    int (*libinitfunc) ();
    //    char *error;
    //    char initfuncname[200];

    {
        int slen = snprintf(modulenameLC, STRINGMAXLEN_LIBRARYNAME, "%s", modulename);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_LIBRARYNAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    /*    for(n=0; n<strlen(modulenameLC); n++)
        {
            c = modulenameLC[n];
            modulenameLC[n] = tolower(c);
        }
    */

    //    sprintf(libname, "%s/lib/lib%s.so", data.sourcedir, modulenameLC);
    {
        int slen = snprintf(libname, STRINGMAXLEN_LIBRARYNAME,
                            "/usr/local/lib/lib%s.so", modulenameLC);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_LIBRARYNAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    printf("libname = %s\n", libname);


    printf("[%5d] Loading shared object \"%s\"\n", DLib_index, libname);

    load_sharedobj(libname);

    return RETURN_SUCCESS;
}





static errno_t load_module_shared_ALL()
{
    char libname[500];
    char dirname[200];
    DIR           *d;
    struct dirent *dir;
    int iter;
    int loopOK;
    int itermax;

    sprintf(dirname, "%s/lib", data.sourcedir);
    printf("LOAD MODULES SHARED ALL: %s\n", dirname);

    loopOK = 0;
    iter = 0;
    itermax = 4; // number of passes
    while((loopOK == 0) && (iter < itermax))
    {
        loopOK = 1;
        d = opendir(dirname);
        if(d)
        {
            while((dir = readdir(d)) != NULL)
            {
                char *dot = strrchr(dir->d_name, '.');
                if(dot && !strcmp(dot, ".so"))
                {
                    sprintf(libname, "%s/lib/%s", data.sourcedir, dir->d_name);
                    //printf("%02d   (re-?) LOADING shared object  %40s -> %s\n", DLib_index, dir->d_name, libname);
                    //fflush(stdout);

                    //printf("[%5d] Loading shared object \"%s\"\n", DLib_index, libname);
                    DLib_handle[DLib_index] = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
                    if(!DLib_handle[DLib_index])
                    {
                        fprintf(stderr, KMAG
                                "        WARNING: linker pass # %d, module # %d\n          %s\n" KRES, iter,
                                DLib_index, dlerror());
                        fflush(stderr);
                        //exit(EXIT_FAILURE);
                        loopOK = 0;
                    }
                    else
                    {
                        dlerror();
                        // increment number of libs dynamically loaded
                        DLib_index ++;
                    }


                }
            }

            closedir(d);
        }
        if(iter > 0)
            if(loopOK == 1)
            {
                printf(KGRN "        Linker pass #%d successful\n" KRES, iter);
            }
        iter++;
    }

    if(loopOK != 1)
    {
        printf("Some libraries could not be loaded -> EXITING\n");
        exit(2);
    }

    //printf("All libraries successfully loaded\n");


    return RETURN_SUCCESS;
}






/**
 * @brief command help\n
 *
 * @param[in] cmdkey Commmand name
 */


static errno_t help_command(
    const char *restrict cmdkey
)
{
    int cOK = 0;

    for(unsigned int i = 0; i < data.NBcmd; i++)
    {
        if(!strcmp(cmdkey, data.cmd[i].key))
        {
            printf("\n");
            printf("key       :    %s\n", data.cmd[i].key);
            printf("module    :    %s\n", data.cmd[i].module);
            printf("info      :    %s\n", data.cmd[i].info);
            printf("syntax    :    %s\n", data.cmd[i].syntax);
            printf("example   :    %s\n", data.cmd[i].example);
            printf("C call    :    %s\n", data.cmd[i].Ccall);
            printf("\n");
            cOK = 1;
        }
    }
    if(cOK == 0)
    {
        printf("\tCommand \"%s\" does not exist\n", cmdkey);
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






