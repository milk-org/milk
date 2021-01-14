/**
 * @file processtools.c
 * @brief Tools to manage processes
 *
 *
 * Manages structure PROCESSINFO.
 *
 * @see @ref page_ProcessInfoStructure
 *
 *
 */


#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif


static int CTRLscreenExitLine = 0; // for debugging


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif


/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/file.h>
#include <malloc.h>
#include <sys/mman.h> // mmap()

#include <time.h>
#include <signal.h>
#include <termios.h>
#include <sys/ioctl.h>

#include <unistd.h>    // getpid()
#include <sys/types.h>

#include <sys/stat.h>

#include <ncurses.h>
#include <fcntl.h>
#include <ctype.h>

#include <dirent.h>

#include <wchar.h>
#include <locale.h>

#include <pthread.h>


#include "CommandLineInterface/timeutils.h"

#include "CLIcore.h"
#include "COREMOD_tools/COREMOD_tools.h"
#define SHAREDPROCDIR data.shmdir


#include <processtools.h>

#ifdef USE_HWLOC
#include <hwloc.h>
#endif



/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */




// What do we want to compute/print ?
#define CMDPROC_CONTEXTSWITCH	1
#define CMDPROC_CPUUSE	1
#define CMDPROC_MEMUSE	1

#define CMDPROC_PROCSTAT 1



#define PROCCTRL_DISPLAYMODE_HELP      1
#define PROCCTRL_DISPLAYMODE_CTRL      2
#define PROCCTRL_DISPLAYMODE_RESOURCES 3
#define PROCCTRL_DISPLAYMODE_TRIGGER   4
#define PROCCTRL_DISPLAYMODE_TIMING    5
#define PROCCTRL_DISPLAYMODE_HTOP      6
#define PROCCTRL_DISPLAYMODE_IOTOP     7
#define PROCCTRL_DISPLAYMODE_ATOP      8



/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */






static PROCESSINFOLIST *pinfolist;

static int wrow, wcol;
static int wcolmax; // max number of cols






#define NBtopMax 5000

/*
static int   toparray_PID[NBtopMax];
static char  toparray_USER[NBtopMax][32];
static char  toparray_PR[NBtopMax][8];
static int   toparray_NI[NBtopMax];
static char  toparray_VIRT[NBtopMax][32];
static char  toparray_RES[NBtopMax][32];
static char  toparray_SHR[NBtopMax][32];
static char  toparray_S[NBtopMax][8];
static float toparray_CPU[NBtopMax];
static float toparray_MEM[NBtopMax];
static char  toparray_TIME[NBtopMax][32];
static char  toparray_COMMAND[NBtopMax][32];

static int NBtopP; // number of processes scanned by top
*/



// timing info collected to optimize this program
static struct timespec t1;
static struct timespec t2;
static struct timespec tdiff;

// timing categories
static double scantime_cpuset;
static double scantime_status;
static double scantime_stat;
static double scantime_pstree;
static double scantime_top;
static double scantime_CPUload;
static double scantime_CPUpcnt;



/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */



errno_t processinfo_procdirname(char *procdname)
{
    int procdirOK = 0;
    DIR *tmpdir;

    // first, we try the env variable if it exists
    char *MILK_PROC_DIR = getenv("MILK_PROC_DIR");
    if(MILK_PROC_DIR != NULL)
    {
        printf(" [ MILK_PROC_DIR ] '%s'\n", MILK_PROC_DIR);

        {
            int slen = snprintf(procdname, STRINGMAXLEN_FULLFILENAME, "%s", MILK_PROC_DIR);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_FULLFILENAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        // does this direcory exist ?
        tmpdir = opendir(procdname);
        if(tmpdir) // directory exits
        {
            procdirOK = 1;
            closedir(tmpdir);
        }
        else
        {
            printf(" [ WARNING ] '%s' does not exist\n", MILK_PROC_DIR);
        }
    }

    // second, we try SHAREDPROCDIR default
    if(procdirOK == 0)
    {
        tmpdir = opendir(SHAREDPROCDIR);
        if(tmpdir) // directory exits
        {
            sprintf(procdname, "%s", SHAREDPROCDIR);
            procdirOK = 1;
            closedir(tmpdir);
        }
    }

    // if all above fails, set to /tmp
    if(procdirOK == 0)
    {
        tmpdir = opendir("/tmp");
        if(!tmpdir)
        {
            exit(EXIT_FAILURE);
        }
        else
        {
            sprintf(procdname, "/tmp");
            procdirOK = 1;
        }
    }

    return RETURN_SUCCESS;
}



// High level processinfo function

PROCESSINFO *processinfo_setup(
    char *pinfoname,	// short name for the processinfo instance, avoid spaces, name should be human-readable
    char  descriptionstring[200],
    char  msgstring[200],
    const char *functionname,
    const char *filename,
    int   linenumber
)
{
    static PROCESSINFO
    *processinfo; // Only one instance of processinfo created by process
    // subsequent calls to this function will re-use the same processinfo structure

    DEBUG_TRACEPOINT(" ");



    DEBUG_TRACEPOINT(" ");
    if(data.processinfoActive == 0)
    {
        //        PROCESSINFO *processinfo;
        DEBUG_TRACEPOINT(" ");

        char pinfoname0[STRINGMAXLEN_PROCESSINFO_NAME];
        {
            int slen = snprintf(pinfoname0, STRINGMAXLEN_PROCESSINFO_NAME, "%s", pinfoname);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_PROCESSINFO_NAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        DEBUG_TRACEPOINT(" ");

        processinfo = processinfo_shm_create(pinfoname0, 0);


        processinfo_CatchSignals();
    }


    DEBUG_TRACEPOINT(" ");

    processinfo->loopstat = 0; // loop initialization
    strcpy(processinfo->source_FUNCTION, functionname);
    strcpy(processinfo->source_FILE,     filename);
    processinfo->source_LINE = linenumber;
    strcpy(processinfo->description, descriptionstring);
    processinfo_WriteMessage(processinfo, msgstring);
    data.processinfoActive = 1;

    processinfo->loopcntMax = -1;     // infinite loop

    processinfo->MeasureTiming =  0;  // default: do not measure timing
    processinfo->RT_priority   = -1;  // default: do not assign RT priority

    DEBUG_TRACEPOINT(" ");

    return processinfo;
}




// report error
// should be followed by return(EXIT_FAILURE) call
//
errno_t processinfo_error(
    PROCESSINFO *processinfo,
    char *errmsgstring
)
{
    processinfo->loopstat = 4; // ERROR
    processinfo_WriteMessage(processinfo, errmsgstring);
    processinfo_cleanExit(processinfo);
    return RETURN_SUCCESS;
}






errno_t processinfo_loopstart(
    PROCESSINFO *processinfo
)
{
    processinfo->loopcnt = 0;
    processinfo->loopstat = 1;

    if(processinfo->RT_priority > -1)
    {
        struct sched_param schedpar;
        // ===========================
        // Set realtime priority
        // ===========================
        schedpar.sched_priority = processinfo->RT_priority;
#ifndef __MACH__
        if(seteuid(data.euid) != 0)     //This goes up to maximum privileges
        {
            PRINT_ERROR("seteuid error");
        }
        sched_setscheduler(0, SCHED_FIFO,
                           &schedpar); //other option is SCHED_RR, might be faster
        if(seteuid(data.ruid) != 0)     //Go back to normal privileges
        {
            PRINT_ERROR("seteuid error");
        }
#endif
    }


    return RETURN_SUCCESS;
}




// returns loop status
// 0 if loop should exit, 1 otherwise

int processinfo_loopstep(
    PROCESSINFO *processinfo
)
{
    int loopstatus = 1;

    while(processinfo->CTRLval == 1)   // pause
    {
        usleep(50);
    }
    if(processinfo->CTRLval == 2)   // single iteration
    {
        processinfo->CTRLval = 1;
    }
    if(processinfo->CTRLval == 3)   // exit loop
    {
        loopstatus = 0;
    }

    if(data.signal_INT == 1)    // CTRL-C
    {
        loopstatus = 0;
    }

    if(data.signal_HUP == 1)    // terminal has disappeared
    {
        loopstatus = 0;
    }

    if(processinfo->loopcntMax != -1)
        if(processinfo->loopcnt >= processinfo->loopcntMax)
        {
            loopstatus = 0;
        }


    return loopstatus;
}





int processinfo_compute_status(
    PROCESSINFO *processinfo
)
{
    int compstatus = 1;

    // CTRLval = 5 will disable computations in loop (usually for testing)
    if(processinfo->CTRLval == 5)
    {
        compstatus = 0;
    }

    return compstatus;
}










/**
 * ## Purpose
 *
 * Read/create processinfo list
 *
 * ## Description
 *
 * If list does not exist, create it and return index = 0
 *
 * If list exists, return first available index
 *
 *
 */

long processinfo_shm_list_create()
{
    char  SM_fname[STRINGMAXLEN_FULLFILENAME];
    long pindex = 0;

    char  procdname[200];
    processinfo_procdirname(procdname);

    WRITE_FULLFILENAME(SM_fname, "%s/processinfo.list.shm", procdname);

    /*
    * Check if a file exist using stat() function.
    * return 1 if the file exist otherwise return 0.
    */
    struct stat buffer;
    int exists = stat(SM_fname, &buffer);

    if(exists == -1)
    {
        printf("CREATING PROCESSINFO LIST\n");

        size_t sharedsize = 0; // shared memory size in bytes
        int SM_fd; // shared memory file descriptor

        sharedsize = sizeof(PROCESSINFOLIST);

        SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
        if(SM_fd == -1)
        {
            perror("Error opening file for writing");
            exit(0);
        }

        int result;
        result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
        if(result == -1)
        {
            close(SM_fd);
            fprintf(stderr, "Error calling lseek() to 'stretch' the file");
            exit(0);
        }

        result = write(SM_fd, "", 1);
        if(result != 1)
        {
            close(SM_fd);
            perror("Error writing last byte of the file");
            exit(0);
        }

        pinfolist = (PROCESSINFOLIST *) mmap(0, sharedsize, PROT_READ | PROT_WRITE,
                                             MAP_SHARED, SM_fd, 0);
        if(pinfolist == MAP_FAILED)
        {
            close(SM_fd);
            perror("Error mmapping the file");
            exit(0);
        }

        for(pindex = 0; pindex < PROCESSINFOLISTSIZE; pindex++)
        {
            pinfolist->active[pindex] = 0;
        }

        pindex = 0;
    }
    else
    {
        int SM_fd;
        //struct stat file_stat;

        pinfolist = (PROCESSINFOLIST *)processinfo_shm_link(SM_fname, &SM_fd);
        while((pinfolist->active[pindex] != 0) && (pindex < PROCESSINFOLISTSIZE))
        {
            pindex ++;
        }

        if(pindex == PROCESSINFOLISTSIZE)
        {
            printf("ERROR: pindex reaches max value\n");
            exit(0);
        }
    }


    printf("pindex = %ld\n", pindex);

    return pindex;
}






/**
 * Create PROCESSINFO structure in shared memory
 *
 * The structure holds real-time information about a process, so its status can be monitored and controlled
 * See structure PROCESSINFO in CLLIcore.h for details
 *
*/

PROCESSINFO *processinfo_shm_create(
    const char *pname,
    int CTRLval
)
{
    size_t sharedsize = 0; // shared memory size in bytes
    int SM_fd; // shared memory file descriptor
    PROCESSINFO *pinfo;

    static int LogFileCreated =
        0; // toggles to 1 when created. To avoid re-creating file on same process

    sharedsize = sizeof(PROCESSINFO);

    char  SM_fname[STRINGMAXLEN_FULLFILENAME];
    pid_t PID;


    PID = getpid();

    long pindex;
    pindex = processinfo_shm_list_create();

    pinfolist->PIDarray[pindex] = PID;
    strncpy(pinfolist->pnamearray[pindex], pname, STRINGMAXLEN_PROCESSINFO_NAME);

    char  procdname[STRINGMAXLEN_FULLFILENAME];
    processinfo_procdirname(procdname);

    WRITE_FULLFILENAME(SM_fname, "%s/proc.%s.%06d.shm", procdname, pname,
                       (int) PID);

    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if(SM_fd == -1)
    {
        perror("Error opening file for writing");
        exit(0);
    }

    int result;
    result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
    if(result == -1)
    {
        close(SM_fd);
        fprintf(stderr, "Error calling lseek() to 'stretch' the file");
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if(result != 1)
    {
        close(SM_fd);
        perror("Error writing last byte of the file");
        exit(0);
    }

    pinfo = (PROCESSINFO *) mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED,
                                 SM_fd, 0);
    if(pinfo == MAP_FAILED)
    {
        close(SM_fd);
        perror("Error mmapping the file");
        exit(0);
    }

    printf("created processinfo entry at %s\n", SM_fname);
    printf("shared memory space = %ld bytes\n", sharedsize); //TEST

    clock_gettime(CLOCK_REALTIME, &pinfo->createtime);
    strcpy(pinfo->name, pname);

    pinfolist->active[pindex] = 1;

    char tmuxname[100];
    FILE *fpout;
    int notmux = 0;

    fpout = popen("tmuxsessionname", "r");
    if(fpout == NULL)
    {
        printf("WARNING: cannot run command \"tmuxsessionname\"\n");
    }
    else
    {
        if(fgets(tmuxname, 100, fpout) == NULL)
        {
            //printf("WARNING: fgets error\n");
            notmux = 1;
        }
        pclose(fpout);
    }
    // remove line feed
    if(strlen(tmuxname) > 0)
    {
        //  printf("tmux name : %s\n", tmuxname);
        //  printf("len: %d\n", (int) strlen(tmuxname));
        fflush(stdout);

        if(tmuxname[strlen(tmuxname) - 1] == '\n')
        {
            tmuxname[strlen(tmuxname) - 1] = '\0';
        }
        else
        {
            printf("tmux name empty\n");
        }
    }
    else
    {
        notmux = 1;
    }

    if(notmux == 1)
    {
        sprintf(tmuxname, " ");
    }

    // force last char to be term, just in case
    tmuxname[99] = '\0';

    printf("tmux name : %s\n", tmuxname);

    strncpy(pinfo->tmuxname, tmuxname, 100);

    // set control value (default 0)
    // 1 : pause
    // 2 : increment single step (will go back to 1)
    // 3 : exit loop
    pinfo->CTRLval = CTRLval;


    pinfo->MeasureTiming = 1;

    // initialize timer indexes and counters
    pinfo->timerindex = 0;
    pinfo->timingbuffercnt = 0;

    // disable timer limit feature
    pinfo->dtiter_limit_enable = 0;
    pinfo->dtexec_limit_enable = 0;

    data.pinfo = pinfo;
    pinfo->PID = PID;


    // create logfile
    //char logfilename[300];
    struct timespec tnow;

    clock_gettime(CLOCK_REALTIME, &tnow);

    sprintf(pinfo->logfilename, "%s/proc.%s.%06d.%09ld.logfile", procdname,
            pinfo->name, (int) pinfo->PID, tnow.tv_sec);

    if(LogFileCreated == 0)
    {
        pinfo->logFile = fopen(pinfo->logfilename, "w");
        LogFileCreated = 1;
    }


    char msgstring[300];
    sprintf(msgstring, "LOG START %s", pinfo->logfilename);
    processinfo_WriteMessage(pinfo, msgstring);

    return pinfo;
}



PROCESSINFO *processinfo_shm_link(const char *pname, int *fd)
{
    struct stat file_stat;

    *fd = open(pname, O_RDWR);
    if(*fd == -1)
    {
        perror("Error opening file");
        exit(0);
    }
    fstat(*fd, &file_stat);
    //printf("[%d] File %s size: %zd\n", __LINE__, pname, file_stat.st_size);

    PROCESSINFO *pinfolist = (PROCESSINFO *) mmap(0, file_stat.st_size,
                             PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    if(pinfolist == MAP_FAILED)
    {
        close(*fd);
        fprintf(stderr, "Error mmapping the file");
        exit(0);
    }

    return pinfolist;
}

int processinfo_shm_close(PROCESSINFO *pinfo, int fd)
{
    struct stat file_stat;
    fstat(fd, &file_stat);
    munmap(pinfo, file_stat.st_size);
    close(fd);
    return EXIT_SUCCESS;
}






int processinfo_cleanExit(PROCESSINFO *processinfo)
{

    if(processinfo->loopstat != 4)
    {
        struct timespec tstop;
        struct tm *tstoptm;
        char msgstring[200];

        clock_gettime(CLOCK_REALTIME, &tstop);
        tstoptm = gmtime(&tstop.tv_sec);

        if(processinfo->CTRLval == 3)   // loop exit from processinfo control
        {
            sprintf(msgstring, "CTRLexit %02d:%02d:%02d.%03d", tstoptm->tm_hour,
                    tstoptm->tm_min, tstoptm->tm_sec, (int)(0.000001 * (tstop.tv_nsec)));
            strncpy(processinfo->statusmsg, msgstring, 200);
        }

        if(processinfo->loopstat == 1)
        {
            sprintf(msgstring, "Loop exit %02d:%02d:%02d.%03d", tstoptm->tm_hour,
                    tstoptm->tm_min, tstoptm->tm_sec, (int)(0.000001 * (tstop.tv_nsec)));
            strncpy(processinfo->statusmsg, msgstring, 200);
        }

        processinfo->loopstat = 3; // clean exit
    }

    return 0;
}






int processinfo_SIGexit(PROCESSINFO *processinfo, int SignalNumber)
{
    char       timestring[200];
    struct     timespec tstop;
    struct tm *tstoptm;
    char       msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

    clock_gettime(CLOCK_REALTIME, &tstop);
    tstoptm = gmtime(&tstop.tv_sec);

    sprintf(timestring, "%02d:%02d:%02d.%03d", tstoptm->tm_hour, tstoptm->tm_min,
            tstoptm->tm_sec, (int)(0.000001 * (tstop.tv_nsec)));
    processinfo->loopstat = 3; // clean exit


    char SIGstr[12];
    int SIGflag = 0;
    switch(SignalNumber)
    {

        case SIGHUP :  // Hangup detected on controlling terminal or death of controlling process
            strcpy(SIGstr, "SIGHUP");
            SIGflag = 1;
            break;

        case SIGINT :  // Interrupt from keyboard
            strcpy(SIGstr, "SIGINT");
            SIGflag = 1;
            break;

        case SIGQUIT :  // Quit from keyboard
            strcpy(SIGstr, "SIGQUIT");
            SIGflag = 1;
            break;

        case SIGILL :  // Illegal Instruction
            strcpy(SIGstr, "SIGILL");
            SIGflag = 1;
            break;

        case SIGABRT :  // Abort signal from abort
            strcpy(SIGstr, "SIGABRT");
            SIGflag = 1;
            break;

        case SIGFPE :  // Floating-point exception
            strcpy(SIGstr, "SIGFPE");
            SIGflag = 1;
            break;

        case SIGKILL :  // Kill signal
            strcpy(SIGstr, "SIGKILL");
            SIGflag = 1;
            break;

        case SIGSEGV :  // Invalid memory reference
            strcpy(SIGstr, "SIGSEGV");
            SIGflag = 1;
            break;

        case SIGPIPE :  // Broken pipe: write to pipe with no readers
            strcpy(SIGstr, "SIGPIPE");
            SIGflag = 1;
            break;

        case SIGALRM :  // Timer signal from alarm
            strcpy(SIGstr, "SIGALRM");
            SIGflag = 1;
            break;

        case SIGTERM :  // Termination signal
            strcpy(SIGstr, "SIGTERM");
            SIGflag = 1;
            break;

        case SIGUSR1 :  // User-defined signal 1
            strcpy(SIGstr, "SIGUSR1");
            SIGflag = 1;
            break;

        case SIGUSR2 :  // User-defined signal 1
            strcpy(SIGstr, "SIGUSR2");
            SIGflag = 1;
            break;

        case SIGCHLD :  // Child stopped or terminated
            strcpy(SIGstr, "SIGCHLD");
            SIGflag = 1;
            break;

        case SIGCONT :  // Continue if stoppedshmimTCPtransmit
            strcpy(SIGstr, "SIGCONT");
            SIGflag = 1;
            break;

        case SIGSTOP :  // Stop process
            strcpy(SIGstr, "SIGSTOP");
            SIGflag = 1;
            break;

        case SIGTSTP :  // Stop typed at terminal
            strcpy(SIGstr, "SIGTSTP");
            SIGflag = 1;
            break;

        case SIGTTIN :  // Terminal input for background process
            strcpy(SIGstr, "SIGTTIN");
            SIGflag = 1;
            break;

        case SIGTTOU :  // Terminal output for background process
            strcpy(SIGstr, "SIGTTOU");
            SIGflag = 1;
            break;

        case SIGBUS :  // Bus error (bad memory access)
            strcpy(SIGstr, "SIGBUS");
            SIGflag = 1;
            break;

        case SIGPOLL :  // Pollable event (Sys V).
            strcpy(SIGstr, "SIGPOLL");
            SIGflag = 1;
            break;

        case SIGPROF :  // Profiling timer expired
            strcpy(SIGstr, "SIGPROF");
            SIGflag = 1;
            break;

        case SIGSYS :  // Bad system call (SVr4)
            strcpy(SIGstr, "SIGSYS");
            SIGflag = 1;
            break;

        case SIGTRAP :  // Trace/breakpoint trap
            strcpy(SIGstr, "SIGTRAP");
            SIGflag = 1;
            break;

        case SIGURG :  // Urgent condition on socket (4.2BSD)
            strcpy(SIGstr, "SIGURG");
            SIGflag = 1;
            break;

        case SIGVTALRM :  // Virtual alarm clock (4.2BSD)
            strcpy(SIGstr, "SIGVTALRM");
            SIGflag = 1;
            break;

        case SIGXCPU :  // CPU time limit exceeded (4.2BSD)
            strcpy(SIGstr, "SIGXCPU");
            SIGflag = 1;
            break;

        case SIGXFSZ :  // File size limit exceeded (4.2BSD)
            strcpy(SIGstr, "SIGXFSZ");
            SIGflag = 1;
            break;
    }


    if(SIGflag == 1)
    {
        int slen = snprintf(msgstring, STRINGMAXLEN_PROCESSINFO_STATUSMSG, "%s at %s",
                            SIGstr, timestring);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_IMGNAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }

        processinfo_WriteMessage(processinfo, msgstring);
    }

    return 0;
}







int processinfo_WriteMessage(
    PROCESSINFO *processinfo,
    const char  *msgstring
)
{
    struct timespec tnow;
    struct tm *tmnow;

    DEBUG_TRACEPOINT(" ");

    clock_gettime(CLOCK_REALTIME, &tnow);
    tmnow = gmtime(&tnow.tv_sec);

    strcpy(processinfo->statusmsg, msgstring);

    DEBUG_TRACEPOINT(" ");

    fprintf(processinfo->logFile,
            "%02d:%02d:%02d.%06d  %8ld.%09ld  %06d  %s\n",
            tmnow->tm_hour, tmnow->tm_min, tmnow->tm_sec, (int)(0.001 * (tnow.tv_nsec)),
            tnow.tv_sec, tnow.tv_nsec,
            (int) processinfo->PID,
            msgstring);

    DEBUG_TRACEPOINT(" ");
    fflush(processinfo->logFile);
    return 0;
}




int processinfo_CatchSignals()
{
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

    return 0;
}



int processinfo_ProcessSignals(PROCESSINFO *processinfo)
{
    int loopOK = 1;
    // process signals

    if(data.signal_TERM == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGTERM);
    }

    if(data.signal_INT == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGINT);
    }

    if(data.signal_ABRT == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGABRT);
    }

    if(data.signal_BUS == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGBUS);
    }

    if(data.signal_SEGV == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGSEGV);
    }

    if(data.signal_HUP == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGHUP);
    }

    if(data.signal_PIPE == 1)
    {
        loopOK = 0;
        processinfo_SIGexit(processinfo, SIGPIPE);
    }

    return loopOK;
}





/** @brief Update ouput stream at completion of processinfo-enabled loop iteration
 *
 */
errno_t processinfo_update_output_stream(
    PROCESSINFO *processinfo,
    imageID outstreamID
)
{
    imageID IDin;

    IDin = processinfo->triggerstreamID;
    if(IDin > -1)
    {
        int sptisize = data.image[IDin].md[0].NBproctrace - 1;

        // copy streamproctrace from input to output
        memcpy(&data.image[outstreamID].streamproctrace[1],
               &data.image[IDin].streamproctrace[0], sizeof(STREAM_PROC_TRACE)*sptisize);
    }

    struct timespec ts;
    if(clock_gettime(CLOCK_REALTIME, &ts) == -1)
    {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    // write first streamproctrace entry
    data.image[outstreamID].streamproctrace[0].triggermode      =
        processinfo->triggermode;

    data.image[outstreamID].streamproctrace[0].procwrite_PID    = getpid();

    data.image[outstreamID].streamproctrace[0].trigger_inode    =
        processinfo->triggerstreaminode;

    data.image[outstreamID].streamproctrace[0].ts_procstart     =
        processinfo->texecstart[processinfo->timerindex];

    data.image[outstreamID].streamproctrace[0].ts_streamupdate  = ts;

    data.image[outstreamID].streamproctrace[0].trigsemindex     =
        processinfo->triggersem;

    data.image[outstreamID].streamproctrace[0].triggerstatus    =
        processinfo->triggerstatus;

    if(IDin > -1)
    {
        data.image[outstreamID].streamproctrace[0].cnt0             =
            data.image[IDin].md[0].cnt0;
    }

    DEBUG_TRACEPOINT(" ");

    data.image[outstreamID].md[0].cnt0++;
    data.image[outstreamID].md[0].write = 0;
    ImageStreamIO_sempost(&data.image[outstreamID], -1); // post all semaphores

    return RETURN_SUCCESS;
}







int processinfo_exec_start(
    PROCESSINFO *processinfo
)
{
    DEBUG_TRACEPOINT(" ");
    if(processinfo->MeasureTiming == 1)
    {

        processinfo->timerindex ++;
        if(processinfo->timerindex == PROCESSINFO_NBtimer)
        {
            processinfo->timerindex = 0;
            processinfo->timingbuffercnt++;
        }

        clock_gettime(CLOCK_REALTIME,
                      &processinfo->texecstart[processinfo->timerindex]);

        if(processinfo->dtiter_limit_enable != 0)
        {
            long dtiter;
            int timerindexlast;

            if(processinfo->timerindex == 0)
            {
                timerindexlast = PROCESSINFO_NBtimer - 1;
            }
            else
            {
                timerindexlast = processinfo->timerindex - 1;
            }

            dtiter = processinfo->texecstart[processinfo->timerindex].tv_nsec -
                     processinfo->texecstart[timerindexlast].tv_nsec;
            dtiter += 1000000000 * (processinfo->texecstart[processinfo->timerindex].tv_sec
                                    - processinfo->texecstart[timerindexlast].tv_sec);



            if(dtiter > processinfo->dtiter_limit_value)
            {
                char msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

                {
                    int slen = snprintf(msgstring,
                                        STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                                        "dtiter %4ld  %4d %6.1f us  > %6.1f us",
                                        processinfo->dtiter_limit_cnt,
                                        processinfo->timerindex,
                                        0.001 * dtiter,
                                        0.001 * processinfo->dtiter_limit_value);
                    if(slen < 1)
                    {
                        PRINT_ERROR("snprintf wrote <1 char");
                        abort(); // can't handle this error any other way
                    }
                    if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
                    {
                        PRINT_ERROR("snprintf string truncation");
                        abort(); // can't handle this error any other way
                    }
                }

                processinfo_WriteMessage(processinfo, msgstring);

                if(processinfo->dtiter_limit_enable == 2)   // pause process due to timing limit
                {
                    processinfo->CTRLval = 1;
                    sprintf(msgstring, "dtiter lim -> paused");
                    processinfo_WriteMessage(processinfo, msgstring);
                }
                processinfo->dtiter_limit_cnt ++;
            }
        }
    }
    DEBUG_TRACEPOINT(" ");
    return 0;
}



int processinfo_exec_end(
    PROCESSINFO *processinfo
)
{
    int loopOK = 1;

    DEBUG_TRACEPOINT("End of execution loop, measure timing = %d",
                     processinfo->MeasureTiming);
    if(processinfo->MeasureTiming == 1)
    {
        clock_gettime(CLOCK_REALTIME, &processinfo->texecend[processinfo->timerindex]);

        if(processinfo->dtexec_limit_enable != 0)
        {
            long dtexec;

            dtexec = processinfo->texecend[processinfo->timerindex].tv_nsec -
                     processinfo->texecstart[processinfo->timerindex].tv_nsec;
            dtexec += 1000000000 * (processinfo->texecend[processinfo->timerindex].tv_sec -
                                    processinfo->texecend[processinfo->timerindex].tv_sec);

            if(dtexec > processinfo->dtexec_limit_value)
            {
                char msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

                {
                    int slen = snprintf(msgstring,
                                        STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                                        "dtexec %4ld  %4d %6.1f us  > %6.1f us",
                                        processinfo->dtexec_limit_cnt,
                                        processinfo->timerindex,
                                        0.001 * dtexec, 0.001 * processinfo->dtexec_limit_value);
                    if(slen < 1)
                    {
                        PRINT_ERROR("snprintf wrote <1 char");
                        abort(); // can't handle this error any other way
                    }
                    if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
                    {
                        PRINT_ERROR("snprintf string truncation");
                        abort(); // can't handle this error any other way
                    }
                }
                processinfo_WriteMessage(processinfo, msgstring);

                if(processinfo->dtexec_limit_enable == 2)   // pause process due to timing limit
                {
                    processinfo->CTRLval = 1;
                    {
                        int slen = snprintf(msgstring, STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                                            "dtexec lim -> paused");
                        if(slen < 1)
                        {
                            PRINT_ERROR("snprintf wrote <1 char");
                            abort(); // can't handle this error any other way
                        }
                        if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
                        {
                            PRINT_ERROR("snprintf string truncation");
                            abort(); // can't handle this error any other way
                        }
                    }

                    processinfo_WriteMessage(processinfo, msgstring);
                }
                processinfo->dtexec_limit_cnt ++;
            }
        }
    }
    DEBUG_TRACEPOINT("End of execution loop: check signals");
    loopOK = processinfo_ProcessSignals(processinfo);

    processinfo->loopcnt++;

    return loopOK;  // returns 0 if signal stops loop
}






static int processtools__print_header(const char *str, char c)
{
    long n;
    long i;

    attron(A_BOLD);
    n = strlen(str);
    for(i = 0; i < (wcol - n) / 2; i++)
    {
        printw("%c", c);
    }
    printw("%s", str);
    for(i = 0; i < (wcol - n) / 2 - 1; i++)
    {
        printw("%c", c);
    }
    printw("\n");
    attroff(A_BOLD);


    return(0);
}



/**
 * INITIALIZE ncurses
 *
 */
static errno_t initncurses()
{
    if(initscr() == NULL)
    {
        fprintf(stderr, "Error initialising ncurses.\n");
        exit(EXIT_FAILURE);
    }
    getmaxyx(stdscr, wrow, wcol);		/* get the number of rows and columns */
    wcolmax = wcol;

    cbreak();
    // disables line buffering and erase/kill character-processing (interrupt and flow control characters are unaffected),
    // making characters typed by the user immediately available to the program

    keypad(stdscr, TRUE);
    // enable F1, F2 etc..

    nodelay(stdscr, TRUE);
    curs_set(0);


    noecho();			/* Don't echo() while we do getch */



    init_color(COLOR_GREEN, 700, 1000, 700);
    init_color(COLOR_YELLOW, 1000, 1000, 700);

    start_color();



    //  color background
    init_pair(1, COLOR_BLACK, COLOR_WHITE);
    init_pair(2, COLOR_BLACK, COLOR_GREEN);
    init_pair(3, COLOR_BLACK, COLOR_YELLOW);
    init_pair(4, COLOR_WHITE, COLOR_RED);
    init_pair(5, COLOR_WHITE, COLOR_BLUE);

    init_pair(6, COLOR_GREEN, COLOR_BLACK);
    init_pair(7, COLOR_YELLOW, COLOR_BLACK);
    init_pair(8, COLOR_RED, COLOR_BLACK);
    init_pair(9, COLOR_BLACK, COLOR_RED);


    return RETURN_SUCCESS;
}




/**
 * ## Purpose
 *
 * detects the number of CPU and fill the cpuids
 *
 * ## Description
 *
 * populates cpuids array with the global system PU numbers in the physical order:
 * [PU0 of CPU0, PU1 of CPU0, ... PU0 of CPU1, PU1 of CPU1, ...]
 *
 */

int GetNumberCPUs(PROCINFOPROC *pinfop)
{
    int pu_index = 0;

#ifdef USE_HWLOC

    static int initStatus = 0;

    if(initStatus == 0)
    {
        initStatus = 1;
        unsigned int depth = 0;
        hwloc_topology_t topology;

        /* Allocate and initialize topology object. */
        hwloc_topology_init(&topology);

        /* ... Optionally, put detection configuration here to ignore
           some objects types, define a synthetic topology, etc....
           The default is to detect all the objects of the machine that
           the caller is allowed to access.  See Configure Topology
           Detection. */

        /* Perform the topology detection. */
        hwloc_topology_load(topology);

        depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
        pinfop->NBcpusocket = hwloc_get_nbobjs_by_depth(topology, depth);

        depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
        pinfop->NBcpus = hwloc_get_nbobjs_by_depth(topology, depth);

        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, depth, 0);
        do
        {
            pinfop->CPUids[pu_index] = obj->os_index;
            ++pu_index;
            obj = obj->next_cousin;
        }
        while(obj != NULL);

        hwloc_topology_destroy(topology);
    }

#else

    FILE *fpout;
    char outstring[16];
    char buf[100];

    //unsigned int tmp_index = 0;

    fpout = popen("getconf _NPROCESSORS_ONLN", "r");
    if(fpout == NULL)
    {
        printf("WARNING: cannot run command \"tmuxsessionname\"\n");
    }
    else
    {
        if(fgets(outstring, 16, fpout) == NULL)
        {
            printf("WARNING: fgets error\n");
        }
        pclose(fpout);
    }
    pinfop->NBcpus = atoi(outstring);

    fpout = popen("cat /proc/cpuinfo |grep \"physical id\" | awk '{ print $NF }'",
                  "r");
    pu_index = 0;
    pinfop->NBcpusocket = 1;
    while((fgets(buf, sizeof(buf), fpout) != NULL)  &&
            (pu_index < pinfop->NBcpus))
    {
        pinfop->CPUids[pu_index] = pu_index;
        pinfop->CPUphys[pu_index] = atoi(buf);

        //printf("cpu %2d belongs to Physical CPU %d\n", pu_index, pinfop->CPUphys[pu_index] );
        if(pinfop->CPUphys[pu_index] + 1 > pinfop->NBcpusocket)
        {
            pinfop->NBcpusocket = pinfop->CPUphys[pu_index] + 1;
        }

        pu_index++;
    }

#endif

    return(pinfop->NBcpus);
}





// unused
/*


static long getTopOutput()
{
	long NBtop = 0;

    char outstring[200];
    char command[200];
    FILE * fpout;
	int ret;

	clock_gettime(CLOCK_REALTIME, &t1);

    sprintf(command, "top -H -b -n 1");
    fpout = popen (command, "r");
    if(fpout==NULL)
    {
        printf("WARNING: cannot run command \"%s\"\n", command);
    }
    else
    {
		int startScan = 0;
		ret = 12;
        while( (fgets(outstring, 100, fpout) != NULL) && (NBtop<NBtopMax) && (ret==12) )
           {
			   if(startScan == 1)
			   {
				   ret = sscanf(outstring, "%d %s %s %d %s %s %s %s %f %f %s %s\n",
						&toparray_PID[NBtop],
						toparray_USER[NBtop],
						toparray_PR[NBtop],
						&toparray_NI[NBtop],
						 toparray_VIRT[NBtop],
						 toparray_RES[NBtop],
						 toparray_SHR[NBtop],
						 toparray_S[NBtop],
						&toparray_CPU[NBtop],
						&toparray_MEM[NBtop],
						 toparray_TIME[NBtop],
						 toparray_COMMAND[NBtop]
						);
				   NBtop++;
			   }

				if(strstr(outstring, "USER")!=NULL)
					startScan = 1;
		   }
        pclose(fpout);
    }
    clock_gettime(CLOCK_REALTIME, &t2);
	tdiff = timespec_diff(t1, t2);
	scantime_top += 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;

	return NBtop;
}



*/





static int GetCPUloads(PROCINFOPROC *pinfop)
{
    char      *line = NULL;
    size_t     maxstrlen = 256;
    FILE      *fp;
    ssize_t    read;
    int        cpu;
    long long  vall0, vall1, vall2, vall3, vall4, vall5, vall6, vall7, vall8;
    long long  v0, v1, v2, v3, v4, v5, v6, v7, v8;
    char       string0[80];

    static int cnt = 0;

    clock_gettime(CLOCK_REALTIME, &t1);

    line = (char *)malloc(sizeof(char) * maxstrlen);

    fp = fopen("/proc/stat", "r");
    if(fp == NULL)
    {
        exit(EXIT_FAILURE);
    }

    cpu = 0;

    read = getline(&line, &maxstrlen, fp);
    if(read == -1)
    {
        printf("[%s][%d]  ERROR: cannot read file\n", __FILE__, __LINE__);
        exit(EXIT_SUCCESS);
    }

    while(((read = getline(&line, &maxstrlen, fp)) != -1)
            && (cpu < pinfop->NBcpus))
    {

        sscanf(line, "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld",
               string0, &vall0, &vall1, &vall2, &vall3, &vall4, &vall5, &vall6, &vall7,
               &vall8);

        v0 = vall0 - pinfop->CPUcnt0[cpu];
        v1 = vall1 - pinfop->CPUcnt1[cpu];
        v2 = vall2 - pinfop->CPUcnt2[cpu];
        v3 = vall3 - pinfop->CPUcnt3[cpu];
        v4 = vall4 - pinfop->CPUcnt4[cpu];
        v5 = vall5 - pinfop->CPUcnt5[cpu];
        v6 = vall6 - pinfop->CPUcnt6[cpu];
        v7 = vall7 - pinfop->CPUcnt7[cpu];
        v8 = vall8 - pinfop->CPUcnt8[cpu];

        pinfop->CPUcnt0[cpu] = vall0;
        pinfop->CPUcnt1[cpu] = vall1;
        pinfop->CPUcnt2[cpu] = vall2;
        pinfop->CPUcnt3[cpu] = vall3;
        pinfop->CPUcnt4[cpu] = vall4;
        pinfop->CPUcnt5[cpu] = vall5;
        pinfop->CPUcnt6[cpu] = vall6;
        pinfop->CPUcnt7[cpu] = vall7;
        pinfop->CPUcnt8[cpu] = vall8;

        pinfop->CPUload[cpu] = (1.0 * v0 + v1 + v2 + v4 + v5 + v6) /
                               (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8);
        cpu++;
    }
    free(line);
    fclose(fp);
    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = timespec_diff(t1, t2);
    scantime_CPUload += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;


    clock_gettime(CLOCK_REALTIME, &t1);

    // number of process per CPU -> we can get that from ps
    char command[STRINGMAXLEN_COMMAND];
    char psoutfname[STRINGMAXLEN_FULLFILENAME];
    char procdname[200];
    processinfo_procdirname(procdname);

    WRITE_FULLFILENAME(psoutfname, "%s/_psoutput.txt", procdname);

    // use ps command to scan processes, store result in file psoutfname

    //    sprintf(command, "echo \"%5d CREATE\" >> cmdlog.txt\n", cnt);
    //    system(command);


    EXECUTE_SYSTEM_COMMAND("{ if [ ! -f %s/_psOKlock ]; then touch %s/_psOKlock; ps -e -o pid,psr,cpu,cmd > %s; fi; rm %s/_psOKlock &> /dev/null; }",
                           procdname, procdname, psoutfname, procdname);

    //    sprintf(command, "echo \"%5d CREATED\" >> cmdlog.txt\n", cnt);
    //    system(command);


    // read and process psoutfname file

    if(access(psoutfname, F_OK) != -1)
    {

        //        sprintf(command, "echo \"%5d READ\" >> cmdlog.txt\n", cnt);
        //        system(command);

        for(cpu = 0; cpu < pinfop->NBcpus; cpu++)
        {
            char outstring[STRINGMAXLEN_DEFAULT];
            FILE *fpout;
            {
                int slen = snprintf(command, STRINGMAXLEN_COMMAND,
                                    "CORENUM=%d; cat %s | grep -E  \"^[[:space:]][[:digit:]]+[[:space:]]+${CORENUM}\"|wc -l",
                                    cpu, psoutfname);
                if(slen < 1)
                {
                    PRINT_ERROR("snprintf wrote <1 char");
                    abort(); // can't handle this error any other way
                }
                if(slen >= STRINGMAXLEN_COMMAND)
                {
                    PRINT_ERROR("snprintf string truncation");
                    abort(); // can't handle this error any other way
                }
            }
            fpout = popen(command, "r");
            if(fpout == NULL)
            {
                printf("WARNING: cannot run command \"%s\"\n", command);
            }
            else
            {
                if(fgets(outstring, STRINGMAXLEN_DEFAULT, fpout) == NULL)
                {
                    printf("WARNING: fgets error\n");
                }
                pclose(fpout);
                pinfop->CPUpcnt[cpu] = atoi(outstring);
            }
        }
        //        sprintf(command, "echo \"%5d REMOVE\" >> cmdlog.txt\n", cnt);
        //        system(command);
        remove(psoutfname);
    }
    cnt++;


    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = timespec_diff(t1, t2);
    scantime_CPUpcnt += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    return(cpu);
}
















// for Display Modes 2 and 3
//

static int PIDcollectSystemInfo(PROCESSINFODISP *pinfodisp, int level)
{

    // COLLECT INFO FROM SYSTEM
    FILE *fp;
    char fname[STRINGMAXLEN_FULLFILENAME];


    DEBUG_TRACEPOINT(" ");

    // cpuset

    int PID = pinfodisp->PID;

    DEBUG_TRACEPOINT(" ");

    clock_gettime(CLOCK_REALTIME, &t1);

    WRITE_FULLFILENAME(fname, "/proc/%d/task/%d/cpuset", PID, PID);

    fp = fopen(fname, "r");
    if(fp == NULL)
    {
        return -1;
    }
    if(fscanf(fp, "%s", pinfodisp->cpuset) != 1)
    {
        PRINT_ERROR("fscanf returns value != 1");
    }
    fclose(fp);
    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = timespec_diff(t1, t2);
    scantime_cpuset += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;


    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char string0[200];
    char string1[300];


    DEBUG_TRACEPOINT(" ");

    clock_gettime(CLOCK_REALTIME, &t1);
    if(level == 0)
    {
        //FILE *fpout;
        //char command[200];
        //char outstring[200];

        pinfodisp->subprocPIDarray[0] = PID;
        pinfodisp->NBsubprocesses = 1;

        // if(pinfodisp->threads > 1) // look for children
        // {
        DIR *dp;
        struct dirent *ep;
        char dirname[STRINGMAXLEN_FULLFILENAME];

        // fprintf(stderr, "reading /proc/%d/task\n", PID);
        WRITE_FULLFILENAME(dirname, "/proc/%d/task/", PID);
        //sprintf(dirname, "/proc/%d/task/", PID);
        dp = opendir(dirname);

        if(dp != NULL)
        {
            while((ep = readdir(dp)))
            {
                if(ep->d_name[0] != '.')
                {
                    int subPID = atoi(ep->d_name);
                    if(subPID != PID)
                    {
                        pinfodisp->subprocPIDarray[pinfodisp->NBsubprocesses] = atoi(ep->d_name);
                        pinfodisp->NBsubprocesses++;
                    }
                }
            }
            closedir(dp);
        }
        else
        {
            return -1;
        }
        // }
        // fprintf(stderr, "%d threads found\n", pinfodisp->NBsubprocesses);
        pinfodisp->threads = pinfodisp->NBsubprocesses;
    }
    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = timespec_diff(t1, t2);
    scantime_pstree += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;


    // read /proc/PID/status
#ifdef CMDPROC_PROCSTAT
    for(int spindex = 0; spindex < pinfodisp->NBsubprocesses; spindex++)
    {
        clock_gettime(CLOCK_REALTIME, &t1);
        PID = pinfodisp->subprocPIDarray[spindex];


        WRITE_FULLFILENAME(fname, "/proc/%d/status", PID);
        fp = fopen(fname, "r");
        if(fp == NULL)
        {
            return -1;
        }

        while((read = getline(&line, &len, fp)) != -1)
        {
            if(sscanf(line, "%31[^:]: %s", string0, string1) == 2)
            {
                if(spindex == 0)
                {
                    if(strcmp(string0, "Cpus_allowed_list") == 0)
                    {
                        strcpy(pinfodisp->cpusallowed, string1);
                    }

                    if(strcmp(string0, "Threads") == 0)
                    {
                        pinfodisp->threads = atoi(string1);
                    }
                }

                if(strcmp(string0, "VmRSS") == 0)
                {
                    pinfodisp->VmRSSarray[spindex] = atol(string1);
                }

                if(strcmp(string0, "nonvoluntary_ctxt_switches") == 0)
                {
                    pinfodisp->ctxtsw_nonvoluntary[spindex] = atoi(string1);
                }
                if(strcmp(string0, "voluntary_ctxt_switches") == 0)
                {
                    pinfodisp->ctxtsw_voluntary[spindex] = atoi(string1);
                }
            }
        }

        fclose(fp);
        if(line)
        {
            free(line);
        }
        line = NULL;
        len = 0;

        clock_gettime(CLOCK_REALTIME, &t2);
        tdiff = timespec_diff(t1, t2);
        scantime_status += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;





        // read /proc/PID/stat
        clock_gettime(CLOCK_REALTIME, &t1);
        WRITE_FULLFILENAME(fname, "/proc/%d/stat", PID);

        int           stat_pid;       // (1) The process ID.
        char
        stat_comm[20];  // (2) The filename of the executable, in parentheses.
        char          stat_state;     // (3)
        /* One of the following characters, indicating process state:
                            R  Running
                            S  Sleeping in an interruptible wait
                            D  Waiting in uninterruptible disk sleep
                            Z  Zombie
                            T  Stopped (on a signal) or (before Linux 2.6.33)
                            trace stopped
                            t  Tracing stop (Linux 2.6.33 onward)
                            W  Paging (only before Linux 2.6.0)
                            X  Dead (from Linux 2.6.0 onward)
                            x  Dead (Linux 2.6.33 to 3.13 only)
                            K  Wakekill (Linux 2.6.33 to 3.13 only)
                            W  Waking (Linux 2.6.33 to 3.13 only)
                            P  Parked (Linux 3.9 to 3.13 only)
                    */
        int           stat_ppid;      // (4) The PID of the parent of this process.
        int           stat_pgrp;      // (5) The process group ID of the process
        int           stat_session;   // (6) The session ID of the process
        int           stat_tty_nr;    // (7) The controlling terminal of the process
        int
        stat_tpgid;     // (8) The ID of the foreground process group of the controlling terminal of the process
        unsigned int  stat_flags;     // (9) The kernel flags word of the process
        unsigned long
        stat_minflt;    // (10) The number of minor faults the process has made which have not required loading a memory page from disk
        unsigned long
        stat_cminflt;   // (11) The number of minor faults that the process's waited-for children have made
        unsigned long
        stat_majflt;    // (12) The number of major faults the process has made which have required loading a memory page from disk
        unsigned long
        stat_cmajflt;   // (13) The number of major faults that the process's waited-for children have made
        unsigned long
        stat_utime;     // (14) Amount of time that this process has been scheduled in user mode, measured in clock ticks (divide by sysconf(_SC_CLK_TCK)).
        unsigned long
        stat_stime;     // (15) Amount of time that this process has been scheduled in kernel mode, measured in clock ticks
        long
        stat_cutime;       // (16) Amount of time that this process's waited-for children have been scheduled in user mode, measured in clock ticks
        long
        stat_cstime;       // (17) Amount of time that this process's waited-for children have been scheduled in kernel mode, measured in clock ticks
        long
        stat_priority;     // (18) (Explanation for Linux 2.6) For processes running a
        /*                  real-time scheduling policy (policy below; see
                            sched_setscheduler(2)), this is the negated schedul‐
                            ing priority, minus one; that is, a number in the
                            range -2 to -100, corresponding to real-time priori‐
                            ties 1 to 99.  For processes running under a non-
                            real-time scheduling policy, this is the raw nice
                            value (setpriority(2)) as represented in the kernel.
                            The kernel stores nice values as numbers in the
                            range 0 (high) to 39 (low), corresponding to the
                            user-visible nice range of -20 to 19.

                            Before Linux 2.6, this was a scaled value based on
                            the scheduler weighting given to this process.*/
        long
        stat_nice;         // (19) The nice value (see setpriority(2)), a value in the range 19 (low priority) to -20 (high priority)
        long          stat_num_threads;  // (20) Number of threads in this process
        long          stat_itrealvalue;  // (21) hard coded as 0
        unsigned long long
        stat_starttime; // (22) The time the process started after system boot in clock ticks
        unsigned long stat_vsize;        // (23)  Virtual memory size in bytes
        long
        stat_rss;          // (24) Resident Set Size: number of pages the process has in real memory
        unsigned long
        stat_rsslim;       // (25) Current soft limit in bytes on the rss of the process
        unsigned long
        stat_startcode;    // (26) The address above which program text can run
        unsigned long
        stat_endcode;      // (27) The address below which program text can run
        unsigned long
        stat_startstack;   // (28) The address of the start (i.e., bottom) of the stack
        unsigned long
        stat_kstkesp;      // (29) The current value of ESP (stack pointer), as found in the kernel stack page for the process
        unsigned long stat_kstkeip;      // (30) The current EIP (instruction pointer)
        unsigned long
        stat_signal;       // (31) The bitmap of pending signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead
        unsigned long
        stat_blocked;      // (32) The bitmap of blocked signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long
        stat_sigignore;    // (33) The bitmap of ignored signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long
        stat_sigcatch;     // (34) The bitmap of ignored signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long
        stat_wchan;        // (35) This is the "channel" in which the process is waiting.  It is the address of a location in the kernel where the process is sleeping.  The corresponding symbolic name can be found in /proc/[pid]/wchan.
        unsigned long
        stat_nswap;        // (36) Number of pages swapped (not maintained)
        unsigned long
        stat_cnswap;       // (37) Cumulative nswap for child processes (not maintained)
        int           stat_exit_signal;  // (38) Signal to be sent to parent when we die
        int           stat_processor;    // (39) CPU number last executed on
        unsigned int
        stat_rt_priority;  // (40) Real-time scheduling priority, a number in the range 1 to 99 for processes scheduled under a real-time policy, or 0, for non-real-time processes (see  sched_setscheduler(2)).
        unsigned int
        stat_policy;       // (41) Scheduling policy (see sched_setscheduler(2))
        unsigned long long
        stat_delayacct_blkio_ticks; // (42) Aggregated block I/O delays, measured in clock ticks
        unsigned long
        stat_guest_time;   // (43) Guest time of the process (time spent running a virtual CPU for a guest operating system), measured in clock ticks
        long
        stat_cguest_time;  // (44) Guest time of the process's children, measured in clock ticks (divide by sysconf(_SC_CLK_TCK)).
        unsigned long
        stat_start_data;   // (45) Address above which program initialized and uninitialized (BSS) data are placed
        unsigned long
        stat_end_data;     // (46) ddress below which program initialized and uninitialized (BSS) data are placed
        unsigned long
        stat_start_brk;    // (47) Address above which program heap can be expanded with brk(2)
        unsigned long
        stat_arg_start;    // (48) Address above which program command-line arguments (argv) are placed
        unsigned long
        stat_arg_end;      // (49) Address below program command-line arguments (argv) are placed
        unsigned long
        stat_env_start;    // (50) Address above which program environment is placed
        unsigned long
        stat_env_end;      // (51) Address below which program environment is placed
        long
        stat_exit_code;    // (52) The thread's exit status in the form reported by waitpid(2)





        fp = fopen(fname, "r");
        int Nfields;
        if(fp == NULL)
        {
            return -1;
        }

        Nfields = fscanf(fp,
                         "%d %s %c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu %ld %ld %ld %ld %ld %ld %llu %lu %ld %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %d %d %u %u %llu %lu %ld %lu %lu %lu %lu %lu %lu %lu %ld\n",
                         &stat_pid,      //  1
                         stat_comm,
                         &stat_state,
                         &stat_ppid,
                         &stat_pgrp,
                         &stat_session,
                         &stat_tty_nr,
                         &stat_tpgid,
                         &stat_flags,
                         &stat_minflt,   //  10
                         &stat_cminflt,
                         &stat_majflt,
                         &stat_cmajflt,
                         &stat_utime,
                         &stat_stime,
                         &stat_cutime,
                         &stat_cstime,
                         &stat_priority,
                         &stat_nice,
                         &stat_num_threads,  // 20
                         &stat_itrealvalue,
                         &stat_starttime,
                         &stat_vsize,
                         &stat_rss,
                         &stat_rsslim,
                         &stat_startcode,
                         &stat_endcode,
                         &stat_startstack,
                         &stat_kstkesp,
                         &stat_kstkeip,  // 30
                         &stat_signal,
                         &stat_blocked,
                         &stat_sigignore,
                         &stat_sigcatch,
                         &stat_wchan,
                         &stat_nswap,
                         &stat_cnswap,
                         &stat_exit_signal,
                         &stat_processor,
                         &stat_rt_priority,  // 40
                         &stat_policy,
                         &stat_delayacct_blkio_ticks,
                         &stat_guest_time,
                         &stat_cguest_time,
                         &stat_start_data,
                         &stat_end_data,
                         &stat_start_brk,
                         &stat_arg_start,
                         &stat_arg_end,
                         &stat_env_start,   // 50
                         &stat_env_end,
                         &stat_exit_code
                        );
        if(Nfields != 52)
        {
            PRINT_ERROR("fscanf returns value != 1");
            pinfodisp->processorarray[spindex] = stat_processor;
            pinfodisp->rt_priority = stat_rt_priority;
        }
        else
        {
            pinfodisp->processorarray[spindex] = stat_processor;
            pinfodisp->rt_priority = stat_rt_priority;
        }
        fclose(fp);

        pinfodisp->sampletimearray[spindex] = 1.0 * t1.tv_sec + 1.0e-9 * t1.tv_nsec;

        pinfodisp->cpuloadcntarray[spindex] = (stat_utime + stat_stime);
        pinfodisp->memload = 0.0;

        clock_gettime(CLOCK_REALTIME, &t2);
        tdiff = timespec_diff(t1, t2);
        scantime_stat += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
    }
#endif


    DEBUG_TRACEPOINT(" ");

    return 0;

}






/**
 * ## Purpose
 *
 * Creates list of CPU sets
 *
 * ## Description
 *
 * Uses command: cset set -l
 *
 *
 */

int processinfo_CPUsets_List(STRINGLISTENTRY *CPUsetList)
{
    char line[200];
    FILE *fp;
    int NBsetMax = 1000;
    int setindex;
    char word[200];
    char word1[200];
    int NBset = 0;

    EXECUTE_SYSTEM_COMMAND("cset set -l | awk '/root/{stop=1} stop==1{print $0}' > _tmplist.txt");

    // first scan: get number of entries
    fp = fopen("_tmplist.txt", "r");
    while(NBset < NBsetMax)
    {
        if(fgets(line, 199, fp) == NULL)
        {
            break;
        }
        NBset++;
//		printf("%3d: %s", NBset, line);
    }
    fclose(fp);


    setindex = 0;
    fp = fopen("_tmplist.txt", "r");
    while(1)
    {
        if(fgets(line, 199, fp) == NULL)
        {
            break;
        }
        sscanf(line, "%s %s", word, word1);
        strcpy(CPUsetList[setindex].name, word);
        strcpy(CPUsetList[setindex].description, word1);
        setindex++;
    }
    fclose(fp);

    return NBset;
}





int processinfo_SelectFromList(STRINGLISTENTRY *StringList, int NBelem)
{
    int selected = 0;
    long i;
    char buff[100];
    int inputOK;
    char *p;
    int strlenmax = 20;

    printf("%d entries in list:\n", NBelem);
    fflush(stdout);
    for(i = 0; i < NBelem; i++)
    {
        printf("   %3ld   : %16s   %s\n", i, StringList[i].name,
               StringList[i].description);
        fflush(stdout);
    }


    inputOK = 0;

    while(inputOK == 0)
    {
        printf("\nEnter a number: ");
        fflush(stdout);

        int stringindex = 0;
        char c;
        while(((c = getchar()) != 13) && (stringindex < strlenmax - 1))
        {
            buff[stringindex] = c;
            if(c == 127) // delete key
            {
                putchar(0x8);
                putchar(' ');
                putchar(0x8);
                stringindex --;
            }
            else
            {
                putchar(c);  // echo on screen
                stringindex++;
            }
            if(stringindex < 0)
            {
                stringindex = 0;
            }
        }
        buff[stringindex] = '\0';

        selected = strtol(buff, &p, strlenmax);

        if((selected < 0) || (selected > NBelem - 1))
        {
            printf("\nError: number not valid. Must be >= 0 and < %d\n", NBelem);
            inputOK = 0;
        }
        else
        {
            inputOK = 1;
        }
    }

    printf("Selected entry : %s\n", StringList[selected].name);


    return selected;
}












/**
 * ## Purpose
 *
 * Scan function for processinfo CTRL
 *
 * ## Description
 *
 * Runs in background loop as thread initiated by processinfo_CTRL
 *
 */
void *processinfo_scan(
    void *thptr
)
{
    PROCINFOPROC *pinfop;

    pinfop = (PROCINFOPROC *) thptr;

    long pindex;
    long pindexdisp;

    pinfop->loopcnt = 0;

    // timing
    static int       firstIter = 1;
    static struct    timespec t0;
    struct timespec  t1;
    double           tdiffv;
    struct timespec  tdiff;


    char  procdname[200];
    processinfo_procdirname(procdname);

    pinfop->scanPID = getpid();

    pinfop->scandebugline = __LINE__;

    while(pinfop->loop == 1)
    {
        DEBUG_TRACEPOINT(" ");

        pinfop->scandebugline = __LINE__;

        DEBUG_TRACEPOINT(" ");

        // timing measurement
        clock_gettime(CLOCK_REALTIME, &t1);
        if(firstIter == 1)
        {
            tdiffv = 0.1;
            firstIter = 0;
        }
        else
        {
            tdiff = timespec_diff(t0, t1);
            tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        }
        clock_gettime(CLOCK_REALTIME, &t0);
        pinfop->dtscan = tdiffv;

        DEBUG_TRACEPOINT(" ");



        pinfop->scandebugline = __LINE__;

        pinfop->SCANBLOCK_requested = 1;  // request scan
        //system("echo \"scanblock request write 1\" > steplog.sRQw1.txt");//TEST

        while(pinfop->SCANBLOCK_OK == 0)    // wait for display to OK scan
        {
            //system("echo \"scanblock OK read 0\" > steplog.sOKr0.txt");//TEST
            usleep(100);
            pinfop->scandebugline = __LINE__;
            if(pinfop->loop == 0)
            {
                int line = __LINE__;
                pthread_exit(&line);
            }
        }
        pinfop->SCANBLOCK_requested = 0; // acknowledge that request has been granted
        //system("echo \"scanblock request write 0\" > steplog.sRQw0.txt");//TEST

        DEBUG_TRACEPOINT(" ");

        // LOAD / UPDATE process information
        // This step re-mmaps pinfo and rebuilds list, so we need to ensure it is run exclusively of the dislpay
        //
        pinfop->scandebugline = __LINE__;

        DEBUG_TRACEPOINT(" ");

        for(pindex = 0; pindex < pinfop->NBpinfodisp; pindex++)
        {
            if(pinfop->loop == 1)
            {

                DEBUG_TRACEPOINT(" ");

                char SM_fname[STRINGMAXLEN_FULLFILENAME];    // shared memory file name
                struct stat file_stat;

                pinfop->scandebugline = __LINE__;
                pinfop->PIDarray[pindex] = pinfolist->PIDarray[pindex];

                // SHOULD WE (RE)LOAD ?
                if(pinfolist->active[pindex] == 0)   // inactive
                {
                    pinfop->updatearray[pindex] = 0;
                }

                if((pinfolist->active[pindex] == 1)
                        || (pinfolist->active[pindex] == 2))   // active or crashed
                {
                    pinfop->updatearray[pindex] = 1;

                }
                //    if(pinfolist->active[pindex] == 2) // mmap crashed, file may still be present
                //        updatearray[pindex] = 1;

                if(pinfolist->active[pindex] == 3)   // file has gone away
                {
                    pinfop->updatearray[pindex] = 0;
                }

                DEBUG_TRACEPOINT(" ");


                pinfop->scandebugline = __LINE__;


                // check if process info file exists
                WRITE_FULLFILENAME(SM_fname, "%s/proc.%s.%06d.shm", procdname,
                                   pinfolist->pnamearray[pindex], (int) pinfolist->PIDarray[pindex]);

                // Does file exist ?
                if(stat(SM_fname, &file_stat) == -1 && errno == ENOENT)
                {
                    // if not, don't (re)load and remove from process info list
                    pinfolist->active[pindex] = 0;
                    pinfop->updatearray[pindex] = 0;
                }

                DEBUG_TRACEPOINT(" ");


                if(pinfolist->active[pindex] == 1)
                {
                    // check if process still exists
                    struct stat sts;
                    char procfname[STRINGMAXLEN_FULLFILENAME];

                    WRITE_FULLFILENAME(procfname, "/proc/%d", (int) pinfolist->PIDarray[pindex]);
                    if(stat(procfname, &sts) == -1 && errno == ENOENT)
                    {
                        // process doesn't exist -> flag as inactive
                        pinfolist->active[pindex] = 2;
                    }
                }

                DEBUG_TRACEPOINT(" ");

                pinfop->scandebugline = __LINE__;

                if((pindex < pinfop->NBpinfodisp) && (pinfop->updatearray[pindex] == 1))
                {
                    // (RE)LOAD
                    //struct stat file_stat;

                    DEBUG_TRACEPOINT(" ");

                    // if already mmapped, first unmap
                    if(pinfop->pinfommapped[pindex] == 1)
                    {
                        processinfo_shm_close(pinfop->pinfoarray[pindex], pinfop->fdarray[pindex]);
                        pinfop->pinfommapped[pindex] = 0;
                    }


                    DEBUG_TRACEPOINT(" ");


                    // COLLECT INFORMATION FROM PROCESSINFO FILE
                    pinfop->pinfoarray[pindex] = processinfo_shm_link(SM_fname,
                                                 &pinfop->fdarray[pindex]);

                    if(pinfop->pinfoarray[pindex] == MAP_FAILED)
                    {
                        close(pinfop->fdarray[pindex]);
                        endwin();
                        fprintf(stderr, "[%d] Error mapping file %s\n", __LINE__, SM_fname);
                        pinfolist->active[pindex] = 3;
                        pinfop->pinfommapped[pindex] = 0;
                    }
                    else
                    {
                        pinfop->pinfommapped[pindex] = 1;
                        strncpy(pinfop->pinfodisp[pindex].name, pinfop->pinfoarray[pindex]->name,
                                40 - 1);

                        struct tm *createtm;
                        createtm      = gmtime(&pinfop->pinfoarray[pindex]->createtime.tv_sec);
                        pinfop->pinfodisp[pindex].createtime_hr = createtm->tm_hour;
                        pinfop->pinfodisp[pindex].createtime_min = createtm->tm_min;
                        pinfop->pinfodisp[pindex].createtime_sec = createtm->tm_sec;
                        pinfop->pinfodisp[pindex].createtime_ns =
                            pinfop->pinfoarray[pindex]->createtime.tv_nsec;

                        pinfop->pinfodisp[pindex].loopcnt = pinfop->pinfoarray[pindex]->loopcnt;
                    }

                    DEBUG_TRACEPOINT(" ");

                    pinfop->pinfodisp[pindex].active = pinfolist->active[pindex];
                    pinfop->pinfodisp[pindex].PID = pinfolist->PIDarray[pindex];

                    pinfop->pinfodisp[pindex].updatecnt ++;

                    // pinfop->updatearray[pindex] == 0; // by default, no need to re-connect

                    DEBUG_TRACEPOINT(" ");

                }

                pinfop->scandebugline = __LINE__;
            }
            else
            {
                int line = __LINE__;
                pthread_exit(&line);
            }
        }


        /** ### Build a time-sorted list of processes
          *
          *
          *
          */
        int index;

        DEBUG_TRACEPOINT(" ");

        pinfop->NBpindexActive = 0;
        for(pindex = 0; pindex < PROCESSINFOLISTSIZE; pindex++)
            if(pinfolist->active[pindex] != 0)
            {
                pinfop->pindexActive[pinfop->NBpindexActive] = pindex;
                pinfop->NBpindexActive++;
            }

        if(pinfop->NBpindexActive > 0)
        {

            double *timearray;
            long *indexarray;
            timearray  = (double *) malloc(sizeof(double) * pinfop->NBpindexActive);
            indexarray = (long *)   malloc(sizeof(long)  * pinfop->NBpindexActive);
            int listcnt = 0;
            for(index = 0; index < pinfop->NBpindexActive; index++)
            {
                pindex = pinfop->pindexActive[index];
                if(pinfop->pinfommapped[pindex] == 1)
                {
                    indexarray[index] = pindex;
                    // minus sign for most recent first
                    //printw("index  %ld  ->  pindex  %ld\n", index, pindex);
                    timearray[index] = -1.0 * pinfop->pinfoarray[pindex]->createtime.tv_sec - 1.0e-9
                                       * pinfop->pinfoarray[pindex]->createtime.tv_nsec;
                    listcnt++;
                }
            }
            DEBUG_TRACEPOINT(" ");

            pinfop->NBpindexActive = listcnt;
            quick_sort2l_double(timearray, indexarray, pinfop->NBpindexActive);

            for(index = 0; index < pinfop->NBpindexActive; index++)
            {
                pinfop->sorted_pindex_time[index] = indexarray[index];
            }

            DEBUG_TRACEPOINT(" ");

            free(timearray);
            free(indexarray);

        }

        pinfop->scandebugline = __LINE__;

        pinfop->SCANBLOCK_OK = 0; // let display thread we're done
        //system("echo \"scanblock OK write 0\" > steplog.sOKw0.txt");//TEST






        pinfop->scandebugline = __LINE__;

        DEBUG_TRACEPOINT(" ");



        if(pinfop->DisplayMode ==
                PROCCTRL_DISPLAYMODE_RESOURCES)   // only compute of displayed processes
        {
            DEBUG_TRACEPOINT(" ");
            pinfop->scandebugline = __LINE__;
            GetCPUloads(pinfop);
            pinfop->scandebugline = __LINE__;
            // collect required info for display
            for(pindexdisp = 0; pindexdisp < pinfop->NBpinfodisp ; pindexdisp++)
            {
                if(pinfop->loop == 1)
                {
                    DEBUG_TRACEPOINT(" ");

                    if(pinfolist->active[pindexdisp] != 0)
                    {
                        pinfop->scandebugline = __LINE__;

                        if(pinfop->pinfodisp[pindexdisp].NBsubprocesses !=
                                0)   // pinfop->pinfodisp[pindex].NBsubprocesses should never be zero - should be at least 1 (for main process)
                        {

                            int spindex; // sub process index, 0 for main

                            if(pinfop->psysinfostatus[pindexdisp] != -1)
                            {
                                for(spindex = 0; spindex < pinfop->pinfodisp[pindexdisp].NBsubprocesses;
                                        spindex++)
                                {
                                    // place info in subprocess arrays
                                    pinfop->pinfodisp[pindexdisp].sampletimearray_prev[spindex] =
                                        pinfop->pinfodisp[pindexdisp].sampletimearray[spindex];
                                    // Context Switches

                                    pinfop->pinfodisp[pindexdisp].ctxtsw_voluntary_prev[spindex]    =
                                        pinfop->pinfodisp[pindexdisp].ctxtsw_voluntary[spindex];
                                    pinfop->pinfodisp[pindexdisp].ctxtsw_nonvoluntary_prev[spindex] =
                                        pinfop->pinfodisp[pindexdisp].ctxtsw_nonvoluntary[spindex];


                                    // CPU use
                                    pinfop->pinfodisp[pindexdisp].cpuloadcntarray_prev[spindex] =
                                        pinfop->pinfodisp[pindexdisp].cpuloadcntarray[spindex];

                                }
                            }


                            pinfop->scandebugline = __LINE__;

                            pinfop->psysinfostatus[pindex] = PIDcollectSystemInfo(&
                                                             (pinfop->pinfodisp[pindexdisp]), 0);

                            if(pinfop->psysinfostatus[pindexdisp] != -1)
                            {
                                char cpuliststring[200];
                                char cpustring[16];

                                for(spindex = 0; spindex < pinfop->pinfodisp[pindexdisp].NBsubprocesses;
                                        spindex++)
                                {
                                    if(pinfop->pinfodisp[pindexdisp].sampletimearray[spindex] !=
                                            pinfop->pinfodisp[pindexdisp].sampletimearray_prev[spindex])
                                    {
                                        // get CPU and MEM load

                                        // THIS DOES NOT WORK ON TICKLESS KERNEL
                                        pinfop->pinfodisp[pindexdisp].subprocCPUloadarray[spindex] =
                                            100.0 * (
                                                (1.0 * pinfop->pinfodisp[pindexdisp].cpuloadcntarray[spindex]
                                                 - pinfop->pinfodisp[pindexdisp].cpuloadcntarray_prev[spindex])
                                                / sysconf(_SC_CLK_TCK))
                                            / (pinfop->pinfodisp[pindexdisp].sampletimearray[spindex]
                                               - pinfop->pinfodisp[pindexdisp].sampletimearray_prev[spindex]);

                                        pinfop->pinfodisp[pindexdisp].subprocCPUloadarray_timeaveraged[spindex] =
                                            0.9 * pinfop->pinfodisp[pindexdisp].subprocCPUloadarray_timeaveraged[spindex]
                                            + 0.1 * pinfop->pinfodisp[pindexdisp].subprocCPUloadarray[spindex];
                                    }
                                }

                                sprintf(cpuliststring, ",%s,", pinfop->pinfodisp[pindexdisp].cpusallowed);

                                pinfop->scandebugline = __LINE__;

                                int cpu;
                                for(cpu = 0; cpu < pinfop->NBcpus; cpu++)
                                {
                                    int cpuOK = 0;
                                    int cpumin, cpumax;

                                    sprintf(cpustring, ",%d,", pinfop->CPUids[cpu]);
                                    if(strstr(cpuliststring, cpustring) != NULL)
                                    {
                                        cpuOK = 1;
                                    }


                                    for(cpumin = 0; cpumin <= pinfop->CPUids[cpu]; cpumin++)
                                        for(cpumax = pinfop->CPUids[cpu]; cpumax < pinfop->NBcpus; cpumax++)
                                        {
                                            sprintf(cpustring, ",%d-%d,", cpumin, cpumax);
                                            if(strstr(cpuliststring, cpustring) != NULL)
                                            {
                                                cpuOK = 1;
                                            }
                                        }
                                    pinfop->pinfodisp[pindexdisp].cpuOKarray[cpu] = cpuOK;
                                }
                            }

                        }

                    }
                }
                else
                {
                    DEBUG_TRACEPOINT(" ");
                    int line = __LINE__;
                    pthread_exit(&line);
                }
            } // end of if(pinfop->DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES)

            pinfop->scandebugline = __LINE__;

        } // end of DisplayMode PROCCTRL_DISPLAYMODE_RESOURCES


        DEBUG_TRACEPOINT(" ");


        pinfop->loopcnt++;


        int loopcntiter = 0;
        int NBloopcntiter = 10;
        while((pinfop->loop == 1) && (loopcntiter < NBloopcntiter))
        {
            usleep(pinfop->twaitus / NBloopcntiter);
            loopcntiter++;
        }

        if(pinfop->loop == 0)
        {
            int line = __LINE__;
            pthread_exit(&line);
        }
    }


    if(pinfop->loop == 0)
    {
        int line = __LINE__;
        pthread_exit(&line);
    }

    return NULL;
}






void processinfo_CTRLscreen_atexit()
{
    //echo();
    //endwin();

    printf("EXIT from processinfo_CTRLscreen at line %d\n", CTRLscreenExitLine);
}







void processinfo_CTRLscreen_handle_winch(int __attribute__((unused)) sig)
{
    endwin();
    refresh();
    clear();

    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    wrow = w.ws_row;
    wcol = w.ws_col;

    if(wcol > wcolmax)
    {
        wcol = wcolmax;
    }
}




/**
 * ## Purpose
 *
 * Control screen for PROCESSINFO structures
 *
 * ## Description
 *
 * Relies on ncurses for display\n
 *
 *
 */
errno_t processinfo_CTRLscreen()
{
    long pindex, index;

    PROCINFOPROC
    procinfoproc;  // Main structure - holds everything that needs to be shared with other functions and scan thread
    pthread_t threadscan;

    int cpusocket;

    char pselected_FILE[200];
    char pselected_FUNCTION[200];
    int  pselected_LINE;

    // timers
    struct timespec t1loop;
    struct timespec t2loop;
    //struct timespec tdiffloop;

    struct timespec t01loop;
    struct timespec t02loop;
    struct timespec t03loop;
    struct timespec t04loop;
    struct timespec t05loop;
    struct timespec t06loop;
    struct timespec t07loop;


    float frequ = 32.0; // Hz
    char  monstring[200];

    // list of active indices
    int   pindexActiveSelected;
    int   pindexSelected;


    int listindex;

    int ToggleValue;

    DEBUG_TRACEPOINT(" ");

    char  procdname[200];
    processinfo_procdirname(procdname);

    processinfo_CatchSignals();


    struct sigaction sa;

    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sa.sa_handler = processinfo_CTRLscreen_handle_winch;
    if(sigaction(SIGWINCH, &sa, NULL) == -1)
    {
        printf("can't handle SIGWINCH");
        exit(EXIT_FAILURE);
    }


    setlocale(LC_ALL, "");



    // initialize procinfoproc entries
    procinfoproc.loopcnt = 0;
    for(pindex = 0; pindex < PROCESSINFOLISTSIZE; pindex++)
    {
        procinfoproc.pinfoarray[pindex]         = NULL;
        procinfoproc.pinfommapped[pindex]       = 0; // 1 if mmapped, 0 otherwise
        procinfoproc.PIDarray[pindex]           = 0; // used to track changes
        procinfoproc.updatearray[pindex]        = 1; // initialize: load all
        procinfoproc.fdarray[pindex]            = 0; // file descriptors
        procinfoproc.loopcntarray[pindex]       = 0;
        procinfoproc.loopcntoffsetarray[pindex] = 0;
        procinfoproc.selectedarray[pindex]      = 0; // initially not selected
        procinfoproc.sorted_pindex_time[pindex] = pindex;

        procinfoproc.pindexActive[pindex]       = 0;
        procinfoproc.psysinfostatus[pindex]     = 0;
    }
    procinfoproc.NBcpus      = 1;
    procinfoproc.NBcpusocket = 1;
    for(int cpu = 0; cpu < MAXNBCPU; cpu++)
    {

        procinfoproc.CPUload[cpu] = 0.0;

        procinfoproc.CPUcnt0[cpu] = 0;
        procinfoproc.CPUcnt1[cpu] = 0;
        procinfoproc.CPUcnt2[cpu] = 0;
        procinfoproc.CPUcnt3[cpu] = 0;
        procinfoproc.CPUcnt4[cpu] = 0;
        procinfoproc.CPUcnt5[cpu] = 0;
        procinfoproc.CPUcnt6[cpu] = 0;
        procinfoproc.CPUcnt7[cpu] = 0;
        procinfoproc.CPUcnt8[cpu] = 0;

        procinfoproc.CPUids[cpu]  = cpu;
        procinfoproc.CPUphys[cpu] = 0;
        procinfoproc.CPUpcnt[cpu] = 0;

    }

    STRINGLISTENTRY *CPUsetList;
    int NBCPUset;
    CPUsetList = (STRINGLISTENTRY *)malloc(sizeof(STRINGLISTENTRY) * 1000);
    NBCPUset = processinfo_CPUsets_List(CPUsetList);


    // Create / read process list
    //
    if(processinfo_shm_list_create() == 0)
    {
        printf("==== NO PROCESS TO DISPLAY -> EXITING ====\n");
        return(0);
    }


    // copy pointer
    procinfoproc.pinfolist = pinfolist;

    procinfoproc.NBcpus = GetNumberCPUs(&procinfoproc);
    GetCPUloads(&procinfoproc);




    // INITIALIZE ncurses
    initncurses();


    //atexit( processinfo_CTRLscreen_atexit );

    // set print string lengths
    char string[200]; // string to be printed. Used to keep track of total length
    int pstrlen_total; // Used to keep track of total length
    int pstrlen_total_max;

    int pstrlen_status  = 10;
    int pstrlen_pid     =  7;
    int pstrlen_pname   = 25;
    int pstrlen_state   =  5;
    // Clevel :  2
    // tstart : 12
    int pstrlen_tmux    = 16;
    int pstrlen_loopcnt = 10;
    int pstrlen_descr   = 25;

    int pstrlen_msg     = 35;
    int pstrlen_msg_min = 10;
    int pstrlen_msg_max = 50;

    int pstrlen_cset    = 10;

    int pstrlen_inode     = 10;
    int pstrlen_trigstreamname = 16;
    int pstrlen_missedfr  = 4;
    int pstrlen_missedfrc = 12;
    int pstrlen_tocnt     = 10;


    //	int pstrlen_total = 28 + pstrlen_status + pstrlen_pid + pstrlen_pname + pstrlen_state + pstrlen_tmux + pstrlen_loopcnt + pstrlen_descr + pstrlen_msg;



    clear();

    // redirect stderr to /dev/null

    int backstderr, newstderr;

    fflush(stderr);
    backstderr = dup(STDERR_FILENO);
    newstderr = open("/dev/null", O_WRONLY);
    dup2(newstderr, STDERR_FILENO);
    close(newstderr);




    procinfoproc.NBpinfodisp = wrow - 5;
    procinfoproc.pinfodisp = (PROCESSINFODISP *) malloc(sizeof(
                                 PROCESSINFODISP) * procinfoproc.NBpinfodisp);
    for(pindex = 0; pindex < procinfoproc.NBpinfodisp; pindex++)
    {
        procinfoproc.pinfodisp[pindex].NBsubprocesses =
            1;  // by default, each process is assumed to be single-threaded

        procinfoproc.pinfodisp[pindex].active         = 0;
        procinfoproc.pinfodisp[pindex].PID            = 0;
        strcpy(procinfoproc.pinfodisp[pindex].name, "null");
        procinfoproc.pinfodisp[pindex].updatecnt      = 0;

        procinfoproc.pinfodisp[pindex].loopcnt         = 0;
        procinfoproc.pinfodisp[pindex].loopstat        = 0;

        procinfoproc.pinfodisp[pindex].createtime_hr   = 0;
        procinfoproc.pinfodisp[pindex].createtime_min  = 0;
        procinfoproc.pinfodisp[pindex].createtime_sec  = 0;
        procinfoproc.pinfodisp[pindex].createtime_ns   = 0;

        strcpy(procinfoproc.pinfodisp[pindex].cpuset, "null");
        strcpy(procinfoproc.pinfodisp[pindex].cpusallowed, "null");
        for(int cpu = 0; cpu < MAXNBCPU; cpu++)
        {
            procinfoproc.pinfodisp[pindex].cpuOKarray[cpu] = 0;
        }
        procinfoproc.pinfodisp[pindex].threads         = 0;


        procinfoproc.pinfodisp[pindex].rt_priority     = 0;
        procinfoproc.pinfodisp[pindex].memload         = 0.0;


        strcpy(procinfoproc.pinfodisp[pindex].statusmsg, "");
        strcpy(procinfoproc.pinfodisp[pindex].tmuxname, "");


        procinfoproc.pinfodisp[pindex].NBsubprocesses = 1;
        for(int spi = 0; spi < MAXNBSUBPROCESS; spi++)
        {
            procinfoproc.pinfodisp[pindex].sampletimearray[spi]          = 0.0;
            procinfoproc.pinfodisp[pindex].sampletimearray_prev[spi]     = 0.0;

            procinfoproc.pinfodisp[pindex].ctxtsw_voluntary[spi]         = 0;
            procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary[spi]      = 0;
            procinfoproc.pinfodisp[pindex].ctxtsw_voluntary_prev[spi]    = 0;
            procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spi] = 0;

            procinfoproc.pinfodisp[pindex].cpuloadcntarray[spi]          = 0;
            procinfoproc.pinfodisp[pindex].cpuloadcntarray_prev[spi]     = 0;
            procinfoproc.pinfodisp[pindex].subprocCPUloadarray[spi]      = 0.0;
            procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spi] = 0.0;

            procinfoproc.pinfodisp[pindex].VmRSSarray[spi]               = 0;
            procinfoproc.pinfodisp[pindex].processorarray[spi]           = 0;

            procinfoproc.pinfodisp[pindex].subprocPIDarray[spi]          = 0;

        }
    }

    pindexActiveSelected = 0;
    procinfoproc.DisplayMode = PROCCTRL_DISPLAYMODE_CTRL; // default upon startup
    // display modes:
    // 2: overview
    // 3: CPU affinity

    // Start scan thread
    procinfoproc.loop = 1;
    procinfoproc.twaitus = 1000000; // 1 sec

    procinfoproc.SCANBLOCK_requested = 0;
    procinfoproc.SCANBLOCK_OK = 0;

    pthread_create(&threadscan, NULL, processinfo_scan, (void *) &procinfoproc);




    // wait for first scan to be completed
    procinfoproc.SCANBLOCK_OK = 1;
    while(procinfoproc.loopcnt < 1)
    {
        //printf("procinfoproc.loopcnt  = %ld\n", (long) procinfoproc.loopcnt);
        usleep(10000);
    }




    int loopOK = 1;
    int freeze = 0;
    long cnt = 0;
    int MonMode = 0;
    int TimeSorted =
        1;  // by default, sort processes by start time (most recent at top)
    int dispindexMax = 0;


    clear();
    int Xexit = 0; // toggles to 1 when users types x

    pindexSelected = 0;
    int pindexSelectedOK = 0; // no process selected by cursor

    while(loopOK == 1)
    {
        int pid;
        //char command[200];

        DEBUG_TRACEPOINT(" ");

        if(procinfoproc.SCANBLOCK_requested == 1)
        {
            //system("echo \"scanblock request read 1\" > steplog.dQRr1.txt");//TEST

            procinfoproc.SCANBLOCK_OK = 1;  // issue OK to scan thread
            //system("echo \"scanblock OK write 1\" > steplog.dOKw1.txt");//TEST

            // wait for scan thread to have completed scan
            while(procinfoproc.SCANBLOCK_OK == 1)
            {
                //system("echo \"scanblock OK read 1\" > steplog.dOKr1.txt");//TEST
                usleep(100);
            }
            //system("echo \"scanblock OK read 0\" > steplog.dOKr0.txt");//TEST
        }
        //		else
        //			system("echo \"scanblock request read 0\" > steplog.dQRr0.txt");//TEST


        usleep((long)(1000000.0 / frequ));
        int ch = getch();

        clock_gettime(CLOCK_REALTIME, &t1loop);

        scantime_cpuset = 0.0;
        scantime_status = 0.0;
        scantime_stat = 0.0;
        scantime_pstree = 0.0;
        scantime_top = 0.0;
        scantime_CPUload = 0.0;
        scantime_CPUpcnt = 0.0;

        DEBUG_TRACEPOINT(" ");


        if(freeze == 0)
        {
            attron(A_BOLD);
            sprintf(monstring, "Mode %d   PRESS x TO STOP MONITOR", MonMode);
            processtools__print_header(monstring, '-');
            attroff(A_BOLD);
        }

        int selectedOK = 0; // goes to 1 if at least one process is selected
        switch(ch)
        {
            case 'f':     // Freeze screen (toggle)
                if(freeze == 0)
                {
                    freeze = 1;
                }
                else
                {
                    freeze = 0;
                }
                break;

            case 'x':     // Exit control screen
                loopOK = 0;
                Xexit = 1;
                break;

            case ' ':     // Mark current PID as selected (if none selected, other commands only apply to highlighted process)
                pindex = pindexSelected;
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    procinfoproc.selectedarray[pindex] = 0;
                }
                else
                {
                    procinfoproc.selectedarray[pindex] = 1;
                }
                break;

            case 'u':    // undelect all
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    procinfoproc.selectedarray[pindex] = 0;
                }
                break;




            case KEY_UP:
                pindexActiveSelected --;
                if(pindexActiveSelected < 0)
                {
                    pindexActiveSelected = 0;
                }
                if(TimeSorted == 0)
                {
                    pindexSelected = procinfoproc.pindexActive[pindexActiveSelected];
                }
                else
                {
                    pindexSelected = procinfoproc.sorted_pindex_time[pindexActiveSelected];
                }
                break;

            case KEY_DOWN:
                pindexActiveSelected ++;
                if(pindexActiveSelected > procinfoproc.NBpindexActive - 1)
                {
                    pindexActiveSelected = procinfoproc.NBpindexActive - 1;
                }
                if(TimeSorted == 0)
                {
                    pindexSelected = procinfoproc.pindexActive[pindexActiveSelected];
                }
                else
                {
                    pindexSelected = procinfoproc.sorted_pindex_time[pindexActiveSelected];
                }
                break;


            case KEY_RIGHT:
                procinfoproc.DisplayDetailedMode = 1;
                break;


            case KEY_LEFT:
                procinfoproc.DisplayDetailedMode = 0;
                break;


            case KEY_PPAGE:
                pindexActiveSelected -= 10;
                if(pindexActiveSelected < 0)
                {
                    pindexActiveSelected = 0;
                }
                if(TimeSorted == 0)
                {
                    pindexSelected = procinfoproc.pindexActive[pindexActiveSelected];
                }
                else
                {
                    pindexSelected = procinfoproc.sorted_pindex_time[pindexActiveSelected];
                }
                break;

            case KEY_NPAGE:
                pindexActiveSelected += 10;
                if(pindexActiveSelected > procinfoproc.NBpindexActive - 1)
                {
                    pindexActiveSelected = procinfoproc.NBpindexActive - 1;
                }
                if(TimeSorted == 0)
                {
                    pindexSelected = procinfoproc.pindexActive[pindexActiveSelected];
                }
                else
                {
                    pindexSelected = procinfoproc.sorted_pindex_time[pindexActiveSelected];
                }
                break;





            case 'T':
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        pid = pinfolist->PIDarray[pindex];
                        kill(pid, SIGTERM);
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    pid = pinfolist->PIDarray[pindex];
                    kill(pid, SIGTERM);
                }
                break;

            case 'K':
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        pid = pinfolist->PIDarray[pindex];
                        kill(pid, SIGKILL);
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    pid = pinfolist->PIDarray[pindex];
                    kill(pid, SIGKILL);
                }
                break;

            case 'I':
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        pid = pinfolist->PIDarray[pindex];
                        kill(pid, SIGINT);
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    pid = pinfolist->PIDarray[pindex];
                    kill(pid, SIGINT);
                }
                break;

            case 'r':
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        if(pinfolist->active[pindex] != 1)
                        {
                            char SM_fname[STRINGMAXLEN_FULLFILENAME];
                            WRITE_FULLFILENAME(SM_fname, "%s/proc.%s.%06d.shm", procdname,
                                               pinfolist->pnamearray[pindex], (int) pinfolist->PIDarray[pindex]);
                            remove(SM_fname);
                        }
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    if(pinfolist->active[pindex] != 1)
                    {
                        remove(procinfoproc.pinfoarray[pindex]->logfilename);

                        char SM_fname[STRINGMAXLEN_FULLFILENAME];
                        WRITE_FULLFILENAME(SM_fname, "%s/proc.%s.%06d.shm", procdname,
                                           pinfolist->pnamearray[pindex], (int) pinfolist->PIDarray[pindex]);
                        remove(SM_fname);
                    }
                }
                break;

            case 'R':
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(pinfolist->active[pindex] != 1)
                    {
                        remove(procinfoproc.pinfoarray[pindex]->logfilename);

                        char SM_fname[STRINGMAXLEN_FULLFILENAME];
                        WRITE_FULLFILENAME(SM_fname, "%s/proc.%s.%06d.shm", procdname,
                                           pinfolist->pnamearray[pindex], (int) pinfolist->PIDarray[pindex]);
                        remove(SM_fname);
                    }
                }
                break;

            // loop controls
            case 'p': // pause toggle
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        if(procinfoproc.pinfoarray[pindex]->CTRLval == 0)
                        {
                            procinfoproc.pinfoarray[pindex]->CTRLval = 1;
                        }
                        else
                        {
                            procinfoproc.pinfoarray[pindex]->CTRLval = 0;
                        }
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    if(procinfoproc.pinfoarray[pindex]->CTRLval == 0)
                    {
                        procinfoproc.pinfoarray[pindex]->CTRLval = 1;
                    }
                    else
                    {
                        procinfoproc.pinfoarray[pindex]->CTRLval = 0;
                    }
                }
                break;

            case 'c': // compute toggle (toggles between 0-run and 5-run-without-compute)
                DEBUG_TRACEPOINT(" ");
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        if(procinfoproc.pinfoarray[pindex]->CTRLval ==
                                0) // if running, turn compute to off
                        {
                            procinfoproc.pinfoarray[pindex]->CTRLval = 5;
                        }
                        else if(procinfoproc.pinfoarray[pindex]->CTRLval ==
                                5)  // if compute off, turn compute back on
                        {
                            procinfoproc.pinfoarray[pindex]->CTRLval = 0;
                        }
                    }
                }
                DEBUG_TRACEPOINT(" ");
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    if(procinfoproc.pinfoarray[pindex]->CTRLval ==
                            0) // if running, turn compute to off
                    {
                        procinfoproc.pinfoarray[pindex]->CTRLval = 5;
                    }
                    else if(procinfoproc.pinfoarray[pindex]->CTRLval ==
                            5)  // if procinfoproccompute off, turn compute back on
                    {
                        procinfoproc.pinfoarray[pindex]->CTRLval = 0;
                    }
                }
                DEBUG_TRACEPOINT(" ");
                break;

            case 's': // step
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        procinfoproc.pinfoarray[pindex]->CTRLval = 2;
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    procinfoproc.pinfoarray[pindex]->CTRLval = 2;
                }
                break;




            case '>': // move to other cpuset
                pindex = pindexSelected;
                if(pinfolist->active[pindex] == 1)
                {
                    endwin();
                    if(system("clear") != 0) // clear screen
                    {
                        PRINT_ERROR("system() returns non-zero value");
                    }
                    printf("CURRENT cpu set : %s\n",  procinfoproc.pinfodisp[pindex].cpuset);
                    listindex = processinfo_SelectFromList(CPUsetList, NBCPUset);

                    EXECUTE_SYSTEM_COMMAND("sudo cset proc -m %d %s", pinfolist->PIDarray[pindex],
                                           CPUsetList[listindex].name);

                    initncurses();
                }
                break;

            case '<': // move to same cpuset
                pindex = pindexSelected;
                if(pinfolist->active[pindex] == 1)
                {
                    endwin();

                    EXECUTE_SYSTEM_COMMAND("sudo cset proc -m %d root &> /dev/null",
                                           pinfolist->PIDarray[pindex]);
                    EXECUTE_SYSTEM_COMMAND("sudo cset proc --force -m %d %s &> /dev/null",
                                           pinfolist->PIDarray[pindex], procinfoproc.pinfodisp[pindex].cpuset);

                    initncurses();
                }
                break;


            case 'e': // exit
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        procinfoproc.pinfoarray[pindex]->CTRLval = 3;
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    procinfoproc.pinfoarray[pindex]->CTRLval = 3;
                }
                break;

            case 'z': // apply current value as offset (zero loop counter)
                selectedOK = 0;
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        procinfoproc.loopcntoffsetarray[pindex] =
                            procinfoproc.pinfoarray[pindex]->loopcnt;
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    procinfoproc.loopcntoffsetarray[pindex] =
                        procinfoproc.pinfoarray[pindex]->loopcnt;
                }
                break;

            case 'Z': // revert to original counter value
                for(index = 0; index < procinfoproc.NBpindexActive; index++)
                {
                    pindex = procinfoproc.pindexActive[index];
                    if(procinfoproc.selectedarray[pindex] == 1)
                    {
                        selectedOK = 1;
                        procinfoproc.loopcntoffsetarray[pindex] = 0;
                    }
                }
                if((selectedOK == 0) && (pindexSelectedOK == 1))
                {
                    pindex = pindexSelected;
                    procinfoproc.loopcntoffsetarray[pindex] = 0;
                }
                break;

            case 't':
                endwin();
                EXECUTE_SYSTEM_COMMAND("tmux a -t %s",
                                       procinfoproc.pinfoarray[pindexSelected]->tmuxname);
                initncurses();
                break;

            case 'a':
                pindex = pindexSelected;
                if(pinfolist->active[pindex] == 1)
                {
                    endwin();
                    EXECUTE_SYSTEM_COMMAND("watch -n 0.1 cat /proc/%d/status",
                                           (int) pinfolist->PIDarray[pindex]);
                    initncurses();
                }
                break;

            case 'd':
                pindex = pindexSelected;
                if(pinfolist->active[pindex] == 1)
                {
                    endwin();
                    EXECUTE_SYSTEM_COMMAND("watch -n 0.1 cat /proc/%d/sched",
                                           (int) pinfolist->PIDarray[pindex]);
                    EXECUTE_SYSTEM_COMMAND("watch -n 0.1 cat /proc/%d/sched",
                                           (int) pinfolist->PIDarray[pindex]);
                    initncurses();
                }
                break;


            case 'o':
                if(TimeSorted == 1)
                {
                    TimeSorted = 0;
                }
                else
                {
                    TimeSorted = 1;
                }
                break;


            case 'L': // toggle time limit (iter)
                pindex = pindexSelected;
                ToggleValue = procinfoproc.pinfoarray[pindex]->dtiter_limit_enable;
                if(ToggleValue == 0)
                {
                    procinfoproc.pinfoarray[pindex]->dtiter_limit_enable = 1;
                    procinfoproc.pinfoarray[pindex]->dtiter_limit_value = (long)(
                                1.5 * procinfoproc.pinfoarray[pindex]->dtmedian_iter_ns);
                    procinfoproc.pinfoarray[pindex]->dtiter_limit_cnt = 0;
                }
                else
                {
                    ToggleValue ++;
                    if(ToggleValue == 3)
                    {
                        ToggleValue = 0;
                    }
                    procinfoproc.pinfoarray[pindex]->dtiter_limit_enable = ToggleValue;
                }
                break;;

            case 'M' : // toggle time limit (exec)
                pindex = pindexSelected;
                ToggleValue = procinfoproc.pinfoarray[pindex]->dtexec_limit_enable;
                if(ToggleValue == 0)
                {
                    procinfoproc.pinfoarray[pindex]->dtexec_limit_enable = 1;
                    procinfoproc.pinfoarray[pindex]->dtexec_limit_value = (long)(
                                1.5 * procinfoproc.pinfoarray[pindex]->dtmedian_exec_ns + 20000);
                    procinfoproc.pinfoarray[pindex]->dtexec_limit_cnt = 0;
                }
                else
                {
                    ToggleValue ++;
                    if(ToggleValue == 3)
                    {
                        ToggleValue = 0;
                    }
                    procinfoproc.pinfoarray[pindex]->dtexec_limit_enable = ToggleValue;
                }
                break;;


            case 'm' : // message
                pindex = pindexSelected;
                if(pinfolist->active[pindex] == 1)
                {
                    endwin();
                    EXECUTE_SYSTEM_COMMAND("clear; tail -f %s",
                                           procinfoproc.pinfoarray[pindex]->logfilename);
                    initncurses();
                }
                break;


            // ============ SCREENS

            case 'h': // help
                procinfoproc.DisplayMode = PROCCTRL_DISPLAYMODE_HELP;
                break;

            case KEY_F(2): // control
                procinfoproc.DisplayMode = PROCCTRL_DISPLAYMODE_CTRL;
                break;

            case KEY_F(3): // resources
                procinfoproc.DisplayMode = PROCCTRL_DISPLAYMODE_RESOURCES;
                break;

            case KEY_F(4): // triggering
                procinfoproc.DisplayMode = PROCCTRL_DISPLAYMODE_TRIGGER;
                break;

            case KEY_F(5): // timing
                procinfoproc.DisplayMode = PROCCTRL_DISPLAYMODE_TIMING;
                break;

            case KEY_F(6): // htop
                endwin();
                if(system("htop") != 0)
                {
                    PRINT_ERROR("system() returns non-zero value");
                }
                initncurses();
                break;

            case KEY_F(7): // iotop
                endwin();
                if(system("sudo iotop -o") != 0)
                {
                    PRINT_ERROR("system() returns non-zero value");
                }
                initncurses();
                break;

            case KEY_F(8): // atop
                endwin();
                if(system("sudo atop") != 0)
                {
                    PRINT_ERROR("system() returns non-zero value");
                }
                initncurses();
                break;




            // ============ SCANNING

            case '{': // slower scan update
                procinfoproc.twaitus = (int)(1.2 * procinfoproc.twaitus);
                if(procinfoproc.twaitus > 1000000)
                {
                    procinfoproc.twaitus = 1000000;
                }
                break;

            case '}': // faster scan update
                procinfoproc.twaitus = (int)(0.83333333333333333333 * procinfoproc.twaitus);
                if(procinfoproc.twaitus < 1000)
                {
                    procinfoproc.twaitus = 1000;
                }
                break;


            // ============ DISPLAY

            case '-': // slower display update
                frequ *= 0.5;
                if(frequ < 1.0)
                {
                    frequ = 1.0;
                }
                if(frequ > 64.0)
                {
                    frequ = 64.0;
                }
                break;


            case '+': // faster display update
                frequ *= 2.0;
                if(frequ < 1.0)
                {
                    frequ = 1.0;
                }
                if(frequ > 64.0)
                {
                    frequ = 64.0;
                }
                break;

        }
        clock_gettime(CLOCK_REALTIME, &t01loop);

        DEBUG_TRACEPOINT(" ");

        if(freeze == 0)
        {
            erase();

            if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_HELP)
            {
                int attrval = A_BOLD;

                attron(attrval);
                printw("    x");
                attroff(attrval);
                printw("    Exit\n");


                printw("\n");
                printw("============ SCREENS \n");

                attron(attrval);
                printw("     h");
                attroff(attrval);
                printw("   Help screen\n");

                attron(attrval);
                printw("    F2");
                attroff(attrval);
                printw("   Process control screen\n");

                attron(attrval);
                printw("    F3");
                attroff(attrval);
                printw("   Process CPU and MEM resources screen\n");
                attron(attrval);

                printw("    F4");
                attroff(attrval);
                printw("   Process syncing\n");

                printw("    F5");
                attroff(attrval);
                printw("   Process timing screen\n");

                attron(attrval);
                printw("    F6");
                attroff(attrval);
                printw("   htop        Type F10 to exit\n");

                attron(attrval);
                printw("    F7");
                attroff(attrval);
                printw("   iotop       Type q to exit\n");

                attron(attrval);
                printw("    F8");
                attroff(attrval);
                printw("   atop        Type q to exit\n");




                printw("\n");
                printw("============ SCANNING \n");

                attron(attrval);
                printw("    }");
                attroff(attrval);
                printw("    Increase scan frequency\n");

                attron(attrval);
                printw("    {");
                attroff(attrval);
                printw("    Decrease scan frequency\n");




                printw("\n");
                printw("============ DISPLAY \n");

                attron(attrval);
                printw("    +");
                attroff(attrval);
                printw("    Increase display frequency\n");

                attron(attrval);
                printw("    -");
                attroff(attrval);
                printw("    Decrease display frequency\n");

                attron(attrval);
                printw("    f");
                attroff(attrval);
                printw("    Freeze\n");

                attron(attrval);
                printw("    r");
                attroff(attrval);
                printw("    Remove selected inactive process log\n");

                attron(attrval);
                printw("    R");
                attroff(attrval);
                printw("    Remove all inactive processes logs\n");

                attron(attrval);
                printw("    o");
                attroff(attrval);
                printw("    sort processes (toggle)\n");

                attron(attrval);
                printw("SPACE");
                attroff(attrval);
                printw("    Select this process\n");

                attron(attrval);
                printw("    u");
                attroff(attrval);
                printw("    Unselect all processes\n");



                printw("\n");
                printw("============ PROCESS DETAILS \n");

                attron(attrval);
                printw("    t");
                attroff(attrval);
                printw("    Connect to tmux session\n");

                attron(attrval);
                printw("    a");
                attroff(attrval);
                printw("    process stat\n");

                attron(attrval);
                printw("    d");
                attroff(attrval);
                printw("    process sched\n");




                printw("\n");
                printw("============ LOOP CONTROL \n");

                attron(attrval);
                printw("    p");
                attroff(attrval);
                printw("    pause (toggle C0 - C1)\n");

                attron(attrval);
                printw("    c");
                attroff(attrval);
                printw("    compute on/off (toggle C0 - C5)\n");

                attron(attrval);
                printw("    s");
                attroff(attrval);
                printw("    step\n");

                attron(attrval);
                printw("    e");
                attroff(attrval);
                printw("    clean exit\n");

                attron(attrval);
                printw("    T");
                attroff(attrval);
                printw("    SIGTERM\n");

                attron(attrval);
                printw("    K");
                attroff(attrval);
                printw("    SIGKILL\n");

                attron(attrval);
                printw("    I");
                attroff(attrval);
                printw("    SIGINT\n");




                printw("\n");
                printw("============ COUNTERS, TIMERS \n");

                attron(attrval);
                printw("    z");
                attroff(attrval);
                printw("    zero this selected counter\n");

                attron(attrval);
                printw("    Z");
                attroff(attrval);
                printw("    zero all selected counters\n");

                attron(attrval);
                printw("    L");
                attroff(attrval);
                printw("    Enable iteration time limit\n");

                attron(attrval);
                printw("    M");
                attroff(attrval);
                printw("    Enable execution time limit\n");



                printw("\n");
                printw("============ AFFINITY \n");

                attron(attrval);
                printw("    >");
                attroff(attrval);
                printw("    Move to other CPU set\n");

                attron(attrval);
                printw("    <");
                attroff(attrval);
                printw("    Move back to same CPU set\n");


                printw("\n\n");
            }
            else
            {
                DEBUG_TRACEPOINT(" ");

                printw("pindexSelected = %d    %d\n", pindexSelected, pindexSelectedOK);

                printw("[PID %d   SCAN TID %d]  %2d cpus   %2d processes tracked    Display Mode %d\n",
                       CLIPID, (int) procinfoproc.scanPID, procinfoproc.NBcpus,
                       procinfoproc.NBpindexActive, procinfoproc.DisplayMode);

                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_HELP)
                {
                    attron(A_REVERSE);
                    printw("[h] Help");
                    attroff(A_REVERSE);
                }
                else
                {
                    printw("[h] Help");
                }
                printw("   ");

                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_CTRL)
                {
                    attron(A_REVERSE);
                    printw("[F2] CTRL");
                    attroff(A_REVERSE);
                }
                else
                {
                    printw("[F2] CTRL");
                }
                printw("   ");

                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES)
                {
                    attron(A_REVERSE);
                    printw("[F3] Resources");
                    attroff(A_REVERSE);
                }
                else
                {
                    printw("[F3] Resources");
                }
                printw("   ");


                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER)
                {
                    attron(A_REVERSE);
                    printw("[F4] Triggering");
                    attroff(A_REVERSE);
                }
                else
                {
                    printw("[F4] Triggering");
                }
                printw("   ");


                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_TIMING)
                {
                    attron(A_REVERSE);
                    printw("[F5] Timing");
                    attroff(A_REVERSE);
                }
                else
                {
                    printw("[F5] Timing");
                }
                printw("   ");

                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_HTOP)
                {
                    attron(A_REVERSE);
                    printw("[F6] htop (F10 to exit)");
                    attroff(A_REVERSE);
                }
                else
                {
                    printw("[F6] htop (F10 to exit)");
                }
                printw("   ");

                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_IOTOP)
                {
                    attron(A_REVERSE);
                    printw("[F7] iotop (q to exit)");
                    attroff(A_REVERSE);
                }
                else
                {
                    printw("[F7] iotop (q to exit)");
                }
                printw("   ");

                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_ATOP)
                {
                    attron(A_REVERSE);
                    printw("[F8] atop (q to exit)");
                    attroff(A_REVERSE);
                }
                else
                {
                    printw("[F8] atop (q to exit)");
                }
                printw("   ");


                printw("\n");

                DEBUG_TRACEPOINT(" ");

                printw("Display frequ = %2d Hz  [%ld] fscan=%5.2f Hz ( %5.2f Hz %5.2f %% busy )\n",
                       (int)(frequ + 0.5),
                       procinfoproc.loopcnt,
                       1.0 / procinfoproc.dtscan,
                       1000000.0 / procinfoproc.twaitus,
                       100.0 * (procinfoproc.dtscan - 1.0e-6 * procinfoproc.twaitus) /
                       procinfoproc.dtscan);


                DEBUG_TRACEPOINT(" ");


                if((pindexSelected >= 0) && (pindexSelected < PROCESSINFOLISTSIZE))
                {
                    if(procinfoproc.pinfommapped[pindexSelected] == 1)
                    {

                        strcpy(pselected_FILE, procinfoproc.pinfoarray[pindexSelected]->source_FILE);
                        strcpy(pselected_FUNCTION,
                               procinfoproc.pinfoarray[pindexSelected]->source_FUNCTION);
                        pselected_LINE = procinfoproc.pinfoarray[pindexSelected]->source_LINE;

                        printw("Source Code: %s line %d (function %s)\n", pselected_FILE,
                               pselected_LINE, pselected_FUNCTION);
                    }
                    else
                    {
                        sprintf(pselected_FILE, "?");
                        sprintf(pselected_FUNCTION, "?");
                        pselected_LINE = 0;
                        printw("\n");
                    }
                }
                else
                {
                    printw("---\n");
                }

                printw("\n");

                clock_gettime(CLOCK_REALTIME, &t02loop);

                DEBUG_TRACEPOINT(" ");

                clock_gettime(CLOCK_REALTIME, &t03loop);




                clock_gettime(CLOCK_REALTIME, &t04loop);

                /** ### Display
                 *
                 *
                 *
                 */



                int dispindex;
                if(TimeSorted == 0)
                {
                    dispindexMax = wrow - 4;
                }
                else
                {
                    dispindexMax = procinfoproc.NBpindexActive;
                }

                DEBUG_TRACEPOINT(" ");

                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES)
                {
                    int cpu;
                    DEBUG_TRACEPOINT(" ");

                    // List CPUs

                    // Measure CPU loads, Display
                    int ColorCode;

                    // color limits for load
                    int CPUloadLim0 = 3;
                    int CPUloadLim1 = 40;
                    int CPUloadLim2 = 60;
                    int CPUloadLim3 = 80;

                    // color limits for # processes
                    int CPUpcntLim0 = 1;
                    int CPUpcntLim1 = 2;
                    int CPUpcntLim2 = 4;
                    int CPUpcntLim3 = 8;

                    DEBUG_TRACEPOINT(" ");


                    // List CPUs
                    printw(" %*.*s %*.*s %-*.*s    %-*.*s              ",
                           pstrlen_status,  pstrlen_status,  " ",
                           pstrlen_pid,     pstrlen_pid,     " ",
                           pstrlen_pname,   pstrlen_pname,   " ",
                           pstrlen_cset,    pstrlen_cset,    " "
                          );


                    for(cpusocket = 0; cpusocket < procinfoproc.NBcpusocket; cpusocket++)
                    {
                        if(cpusocket > 0)
                        {
                            printw("    ");
                        }
                        for(cpu = 0; cpu < procinfoproc.NBcpus; cpu++)
                            if(procinfoproc.CPUphys[cpu] == cpusocket)
                            {
                                printw("|%02d", procinfoproc.CPUids[cpu]);
                            }
                        printw("|");
                    }
                    printw(" <- %2d sockets %2d CPUs\n", procinfoproc.NBcpusocket,
                           procinfoproc.NBcpus);

                    // List CPU # processes
                    printw(" %*.*s %*.*s %-*.*s    %-*.*s              ",
                           pstrlen_status,  pstrlen_status,  " ",
                           pstrlen_pid,     pstrlen_pid,     " ",
                           pstrlen_pname,   pstrlen_pname,   " ",
                           pstrlen_cset,    pstrlen_cset,    " "
                          );



                    for(cpusocket = 0; cpusocket < procinfoproc.NBcpusocket; cpusocket++)
                    {
                        if(cpusocket > 0)
                        {
                            printw("    ");
                        }

                        for(cpu = 0; cpu < procinfoproc.NBcpus; cpu++)
                            if(procinfoproc.CPUphys[cpu] == cpusocket)
                            {
                                int vint = procinfoproc.CPUpcnt[procinfoproc.CPUids[cpu]];
                                if(vint > 99)
                                {
                                    vint = 99;
                                }

                                ColorCode = 0;
                                if(vint > CPUpcntLim1)
                                {
                                    ColorCode = 2;
                                }
                                if(vint > CPUpcntLim2)
                                {
                                    ColorCode = 3;
                                }
                                if(vint > CPUpcntLim3)
                                {
                                    ColorCode = 4;
                                }
                                if(vint < CPUpcntLim0)
                                {
                                    ColorCode = 5;
                                }

                                printw("|");
                                if(ColorCode != 0)
                                {
                                    attron(COLOR_PAIR(ColorCode));
                                }
                                printw("%02d", vint);
                                if(ColorCode != 0)
                                {
                                    attroff(COLOR_PAIR(ColorCode));
                                }
                            }
                        printw("|");
                    }

                    printw(" <- PROCESSES\n");


                    DEBUG_TRACEPOINT(" ");


                    // Print CPU LOAD
                    printw(" %*.*s %*.*s %-*.*s PR %-*.*s  #T  ctxsw   ",
                           pstrlen_status,  pstrlen_status,  "STATUS",
                           pstrlen_pid,     pstrlen_pid,     "PID",
                           pstrlen_pname,   pstrlen_pname,   "pname",
                           pstrlen_cset,    pstrlen_cset,    "cset",
                           procinfoproc.NBcpus);

                    for(cpusocket = 0; cpusocket < procinfoproc.NBcpusocket; cpusocket++)
                    {
                        if(cpusocket > 0)
                        {
                            printw("    ");
                        }
                        for(cpu = 0; cpu < procinfoproc.NBcpus; cpu++)
                            if(procinfoproc.CPUphys[cpu] == cpusocket)
                            {
                                int vint = (int)(100.0 * procinfoproc.CPUload[procinfoproc.CPUids[cpu]]);
                                if(vint > 99)
                                {
                                    vint = 99;
                                }

                                ColorCode = 0;
                                if(vint > CPUloadLim1)
                                {
                                    ColorCode = 2;
                                }
                                if(vint > CPUloadLim2)
                                {
                                    ColorCode = 3;
                                }
                                if(vint > CPUloadLim3)
                                {
                                    ColorCode = 4;
                                }
                                if(vint < CPUloadLim0)
                                {
                                    ColorCode = 5;
                                }

                                printw("|");
                                if(ColorCode != 0)
                                {
                                    attron(COLOR_PAIR(ColorCode));
                                }
                                printw("%02d", vint);
                                if(ColorCode != 0)
                                {
                                    attroff(COLOR_PAIR(ColorCode));
                                }
                            }
                        printw("|");
                    }

                    printw(" <- CPU LOAD\n");
                    printw("\n");
                }



                // print header for display mode PROCCTRL_DISPLAYMODE_CTRL
                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_CTRL)
                {
                    DEBUG_TRACEPOINT(" ");
                    printw("\n");
                    printw("\n");
                    printw(" %*.*s %*.*s %-*.*s %-*.*s C# tstart       %-*.*s %-*.*s   %-*.*s   %-*.*s\n",
                           pstrlen_status,  pstrlen_status,  "STATUS",
                           pstrlen_pid,     pstrlen_pid,     "PID",
                           pstrlen_pname,   pstrlen_pname,   "pname",
                           pstrlen_state,   pstrlen_state,   "state",
                           pstrlen_tmux,    pstrlen_tmux,    "tmux sess",
                           pstrlen_loopcnt, pstrlen_loopcnt, "loopcnt",
                           pstrlen_descr,   pstrlen_descr,   "Description",
                           pstrlen_msg,     pstrlen_msg,     "Message"
                          );
                    printw("\n");
                    DEBUG_TRACEPOINT(" ");
                }


                // print header for display mode PROCCTRL_DISPLAYMODE_TRIGGER
                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER)
                {
                    DEBUG_TRACEPOINT(" ");
                    printw("\n");
                    printw("\n");
                    printw(" %*.*s %*.*s %-*.*s %*.*s %*.*s mode sem %*.*s  %*.*s  %*.*s\n",
                           pstrlen_status,   pstrlen_status,   "STATUS",
                           pstrlen_pid,      pstrlen_pid,      "PID",
                           pstrlen_pname,    pstrlen_pname,    "pname",
                           pstrlen_inode,    pstrlen_inode,    "inode",
                           pstrlen_trigstreamname,    pstrlen_trigstreamname,    "stream",
                           pstrlen_missedfr, pstrlen_missedfr, "miss",
                           pstrlen_missedfrc, pstrlen_missedfrc, "misscumul",
                           pstrlen_tocnt,    pstrlen_tocnt,     "timeouts"
                          );
                    printw("\n");
                    DEBUG_TRACEPOINT(" ");
                }


                // print header for display mode PROCCTRL_DISPLAYMODE_TIMING
                if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_TIMING)
                {
                    DEBUG_TRACEPOINT(" ");
                    printw("\n");
                    printw("\n");
                    printw("   STATUS    PID   process name                    \n");
                    printw("\n");
                    DEBUG_TRACEPOINT(" ");
                }

                DEBUG_TRACEPOINT(" ");

                clock_gettime(CLOCK_REALTIME, &t05loop);


                // ===========================================================================
                // ============== PRINT INFORMATION FOR EACH PROCESS =========================
                // ===========================================================================
                pstrlen_total_max = 0;
                pindexSelectedOK = 0;



                for(dispindex = 0; dispindex < dispindexMax; dispindex++)
                {
                    if(TimeSorted == 0)
                    {
                        pindex = dispindex;
                    }
                    else
                    {
                        pindex = procinfoproc.sorted_pindex_time[dispindex];
                    }

                    if(pindex < procinfoproc.NBpinfodisp)
                    {
                        DEBUG_TRACEPOINT("%d %d   %ld %ld", dispindex, dispindexMax, pindex,
                                         procinfoproc.NBpinfodisp);

                        if(pindex == pindexSelected)
                        {
                            attron(A_REVERSE);
                            pindexSelectedOK = 1;
                        }

                        if(procinfoproc.selectedarray[pindex] == 1)
                        {
                            printw("*");
                        }
                        else
                        {
                            printw(" ");
                        }
                        pstrlen_total = 1;

                        DEBUG_TRACEPOINT("procinfoproc.selectedarray[pindex] = %d",
                                         procinfoproc.selectedarray[pindex]);

                        if(pinfolist->active[pindex] == 1)
                        {
                            sprintf(string, "%-*.*s", pstrlen_status, pstrlen_status, "ACTIVE");
                            pstrlen_total += strlen(string);
                            attron(COLOR_PAIR(2));
                            printw("%s", string);
                            attroff(COLOR_PAIR(2));
                        }


                        DEBUG_TRACEPOINT("pinfolist->active[pindex] = %d", pinfolist->active[pindex]);


                        if(pinfolist->active[pindex] == 2)  // not active: error, crashed or terminated
                        {
                            switch(procinfoproc.pinfoarray[pindex]->loopstat)
                            {
                                case 3: // clean exit
                                    sprintf(string, "%-*.*s", pstrlen_status, pstrlen_status, "STOPPED");
                                    pstrlen_total += strlen(string);
                                    attron(COLOR_PAIR(3));
                                    printw("%s", string);
                                    attroff(COLOR_PAIR(3));
                                    break;

                                case 4: // error
                                    sprintf(string, "%-*.*s", pstrlen_status, pstrlen_status, "ERROR");
                                    pstrlen_total += strlen(string);
                                    attron(COLOR_PAIR(3));
                                    printw("%s", string);
                                    attroff(COLOR_PAIR(3));
                                    break;

                                default: // crashed
                                    sprintf(string, "%-*.*s", pstrlen_status, pstrlen_status, "CRASHED");
                                    pstrlen_total += strlen(string);
                                    attron(COLOR_PAIR(4));
                                    printw("%s", string);
                                    attroff(COLOR_PAIR(4));
                                    break;
                            }
                        }


                        DEBUG_TRACEPOINT("%d %d   %ld %ld", dispindex, dispindexMax, pindex,
                                         procinfoproc.NBpinfodisp);

                        if(pinfolist->active[pindex] != 0)
                        {
                            if(pindex == pindexSelected)
                            {
                                attron(A_REVERSE);
                            }

                            sprintf(string, " %-*.*d", pstrlen_pid, pstrlen_pid,
                                    pinfolist->PIDarray[pindex]);
                            pstrlen_total += strlen(string);
                            printw("%s", string);


                            attron(A_BOLD);

                            sprintf(string, " %-*.*s", pstrlen_pname, pstrlen_pname,
                                    procinfoproc.pinfodisp[pindex].name);
                            pstrlen_total += strlen(string);
                            printw("%s", string);
                            attroff(A_BOLD);


                            // ================ DISPLAY MODE PROCCTRL_DISPLAYMODE_CTRL ==================
                            if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_CTRL)
                            {
                                switch(procinfoproc.pinfoarray[pindex]->loopstat)
                                {
                                    case 0:
                                        sprintf(string, " %-*.*s", pstrlen_state, pstrlen_state, "INIT");
                                        break;

                                    case 1:
                                        sprintf(string, " %-*.*s", pstrlen_state, pstrlen_state, "RUN");
                                        break;

                                    case 2:
                                        sprintf(string, " %-*.*s", pstrlen_state, pstrlen_state, "PAUS");
                                        break;

                                    case 3:
                                        sprintf(string, " %-*.*s", pstrlen_state, pstrlen_state, "TERM");
                                        break;

                                    case 4:
                                        sprintf(string, " %-*.*s", pstrlen_state, pstrlen_state, "ERR");
                                        break;

                                    case 5:
                                        sprintf(string, " %-*.*s", pstrlen_state, pstrlen_state, "OFF");
                                        break;

                                    case 6:
                                        sprintf(string, " %-*.*s", pstrlen_state, pstrlen_state, "CRASH");
                                        break;

                                    default:
                                        sprintf(string, " %-*.*s", pstrlen_state, pstrlen_state, "??");
                                }
                                pstrlen_total += strlen(string);
                                printw("%s", string);



                                if(procinfoproc.pinfoarray[pindex]->CTRLval == 0)
                                {
                                    attron(COLOR_PAIR(2));
                                    printw(" C%d", procinfoproc.pinfoarray[pindex]->CTRLval);
                                    attroff(COLOR_PAIR(2));
                                }
                                else
                                {
                                    printw(" C%d", procinfoproc.pinfoarray[pindex]->CTRLval);
                                }
                                pstrlen_total += 3;


                                sprintf(string, " %02d:%02d:%02d.%03d",
                                        procinfoproc.pinfodisp[pindex].createtime_hr,
                                        procinfoproc.pinfodisp[pindex].createtime_min,
                                        procinfoproc.pinfodisp[pindex].createtime_sec,
                                        (int)(0.000001 * (procinfoproc.pinfodisp[pindex].createtime_ns)));
                                pstrlen_total += strlen(string);
                                printw("%s", string);



                                sprintf(string, " %-*.*s", pstrlen_tmux, pstrlen_tmux,
                                        procinfoproc.pinfoarray[pindex]->tmuxname);
                                pstrlen_total += strlen(string);
                                printw("%s", string);


                                sprintf(string, " %- *.*ld", pstrlen_loopcnt, pstrlen_loopcnt,
                                        procinfoproc.pinfoarray[pindex]->loopcnt -
                                        procinfoproc.loopcntoffsetarray[pindex]);
                                pstrlen_total += strlen(string);
                                //if(procinfoproc.pinfoarray[pindex]->loopcnt == procinfoproc.loopcntarray[pindex])
                                if(procinfoproc.pinfoarray[pindex]->loopcnt ==
                                        procinfoproc.loopcntarray[pindex])
                                {
                                    // loopcnt has not changed
                                    printw("%s", string);
                                }
                                else
                                {
                                    // loopcnt has changed
                                    attron(COLOR_PAIR(2));
                                    printw("%s", string);
                                    attroff(COLOR_PAIR(2));
                                }

                                procinfoproc.loopcntarray[pindex] = procinfoproc.pinfoarray[pindex]->loopcnt;

                                printw(" | ");
                                pstrlen_total += 3;

                                sprintf(string, "%-*.*s", pstrlen_descr, pstrlen_descr,
                                        procinfoproc.pinfoarray[pindex]->description);
                                pstrlen_total += strlen(string);
                                printw("%s", string);

                                printw(" | ");
                                pstrlen_total += 3;

                                if((procinfoproc.pinfoarray[pindex]->loopstat == 4)
                                        || (procinfoproc.pinfoarray[pindex]->loopstat == 6)) // ERROR or CRASH
                                {
                                    attron(COLOR_PAIR(4));
                                }

                                sprintf(string, "%-*.*s", pstrlen_msg, pstrlen_msg,
                                        procinfoproc.pinfoarray[pindex]->statusmsg);
                                pstrlen_total += strlen(string);
                                printw("%s", string);

                                if((procinfoproc.pinfoarray[pindex]->loopstat == 4)
                                        || (procinfoproc.pinfoarray[pindex]->loopstat == 6)) // ERROR
                                {
                                    attroff(COLOR_PAIR(4));
                                }
                            }



                            // ================ DISPLAY MODE PROCCTRL_DISPLAYMODE_RESOURCES ==================
                            if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES)
                            {
                                int cpu;


                                if(procinfoproc.psysinfostatus[pindex] == -1)
                                {
                                    sprintf(string, " no process info available\n");
                                    pstrlen_total += strlen(string);
                                    printw("%s", string);
                                }
                                else
                                {


                                    int spindex; // sub process index, 0 for main
                                    for(spindex = 0; spindex < procinfoproc.pinfodisp[pindex].NBsubprocesses;
                                            spindex++)
                                    {
                                        //int TID; // thread ID



                                        if(spindex > 0)
                                        {
                                            //TID = procinfoproc.pinfodisp[pindex].subprocPIDarray[spindex];
                                            sprintf(string, " %*.*s %-*.*d %-*.*s",
                                                    pstrlen_status, pstrlen_status, "|---",
                                                    pstrlen_pid, pstrlen_pid,
                                                    procinfoproc.pinfodisp[pindex].subprocPIDarray[spindex],
                                                    pstrlen_pname, pstrlen_pname, procinfoproc.pinfodisp[pindex].name
                                                   );
                                            pstrlen_total += strlen(string);
                                            printw("%s", string);
                                        }
                                        else
                                        {
                                            //TID = procinfoproc.pinfodisp[pindex].PID;
                                            procinfoproc.pinfodisp[pindex].subprocPIDarray[0] =
                                                procinfoproc.pinfodisp[pindex].PID;
                                        }

                                        sprintf(string, " %2d", procinfoproc.pinfodisp[pindex].rt_priority);
                                        pstrlen_total += strlen(string);
                                        printw("%s", string);

                                        sprintf(string, " %-*.*s", pstrlen_cset, pstrlen_cset,
                                                procinfoproc.pinfodisp[pindex].cpuset);
                                        pstrlen_total += strlen(string);
                                        printw("%s", string);

                                        sprintf(string, " %2dx ", procinfoproc.pinfodisp[pindex].threads);
                                        pstrlen_total += strlen(string);
                                        printw("%s", string);


                                        // Context Switches
#ifdef CMDPROC_CONTEXTSWITCH
                                        if(procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex] !=
                                                procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary[spindex])
                                        {
                                            attron(COLOR_PAIR(4));
                                        }
                                        else if(procinfoproc.pinfodisp[pindex].ctxtsw_voluntary_prev[spindex] !=
                                                procinfoproc.pinfodisp[pindex].ctxtsw_voluntary[spindex])
                                        {
                                            attron(COLOR_PAIR(3));
                                        }

                                        sprintf(string, " +%02ld +%02ld",
                                                labs(procinfoproc.pinfodisp[pindex].ctxtsw_voluntary[spindex]    -
                                                     procinfoproc.pinfodisp[pindex].ctxtsw_voluntary_prev[spindex]) % 100,
                                                labs(procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary[spindex] -
                                                     procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex]) % 100
                                               );
                                        pstrlen_total += strlen(string);
                                        printw("%s", string);

                                        if(procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex] !=
                                                procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary[spindex])
                                        {
                                            attroff(COLOR_PAIR(4));
                                        }
                                        else if(procinfoproc.pinfodisp[pindex].ctxtsw_voluntary_prev[spindex] !=
                                                procinfoproc.pinfodisp[pindex].ctxtsw_voluntary[spindex])
                                        {
                                            attroff(COLOR_PAIR(3));
                                        }
                                        printw(" ");
#endif



                                        // CPU use
#ifdef CMDPROC_CPUUSE
                                        int cpuColor = 0;

                                        //					if(pinfodisp[pindex].subprocCPUloadarray[spindex]>5.0)
                                        cpuColor = 1;
                                        if(procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex] >
                                                10.0)
                                        {
                                            cpuColor = 2;
                                        }
                                        if(procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex] >
                                                20.0)
                                        {
                                            cpuColor = 3;
                                        }
                                        if(procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex] >
                                                40.0)
                                        {
                                            cpuColor = 4;
                                        }
                                        if(procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex] <
                                                1.0)
                                        {
                                            cpuColor = 5;
                                        }


                                        // First group of cores (physical CPU 0)
                                        for(cpu = 0; cpu < procinfoproc.NBcpus / procinfoproc.NBcpusocket; cpu++)
                                        {
                                            printw("|");
                                            pstrlen_total += 1;

                                            if(procinfoproc.CPUids[cpu] ==
                                                    procinfoproc.pinfodisp[pindex].processorarray[spindex])
                                            {
                                                attron(COLOR_PAIR(cpuColor));
                                            }

                                            if(procinfoproc.pinfodisp[pindex].cpuOKarray[cpu] == 1)
                                            {
                                                printw("%2d", procinfoproc.CPUids[cpu]);
                                            }
                                            else
                                            {
                                                printw("  ");
                                            }
                                            pstrlen_total += 2;


                                            if(procinfoproc.CPUids[cpu] ==
                                                    procinfoproc.pinfodisp[pindex].processorarray[spindex])
                                            {
                                                attroff(COLOR_PAIR(cpuColor));
                                            }
                                        }

                                        sprintf(string, "|    ");
                                        pstrlen_total += strlen(string);
                                        printw("%s", string);


                                        // Second group of cores (physical CPU 0)
                                        for(cpu = procinfoproc.NBcpus / procinfoproc.NBcpusocket;
                                                cpu < procinfoproc.NBcpus; cpu++)
                                        {
                                            printw("|");
                                            pstrlen_total += 1;

                                            if(procinfoproc.CPUids[cpu] ==
                                                    procinfoproc.pinfodisp[pindex].processorarray[spindex])
                                            {
                                                attron(COLOR_PAIR(cpuColor));
                                            }

                                            if(procinfoproc.pinfodisp[pindex].cpuOKarray[cpu] == 1)
                                            {
                                                printw("%2d", procinfoproc.CPUids[cpu]);
                                            }
                                            else
                                            {
                                                printw("  ");
                                            }
                                            pstrlen_total += 2;

                                            if(procinfoproc.CPUids[cpu] ==
                                                    procinfoproc.pinfodisp[pindex].processorarray[spindex])
                                            {
                                                attroff(COLOR_PAIR(cpuColor));
                                            }
                                        }
                                        printw("| ");
                                        pstrlen_total += 2;


                                        attron(COLOR_PAIR(cpuColor));
                                        sprintf(string, "%5.1f %6.2f",
                                                procinfoproc.pinfodisp[pindex].subprocCPUloadarray[spindex],
                                                procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex]);
                                        pstrlen_total += strlen(string);
                                        printw("%s", string);
                                        attroff(COLOR_PAIR(cpuColor));
#endif


                                        // Memory use
#ifdef CMDPROC_MEMUSE
                                        int memColor = 0;

                                        int kBcnt, MBcnt, GBcnt;

                                        kBcnt = procinfoproc.pinfodisp[pindex].VmRSSarray[spindex];
                                        MBcnt = kBcnt / 1024;
                                        kBcnt = kBcnt - MBcnt * 1024;

                                        GBcnt = MBcnt / 1024;
                                        MBcnt = MBcnt - GBcnt * 1024;

                                        //if(pinfodisp[pindex].subprocMEMloadarray[spindex]>0.5)
                                        memColor = 1;
                                        if(procinfoproc.pinfodisp[pindex].VmRSSarray[spindex] > 10 * 1024)      // 10 MB
                                        {
                                            memColor = 2;
                                        }
                                        if(procinfoproc.pinfodisp[pindex].VmRSSarray[spindex] > 100 *
                                                1024)     // 100 MB
                                        {
                                            memColor = 3;
                                        }
                                        if(procinfoproc.pinfodisp[pindex].VmRSSarray[spindex] > 1024 * 1024)    // 1 GB
                                        {
                                            memColor = 4;
                                        }
                                        if(procinfoproc.pinfodisp[pindex].VmRSSarray[spindex] < 1024)           // 1 MB
                                        {
                                            memColor = 5;
                                        }

                                        printw(" ");
                                        pstrlen_total += 1;

                                        attron(COLOR_PAIR(memColor));
                                        if(GBcnt > 0)
                                        {
                                            sprintf(string, "%3d GB ", GBcnt);
                                            pstrlen_total += strlen(string);
                                            printw("%s", string);
                                        }
                                        else
                                        {
                                            sprintf(string, "       ");
                                            pstrlen_total += strlen(string);
                                            printw("%s", string);
                                        }

                                        if(MBcnt > 0)
                                        {
                                            sprintf(string, "%4d MB ", MBcnt);
                                            pstrlen_total += strlen(string);
                                            printw("%s", string);
                                        }
                                        else
                                        {
                                            sprintf(string, "       ");
                                            pstrlen_total += strlen(string);
                                            printw("%s", string);
                                        }

                                        if(kBcnt > 0)
                                        {
                                            sprintf(string, "%4d kB ", kBcnt);
                                            pstrlen_total += strlen(string);
                                            printw("%s", string);
                                        }
                                        else
                                        {
                                            sprintf(string, "       ");
                                            pstrlen_total += strlen(string);
                                            printw("%s", string);
                                        }
                                        attroff(COLOR_PAIR(memColor));
#endif

                                        if(pindex == pindexSelected)
                                        {
                                            attroff(A_REVERSE);
                                        }

                                        printw("\n");
                                        // end of line
                                        if(pstrlen_total > pstrlen_total_max)
                                        {
                                            pstrlen_total_max = pstrlen_total;
                                        }
                                        //printw("len = %d %d / %d\n", pstrlen_total, pstrlen_total_max, wcol);
                                        pstrlen_total = 0;



                                    }
                                    if(procinfoproc.pinfodisp[pindex].NBsubprocesses == 0)
                                    {
                                        printw("  ERROR: procinfoproc.pinfodisp[pindex].NBsubprocesses = %d\n",
                                               (int) procinfoproc.pinfodisp[pindex].NBsubprocesses);

                                        if(pindex == pindexSelected)
                                        {
                                            attroff(A_REVERSE);
                                        }
                                    }

                                }


                            }



                            // ================ DISPLAY MODE PROCCTRL_DISPLAYMODE_TRIGGER ==================
                            if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER)
                            {
                                printw("%*d ", pstrlen_inode,
                                       procinfoproc.pinfoarray[pindex]->triggerstreaminode);
                                printw("%*s ", pstrlen_trigstreamname,
                                       procinfoproc.pinfoarray[pindex]->triggerstreamname);

                                switch(procinfoproc.pinfoarray[pindex]->triggermode)
                                {

                                    case PROCESSINFO_TRIGGERMODE_IMMEDIATE :
                                        printw(" IMME   ");
                                        break;

                                    case PROCESSINFO_TRIGGERMODE_CNT0 :
                                        printw(" CNT0   ");
                                        break;

                                    case PROCESSINFO_TRIGGERMODE_CNT1 :
                                        printw(" CNT1   ");
                                        break;

                                    case PROCESSINFO_TRIGGERMODE_SEMAPHORE :
                                        printw(" SEMA %2d",
                                               procinfoproc.pinfoarray[pindex]->triggersem
                                              );

                                        break;

                                    case PROCESSINFO_TRIGGERMODE_DELAY :
                                        printw(" DELA   ");
                                        break;

                                    default :
                                        printw(" %04d   ",  procinfoproc.pinfoarray[pindex]->triggermode);
                                }

                                printw("  %*d ", pstrlen_missedfr,
                                       procinfoproc.pinfoarray[pindex]->triggermissedframe);
                                printw("  %*llu ", pstrlen_missedfrc,
                                       procinfoproc.pinfoarray[pindex]->triggermissedframe_cumul);
                                printw("  %*llu ", pstrlen_tocnt,
                                       procinfoproc.pinfoarray[pindex]->trigggertimeoutcnt);
                            }


                            // ================ DISPLAY MODE PROCCTRL_DISPLAYMODE_TIMING ==================
                            if(procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_TIMING)
                            {
                                if(procinfoproc.pinfoarray[pindex]->MeasureTiming == 1)
                                {
                                    printw(" ON");
                                }
                                else
                                {
                                    printw("OFF");
                                }

                                if(procinfoproc.pinfoarray[pindex]->MeasureTiming == 1)
                                {
                                    long *dtiter_array;
                                    long *dtexec_array;
                                    //int dtindex;


                                    printw(" %3d ..%02ld  ", procinfoproc.pinfoarray[pindex]->timerindex,
                                           procinfoproc.pinfoarray[pindex]->timingbuffercnt % 100);

                                    // compute timing stat
                                    dtiter_array = (long *) malloc(sizeof(long) * (PROCESSINFO_NBtimer - 1));
                                    dtexec_array = (long *) malloc(sizeof(long) * (PROCESSINFO_NBtimer - 1));

                                    int tindex;
                                    //dtindex = 0;

                                    // we exclude the current timerindex, as timers may not all be written
                                    for(tindex = 0; tindex < PROCESSINFO_NBtimer - 1; tindex++)
                                    {
                                        int ti0, ti1;

                                        ti1 = procinfoproc.pinfoarray[pindex]->timerindex - tindex;
                                        ti0 = ti1 - 1;

                                        if(ti0 < 0)
                                        {
                                            ti0 += PROCESSINFO_NBtimer;
                                        }

                                        if(ti1 < 0)
                                        {
                                            ti1 += PROCESSINFO_NBtimer;
                                        }

                                        dtiter_array[tindex] = (procinfoproc.pinfoarray[pindex]->texecstart[ti1].tv_nsec
                                                                - procinfoproc.pinfoarray[pindex]->texecstart[ti0].tv_nsec) + 1000000000 *
                                                               (procinfoproc.pinfoarray[pindex]->texecstart[ti1].tv_sec -
                                                                procinfoproc.pinfoarray[pindex]->texecstart[ti0].tv_sec);

                                        dtexec_array[tindex] = (procinfoproc.pinfoarray[pindex]->texecend[ti0].tv_nsec -
                                                                procinfoproc.pinfoarray[pindex]->texecstart[ti0].tv_nsec) + 1000000000 *
                                                               (procinfoproc.pinfoarray[pindex]->texecend[ti0].tv_sec -
                                                                procinfoproc.pinfoarray[pindex]->texecstart[ti0].tv_sec);
                                    }



                                    quick_sort_long(dtiter_array, PROCESSINFO_NBtimer - 1);
                                    quick_sort_long(dtexec_array, PROCESSINFO_NBtimer - 1);

                                    int colorcode;

                                    if(procinfoproc.pinfoarray[pindex]->dtiter_limit_enable != 0)
                                    {
                                        if(procinfoproc.pinfoarray[pindex]->dtiter_limit_cnt == 0)
                                        {
                                            colorcode = COLOR_PAIR(2);
                                        }
                                        else
                                        {
                                            colorcode = COLOR_PAIR(4);
                                        }
                                        attron(colorcode);
                                    }
                                    printw("ITERlim %d/%5ld/%4ld",
                                           procinfoproc.pinfoarray[pindex]->dtiter_limit_enable,
                                           (long)(0.001 * procinfoproc.pinfoarray[pindex]->dtiter_limit_value),
                                           procinfoproc.pinfoarray[pindex]->dtiter_limit_cnt);
                                    if(procinfoproc.pinfoarray[pindex]->dtiter_limit_enable != 0)
                                    {
                                        attroff(colorcode);
                                    }

                                    printw("  ");

                                    if(procinfoproc.pinfoarray[pindex]->dtexec_limit_enable != 0)
                                    {
                                        if(procinfoproc.pinfoarray[pindex]->dtexec_limit_cnt == 0)
                                        {
                                            colorcode = COLOR_PAIR(2);
                                        }
                                        else
                                        {
                                            colorcode = COLOR_PAIR(4);
                                        }
                                        attron(colorcode);
                                    }

                                    printw("EXEClim %d/%5ld/%4ld ",
                                           procinfoproc.pinfoarray[pindex]->dtexec_limit_enable,
                                           (long)(0.001 * procinfoproc.pinfoarray[pindex]->dtexec_limit_value),
                                           procinfoproc.pinfoarray[pindex]->dtexec_limit_cnt);
                                    if(procinfoproc.pinfoarray[pindex]->dtexec_limit_enable != 0)
                                    {
                                        attroff(colorcode);
                                    }


                                    float tval;

                                    tval = 0.001 * dtiter_array[(long)(0.5 * PROCESSINFO_NBtimer)];
                                    procinfoproc.pinfoarray[pindex]->dtmedian_iter_ns = dtiter_array[(long)(
                                                0.5 * PROCESSINFO_NBtimer)];
                                    if(tval > 9999.9)
                                    {
                                        printw(" ITER    >10ms ");
                                    }
                                    else
                                    {
                                        printw(" ITER %6.1fus ", tval);
                                    }

                                    tval = 0.001 * dtiter_array[0];
                                    if(tval > 9999.9)
                                    {
                                        printw("[   >10ms -");
                                    }
                                    else
                                    {
                                        printw("[%6.1fus -", tval);
                                    }

                                    tval = 0.001 * dtiter_array[PROCESSINFO_NBtimer - 2];
                                    if(tval > 9999.9)
                                    {
                                        printw("    >10ms ]");
                                    }
                                    else
                                    {
                                        printw(" %6.1fus ]", tval);
                                    }


                                    tval = 0.001 * dtexec_array[(long)(0.5 * PROCESSINFO_NBtimer)];
                                    procinfoproc.pinfoarray[pindex]->dtmedian_exec_ns = dtexec_array[(long)(
                                                0.5 * PROCESSINFO_NBtimer)];
                                    if(tval > 9999.9)
                                    {
                                        printw(" EXEC    >10ms ");
                                    }
                                    else
                                    {
                                        printw(" EXEC %6.1fus ", tval);
                                    }

                                    tval = 0.001 * dtexec_array[0];
                                    if(tval > 9999.9)
                                    {
                                        printw("[   >10ms -");
                                    }
                                    else
                                    {
                                        printw("[%6.1fus -", tval);
                                    }

                                    tval = 0.001 * dtexec_array[PROCESSINFO_NBtimer - 2];
                                    if(tval > 9999.9)
                                    {
                                        printw("    >10ms ]");
                                    }
                                    else
                                    {
                                        printw(" %6.1fus ]", tval);
                                    }


                                    //	printw(" ITER %9.3fus [%9.3f - %9.3f] ", 0.001*dtiter_array[(long) (0.5*PROCESSINFO_NBtimer)], 0.001*dtiter_array[0], 0.001*dtiter_array[PROCESSINFO_NBtimer-2]);





                                    //	printw(" EXEC %9.3fus [%9.3f - %9.3f] ", 0.001*dtexec_array[(long) (0.5*PROCESSINFO_NBtimer)], 0.001*dtexec_array[0], 0.001*dtexec_array[PROCESSINFO_NBtimer-2]);


                                    printw("  busy = %6.2f %%",
                                           100.0 * dtexec_array[(long)(0.5 * PROCESSINFO_NBtimer)] / (dtiter_array[(long)(
                                                       0.5 * PROCESSINFO_NBtimer)] + 1));

                                    free(dtiter_array);
                                    free(dtexec_array);

                                }
                            }


                            if(pindex == pindexSelected)
                            {
                                attroff(A_REVERSE);
                            }
                        }

                    }

                    if((procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_CTRL)
                            || (procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER)
                            || (procinfoproc.DisplayMode == PROCCTRL_DISPLAYMODE_TIMING))
                    {
                        printw("\n");
                        // end of line
                        if(pstrlen_total > pstrlen_total_max)
                        {
                            pstrlen_total_max = pstrlen_total;
                        }
                        //printw("len = %d %d / %d / %d\n", pstrlen_total, pstrlen_total_max, wcol, pstrlen_msg);
                        pstrlen_total = 0;
                    }






                }
            }



            clock_gettime(CLOCK_REALTIME, &t06loop);


            DEBUG_TRACEPOINT(" ");

            clock_gettime(CLOCK_REALTIME, &t07loop);

            cnt++;



            clock_gettime(CLOCK_REALTIME, &t2loop);

            tdiff = timespec_diff(t1loop, t2loop);
            double tdiffvloop = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

            printw("\nLoop time = %9.8f s  ( max rate = %7.2f Hz)\n", tdiffvloop,
                   1.0 / tdiffvloop);




            if(pstrlen_total_max > wcol - 1)
            {
                int testval;

                testval = pstrlen_msg - (pstrlen_total_max - (wcol - 1));
                if(testval < pstrlen_msg_min)
                {
                    testval = pstrlen_msg_min;
                }

                if(pstrlen_msg != testval)
                {
                    pstrlen_msg = testval;
                    refresh();
                    clear();
                }
            }

            if(pstrlen_total_max < wcol - 2)
            {
                pstrlen_msg += (wcol - 2 - pstrlen_total_max);
                if(pstrlen_msg > pstrlen_msg_max)
                {
                    pstrlen_msg = pstrlen_msg_max;
                }
            }



            refresh();
        }



        if((data.signal_TERM == 1) || (data.signal_INT == 1) || (data.signal_ABRT == 1)
                || (data.signal_BUS == 1) || (data.signal_SEGV == 1) || (data.signal_HUP == 1)
                || (data.signal_PIPE == 1))
        {
            loopOK = 0;
        }

    }
    endwin();


    // Why did we exit ?

    printf("loopOK = 0 -> exit\n");


    if(Xexit == 1)    // normal exit
    {
        printf("[%4d] User typed x -> exiting\n", __LINE__);
    }
    else if(data.signal_TERM == 1)
    {
        printf("[%4d] Received signal TERM\n", __LINE__);
    }
    else if(data.signal_INT == 1)
    {
        printf("[%4d] Received signal INT\n", __LINE__);
    }
    else if(data.signal_ABRT == 1)
    {
        printf("[%4d] Received signal ABRT\n", __LINE__);
    }
    else if(data.signal_BUS == 1)
    {
        printf("[%4d] Received signal BUS\n", __LINE__);
    }
    else if(data.signal_SEGV == 1)
    {
        printf("[%4d] Received signal SEGV\n", __LINE__);
    }
    else if(data.signal_HUP == 1)
    {
        printf("[%4d] Received signal HUP\n", __LINE__);
    }
    else if(data.signal_PIPE == 1)
    {
        printf("[%4d] Received signal PIPE\n", __LINE__);
    }


    procinfoproc.loop = 0;



    int *line;
    int ret = -1;

    while(ret != 0)
    {
        ret = pthread_tryjoin_np(threadscan, (void **)&line);
        /*
                if(ret==EBUSY){
                    printf("Waiting for thread to complete - currently at line %d\n", procinfoproc.scandebugline);
                }
                */
        usleep(10000);
    }




    // cleanup
    for(pindex = 0; pindex < procinfoproc.NBpinfodisp; pindex++)
    {
        if(procinfoproc.pinfommapped[pindex] == 1)
        {
            processinfo_shm_close(procinfoproc.pinfoarray[pindex],
                                  procinfoproc.fdarray[pindex]);
            procinfoproc.pinfommapped[pindex] = 0;
        }

    }


    free(procinfoproc.pinfodisp);

    free(CPUsetList);

    fflush(stderr);
    dup2(backstderr, STDERR_FILENO);
    close(backstderr);


    return RETURN_SUCCESS;
}
