
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



#ifdef STANDALONE
#include "standalone_dependencies.h"
#else
#include <00CORE/00CORE.h>
#include <CommandLineInterface/CLIcore.h>
#include "COREMOD_tools/COREMOD_tools.h"
#include "info/info.h"
#endif

#include <processtools.h>

#ifdef USE_HWLOC
#include <hwloc.h>
#endif



// What do we want to compute/print ?
#define CMDPROC_CONTEXTSWITCH	1
#define CMDPROC_CPUUSE	1
#define CMDPROC_MEMUSE	1

//#define CMDPROC_PROCSTAT 1

/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */






static PROCESSINFOLIST *pinfolist;

static int wrow, wcol;






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
    char  SM_fname[200];
	long pindex = 0;

    sprintf(SM_fname, "%s/processinfo.list.shm", SHAREDMEMDIR);


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
        if (SM_fd == -1) {
            perror("Error opening file for writing");
            exit(0);
        }

        int result;
        result = lseek(SM_fd, sharedsize-1, SEEK_SET);
        if (result == -1) {
            close(SM_fd);
            fprintf(stderr, "Error calling lseek() to 'stretch' the file");
            exit(0);
        }

        result = write(SM_fd, "", 1);
        if (result != 1) {
            close(SM_fd);
            perror("Error writing last byte of the file");
            exit(0);
        }

        pinfolist = (PROCESSINFOLIST*) mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
        if (pinfolist == MAP_FAILED) {
            close(SM_fd);
            perror("Error mmapping the file");
            exit(0);
        }
        
        for(pindex=0; pindex<PROCESSINFOLISTSIZE; pindex++)
			pinfolist->active[pindex] = 0;

        pindex = 0;
    }
    else
    {
		int SM_fd;
		struct stat file_stat;
		
		SM_fd = open(SM_fname, O_RDWR);
		fstat(SM_fd, &file_stat);
        printf("[%d] File %s size: %zd\n", __LINE__, SM_fname, file_stat.st_size);

        pinfolist = (PROCESSINFOLIST*) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
        if (pinfolist == MAP_FAILED) {
            close(SM_fd);
            fprintf(stderr, "Error mmapping the file");
            exit(0);
        }
        
        while((pinfolist->active[pindex] != 0)&&(pindex<PROCESSINFOLISTSIZE))
			pindex ++;
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

PROCESSINFO* processinfo_shm_create(
	char *pname, 
	int CTRLval
	)
{
    size_t sharedsize = 0; // shared memory size in bytes
    int SM_fd; // shared memory file descriptor
    PROCESSINFO *pinfo;
    
    
    sharedsize = sizeof(PROCESSINFO);

	char  SM_fname[200];
    pid_t PID;

    
    PID = getpid();

	long pindex;
    pindex = processinfo_shm_list_create();
    pinfolist->PIDarray[pindex] = PID;
    
    
    
    sprintf(SM_fname, "%s/proc.%06d.shm", SHAREDMEMDIR, (int) PID);    
    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (SM_fd == -1) {
        perror("Error opening file for writing");
        exit(0);
    }

    int result;
    result = lseek(SM_fd, sharedsize-1, SEEK_SET);
    if (result == -1) {
        close(SM_fd);
        fprintf(stderr, "Error calling lseek() to 'stretch' the file");
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if (result != 1) {
        close(SM_fd);
        perror("Error writing last byte of the file");
        exit(0);
    }

    pinfo = (PROCESSINFO*) mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (pinfo == MAP_FAILED) {
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
	fpout = popen ("tmuxsessionname", "r");
	if(fpout==NULL)
	{
		printf("WARNING: cannot run command \"tmuxsessionname\"\n");
	}
	else
	{
		if(fgets(tmuxname, 100, fpout)== NULL)
			printf("WARNING: fgets error\n");
		pclose(fpout);
	}
	// remove line feed
	if(strlen(tmuxname)>0)
	{
		printf("tmux name : %s\n", tmuxname);
		printf("len: %d\n", (int) strlen(tmuxname));
		fflush(stdout);
		
		if(tmuxname[strlen(tmuxname)-1] == '\n')
			tmuxname[strlen(tmuxname)-1] = '\0';
	}
	
	printf("line %d\n", __LINE__);
	fflush(stdout);
	// force last char to be term, just in case
	tmuxname[99] = '\0';
	printf("line %d\n", __LINE__);
	fflush(stdout);
	
	strncpy(pinfo->tmuxname, tmuxname, 100);
	
	printf("line %d\n", __LINE__);
	fflush(stdout);
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
	
#ifndef STANDALONE
	data.pinfo = pinfo;  
#endif
	pinfo->PID = PID;
	
	
	// create logfile
	char logfilename[300];
	struct timespec tnow;
	
    clock_gettime(CLOCK_REALTIME, &tnow);
 
	sprintf(pinfo->logfilename, "/tmp/proc.%s.%06d.%09ld.logfile", pinfo->name, (int) pinfo->PID, tnow.tv_sec);
	pinfo->logFile = fopen(pinfo->logfilename, "w");
	

	
	char msgstring[300];
	sprintf(msgstring, "LOG START %s", pinfo->logfilename);
	processinfo_WriteMessage(pinfo, msgstring);
	
	
	
    return pinfo;
}







int processinfo_cleanExit(PROCESSINFO *processinfo)
{
    struct timespec tstop;
    struct tm *tstoptm;
    char msgstring[200];

    clock_gettime(CLOCK_REALTIME, &tstop);
    tstoptm = gmtime(&tstop.tv_sec);

    if(processinfo->CTRLval == 3) // loop exit from processinfo control
    {
        sprintf(msgstring, "CTRLexit %02d:%02d:%02d.%03d", tstoptm->tm_hour, tstoptm->tm_min, tstoptm->tm_sec, (int) (0.000001*(tstop.tv_nsec)));
		strncpy(processinfo->statusmsg, msgstring, 200);
    }
    
    if(processinfo->loopstat == 1)
   {
        sprintf(msgstring, "Loop exit %02d:%02d:%02d.%03d", tstoptm->tm_hour, tstoptm->tm_min, tstoptm->tm_sec, (int) (0.000001*(tstop.tv_nsec)));
		strncpy(processinfo->statusmsg, msgstring, 200);
	}
	
	processinfo->loopstat = 3; // clean exit

    return 0;
}







int processinfo_SIGexit(PROCESSINFO *processinfo, int SignalNumber)
{
	char       timestring[200];
    struct     timespec tstop;
    struct tm *tstoptm;
    char       msgstring[200];

    clock_gettime(CLOCK_REALTIME, &tstop);
    tstoptm = gmtime(&tstop.tv_sec);
	
	sprintf(timestring, "%02d:%02d:%02d.%03d", tstoptm->tm_hour, tstoptm->tm_min, tstoptm->tm_sec, (int) (0.000001*(tstop.tv_nsec)));
	processinfo->loopstat = 3; // clean exit
	

	switch ( SignalNumber ) {

		case SIGHUP :  // Hangup detected on controlling terminal or death of controlling process
		sprintf(msgstring, "SIGHUP at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;
		
		case SIGINT :  // Interrupt from keyboard
		sprintf(msgstring, "SIGINT at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;
		
		case SIGQUIT :  // Quit from keyboard
		sprintf(msgstring, "SIGQUIT at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGILL :  // Illegal Instruction
		sprintf(msgstring, "SIGILL at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGABRT :  // Abort signal from abort
		sprintf(msgstring, "SIGABRT at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGFPE :  // Floating-point exception
		sprintf(msgstring, "SIGFPE at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGKILL :  // Kill signal
		sprintf(msgstring, "SIGKILL at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGSEGV :  // Invalid memory reference
		sprintf(msgstring, "SIGSEGV at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGPIPE :  // Broken pipe: write to pipe with no readers
		sprintf(msgstring, "SIGPIPE at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGALRM :  // Timer signal from alarm
		sprintf(msgstring, "SIGALRM at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGTERM :  // Termination signal
		sprintf(msgstring, "SIGTERM at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGUSR1 :  // User-defined signal 1
		sprintf(msgstring, "SIGUSR1 at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGUSR2 :  // User-defined signal 1
		sprintf(msgstring, "SIGUSR2 at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGCHLD :  // Child stopped or terminated
		sprintf(msgstring, "SIGCHLD at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGCONT :  // Continue if stoppedshmimTCPtransmit
		sprintf(msgstring, "SIGCONT at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGSTOP :  // Stop process
		sprintf(msgstring, "SIGSTOP at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGTSTP :  // Stop typed at terminal
		sprintf(msgstring, "SIGTSTP at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGTTIN :  // Terminal input for background process
		sprintf(msgstring, "SIGTTIN at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGTTOU :  // Terminal output for background process
		sprintf(msgstring, "SIGTTOU at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGBUS :  // Bus error (bad memory access)
		sprintf(msgstring, "SIGBUS at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGPOLL :  // Pollable event (Sys V).
		sprintf(msgstring, "SIGPOLL at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGPROF :  // Profiling timer expired
		sprintf(msgstring, "SIGPROF at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGSYS :  // Bad system call (SVr4)
		sprintf(msgstring, "SIGSYS at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGTRAP :  // Trace/breakpoint trap
		sprintf(msgstring, "SIGTRAP at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGURG :  // Urgent condition on socket (4.2BSD)
		sprintf(msgstring, "SIGURG at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGVTALRM :  // Virtual alarm clock (4.2BSD)
		sprintf(msgstring, "SIGVTALRM at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGXCPU :  // CPU time limit exceeded (4.2BSD)
		sprintf(msgstring, "SIGXCPU at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;

		case SIGXFSZ :  // File size limit exceeded (4.2BSD)
		sprintf(msgstring, "SIGXFSZ at %s", timestring);
		processinfo_WriteMessage(processinfo, msgstring);
		break;
	}
	
    return 0;
}







int processinfo_WriteMessage(PROCESSINFO *processinfo, const char* msgstring)
{
    struct timespec tnow;
    struct tm *tmnow;
    char msgstringFull[300];
	FILE *fp;
	
    clock_gettime(CLOCK_REALTIME, &tnow);
    tmnow = gmtime(&tnow.tv_sec);

    strcpy(processinfo->statusmsg, msgstring);

   
    fprintf(processinfo->logFile, "%02d:%02d:%02d.%06d  %8ld.%09ld  %06d  %s\n",
            tmnow->tm_hour, tmnow->tm_min, tmnow->tm_sec, (int) (0.001*(tnow.tv_nsec)),
            tnow.tv_sec, tnow.tv_nsec,
            (int) processinfo->PID, 
            msgstring);
    fflush(processinfo->logFile);

    return 0;
}




int processinfo_CatchSignals()
{
#ifndef STANDALONE
    if (sigaction(SIGTERM, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGTERM\n");

    if (sigaction(SIGINT, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGINT\n");

    if (sigaction(SIGABRT, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGABRT\n");

    if (sigaction(SIGBUS, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGBUS\n");

    if (sigaction(SIGSEGV, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGSEGV\n");

    if (sigaction(SIGHUP, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGHUP\n");

    if (sigaction(SIGPIPE, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGPIPE\n");
#endif

    return 0;
}



int processinfo_ProcessSignals(PROCESSINFO *processinfo)
{
    int loopOK = 1;
    // process signals

#ifndef STANDALONE
    if(data.signal_TERM == 1) {
        loopOK = 0;
        if(data.processinfo==1)
            processinfo_SIGexit(processinfo, SIGTERM);
    }

    if(data.signal_INT == 1) {
        loopOK = 0;
        if(data.processinfo==1)
            processinfo_SIGexit(processinfo, SIGINT);
    }

    if(data.signal_ABRT == 1) {
        loopOK = 0;
        if(data.processinfo==1)
            processinfo_SIGexit(processinfo, SIGABRT);
    }

    if(data.signal_BUS == 1) {
        loopOK = 0;
        if(data.processinfo==1)
            processinfo_SIGexit(processinfo, SIGBUS);
    }

    if(data.signal_SEGV == 1) {
        loopOK = 0;
        if(data.processinfo==1)
            processinfo_SIGexit(processinfo, SIGSEGV);
    }

    if(data.signal_HUP == 1) {
        loopOK = 0;
        if(data.processinfo==1)
            processinfo_SIGexit(processinfo, SIGHUP);
    }

    if(data.signal_PIPE == 1) {
        loopOK = 0;
        if(data.processinfo==1)
            processinfo_SIGexit(processinfo, SIGPIPE);
    }
#endif

    return loopOK;
}





int processinfo_exec_start(PROCESSINFO *processinfo)
{
	
	processinfo->timerindex ++;
	if(processinfo->timerindex==PROCESSINFO_NBtimer)
	{
		processinfo->timerindex = 0;
		processinfo->timingbuffercnt++;
	}
	
	clock_gettime(CLOCK_REALTIME, &processinfo->texecstart[processinfo->timerindex]);

    if(processinfo->dtiter_limit_enable != 0)
    {
        long dtiter;
        int timerindexlast;

        if(processinfo->timerindex == 0)
            timerindexlast = PROCESSINFO_NBtimer-1;
        else
            timerindexlast = processinfo->timerindex - 1;
        
        dtiter = processinfo->texecstart[processinfo->timerindex].tv_nsec - processinfo->texecstart[timerindexlast].tv_nsec;
        dtiter += 1000000000*(processinfo->texecstart[processinfo->timerindex].tv_sec - processinfo->texecstart[timerindexlast].tv_sec);
        
        
        
        if(dtiter > processinfo->dtiter_limit_value)
        {
			char msgstring[200];
						
			sprintf(msgstring, "dtiter %4ld  %4d %6.1f us  > %6.1f us", processinfo->dtiter_limit_cnt, processinfo->timerindex, 0.001*dtiter, 0.001*processinfo->dtiter_limit_value);			
			processinfo_WriteMessage(processinfo, msgstring);	
			
			if(processinfo->dtiter_limit_enable == 2) // pause process due to timing limit
			{
				processinfo->CTRLval = 1;
				sprintf(msgstring, "dtiter lim -> paused");
				processinfo_WriteMessage(processinfo, msgstring);
			}
			processinfo->dtiter_limit_cnt ++;
		}
    }

	return 0;
}



int processinfo_exec_end(PROCESSINFO *processinfo)
{
    clock_gettime(CLOCK_REALTIME, &processinfo->texecend[processinfo->timerindex]);

    if(processinfo->dtexec_limit_enable != 0)
    {
        long dtexec;

        dtexec = processinfo->texecend[processinfo->timerindex].tv_nsec - processinfo->texecstart[processinfo->timerindex].tv_nsec;
        dtexec += 1000000000*(processinfo->texecend[processinfo->timerindex].tv_sec - processinfo->texecend[processinfo->timerindex].tv_sec);
        
        if(dtexec > processinfo->dtexec_limit_value)
        {
			char msgstring[200];
			
			sprintf(msgstring, "dtexec %4ld  %4d %6.1f us  > %6.1f us", processinfo->dtexec_limit_cnt, processinfo->timerindex, 0.001*dtexec, 0.001*processinfo->dtexec_limit_value);
			processinfo_WriteMessage(processinfo, msgstring);
			
			if(processinfo->dtexec_limit_enable == 2) // pause process due to timing limit
			{				
				processinfo->CTRLval = 1;
				sprintf(msgstring, "dtexec lim -> paused");
				processinfo_WriteMessage(processinfo, msgstring);
			}
			processinfo->dtexec_limit_cnt ++;
		}
    }

    return 0;
}





/*
static int print_header(const char *str, char c)
{
    long n;
    long i;

    attron(A_BOLD);
    n = strlen(str);
    for(i=0; i<(wcol-n)/2; i++)
        printw("%c",c);
    printw("%s", str);
    for(i=0; i<(wcol-n)/2-1; i++)
        printw("%c",c);
    printw("\n");
    attroff(A_BOLD);


    return(0);
}
*/


/**
 * INITIALIZE ncurses
 * 
 */ 
static int initncurses()
{
    if ( initscr() == NULL ) {
        fprintf(stderr, "Error initialising ncurses.\n");
        exit(EXIT_FAILURE);
    }
    getmaxyx(stdscr, wrow, wcol);		/* get the number of rows and columns */
    cbreak();
    keypad(stdscr, TRUE);		/* We get F1, F2 etc..		*/
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


    return 0;
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

static int GetNumberCPUs(PROCINFOPROC *pinfop)
{
    unsigned int pu_index = 0;

#ifdef USE_HWLOC

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
    do {
        pinfop->CPUids[pu_index] = obj->os_index;
        ++pu_index;
        obj = obj->next_cousin;
    } while (obj != NULL);

#else

    FILE *fpout;
    char outstring[16];
	char buf[100];
	
    unsigned int tmp_index = 0;

    fpout = popen ("getconf _NPROCESSORS_ONLN", "r");
    if(fpout==NULL)
    {
        printf("WARNING: cannot run command \"tmuxsessionname\"\n");
    }
    else
    {
        if(fgets(outstring, 16, fpout)== NULL)
            printf("WARNING: fgets error\n");
        pclose(fpout);
    }
    pinfop->NBcpus = atoi(outstring);

	fpout = popen("cat /proc/cpuinfo |grep \"physical id\" | awk '{ print $NF }'", "r");
	pu_index = 0;
	pinfop->NBcpusocket = 1;
	while ((fgets(buf, sizeof(buf), fpout) != NULL)&&(pu_index<pinfop->NBcpus)) {
		pinfop->CPUids[pu_index] = pu_index;
		pinfop->CPUphys[pu_index] = atoi(buf);
		
		//printf("cpu %2d belongs to Physical CPU %d\n", pu_index, pinfop->CPUphys[pu_index] );
		if(pinfop->CPUphys[pu_index]+1 > pinfop->NBcpusocket)
			pinfop->NBcpusocket = pinfop->CPUphys[pu_index]+1;
		
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
	tdiff = info_time_diff(t1, t2);
	scantime_top += 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;

	return NBtop;
}



*/





static int GetCPUloads(PROCINFOPROC *pinfop)
{
    char * line = NULL;
    FILE *fp;
    ssize_t read;
    size_t len = 0;
    int cpu;
    long long vall0, vall1, vall2, vall3, vall4, vall5, vall6, vall7, vall8;
    long long v0, v1, v2, v3, v4, v5, v6, v7, v8;
    char string0[80];


    clock_gettime(CLOCK_REALTIME, &t1);

    fp = fopen("/proc/stat", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    cpu = 0;
    if(getline(&line, &len, fp) == -1)
    {
        printf("[%s][%d]  ERROR: cannot read file\n", __FILE__, __LINE__);
        exit(0);
    }

    while (((read = getline(&line, &len, fp)) != -1)&&(cpu<pinfop->NBcpus)) {

        sscanf(line, "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld", string0, &vall0, &vall1, &vall2, &vall3, &vall4, &vall5, &vall6, &vall7, &vall8);

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

        pinfop->CPUload[cpu] = (1.0*v0+v1+v2+v4+v5+v6)/(v0+v1+v2+v3+v4+v5+v6+v7+v8);
        cpu++;
    }

    fclose(fp);
    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = info_time_diff(t1, t2);
    scantime_CPUload += 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;


    clock_gettime(CLOCK_REALTIME, &t1);

    // number of process per CPU -> we can get that from top?
    char command[200];


    for(cpu=0; cpu<pinfop->NBcpus; cpu++)
    {
        char outstring[200];
        FILE * fpout;


        sprintf(command, "CORENUM=%d; cat _psoutput.txt | grep -E  \"^[[:space:]][[:digit:]]+[[:space:]]+${CORENUM}\"|wc -l", cpu);
        fpout = popen (command, "r");
        if(fpout==NULL)
        {
            printf("WARNING: cannot run command \"%s\"\n", command);
        }
        else
        {
            if(fgets(outstring, 100, fpout)== NULL)
                printf("WARNING: fgets error\n");
            pclose(fpout);
            pinfop->CPUpcnt[cpu] = atoi(outstring);
        }
    }

    //	psOK=0; if [ $psOK = "1" ]; then ls; fi; psOK=1

    sprintf(command, "{ if [ ! -f _psOKlock ]; then touch _psOKlock; ps -e -o pid,psr,cpu,cmd > _psoutput.txt; fi; rm _psOKlock &> /dev/null; } &");
    if(system(command) != 0)
		printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");

    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = info_time_diff(t1, t2);
    scantime_CPUpcnt += 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;

    return(cpu);
}
















// for Display Mode 2

static int PIDcollectSystemInfo(PROCESSINFODISP *pinfodisp, int level)
{

    // COLLECT INFO FROM SYSTEM
    FILE *fp;
    char fname[200];



    // cpuset
    
    int PID = pinfodisp->PID;
    clock_gettime(CLOCK_REALTIME, &t1);
    sprintf(fname, "/proc/%d/task/%d/cpuset", PID, PID);
    fp=fopen(fname, "r");
    if (fp == NULL)
        return -1;
    if(fscanf(fp, "%s", pinfodisp->cpuset) != 1)
		printERROR(__FILE__,__func__,__LINE__, "fscanf returns value != 1");
    fclose(fp);
	clock_gettime(CLOCK_REALTIME, &t2);
	tdiff = info_time_diff(t1, t2);
	scantime_cpuset += 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;


    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char string0[200];
    char string1[300];

	clock_gettime(CLOCK_REALTIME, &t1);
    if(level == 0)
    {
        FILE *fpout;
        char command[200];
        char outstring[200];

        pinfodisp->subprocPIDarray[0] = PID;
        pinfodisp->NBsubprocesses = 1;

        // if(pinfodisp->threads > 1) // look for children
        // {
			DIR *dp;
			struct dirent *ep;
			char dirname[200];
			
            // fprintf(stderr, "reading /proc/%d/task\n", PID);
			sprintf(dirname, "/proc/%d/task/", PID);
			dp = opendir(dirname);

			if (dp != NULL)
			{
				while (ep = readdir(dp))
					{
						if(ep->d_name[0] != '.')
						{
                            int subPID = atoi(ep->d_name);
                            if(subPID != PID){
                                pinfodisp->subprocPIDarray[pinfodisp->NBsubprocesses] = atoi(ep->d_name);
                                pinfodisp->NBsubprocesses++;
                            }
						}
					}
				closedir(dp);
			} else {
                return -1;
        }   
        // }   
        // fprintf(stderr, "%d threads found\n", pinfodisp->NBsubprocesses);
        pinfodisp->threads = pinfodisp->NBsubprocesses;
	}
	clock_gettime(CLOCK_REALTIME, &t2);
	tdiff = info_time_diff(t1, t2);
	scantime_pstree += 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;


    // read /proc/PID/status
	#idef CMDPROC_PROCSTAT
	for(int spindex = 0; spindex < pinfodisp->NBsubprocesses; spindex++)
    {
    	clock_gettime(CLOCK_REALTIME, &t1);
        PID = pinfodisp->subprocPIDarray[spindex];



        sprintf(fname, "/proc/%d/status", PID);
        fp = fopen(fname, "r");
        if (fp == NULL)
            return -1;

        while ((read = getline(&line, &len, fp)) != -1) {
            if (sscanf(line, "%31[^:]: %s", string0, string1) == 2){
            if(spindex == 0 ) {
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
        if (line)
            free(line);
        line = NULL;
        len = 0;
        
        clock_gettime(CLOCK_REALTIME, &t2);
        tdiff = info_time_diff(t1, t2);
        scantime_status += 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;





        // read /proc/PID/stat
        clock_gettime(CLOCK_REALTIME, &t1);
        sprintf(fname, "/proc/%d/stat", PID);

        int           stat_pid;       // (1) The process ID.
        char          stat_comm[20];  // (2) The filename of the executable, in parentheses.
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
        int           stat_tpgid;     // (8) The ID of the foreground process group of the controlling terminal of the process
        unsigned int  stat_flags;     // (9) The kernel flags word of the process
        unsigned long stat_minflt;    // (10) The number of minor faults the process has made which have not required loading a memory page from disk
        unsigned long stat_cminflt;   // (11) The number of minor faults that the process's waited-for children have made
        unsigned long stat_majflt;    // (12) The number of major faults the process has made which have required loading a memory page from disk
        unsigned long stat_cmajflt;   // (13) The number of major faults that the process's waited-for children have made
        unsigned long stat_utime;     // (14) Amount of time that this process has been scheduled in user mode, measured in clock ticks (divide by sysconf(_SC_CLK_TCK)).
        unsigned long stat_stime;     // (15) Amount of time that this process has been scheduled in kernel mode, measured in clock ticks
        long          stat_cutime;       // (16) Amount of time that this process's waited-for children have been scheduled in user mode, measured in clock ticks
        long          stat_cstime;       // (17) Amount of time that this process's waited-for children have been scheduled in kernel mode, measured in clock ticks
        long          stat_priority;     // (18) (Explanation for Linux 2.6) For processes running a
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
        long          stat_nice;         // (19) The nice value (see setpriority(2)), a value in the range 19 (low priority) to -20 (high priority)
        long          stat_num_threads;  // (20) Number of threads in this process
        long          stat_itrealvalue;  // (21) hard coded as 0
        unsigned long long    stat_starttime; // (22) The time the process started after system boot in clock ticks
        unsigned long stat_vsize;        // (23)  Virtual memory size in bytes
        long          stat_rss;          // (24) Resident Set Size: number of pages the process has in real memory
        unsigned long stat_rsslim;       // (25) Current soft limit in bytes on the rss of the process
        unsigned long stat_startcode;    // (26) The address above which program text can run
        unsigned long stat_endcode;      // (27) The address below which program text can run
        unsigned long stat_startstack;   // (28) The address of the start (i.e., bottom) of the stack
        unsigned long stat_kstkesp;      // (29) The current value of ESP (stack pointer), as found in the kernel stack page for the process
        unsigned long stat_kstkeip;      // (30) The current EIP (instruction pointer)
        unsigned long stat_signal;       // (31) The bitmap of pending signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead
        unsigned long stat_blocked;      // (32) The bitmap of blocked signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long stat_sigignore;    // (33) The bitmap of ignored signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long stat_sigcatch;     // (34) The bitmap of ignored signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long stat_wchan;        // (35) This is the "channel" in which the process is waiting.  It is the address of a location in the kernel where the process is sleeping.  The corresponding symbolic name can be found in /proc/[pid]/wchan.
        unsigned long stat_nswap;        // (36) Number of pages swapped (not maintained)
        unsigned long stat_cnswap;       // (37) Cumulative nswap for child processes (not maintained)
        int           stat_exit_signal;  // (38) Signal to be sent to parent when we die
        int           stat_processor;    // (39) CPU number last executed on
        unsigned int  stat_rt_priority;  // (40) Real-time scheduling priority, a number in the range 1 to 99 for processes scheduled under a real-time policy, or 0, for non-real-time processes (see  sched_setscheduler(2)).
        unsigned int  stat_policy;       // (41) Scheduling policy (see sched_setscheduler(2))
        unsigned long long    stat_delayacct_blkio_ticks; // (42) Aggregated block I/O delays, measured in clock ticks
        unsigned long stat_guest_time;   // (43) Guest time of the process (time spent running a virtual CPU for a guest operating system), measured in clock ticks
        long          stat_cguest_time;  // (44) Guest time of the process's children, measured in clock ticks (divide by sysconf(_SC_CLK_TCK)).
        unsigned long stat_start_data;   // (45) Address above which program initialized and uninitialized (BSS) data are placed
        unsigned long stat_end_data;     // (46) ddress below which program initialized and uninitialized (BSS) data are placed
        unsigned long stat_start_brk;    // (47) Address above which program heap can be expanded with brk(2)
        unsigned long stat_arg_start;    // (48) Address above which program command-line arguments (argv) are placed
        unsigned long stat_arg_end;      // (49) Address below program command-line arguments (argv) are placed
        unsigned long stat_env_start;    // (50) Address above which program environment is placed
        unsigned long stat_env_end;      // (51) Address below which program environment is placed
        long          stat_exit_code;    // (52) The thread's exit status in the form reported by waitpid(2)





        fp = fopen(fname, "r");
        int Nfields;
        if (fp == NULL)
            return -1;

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
        if(Nfields != 52) {
            printERROR(__FILE__,__func__,__LINE__, "fscanf returns value != 1");
            pinfodisp->processorarray[spindex] = stat_processor;
            pinfodisp->rt_priority = stat_rt_priority; 
        }
        else
        {
            pinfodisp->processorarray[spindex] = stat_processor;
            pinfodisp->rt_priority = stat_rt_priority;
        }
        fclose(fp);
        
        pinfodisp->sampletimearray[spindex] = 1.0*t1.tv_sec + 1.0e-9*t1.tv_nsec;
        
        pinfodisp->cpuloadcntarray[spindex] = (stat_utime + stat_stime); 
        pinfodisp->memload = 0.0;
        
        clock_gettime(CLOCK_REALTIME, &t2);
        tdiff = info_time_diff(t1, t2);
        scantime_stat += 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
    }
    #endif

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
	char syscommand[200];
	char line[200];
	FILE *fp;
	int NBsetMax = 1000;
	int setindex;
	char word[200];
	char word1[200];
	int NBset = 0;
	
	sprintf(syscommand, "cset set -l | awk '/root/{stop=1} stop==1{print $0}' > _tmplist.txt");
	if(system(syscommand) != 0)
		printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
	
	
	// first scan: get number of entries
	fp = fopen("_tmplist.txt", "r");
	while ( NBset < NBsetMax ) {
        if (fgets(line, 199, fp) == NULL) break;
        NBset++;
//		printf("%3d: %s", NBset, line);
	}
	fclose(fp);
	
	
	setindex = 0;
	fp = fopen("_tmplist.txt", "r");
	while ( 1 ) {
        if (fgets(line, 199, fp) == NULL) break;
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
    for(i=0; i<NBelem; i++)
    {
        printf("   %3ld   : %16s   %s\n", i, StringList[i].name, StringList[i].description);
        fflush(stdout);
    }


    inputOK = 0;

    while(inputOK == 0)
    {
        printf ("\nEnter a number: ");
        fflush(stdout);

        int stringindex = 0;
        char c;
        while( ((c = getchar()) != 13) && (stringindex<strlenmax-1) )
        {
            buff[stringindex] = c;
            if(c == 127) // delete key
            {
                putchar (0x8);
                putchar (' ');
                putchar (0x8);
                stringindex --;
            }
            else
            {
                putchar(c);  // echo on screen
                stringindex++;
            }
            if(stringindex<0)
                stringindex = 0;
        }
        buff[stringindex] = '\0';

        selected = strtol(buff, &p, strlenmax);

        if((selected<0)||(selected>NBelem-1))
        {
            printf("\nError: number not valid. Must be >= 0 and < %d\n", NBelem);
            inputOK = 0;
        }
        else
            inputOK = 1;
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
void *processinfo_scan(void *thptr)
{
    PROCINFOPROC* pinfop;

    pinfop = (PROCINFOPROC*) thptr;

    long pindex;

    pinfop->loopcnt = 0;

    // timing
    static int firstIter = 1;
    static struct timespec t0;
    struct timespec t1;
    double tdiffv;
    struct timespec tdiff;


    while(pinfop->loop == 1)
    {
        // timing measurement
        clock_gettime(CLOCK_REALTIME, &t1);
        if(firstIter == 1)
        {
            tdiffv = 0.1;
            firstIter = 0;
        }
        else
        {
            tdiff = info_time_diff(t0, t1);
            tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        }
        clock_gettime(CLOCK_REALTIME, &t0);
        pinfop->dtscan = tdiffv;



        // LOAD / UPDATE process information


        for(pindex=0; pindex<pinfop->NBpinfodisp; pindex++)
        {
            char SM_fname[200];    // shared memory file name
            struct stat file_stat;


            pinfop->PIDarray[pindex] = pinfolist->PIDarray[pindex];

            // SHOULD WE (RE)LOAD ?
            if(pinfolist->active[pindex] == 0) // inactive
                pinfop->updatearray[pindex] = 0;

            if((pinfolist->active[pindex] == 1)||(pinfolist->active[pindex] == 2)) // active or crashed
            {
                pinfop->updatearray[pindex] = 1;

            }
            //    if(pinfolist->active[pindex] == 2) // mmap crashed, file may still be present
            //        updatearray[pindex] = 1;

            if(pinfolist->active[pindex] == 3) // file has gone away
                pinfop->updatearray[pindex] = 0;





            // check if process info file exists

            sprintf(SM_fname, "%s/proc.%06d.shm", SHAREDMEMDIR, (int) pinfolist->PIDarray[pindex]);

            // Does file exist ?
            if(stat(SM_fname, &file_stat) == -1 && errno == ENOENT)
            {
                // if not, don't (re)load and remove from process info list
                pinfolist->active[pindex] = 0;
                pinfop->updatearray[pindex] = 0;
            }


            if(pinfolist->active[pindex] == 1)
            {
                // check if process still exists
                struct stat sts;
                char procfname[200];

                sprintf(procfname, "/proc/%d", (int) pinfolist->PIDarray[pindex]);
                if (stat(procfname, &sts) == -1 && errno == ENOENT) {
                    // process doesn't exist -> flag as inactive
                    pinfolist->active[pindex] = 2;
                }
            }





            if((pindex<pinfop->NBpinfodisp)&&(pinfop->updatearray[pindex] == 1))
            {
                // (RE)LOAD
                struct stat file_stat;

                // if already mmapped, first unmap
                if(pinfop->pinfommapped[pindex] == 1)
                {
                    fstat(pinfop->fdarray[pindex], &file_stat);
                    munmap(pinfop->pinfoarray[pindex], file_stat.st_size);
                    close(pinfop->fdarray[pindex]);
                    pinfop->pinfommapped[pindex] == 0;
                }


                // COLLECT INFORMATION FROM PROCESSINFO FILE

                pinfop->fdarray[pindex] = open(SM_fname, O_RDWR);
                fstat(pinfop->fdarray[pindex], &file_stat);
                pinfop->pinfoarray[pindex] = (PROCESSINFO*) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, pinfop->fdarray[pindex], 0);
                if (pinfop->pinfoarray[pindex] == MAP_FAILED) {
                    close(pinfop->fdarray[pindex]);
                    endwin();
                    fprintf(stderr, "[%d] Error mapping file %s\n", __LINE__, SM_fname);
                    pinfolist->active[pindex] = 3;
                }
                else
                {
                    pinfop->pinfommapped[pindex] = 1;
                    strncpy(pinfop->pinfodisp[pindex].name, pinfop->pinfoarray[pindex]->name, 40-1);

                    struct tm *createtm;
                    createtm      = gmtime(&pinfop->pinfoarray[pindex]->createtime.tv_sec);
                    pinfop->pinfodisp[pindex].createtime_hr = createtm->tm_hour;
                    pinfop->pinfodisp[pindex].createtime_min = createtm->tm_min;
                    pinfop->pinfodisp[pindex].createtime_sec = createtm->tm_sec;
                    pinfop->pinfodisp[pindex].createtime_ns = pinfop->pinfoarray[pindex]->createtime.tv_nsec;

                    pinfop->pinfodisp[pindex].loopcnt = pinfop->pinfoarray[pindex]->loopcnt;
                }

                pinfop->pinfodisp[pindex].active = pinfolist->active[pindex];
                pinfop->pinfodisp[pindex].PID = pinfolist->PIDarray[pindex];

                pinfop->pinfodisp[pindex].updatecnt ++;

            }
        }



        /** ### Build a time-sorted list of processes
          *
          *
          *
          */
        int index;

        pinfop->NBpindexActive = 0;
        for(pindex=0; pindex<PROCESSINFOLISTSIZE; pindex++)
            if(pinfolist->active[pindex] != 0)
            {
                pinfop->pindexActive[pinfop->NBpindexActive] = pindex;
                pinfop->NBpindexActive++;
            }
        double *timearray;
        long *indexarray;
        timearray  = (double*) malloc(sizeof(double)*pinfop->NBpindexActive);
        indexarray = (long*)   malloc(sizeof(long)  *pinfop->NBpindexActive);
        int listcnt = 0;
        for(index=0; index<pinfop->NBpindexActive; index++)
        {
            pindex = pinfop->pindexActive[index];
            if(pinfop->pinfommapped[pindex] == 1)
            {
                indexarray[index] = pindex;
                // minus sign for most recent first
                //printw("index  %ld  ->  pindex  %ld\n", index, pindex);
                timearray[index] = -1.0*pinfop->pinfoarray[pindex]->createtime.tv_sec - 1.0e-9*pinfop->pinfoarray[pindex]->createtime.tv_nsec;
                listcnt++;
            }
        }
        pinfop->NBpindexActive = listcnt;
        quick_sort2l_double(timearray, indexarray, pinfop->NBpindexActive);

        for(index=0; index<pinfop->NBpindexActive; index++)
            pinfop->sorted_pindex_time[index] = indexarray[index];

        free(timearray);
        free(indexarray);



        if(pinfop->DisplayMode == 3)
        {
            GetCPUloads(pinfop);


            // collect required info for display
            for(pindex=0; pindex<PROCESSINFOLISTSIZE ; pindex++)
            {
                if(pinfolist->active[pindex] != 0)
                {

                    if(pinfop->pinfodisp[pindex].NBsubprocesses != 0) // pinfop->pinfodisp[pindex].NBsubprocesses should never be zero - should be at least 1 (for main process)
                    {
						
                        int spindex; // sub process index, 0 for main
                    /*
                        if(pinfop->psysinfostatus[pindex] != -1)
                        {
                            for(spindex = 0; spindex < pinfop->pinfodisp[pindex].NBsubprocesses; spindex++)
                            {
                                // place info in subprocess arrays
                                pinfop->pinfodisp[pindex].sampletimearray_prev[spindex] = pinfop->pinfodisp[pindex].sampletimearray[spindex];
                                // Context Switches

                                pinfop->pinfodisp[pindex].ctxtsw_voluntary_prev[spindex]    = pinfop->pinfodisp[pindex].ctxtsw_voluntary[spindex];
                                pinfop->pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex] = pinfop->pinfodisp[pindex].ctxtsw_nonvoluntary[spindex];


                                // CPU use
                                pinfop->pinfodisp[pindex].cpuloadcntarray_prev[spindex] = pinfop->pinfodisp[pindex].cpuloadcntarray[spindex];

                            }
                        }
*/

                        pinfop->psysinfostatus[pindex] = PIDcollectSystemInfo(&(pinfop->pinfodisp[pindex]), 0);
                      /*  if(pinfop->psysinfostatus[pindex] != -1)
                        {
                            char cpuliststring[200];
                            char cpustring[16];

                            for(spindex = 0; spindex < pinfop->pinfodisp[pindex].NBsubprocesses; spindex++)
                            {
                                if( pinfop->pinfodisp[pindex].sampletimearray[spindex] - pinfop->pinfodisp[pindex].sampletimearray_prev[spindex]) {
                                    // get CPU and MEM load
                                    pinfop->pinfodisp[pindex].subprocCPUloadarray[spindex] = 100.0*((1.0*pinfop->pinfodisp[pindex].cpuloadcntarray[spindex]-pinfop->pinfodisp[pindex].cpuloadcntarray_prev[spindex])/sysconf(_SC_CLK_TCK)) /  ( pinfop->pinfodisp[pindex].sampletimearray[spindex] - pinfop->pinfodisp[pindex].sampletimearray_prev[spindex]);
                                    pinfop->pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex] = 0.9 * pinfop->pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex] + 0.1 * pinfop->pinfodisp[pindex].subprocCPUloadarray[spindex];
                                }
                            }

                            sprintf(cpuliststring, ",%s,", pinfop->pinfodisp[pindex].cpusallowed);

                            int cpu;
                            for (cpu = 0; cpu < pinfop->NBcpus; cpu++)
                            {
                                int cpuOK = 0;
                                int cpumin, cpumax;

                                sprintf(cpustring, ",%d,", pinfop->CPUids[cpu]);
                                if(strstr(cpuliststring, cpustring) != NULL)
                                    cpuOK = 1;


                                for(cpumin=0; cpumin<=pinfop->CPUids[cpu]; cpumin++)
                                    for(cpumax=pinfop->CPUids[cpu]; cpumax<pinfop->NBcpus; cpumax++)
                                    {
                                        sprintf(cpustring, ",%d-%d,", cpumin, cpumax);
                                        if(strstr(cpuliststring, cpustring) != NULL)
                                            cpuOK = 1;
                                    }
                                pinfop->pinfodisp[pindex].cpuOKarray[cpu] = cpuOK;
                            }
                        }*/
                        
                    }

                }
            }

        } // end of DisplayMode 3


        pinfop->loopcnt++;
        usleep(pinfop->twaitus);
    }
    printf("Process info scan ended cleanly: %ld scans completed\n", pinfop->loopcnt);

    return NULL;
}






void processinfo_CTRLscreen_atexit()
{
	echo();
	endwin();
	
	printf("EXIT from processinfo_CTRLscreen at line %d\n", CTRLscreenExitLine);
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

int_fast8_t processinfo_CTRLscreen()
{
    long pindex, index;

    PROCINFOPROC procinfoproc;  // Main structure - holds everything that needs to be shared with other functions and scan thread
    pthread_t threadscan;

    int cpusocket;

    char syscommand[300];
    char pselected_FILE[200];
    char pselected_FUNCTION[200];
    int  pselected_LINE;

    // timers
    struct timespec t1loop;
    struct timespec t2loop;
    struct timespec tdiffloop;

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


    processinfo_CatchSignals();

    setlocale(LC_ALL, "");


    for(pindex=0; pindex<PROCESSINFOLISTSIZE; pindex++)
    {
        procinfoproc.updatearray[pindex]   = 1; // initialize: load all
        procinfoproc.pinfommapped[pindex]  = 0;
        procinfoproc.selectedarray[pindex] = 0; // initially not selected
        procinfoproc.loopcntoffsetarray[pindex] = 0;
    }

    STRINGLISTENTRY *CPUsetList;
    int NBCPUset;
    CPUsetList = malloc(1000 * sizeof(STRINGLISTENTRY));
    NBCPUset = processinfo_CPUsets_List(CPUsetList);


    // Create / read process list
    //
    processinfo_shm_list_create();


    // copy pointer
    procinfoproc.pinfolist = pinfolist;

    procinfoproc.NBcpus = GetNumberCPUs(&procinfoproc);
    GetCPUloads(&procinfoproc);


    // INITIALIZE ncurses
    initncurses();
	atexit( processinfo_CTRLscreen_atexit );

	
    procinfoproc.NBpinfodisp = wrow-5;
    procinfoproc.pinfodisp = (PROCESSINFODISP*) malloc(sizeof(PROCESSINFODISP)*procinfoproc.NBpinfodisp);
    for(pindex=0; pindex<procinfoproc.NBpinfodisp; pindex++)
    {
        procinfoproc.pinfodisp[pindex].updatecnt = 0;
        procinfoproc.pinfodisp[pindex].NBsubprocesses = 1;  // by default, each process is assumed to be single-threaded
    }

    pindexActiveSelected = 0;
    procinfoproc.DisplayMode = 2;
    // display modes:
    // 2: overview
    // 3: CPU affinity

    // Start scan thread
    procinfoproc.loop = 1;
    procinfoproc.twaitus = 1000000; // 1 sec
	
	
	pthread_create( &threadscan, NULL, processinfo_scan, (void*) &procinfoproc);




    // wait for first scan to be completed
    while( procinfoproc.loopcnt < 1 )
    {
		printf("procinfoproc.loopcnt  = %ld\n", (long) procinfoproc.loopcnt);
        usleep(1000);
	}




    int loopOK = 1;
    int freeze = 0;
    long cnt = 0;
    int MonMode = 0;
    int TimeSorted = 1;  // by default, sort processes by start time (most recent at top)
    int dispindexMax = 0;


    clear();
    int Xexit = 0; // toggles to 1 when users types x
	
    while( loopOK == 1 )
    {
        int pid;
        char command[200];

        usleep((long) (1000000.0/frequ));
        int ch = getch();

        clock_gettime(CLOCK_REALTIME, &t1loop);

        scantime_cpuset = 0.0;
        scantime_status = 0.0;
        scantime_stat = 0.0;
        scantime_pstree = 0.0;
        scantime_top = 0.0;
        scantime_CPUload = 0.0;
        scantime_CPUpcnt = 0.0;


        if(freeze==0)
        {
            attron(A_BOLD);
            sprintf(monstring, "Mode %d   PRESS x TO STOP MONITOR", MonMode);
            print_header(monstring, '-');
            attroff(A_BOLD);
        }

        int selectedOK = 0; // goes to 1 if at least one process is selected
        switch (ch)
        {
        case 'f':     // Freeze screen (toggle)
            if(freeze==0)
                freeze = 1;
            else
                freeze = 0;
            break;

        case 'x':     // Exit control screen
            loopOK = 0;
            Xexit = 1;
            break;

        case ' ':     // Mark current PID as selected (if none selected, other commands only apply to highlighted process)
            pindex = pindexSelected;
            if(procinfoproc.selectedarray[pindex] == 1)
                procinfoproc.selectedarray[pindex] = 0;
            else
                procinfoproc.selectedarray[pindex] = 1;
            break;

        case 'u':    // undelect all
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                procinfoproc.selectedarray[pindex] = 0;
            }
            break;


        case KEY_UP:
            pindexActiveSelected --;
            if(pindexActiveSelected<0)
                pindexActiveSelected = 0;
            if(TimeSorted == 0)
                pindexSelected = procinfoproc.pindexActive[pindexActiveSelected];
            else
                pindexSelected = procinfoproc.sorted_pindex_time[pindexActiveSelected];
            break;

        case KEY_DOWN:
            pindexActiveSelected ++;
            if(pindexActiveSelected>procinfoproc.NBpindexActive-1)
                pindexActiveSelected = procinfoproc.NBpindexActive-1;
            if(TimeSorted == 0)
                pindexSelected = procinfoproc.pindexActive[pindexActiveSelected];
            else
                pindexSelected = procinfoproc.sorted_pindex_time[pindexActiveSelected];
            break;

        case KEY_PPAGE:
            pindexActiveSelected -= 10;
            if(pindexActiveSelected<0)
                pindexActiveSelected = 0;
            if(TimeSorted == 0)
                pindexSelected = procinfoproc.pindexActive[pindexActiveSelected];
            else
                pindexSelected = procinfoproc.sorted_pindex_time[pindexActiveSelected];
            break;

        case KEY_NPAGE:
            pindexActiveSelected += 10;
            if(pindexActiveSelected>procinfoproc.NBpindexActive-1)
                pindexActiveSelected = procinfoproc.NBpindexActive-1;
            if(TimeSorted == 0)
                pindexSelected = procinfoproc.pindexActive[pindexActiveSelected];
            else
                pindexSelected = procinfoproc.sorted_pindex_time[pindexActiveSelected];
            break;





        case 'T':
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    pid = pinfolist->PIDarray[pindex];
                    kill(pid, SIGTERM);
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                pid = pinfolist->PIDarray[pindex];
                kill(pid, SIGTERM);
            }
            break;

        case 'K':
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    pid = pinfolist->PIDarray[pindex];
                    kill(pid, SIGKILL);
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                pid = pinfolist->PIDarray[pindex];
                kill(pid, SIGKILL);
            }
            break;

        case 'I':
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    pid = pinfolist->PIDarray[pindex];
                    kill(pid, SIGINT);
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                pid = pinfolist->PIDarray[pindex];
                kill(pid, SIGINT);
            }
            break;

        case 'r':
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    if(pinfolist->active[pindex]!=1)
                    {
                        char SM_fname[200];
                        sprintf(SM_fname, "%s/proc.%06d.shm", SHAREDMEMDIR, (int) pinfolist->PIDarray[pindex]);
                        remove(SM_fname);
                    }
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                if(pinfolist->active[pindex]!=1)
                {
                    char SM_fname[200];
                    sprintf(SM_fname, "%s/proc.%06d.shm", SHAREDMEMDIR, (int) pinfolist->PIDarray[pindex]);
                    remove(SM_fname);
                }
            }
            break;

        case 'R':
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(pinfolist->active[pindex]!=1)
                {
                    char SM_fname[200];
                    sprintf(SM_fname, "%s/proc.%06d.shm", SHAREDMEMDIR, (int) pinfolist->PIDarray[pindex]);
                    remove(SM_fname);
                }
            }
            break;

        // loop controls
        case 'p': // pause toggle
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    if(procinfoproc.pinfoarray[pindex]->CTRLval == 0)
                        procinfoproc.pinfoarray[pindex]->CTRLval = 1;
                    else
                        procinfoproc.pinfoarray[pindex]->CTRLval = 0;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                if(procinfoproc.pinfoarray[pindex]->CTRLval == 0)
                    procinfoproc.pinfoarray[pindex]->CTRLval = 1;
                else
                    procinfoproc.pinfoarray[pindex]->CTRLval = 0;
            }
            break;

        case 'c': // compute toggle (toggles between 0-run and 5-run-without-compute) 
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    if(procinfoproc.pinfoarray[pindex]->CTRLval == 0) // if running, turn compute to off
                        procinfoproc.pinfoarray[pindex]->CTRLval = 5;
                    else if (procinfoproc.pinfoarray[pindex]->CTRLval == 5) // if compute off, turn compute back on
                        procinfoproc.pinfoarray[pindex]->CTRLval = 0; 
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                if(procinfoproc.pinfoarray[pindex]->CTRLval == 0) // if running, turn compute to off
                    procinfoproc.pinfoarray[pindex]->CTRLval = 5;
                else if (procinfoproc.pinfoarray[pindex]->CTRLval == 5) // if compute off, turn compute back on
                    procinfoproc.pinfoarray[pindex]->CTRLval = 0; 
            }
            break;

        case 's': // step
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    procinfoproc.pinfoarray[pindex]->CTRLval = 2;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                procinfoproc.pinfoarray[pindex]->CTRLval = 2;
            }
            break;

		


        case '>': // move to other cpuset
            pindex = pindexSelected;
            if(pinfolist->active[pindex]==1)
            {
                endwin();
                if(system("clear") != 0) // clear screen
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
                printf("CURRENT cpu set : %s\n",  procinfoproc.pinfodisp[pindex].cpuset);
                listindex = processinfo_SelectFromList(CPUsetList, NBCPUset);
                sprintf(syscommand,"sudo cset proc -m %d %s", pinfolist->PIDarray[pindex], CPUsetList[listindex].name);
                if(system(syscommand) != 0)
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
                initncurses();
            }
            break;

        case '<': // move to same cpuset
            pindex = pindexSelected;
            if(pinfolist->active[pindex]==1)
            {
                endwin();
                sprintf(syscommand,"sudo cset proc -m %d root &> /dev/null", pinfolist->PIDarray[pindex]);
                if(system(syscommand) != 0)
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
                sprintf(syscommand,"sudo cset proc --force -m %d %s &> /dev/null", pinfolist->PIDarray[pindex], procinfoproc.pinfodisp[pindex].cpuset);
                if(system(syscommand) != 0)
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
                initncurses();
            }
            break;


        case 'e': // exit
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    procinfoproc.pinfoarray[pindex]->CTRLval = 3;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                procinfoproc.pinfoarray[pindex]->CTRLval = 3;
            }
            break;

        case 'z': // apply current value as offset (zero loop counter)
            selectedOK = 0;
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    procinfoproc.loopcntoffsetarray[pindex] = procinfoproc.pinfoarray[pindex]->loopcnt;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                procinfoproc.loopcntoffsetarray[pindex] = procinfoproc.pinfoarray[pindex]->loopcnt;
            }
            break;

        case 'Z': // revert to original counter value
            for(index=0; index<procinfoproc.NBpindexActive; index++)
            {
                pindex = procinfoproc.pindexActive[index];
                if(procinfoproc.selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    procinfoproc.loopcntoffsetarray[pindex] = 0;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                procinfoproc.loopcntoffsetarray[pindex] = 0;
            }
            break;

        case 't':
            endwin();
            sprintf(syscommand, "tmux a -t %s", procinfoproc.pinfoarray[pindexSelected]->tmuxname);
            if(system(syscommand) != 0)
                printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
            initncurses();
            break;

        case 'a':
            pindex = pindexSelected;
            if(pinfolist->active[pindex]==1)
            {
                endwin();
                sprintf(syscommand, "watch -n 0.1 cat /proc/%d/status", (int) pinfolist->PIDarray[pindex]);
                if(system(syscommand) != 0)
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
                initncurses();
            }
            break;

        case 'd':
            pindex = pindexSelected;
            if(pinfolist->active[pindex]==1)
            {
                endwin();
                sprintf(syscommand, "watch -n 0.1 cat /proc/%d/sched", (int) pinfolist->PIDarray[pindex]);
                if(system(syscommand) != 0)
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
                initncurses();
            }
            break;


        case 'o':
            if(TimeSorted == 1)
                TimeSorted = 0;
            else
                TimeSorted = 1;
            break;


        case 'L': // toggle time limit (iter)
            pindex = pindexSelected;
            ToggleValue = procinfoproc.pinfoarray[pindex]->dtiter_limit_enable;
            if(ToggleValue==0)
            {
                procinfoproc.pinfoarray[pindex]->dtiter_limit_enable = 1;
                procinfoproc.pinfoarray[pindex]->dtiter_limit_value = (long) (1.5*procinfoproc.pinfoarray[pindex]->dtmedian_iter_ns);
                procinfoproc.pinfoarray[pindex]->dtiter_limit_cnt = 0;
            }
            else
            {
                ToggleValue ++;
                if(ToggleValue==3)
                    ToggleValue = 0;
                procinfoproc.pinfoarray[pindex]->dtiter_limit_enable = ToggleValue;
            }
            break;;

        case 'M' : // toggle time limit (exec)
            pindex = pindexSelected;
            ToggleValue = procinfoproc.pinfoarray[pindex]->dtexec_limit_enable;
            if(ToggleValue==0)
            {
                procinfoproc.pinfoarray[pindex]->dtexec_limit_enable = 1;
                procinfoproc.pinfoarray[pindex]->dtexec_limit_value = (long) (1.5*procinfoproc.pinfoarray[pindex]->dtmedian_exec_ns + 20000);
                procinfoproc.pinfoarray[pindex]->dtexec_limit_cnt = 0;
            }
            else
            {
                ToggleValue ++;
                if(ToggleValue==3)
                    ToggleValue = 0;
                procinfoproc.pinfoarray[pindex]->dtexec_limit_enable = ToggleValue;
            }
            break;;


        case 'm' : // message
            pindex = pindexSelected;
            if(pinfolist->active[pindex]==1)
            {
                endwin();
                sprintf(syscommand, "clear; tail -f %s", procinfoproc.pinfoarray[pindex]->logfilename);
                //sprintf(syscommand, "ls -l %s", pinfoarray[pindex]->logfilename);
                if(system(syscommand) != 0)
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
                initncurses();
            }
            break;





        // ============ SCREENS

        case 'h': // help
            procinfoproc.DisplayMode = 1;
            break;

        case KEY_F(2): // control
            procinfoproc.DisplayMode = 2;
            break;

        case KEY_F(3): // resources
            procinfoproc.DisplayMode = 3;
            break;

        case KEY_F(4): // timing
            procinfoproc.DisplayMode = 4;
            break;

        case KEY_F(5): // htop
            endwin();
            sprintf(syscommand, "htop");
            if(system(syscommand) != 0)
                printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
            initncurses();
            break;

        case KEY_F(6): // iotop
            endwin();
            sprintf(syscommand, "sudo iotop -o");
            if(system(syscommand) != 0)
                printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
            initncurses();
            break;

        case KEY_F(7): // atop
            endwin();
            sprintf(syscommand, "sudo atop");
            if(system(syscommand) != 0)
                printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
            initncurses();
            break;




        // ============ SCANNING

        case '{': // slower scan update
            procinfoproc.twaitus = (int) (1.2*procinfoproc.twaitus);
            if(procinfoproc.twaitus > 1000000)
                procinfoproc.twaitus = 1000000;
            break;

        case '}': // faster scan update
            procinfoproc.twaitus = (int) (0.83333333333333333333*procinfoproc.twaitus);
            if(procinfoproc.twaitus < 1000)
                procinfoproc.twaitus = 1000;
            break;


        // ============ DISPLAY

        case '-': // slower display update
            frequ *= 0.5;
            if(frequ < 1.0)
                frequ = 1.0;
            if(frequ > 64.0)
                frequ = 64.0;
            break;


        case '+': // faster display update
            frequ *= 2.0;
            if(frequ < 1.0)
                frequ = 1.0;
            if(frequ > 64.0)
                frequ = 64.0;
            break;

        }
        clock_gettime(CLOCK_REALTIME, &t01loop);



        if(freeze==0)
        {
            erase();
			
            if(procinfoproc.DisplayMode == 1)
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
                printw("   Process timing screen\n");

                attron(attrval);
                printw("    F5");
                attroff(attrval);
                printw("   htop        Type F10 to exit\n");

                attron(attrval);
                printw("    F6");
                attroff(attrval);
                printw("   iotop       Type q to exit\n");

                attron(attrval);
                printw("    F7");
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

                printw("%2d cpus   %2d processes tracked    Display Mode %d\n", procinfoproc.NBcpus, procinfoproc.NBpindexActive, procinfoproc.DisplayMode);

                if(procinfoproc.DisplayMode==1)
                {
                    attron(A_REVERSE);
                    printw("[h] Help");
                    attroff(A_REVERSE);
                }
                else
                    printw("[h] Help");
                printw("   ");

                if(procinfoproc.DisplayMode==2)
                {
                    attron(A_REVERSE);
                    printw("[F2] CTRL");
                    attroff(A_REVERSE);
                }
                else
                    printw("[F2] CTRL");
                printw("   ");

                if(procinfoproc.DisplayMode==3)
                {
                    attron(A_REVERSE);
                    printw("[F3] Resources");
                    attroff(A_REVERSE);
                }
                else
                    printw("[F3] Resources");
                printw("   ");

                if(procinfoproc.DisplayMode==4)
                {
                    attron(A_REVERSE);
                    printw("[F4] Timing");
                    attroff(A_REVERSE);
                }
                else
                    printw("[F4] Timing");
                printw("   ");

                if(procinfoproc.DisplayMode==5)
                {
                    attron(A_REVERSE);
                    printw("[F5] htop (F10 to exit)");
                    attroff(A_REVERSE);
                }
                else
                    printw("[F5] htop (F10 to exit)");
                printw("   ");

                if(procinfoproc.DisplayMode==6)
                {
                    attron(A_REVERSE);
                    printw("[F6] iotop (q to exit)");
                    attroff(A_REVERSE);
                }
                else
                    printw("[F6] iotop (q to exit)");
                printw("   ");

                if(procinfoproc.DisplayMode==6)
                {
                    attron(A_REVERSE);
                    printw("[F7] atop (q to exit)");
                    attroff(A_REVERSE);
                }
                else
                    printw("[F7] atop (q to exit)");
                printw("   ");


                printw("\n");



                printw("Display frequ = %2d Hz  [%ld] fscan=%5.2f Hz ( %5.2f Hz %5.2f %% busy )\n", (int) (frequ+0.5), procinfoproc.loopcnt, 1.0/procinfoproc.dtscan, 1000000.0/procinfoproc.twaitus, 100.0*(procinfoproc.dtscan-1.0e-6*procinfoproc.twaitus)/procinfoproc.dtscan);


                if(procinfoproc.pinfommapped[pindexSelected] == 1)
                {

                    strcpy(pselected_FILE, procinfoproc.pinfoarray[pindexSelected]->source_FILE);
                    strcpy(pselected_FUNCTION, procinfoproc.pinfoarray[pindexSelected]->source_FUNCTION);
                    pselected_LINE = procinfoproc.pinfoarray[pindexSelected]->source_LINE;

                    printw("Source Code: %s line %d (function %s)\n", pselected_FILE,  pselected_LINE, pselected_FUNCTION);
                }
                else
                {
                    sprintf(pselected_FILE, "?");
                    sprintf(pselected_FUNCTION, "?");
                    pselected_LINE = 0;
                    printw("\n");
                }

                printw("\n");

                clock_gettime(CLOCK_REALTIME, &t02loop);




                clock_gettime(CLOCK_REALTIME, &t03loop);




                clock_gettime(CLOCK_REALTIME, &t04loop);

                /** ### Display
                 *
                 *
                 *
                 */



                int dispindex;
                if(TimeSorted == 0)
                    dispindexMax = wrow-4;
                else
                    dispindexMax = procinfoproc.NBpindexActive;



                if(procinfoproc.DisplayMode == 3)
                {
                    int cpu;

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


                    // List CPUs
                    printw(
                        "                                                                "
                        "%2d sockets %2d CPUs  ",
                        procinfoproc.NBcpusocket, procinfoproc.NBcpus);

                    for(cpusocket=0; cpusocket < procinfoproc.NBcpusocket; cpusocket++)
                    {
                        if(cpusocket>0)
                            printw("    ");
                        for (cpu = 0; cpu < procinfoproc.NBcpus; cpu++)
                            if(procinfoproc.CPUphys[cpu] == cpusocket)
                                printw("|%02d", procinfoproc.CPUids[cpu]);
                        printw("|");
                    }
                    printw("\n");

                    // List CPU # processes
                    printw("                                                                         PROCESSES  ", procinfoproc.NBcpus);


                    for(cpusocket=0; cpusocket < procinfoproc.NBcpusocket; cpusocket++)
                    {
                        if(cpusocket>0)
                            printw("    ");

                        for (cpu = 0; cpu < procinfoproc.NBcpus; cpu++)
                            if(procinfoproc.CPUphys[cpu] == cpusocket)
                            {
                                int vint = procinfoproc.CPUpcnt[procinfoproc.CPUids[cpu]];
                                if(vint>99)
                                    vint = 99;

                                ColorCode = 0;
                                if(vint>CPUpcntLim1)
                                    ColorCode = 2;
                                if(vint>CPUpcntLim2)
                                    ColorCode = 3;
                                if(vint>CPUpcntLim3)
                                    ColorCode = 4;
                                if(vint<CPUpcntLim0)
                                    ColorCode = 5;

                                printw("|");
                                if(ColorCode != 0)
                                    attron(COLOR_PAIR(ColorCode));
                                printw("%02d", vint);
                                if(ColorCode != 0)
                                    attroff(COLOR_PAIR(ColorCode));
                            }
                        printw("|");
                    }

                    printw("\n");





                    // Print CPU LOAD
                    printw("                                                                          CPU LOAD  ", procinfoproc.NBcpus);
                    for(cpusocket=0; cpusocket < procinfoproc.NBcpusocket; cpusocket++)
                    {
                        if(cpusocket>0)
                            printw("    ");
                        for (cpu = 0; cpu < procinfoproc.NBcpus; cpu++)
                            if(procinfoproc.CPUphys[cpu] == cpusocket)
                            {
                                int vint = (int) (100.0*procinfoproc.CPUload[procinfoproc.CPUids[cpu]]);
                                if(vint>99)
                                    vint = 99;

                                ColorCode = 0;
                                if(vint>CPUloadLim1)
                                    ColorCode = 2;
                                if(vint>CPUloadLim2)
                                    ColorCode = 3;
                                if(vint>CPUloadLim3)
                                    ColorCode = 4;
                                if(vint<CPUloadLim0)
                                    ColorCode = 5;

                                printw("|");
                                if(ColorCode != 0)
                                    attron(COLOR_PAIR(ColorCode));
                                printw("%02d", vint);
                                if(ColorCode != 0)
                                    attroff(COLOR_PAIR(ColorCode));
                            }
                        printw("|");
                    }

                    printw("\n");
                    printw("\n");
                }
                
                // print header for display mode 2
                if(procinfoproc.DisplayMode == 2)
                {
					printw("\n");
					printw("\n");					
					printw("   STATUS    PID   process name                    run status                        tmuxSession     loopcnt        Description                                  Message                                               \n");
					printw("\n");
				}
             
                // print header for display mode 4
                if(procinfoproc.DisplayMode == 4)
                {
					printw("\n");
					printw("\n");					
					printw("   STATUS    PID   process name                    \n");
			//      printw("   ACTIVE   31418  dm00-comb                      1   0 ..00  ITERlim 0/    0/   0  EXEClim 0/    0/   0  ITER    0.0us [   0.0us -    0.0us ] EXEC    0.0us [   0.0us -    0.0us ]  busy =   0.00 %");
					printw("\n");
				}



                clock_gettime(CLOCK_REALTIME, &t05loop);
                


                // ===========================================================================
                // ============== PRINT INFORMATION FOR EACH PROCESS =========================
                // ===========================================================================

                for(dispindex=0; dispindex < dispindexMax; dispindex++)
                {
                    if(TimeSorted == 0)
                        pindex = dispindex;
                    else
                        pindex = procinfoproc.sorted_pindex_time[dispindex];

                    if(pindex<procinfoproc.NBpinfodisp)
                    {

                        if(pindex == pindexSelected)
                            attron(A_REVERSE);

                        if(procinfoproc.selectedarray[pindex]==1)
                            printw("*");
                        else
                            printw(" ");



                        if(pinfolist->active[pindex] == 1)
                        {
                            attron(COLOR_PAIR(2));
                            printw("  ACTIVE");
                            attroff(COLOR_PAIR(2));
                        }

                        if(pinfolist->active[pindex] == 2)  // not active: crashed or terminated
                        {
                            if(procinfoproc.pinfoarray[pindex]->loopstat == 3) // clean exit
                            {
                                attron(COLOR_PAIR(3));
                                printw(" STOPPED");
                                attroff(COLOR_PAIR(3));
                            }
                            else
                            {
                                attron(COLOR_PAIR(4));
                                printw(" CRASHED");
                                attroff(COLOR_PAIR(4));
                            }
                        }
                        
                        
                        


                        if(pinfolist->active[pindex] != 0)
                        {
                            if(pindex == pindexSelected)
                                attron(A_REVERSE);

                            printw("  %6d", pinfolist->PIDarray[pindex]);

                            attron(A_BOLD);
                            printw("  %-30s", procinfoproc.pinfodisp[pindex].name);
                            attroff(A_BOLD);


                            // ================ DISPLAY MODE 2 ==================
                            if( procinfoproc.DisplayMode == 2)
                            {
                                switch (procinfoproc.pinfoarray[pindex]->loopstat)
                                {
                                case 0:
                                    printw("INIT");
                                    break;

                                case 1:
                                    printw(" RUN");
                                    break;

                                case 2:
                                    printw("PAUS");
                                    break;

                                case 3:
                                    printw("TERM");
                                    break;

                                case 4:
                                    printw(" ERR");
                                    break;

                                default:
                                    printw(" ?? ");
                                }
                                
                                
                                if(procinfoproc.pinfoarray[pindex]->CTRLval==0)
                                {
									attron(COLOR_PAIR(2));
									printw(" C%d", procinfoproc.pinfoarray[pindex]->CTRLval );
									attroff(COLOR_PAIR(2));
								}
								else
									printw(" C%d", procinfoproc.pinfoarray[pindex]->CTRLval );


                                printw(" %02d:%02d:%02d.%03d",
                                       procinfoproc.pinfodisp[pindex].createtime_hr,
                                       procinfoproc.pinfodisp[pindex].createtime_min,
                                       procinfoproc.pinfodisp[pindex].createtime_sec,
                                       (int) (0.000001*(procinfoproc.pinfodisp[pindex].createtime_ns)));

                                printw(" %26s", procinfoproc.pinfoarray[pindex]->tmuxname);


                                if(procinfoproc.pinfoarray[pindex]->loopcnt==procinfoproc.loopcntarray[pindex])
                                {   // loopcnt has not changed
                                    printw("  %10ld", procinfoproc.pinfoarray[pindex]->loopcnt-procinfoproc.loopcntoffsetarray[pindex]);
                                }
                                else
                                {   // loopcnt has changed
                                    attron(COLOR_PAIR(2));
                                    printw("  %10ld", procinfoproc.pinfoarray[pindex]->loopcnt-procinfoproc.loopcntoffsetarray[pindex]);
                                    attroff(COLOR_PAIR(2));
                                }

                                procinfoproc.loopcntarray[pindex] = procinfoproc.pinfoarray[pindex]->loopcnt;


                                printw("  %25s", procinfoproc.pinfoarray[pindex]->description);

                                if(procinfoproc.pinfoarray[pindex]->loopstat == 4) // ERROR
                                    attron(COLOR_PAIR(4));
                                printw("  %78s", procinfoproc.pinfoarray[pindex]->statusmsg);
                                if(procinfoproc.pinfoarray[pindex]->loopstat == 4) // ERROR
                                    attroff(COLOR_PAIR(4));
                            }
                            


                            // ================ DISPLAY MODE 3 ==================
                            if( procinfoproc.DisplayMode == 3)
                            {
                                int cpu;


                                if(procinfoproc.psysinfostatus[pindex] == -1)
                                {
                                    printw(" no process info available\n");
                                }
                                else
                                {

                                    int spindex; // sub process index, 0 for main
                                    for(spindex = 0; spindex < procinfoproc.pinfodisp[pindex].NBsubprocesses; spindex++)
                                    {
                                        int TID; // thread ID


                                        if(spindex>0)
                                        {
                                            TID = procinfoproc.pinfodisp[pindex].subprocPIDarray[spindex];
                                            printw("               |---%6d                        ", procinfoproc.pinfodisp[pindex].subprocPIDarray[spindex]);
                                        }
                                        else
                                        {
                                            TID = procinfoproc.pinfodisp[pindex].PID;
                                            procinfoproc.pinfodisp[pindex].subprocPIDarray[0] = procinfoproc.pinfodisp[pindex].PID;
                                        }

                                        printw(" %2d", procinfoproc.pinfodisp[pindex].rt_priority);
                                        printw(" %-10s ", procinfoproc.pinfodisp[pindex].cpuset);
                                        printw(" %2dx ", procinfoproc.pinfodisp[pindex].threads);




                                        // Context Switches
										#ifdef CMDPROC_CONTEXTSWITCH
                                        if(procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex] != procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary[spindex])
                                            attron(COLOR_PAIR(4));
                                        else if(procinfoproc.pinfodisp[pindex].ctxtsw_voluntary_prev[spindex] != procinfoproc.pinfodisp[pindex].ctxtsw_voluntary[spindex])
                                            attron(COLOR_PAIR(3));

                                        printw("ctxsw: +%02ld +%02ld",
                                               abs(procinfoproc.pinfodisp[pindex].ctxtsw_voluntary[spindex]    - procinfoproc.pinfodisp[pindex].ctxtsw_voluntary_prev[spindex])%100,
                                               abs(procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary[spindex] - procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex])%100
                                              );

                                        if(procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex] != procinfoproc.pinfodisp[pindex].ctxtsw_nonvoluntary[spindex])
                                            attroff(COLOR_PAIR(4));
                                        else if(procinfoproc.pinfodisp[pindex].ctxtsw_voluntary_prev[spindex] != procinfoproc.pinfodisp[pindex].ctxtsw_voluntary[spindex])
                                            attroff(COLOR_PAIR(3));
                                        printw(" ");
                                        #endif


                                        
                                        // CPU use
                                        #ifdef CMDPROC_CPUUSE
                                        int cpuColor = 0;

                                        //					if(pinfodisp[pindex].subprocCPUloadarray[spindex]>5.0)
                                        cpuColor = 1;
                                        if(procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex]>10.0)
                                            cpuColor = 2;
                                        if(procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex]>20.0)
                                            cpuColor = 3;
                                        if(procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex]>40.0)
                                            cpuColor = 4;
                                        if(procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex]<1.0)
                                            cpuColor = 5;
                                       

                                        // First group of cores (physical CPU 0)
                                        for (cpu = 0; cpu < procinfoproc.NBcpus / procinfoproc.NBcpusocket; cpu++)
                                        {
                                            printw("|");
                                            if(procinfoproc.CPUids[cpu] == procinfoproc.pinfodisp[pindex].processorarray[spindex])
                                                attron(COLOR_PAIR(cpuColor));

                                            if(procinfoproc.pinfodisp[pindex].cpuOKarray[cpu] == 1)
                                                printw("%2d", procinfoproc.CPUids[cpu]);
                                            else
                                                printw("  ");

                                            if(procinfoproc.CPUids[cpu] == procinfoproc.pinfodisp[pindex].processorarray[spindex])
                                                attroff(COLOR_PAIR(cpuColor));
                                        }
                                        printw("|    ");



                                        // Second group of cores (physical CPU 0)
                                        for (cpu = procinfoproc.NBcpus / procinfoproc.NBcpusocket; cpu < procinfoproc.NBcpus; cpu++)
                                        {
                                            printw("|");
                                            if(procinfoproc.CPUids[cpu] == procinfoproc.pinfodisp[pindex].processorarray[spindex])
                                                attron(COLOR_PAIR(cpuColor));

                                            if(procinfoproc.pinfodisp[pindex].cpuOKarray[cpu] == 1)
                                                printw("%2d", procinfoproc.CPUids[cpu]);
                                            else
                                                printw("  ");

                                            if(procinfoproc.CPUids[cpu] == procinfoproc.pinfodisp[pindex].processorarray[spindex])
                                                attroff(COLOR_PAIR(cpuColor));
                                        }
                                        printw("| ");

                                        attron(COLOR_PAIR(cpuColor));
                                        printw("%5.1f %6.2f",
                                               procinfoproc.pinfodisp[pindex].subprocCPUloadarray[spindex],
                                               procinfoproc.pinfodisp[pindex].subprocCPUloadarray_timeaveraged[spindex]);
                                        attroff(COLOR_PAIR(cpuColor));
                                        #endif


										// Memory use
										#ifdef CMDPROC_MEMUSE
                                        int memColor = 0;

                                        int kBcnt, MBcnt, GBcnt;

                                        kBcnt = procinfoproc.pinfodisp[pindex].VmRSSarray[spindex];
                                        MBcnt = kBcnt/1024;
                                        kBcnt = kBcnt - MBcnt*1024;

                                        GBcnt = MBcnt/1024;
                                        MBcnt = MBcnt - GBcnt*1024;

                                        //if(pinfodisp[pindex].subprocMEMloadarray[spindex]>0.5)
                                        memColor = 1;
                                        if(procinfoproc.pinfodisp[pindex].VmRSSarray[spindex]>100*1024>10)        // 10 MB
                                            memColor = 2;
                                        if(procinfoproc.pinfodisp[pindex].VmRSSarray[spindex]>100*1024)       // 100 MB
                                            memColor = 3;
                                        if(procinfoproc.pinfodisp[pindex].VmRSSarray[spindex]>1024*1024)  // 1 GB
                                            memColor = 4;
                                        if(procinfoproc.pinfodisp[pindex].VmRSSarray[spindex]<1024)            // 1 MB
                                            memColor = 5;

                                        printw(" ");
                                        attron(COLOR_PAIR(memColor));
                                        if(GBcnt>0)
                                            printw("%3d GB ", GBcnt);
                                        else
                                            printw("       ");

                                        if(MBcnt>0)
                                            printw("%4d MB ", MBcnt);
                                        else
                                            printw("       ");

                                        if(kBcnt>0)
                                            printw("%4d kB ", kBcnt);
                                        else
                                            printw("       ");
                                        attroff(COLOR_PAIR(memColor));
                                        #endif

                                        if(pindex == pindexSelected)
                                            attroff(A_REVERSE);

                                        printw("\n");


                                    }
                                    if(procinfoproc.pinfodisp[pindex].NBsubprocesses == 0)
                                    {
										printw("  ERROR: procinfoproc.pinfodisp[pindex].NBsubprocesses = %d\n", (int) procinfoproc.pinfodisp[pindex].NBsubprocesses);

                                        if(pindex == pindexSelected)
                                            attroff(A_REVERSE);
										}
                                    
                                }


                            }
                            
                           


                            // ================ DISPLAY MODE 4 ==================
                            if( procinfoproc.DisplayMode == 4)
                            {

                                printw(" %d", procinfoproc.pinfoarray[pindex]->MeasureTiming);
                                if(procinfoproc.pinfoarray[pindex]->MeasureTiming == 1)
                                {
                                    long *dtiter_array;
                                    long *dtexec_array;
                                    int dtindex;


                                    printw(" %3d ..%02ld  ", procinfoproc.pinfoarray[pindex]->timerindex, procinfoproc.pinfoarray[pindex]->timingbuffercnt % 100);

                                    // compute timing stat
                                    dtiter_array = (long*) malloc(sizeof(long)*(PROCESSINFO_NBtimer-1));
                                    dtexec_array = (long*) malloc(sizeof(long)*(PROCESSINFO_NBtimer-1));

                                    int tindex;
                                    dtindex = 0;

                                    // we exclude the current timerindex, as timers may not all be written
                                    for(tindex=0; tindex<PROCESSINFO_NBtimer-1; tindex++)
                                    {
                                        int ti0, ti1;

                                        ti1 = procinfoproc.pinfoarray[pindex]->timerindex - tindex;
                                        ti0 = ti1 - 1;

                                        if(ti0<0)
                                            ti0 += PROCESSINFO_NBtimer;

                                        if(ti1<0)
                                            ti1 += PROCESSINFO_NBtimer;

                                        dtiter_array[tindex] = (procinfoproc.pinfoarray[pindex]->texecstart[ti1].tv_nsec - procinfoproc.pinfoarray[pindex]->texecstart[ti0].tv_nsec) + 1000000000*(procinfoproc.pinfoarray[pindex]->texecstart[ti1].tv_sec - procinfoproc.pinfoarray[pindex]->texecstart[ti0].tv_sec);

                                        dtexec_array[tindex] = (procinfoproc.pinfoarray[pindex]->texecend[ti0].tv_nsec - procinfoproc.pinfoarray[pindex]->texecstart[ti0].tv_nsec) + 1000000000*(procinfoproc.pinfoarray[pindex]->texecend[ti0].tv_sec - procinfoproc.pinfoarray[pindex]->texecstart[ti0].tv_sec);
                                    }



                                    quick_sort_long(dtiter_array, PROCESSINFO_NBtimer-1);
                                    quick_sort_long(dtexec_array, PROCESSINFO_NBtimer-1);

                                    int colorcode;

                                    if(procinfoproc.pinfoarray[pindex]->dtiter_limit_enable!=0)
                                    {
                                        if(procinfoproc.pinfoarray[pindex]->dtiter_limit_cnt==0)
                                            colorcode = COLOR_PAIR(2);
                                        else
                                            colorcode = COLOR_PAIR(4);
                                        attron(colorcode);
                                    }
                                    printw("ITERlim %d/%5ld/%4ld", procinfoproc.pinfoarray[pindex]->dtiter_limit_enable, (long) (0.001*procinfoproc.pinfoarray[pindex]->dtiter_limit_value), procinfoproc.pinfoarray[pindex]->dtiter_limit_cnt);
                                    if(procinfoproc.pinfoarray[pindex]->dtiter_limit_enable!=0)
                                        attroff(colorcode);

                                    printw("  ");

                                    if(procinfoproc.pinfoarray[pindex]->dtexec_limit_enable!=0)
                                    {
                                        if(procinfoproc.pinfoarray[pindex]->dtexec_limit_cnt==0)
                                            colorcode = COLOR_PAIR(2);
                                        else
                                            colorcode = COLOR_PAIR(4);
                                        attron(colorcode);
                                    }

                                    printw("EXEClim %d/%5ld/%4ld ", procinfoproc.pinfoarray[pindex]->dtexec_limit_enable, (long) (0.001*procinfoproc.pinfoarray[pindex]->dtexec_limit_value), procinfoproc.pinfoarray[pindex]->dtexec_limit_cnt);
                                    if(procinfoproc.pinfoarray[pindex]->dtexec_limit_enable!=0)
                                        attroff(colorcode);


                                    float tval;

                                    tval = 0.001*dtiter_array[(long) (0.5*PROCESSINFO_NBtimer)];
                                    procinfoproc.pinfoarray[pindex]->dtmedian_iter_ns = dtiter_array[(long) (0.5*PROCESSINFO_NBtimer)];
                                    if(tval > 9999.9)
                                        printw(" ITER    >10ms ");
                                    else
                                        printw(" ITER %6.1fus ", tval);

                                    tval = 0.001*dtiter_array[0];
                                    if(tval > 9999.9)
                                        printw("[   >10ms -");
                                    else
                                        printw("[%6.1fus -", tval);

                                    tval = 0.001*dtiter_array[PROCESSINFO_NBtimer-2];
                                    if(tval > 9999.9)
                                        printw("    >10ms ]");
                                    else
                                        printw(" %6.1fus ]", tval);


                                    tval = 0.001*dtexec_array[(long) (0.5*PROCESSINFO_NBtimer)];
                                    procinfoproc.pinfoarray[pindex]->dtmedian_exec_ns = dtexec_array[(long) (0.5*PROCESSINFO_NBtimer)];
                                    if(tval > 9999.9)
                                        printw(" EXEC    >10ms ");
                                    else
                                        printw(" EXEC %6.1fus ", tval);

                                    tval = 0.001*dtexec_array[0];
                                    if(tval > 9999.9)
                                        printw("[   >10ms -");
                                    else
                                        printw("[%6.1fus -", tval);

                                    tval = 0.001*dtexec_array[PROCESSINFO_NBtimer-2];
                                    if(tval > 9999.9)
                                        printw("    >10ms ]");
                                    else
                                        printw(" %6.1fus ]", tval);


                                    //	printw(" ITER %9.3fus [%9.3f - %9.3f] ", 0.001*dtiter_array[(long) (0.5*PROCESSINFO_NBtimer)], 0.001*dtiter_array[0], 0.001*dtiter_array[PROCESSINFO_NBtimer-2]);





                                    //	printw(" EXEC %9.3fus [%9.3f - %9.3f] ", 0.001*dtexec_array[(long) (0.5*PROCESSINFO_NBtimer)], 0.001*dtexec_array[0], 0.001*dtexec_array[PROCESSINFO_NBtimer-2]);


                                    printw("  busy = %6.2f %%", 100.0*dtexec_array[(long) (0.5*PROCESSINFO_NBtimer)] / ( dtiter_array[(long) (0.5*PROCESSINFO_NBtimer)]+1 ) );

                                    free(dtiter_array);
                                    free(dtexec_array);

                                }
                            }


                            if(pindex == pindexSelected)
                                attroff(A_REVERSE);
                        }

                    }

                    if(procinfoproc.DisplayMode == 2)
                        printw("\n");
                    if(procinfoproc.DisplayMode == 4)
                        printw("\n");

                }
            }
            clock_gettime(CLOCK_REALTIME, &t06loop);


            clock_gettime(CLOCK_REALTIME, &t07loop);

            cnt++;



            clock_gettime(CLOCK_REALTIME, &t2loop);

            tdiff = info_time_diff(t1loop, t2loop);
            double tdiffvloop = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;

            printw("\nLoop time = %9.8f s  ( max rate = %7.2f Hz)\n", tdiffvloop, 1.0/tdiffvloop);

            refresh();
        }



#ifndef STANDALONE
        if( (data.signal_TERM == 1) || (data.signal_INT == 1) || (data.signal_ABRT == 1) || (data.signal_BUS == 1) || (data.signal_SEGV == 1) || (data.signal_HUP == 1) || (data.signal_PIPE == 1))
            loopOK = 0;
#endif

    }
    endwin();


	// Why did we exit ?
	
	printf("loopOK = 0 -> exit\n");
	
	
#ifndef STANDALONE
    if ( Xexit == 1 ) // normal exit
        printf("User typed x -> exiting\n");
    else if (data.signal_TERM == 1 )
		printf("Received signal TERM\n");
    else if (data.signal_INT == 1 )
		printf("Received signal INT\n");
    else if (data.signal_ABRT == 1 )
		printf("Received signal ABRT\n");
    else if (data.signal_BUS == 1 )
		printf("Received signal BUS\n");
    else if (data.signal_SEGV == 1 )
		printf("Received signal SEGV\n");
    else if (data.signal_HUP == 1 )
		printf("Received signal HUP\n");
    else if (data.signal_PIPE == 1 )
		printf("Received signal PIPE\n");
#endif
		
		
		
    // cleanup
    for(pindex=0; pindex<procinfoproc.NBpinfodisp; pindex++)
    {
        if(procinfoproc.pinfommapped[pindex] == 1)
        {
            struct stat file_stat;

            fstat(procinfoproc.fdarray[pindex], &file_stat);
            munmap(procinfoproc.pinfoarray[pindex], file_stat.st_size);
            procinfoproc.pinfommapped[pindex] == 0;
            close(procinfoproc.fdarray[pindex]);
        }

    }


    procinfoproc.loop = 0;
    pthread_join(threadscan, NULL);

    free(procinfoproc.pinfodisp);

    return 0;
}
