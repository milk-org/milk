
/**
 * @file processtools.c
 * @brief Tools to manage processes
 * 
 * Manages structure PROCESSINFO
 * 
 * 
 * 
 * Use Template
 * 
 * 
 * At beginning of function:
 * 
 * 
  PROCESSINFO *processinfo;
    if(data.processinfo==1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        sprintf(pinfoname, "process %s to %s", IDinname, IDoutname);
        processinfo = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = 0; // loop initialization

        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE,     __FILE__);
        processinfo->source_LINE = __LINE__;

        char msgstring[200];
        sprintf(msgstring, "%s->%s", IDinname, IDoutname);
        processinfo_WriteMessage(processinfo, msgstring);
    }
 
 // CATCH SIGNALS
 	
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
 
 // pre-loop testing, anything that would prevent loop from starting should issue message

   int loopOK = 1;
 
 if(.... error condition ....)
 {
   sprintf(msgstring, "ERROR: no WFS reference");
        if(data.processinfo == 1)
        {
			processinfo->loopstat = 4; // ERROR
			processinfo_WriteMessage(processinfo, msgstring);
		}	
	loopOK = 0;
	}
 
  
  
 // At loop code
  
  
      if(data.processinfo==1)
        processinfo->loopstat = 1;  // Notify processinfo that we are entering loop
    
   
    long loopcnt = 0;
     
    while(loopOK==1)
    {
      if(data.processinfo==1)
        {
            while(processinfo->CTRLval == 1)  // pause
                usleep(50);

            if(processinfo->CTRLval == 2) // single iteration
                processinfo->CTRLval = 1;

            if(processinfo->CTRLval == 3) // exit loop
            {
                loopOK = 0;
            }
        }
    
    
    // LOOP CODE GOES HERE
    
    
	// OPTIONAL MESSAGE WHILE LOOP RUNNING
	if(data.processinfo==1)
        {
            char msgstring[200];
            sprintf(msgstring, "%d save threads", NBthreads);
            processinfo_WriteMessage(processinfo, msgstring);
        }

     
     
     // process signals

		if(data.signal_TERM == 1){
			loopOK = 0;
			if(data.processinfo==1)
				processinfo_SIGexit(processinfo, SIGTERM);
		}
     
		if(data.signal_INT == 1){
			loopOK = 0;
			if(data.processinfo==1)
				processinfo_SIGexit(processinfo, SIGINT);
		}

		if(data.signal_ABRT == 1){
			loopOK = 0;
			if(data.processinfo==1)
				processinfo_SIGexit(processinfo, SIGABRT);
		}

		if(data.signal_BUS == 1){
			loopOK = 0;
			if(data.processinfo==1)
				processinfo_SIGexit(processinfo, SIGBUS);
		}
		
		if(data.signal_SEGV == 1){
			loopOK = 0;
			if(data.processinfo==1)
				processinfo_SIGexit(processinfo, SIGSEGV);
		}
		
		if(data.signal_HUP == 1){
			loopOK = 0;
			if(data.processinfo==1)
				processinfo_SIGexit(processinfo, SIGHUP);
		}
		
		if(data.signal_PIPE == 1){
			loopOK = 0;
			if(data.processinfo==1)
				processinfo_SIGexit(processinfo, SIGPIPE);
		}	
     
        loopcnt++;
        if(data.processinfo==1)
            processinfo->loopcnt = loopcnt;
    
	}    // end of loop

    if(data.processinfo==1)
        processinfo_cleanExit(processinfo);


 * 
 * 
 * 
 * @author Olivier Guyon
 * @date 24 Aug 2018
 */




#define _GNU_SOURCE


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

#include <00CORE/00CORE.h>
#include <CommandLineInterface/CLIcore.h>
#include "COREMOD_tools/COREMOD_tools.h"



/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

typedef struct
{
	int           active;
	pid_t         PID;
	char          name[40];
	long          updatecnt;

	long          loopcnt;
	int           loopstat;
	
	int           createtime_hr;
	int           createtime_min;
	int           createtime_sec;
	long          createtime_ns;
	
	char          cpuset[16];       /**< cpuset name  */
	char          cpusallowed[20];
	int           threads; 
	long          ctxtsw_voluntary;
	long          ctxtsw_nonvoluntary;

	long          ctxtsw_voluntary_prev[50];
	long          ctxtsw_nonvoluntary_prev[50];
	
	int           processor;
	int           rt_priority;
	
	// sub-processes
	int           NBsubprocesses;
	int           subprocPIDarray[50];
	float         subprocCPUloadarray[50];
	float         subprocMEMloadarray[50];
	
	char          statusmsg[200];
	char          tmuxname[100];
	
} PROCESSINFODISP;




/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */



static PROCESSINFOLIST *pinfolist;

static int wrow, wcol;

static float CPUload[100];
static long long CPUcnt0[100];
static long long CPUcnt1[100];
static long long CPUcnt2[100];
static long long CPUcnt3[100];
static long long CPUcnt4[100];
static long long CPUcnt5[100];
static long long CPUcnt6[100];
static long long CPUcnt7[100];
static long long CPUcnt8[100];

static float CPUpcnt[100];




#define NBtopMax 1000

static int   toparray_PID[NBtopMax];
static char  toparray_USER[32][NBtopMax];
static int   toparray_PR[NBtopMax];
static int   toparray_NI[NBtopMax];
static char  toparray_VIRT[32][NBtopMax];
static char  toparray_RES[32][NBtopMax];
static char  toparray_SHR[32][NBtopMax];
static char  toparray_S[32][NBtopMax];
static float toparray_CPU[NBtopMax];
static float toparray_MEM[NBtopMax];
static char  toparray_TIME[32][NBtopMax];
static char  toparray_COMMAND[32][NBtopMax];









/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */


// 
// if list does not exist, create it and return index = 0
// if list exists, return first available index
//

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

PROCESSINFO* processinfo_shm_create(char *pname, int CTRLval)
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

		case SIGCONT :  // Continue if stopped
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






int processinfo_WriteMessage(PROCESSINFO *processinfo, char* msgstring)
{
	strcpy(processinfo->statusmsg, msgstring);
	
	// TODO: add to logfile
	
	return 0;
}




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






static int GetNumberCPUs()
{
	FILE *fpout;	
	char outstring[16];
	int NBcpus;

	
	fpout = popen ("getconf _NPROCESSORS_ONLN", "r");
	if(fpout==NULL)
	{
		printf("WARNING: cannot run command \"tmuxsessionname\"\n");
	}
	else
	{
		if(fgets(outstring, 100, fpout)== NULL)
			printf("WARNING: fgets error\n");
		pclose(fpout);
	}
	
	NBcpus = atoi(outstring);

	return(NBcpus);
}






static long getTopOutput()
{
	long NBtop = 0;

    char outstring[200];
    char command[200];
    FILE * fpout;
	int ret;
	

    sprintf(command, "top -b -n 1");
    fpout = popen (command, "r");
    if(fpout==NULL)
    {
        printf("WARNING: cannot run command \"%s\"\n", command);
    }
    else
    {
		int startScan = 0;
        while( (fgets(outstring, 100, fpout) != NULL) && (NBtop<NBtopMax) )
           {
			   if(startScan == 1)
			   { 
				   // PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
					// 32412 scexao   -91   0  0.611t 4.063g 3.616g S  80.4  0.8  20:16.25 aol0run

				   ret = sscanf(outstring, "%d %s %d %d %s %s %s %s %f %f %s %s\n",
						&toparray_PID[NBtop],
						toparray_USER[NBtop],
						&toparray_PR[NBtop],
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
						
				// TEST
				printf("%5ld  < %s >  %d\n", NBtop, outstring, ret);
				printf("           process %5d : %4.1f\n", toparray_PID[NBtop], toparray_CPU[NBtop]);
				printf("\n");
						
				   NBtop++;
			   }
			   
				if(strstr(outstring, "USER")!=NULL)
					startScan = 1;
		   }
        pclose(fpout);
    }
    

	return NBtop;
}




static int GetCPUloads()
{
    int NBcpus;
    char * line = NULL;
    FILE *fp;
    ssize_t read;
    size_t len = 0;
    int cpu;
    long long vall0, vall1, vall2, vall3, vall4, vall5, vall6, vall7, vall8;
    long long v0, v1, v2, v3, v4, v5, v6, v7, v8;
    char string0[80];


    NBcpus = GetNumberCPUs();

    fp = fopen("/proc/stat", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    cpu = 0;
    if(getline(&line, &len, fp) == -1)
    {
        printf("[%s][%d]  ERROR: cannot read file\n", __FILE__, __LINE__);
        exit(0);
    }

    while (((read = getline(&line, &len, fp)) != -1)&&(cpu<NBcpus)) {

        sscanf(line, "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld", string0, &vall0, &vall1, &vall2, &vall3, &vall4, &vall5, &vall6, &vall7, &vall8);

        v0 = vall0 - CPUcnt0[cpu];
        v1 = vall1 - CPUcnt1[cpu];
        v2 = vall2 - CPUcnt2[cpu];
        v3 = vall3 - CPUcnt3[cpu];
        v4 = vall4 - CPUcnt4[cpu];
        v5 = vall5 - CPUcnt5[cpu];
        v6 = vall6 - CPUcnt6[cpu];
        v7 = vall7 - CPUcnt7[cpu];
        v8 = vall8 - CPUcnt8[cpu];

        CPUcnt0[cpu] = vall0;
        CPUcnt1[cpu] = vall1;
        CPUcnt2[cpu] = vall2;
        CPUcnt3[cpu] = vall3;
        CPUcnt4[cpu] = vall4;
        CPUcnt5[cpu] = vall5;
        CPUcnt6[cpu] = vall6;
        CPUcnt7[cpu] = vall7;
        CPUcnt8[cpu] = vall8;

        CPUload[cpu] = (1.0*v0+v1+v2+v4+v5+v6)/(v0+v1+v2+v3+v4+v5+v6+v7+v8);
        cpu++;
    }

    fclose(fp);


    // number of process per CPU
    for(cpu=0; cpu<NBcpus; cpu++)
    {
        char outstring[200];
        char command[200];
        FILE * fpout;


        sprintf(command, "CORENUM=%d; ps -e -o pid,psr,cpu,cmd | grep -E  \"^[[:space:]][[:digit:]]+[[:space:]]+${CORENUM}\"|wc -l", cpu);
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
            CPUpcnt[cpu] = atoi(outstring);
        }
	}


    return(cpu);
}







// for Display Mode 2

static int PIDcollectSystemInfo(int PID, int pindex, PROCESSINFODISP *pinfodisp, int level)
{

    // COLLECT INFO FROM SYSTEM
    FILE *fp;
    char fname[200];



    // cpuset

    sprintf(fname, "/proc/%d/task/%d/cpuset", PID, PID);

    fp=fopen(fname, "r");
    if (fp == NULL)
        return -1;

    fscanf(fp, "%s", pinfodisp[pindex].cpuset);
    fclose(fp);




    // read /proc/PID/status

    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char string0[200];
    char string1[200];

    sprintf(fname, "/proc/%d/status", PID);
    fp = fopen(fname, "r");
    if (fp == NULL)
        return -1;

    while ((read = getline(&line, &len, fp)) != -1) {

        if(strncmp(line, "Cpus_allowed_list:", strlen("Cpus_allowed_list:")) == 0)
        {
            sscanf(line, "%s %s", string0, string1);
            strcpy(pinfodisp[pindex].cpusallowed, string1);
        }

        if(strncmp(line, "Threads:", strlen("Threads:")) == 0)
        {
            sscanf(line, "%s %s", string0, string1);
            pinfodisp[pindex].threads = atoi(string1);
        }

        if(strncmp(line, "voluntary_ctxt_switches:", strlen("voluntary_ctxt_switches:")) == 0)
        {
            sscanf(line, "%s %s", string0, string1);
            pinfodisp[pindex].ctxtsw_voluntary = atoi(string1);
        }

        if(strncmp(line, "nonvoluntary_ctxt_switches:", strlen("nonvoluntary_ctxt_switches:")) == 0)
        {
            sscanf(line, "%s %s", string0, string1);
            pinfodisp[pindex].ctxtsw_nonvoluntary = atoi(string1);
        }

    }

    fclose(fp);
    if (line)
        free(line);





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
        if(Nfields != 52)
               {
					printERROR(__FILE__,__func__,__LINE__, "fscanf returns value != 1");
					pinfodisp[pindex].processor = stat_processor;
					pinfodisp[pindex].rt_priority = stat_rt_priority; 
				}
		else
		{
			fclose(fp);
			pinfodisp[pindex].processor = stat_processor;
			pinfodisp[pindex].rt_priority = stat_rt_priority;
		}

/* For testing 
FILE * fpouttest;
					
					sprintf(fname, "out.%d.txt", PID);
					fpouttest = fopen(fname, "w");
					fprintf(fpouttest, "%d\n", Nfields);
					fprintf(fpouttest, "%d\n", pindex);
					fprintf(fpouttest, "%d\n", stat_processor);
					fprintf(fpouttest, "%d\n", stat_rt_priority);
					fprintf(fpouttest, "%d %s %c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu %ld %ld %ld %ld %ld %ld %llu %lu %ld %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %d %d %u %u %llu %lu %ld %lu %lu %lu %lu %lu %lu %lu %ld\n",
                stat_pid,
                stat_comm,
                stat_state,
                stat_ppid,
                stat_pgrp,
                stat_session,
                stat_tty_nr,
                stat_tpgid,
                stat_flags,
                stat_minflt,
                stat_cminflt,
                stat_majflt,
                stat_cmajflt,
                stat_utime,
                stat_stime,
                stat_cutime,
                stat_cstime,
                stat_priority,
                stat_nice,
                stat_num_threads,
                stat_itrealvalue,
                stat_starttime,
                stat_vsize,
                stat_rss,
                stat_rsslim,
                stat_startcode,
                stat_endcode,
                stat_startstack,
                stat_kstkesp,
                stat_kstkeip,
                stat_signal,
                stat_blocked,
                stat_sigignore,
                stat_sigcatch,
                stat_wchan,
                stat_nswap,
                stat_cnswap,
                stat_exit_signal,
                stat_processor,
                stat_rt_priority,
                stat_policy,
                stat_delayacct_blkio_ticks,
                stat_guest_time,
                stat_cguest_time,
                stat_start_data,
                stat_end_data,
                stat_start_brk,
                stat_arg_start,
                stat_arg_end,
                stat_env_start,
                stat_env_end,
                stat_exit_code
               );
					fclose(fpouttest);

*/



    if(level == 0)
    {
        FILE *fpout;
        char command[200];
        char outstring[200];

        pinfodisp[pindex].subprocPIDarray[0] = PID;
        pinfodisp[pindex].NBsubprocesses = 1;

        if(pinfodisp[pindex].threads > 1) // look for children
        {
            char outstringc[200];

            sprintf(command, "pstree -p %d", PID);

            fpout = popen (command, "r");
            if(fpout==NULL)
            {
                printf("WARNING: cannot run command \"%s\"\n", command);
            }
            else
            {
                while(fgets(outstring, 100, fpout) != NULL)
                {
                    int i = 0;
                    int ic = 0;
                    for(i=0; i<strlen(outstring); i++)
                    {
                        if(isdigit(outstring[i]))
                        {
                            outstringc[ic] = outstring[i];
                            ic++;
                        }
                        if(outstring[i] == '(')
                            ic = 0;
                    }
                    outstringc[ic] = '\0';

                    pinfodisp[pindex].subprocPIDarray[pinfodisp[pindex].NBsubprocesses] = atoi(outstringc);
                    pinfodisp[pindex].NBsubprocesses++;
                }
                pclose(fpout);
            }
        }

        // get CPU and MEM load
        // ps -T -o lwp,%cpu,%mem  -p PID

//		sprintf(command, "ps -T -o %%cpu,%%mem  -p %d > test.%d.txt", PID, PID);
	//	system(command);
/*		
        sprintf(command, "ps -T -o %%cpu,%%mem  -p %d", PID);
        
        fpout = popen (command, "r");
        if(fpout==NULL)
        {
            printf("WARNING: cannot run command \"t%s\"\n", command);
        }
        else
        {
            int lcnt = 0;
            while((fgets(outstring, 100, fpout) != NULL)&&(lcnt<pinfodisp[pindex].NBsubprocesses+1))
            {
                if(lcnt>0)
                {
                    sscanf(outstring, "%f %f\n",
                          &pinfodisp[pindex].subprocCPUloadarray[lcnt-1],
                          &pinfodisp[pindex].subprocMEMloadarray[lcnt-1]
                         );
					//pinfodisp[pindex].subprocCPUloadarray[lcnt-1] = 1.0;
				}
				lcnt++;

            }

            pclose(fpout);
        }
        */ 


    }



    return 0;

}







/**
 * Control screen for PROCESSINFO structures
 *
 * Relies on ncurses for display
 *
 */

int_fast8_t processinfo_CTRLscreen()
{
    long pindex, index;

    // these arrays are indexed together
    // the index is different from the displayed order
    // new process takes first available free index
    PROCESSINFO *pinfoarray[PROCESSINFOLISTSIZE];
    int          pinfommapped[PROCESSINFOLISTSIZE];             // 1 if mmapped, 0 otherwise
    pid_t        PIDarray[PROCESSINFOLISTSIZE];  // used to track changes
    int          updatearray[PROCESSINFOLISTSIZE];   // 0: don't load, 1: (re)load
    int          fdarray[PROCESSINFOLISTSIZE];     // file descriptors
    long         loopcntarray[PROCESSINFOLISTSIZE];
    long         loopcntoffsetarray[PROCESSINFOLISTSIZE];
    int          selectedarray[PROCESSINFOLISTSIZE];

    int sorted_pindex_time[PROCESSINFOLISTSIZE];

    // Display fields
    PROCESSINFODISP *pinfodisp;

    char syscommand[200];


    int NBcpus = 0;


	char pselected_FILE[200];
	char pselected_FUNCTION[200];
	int pselected_LINE;

	
	printf("scanning %ld processes\n", getTopOutput());
	
	exit(0);

    for(pindex=0; pindex<PROCESSINFOLISTSIZE; pindex++)
    {
        updatearray[pindex]   = 1; // initialize: load all
        pinfommapped[pindex]  = 0;
        selectedarray[pindex] = 0; // initially not selected
        loopcntoffsetarray[pindex] = 0;
    }


    float frequ = 10.0; // Hz
    char monstring[200];

    // list of active indices
    int pindexActiveSelected;
    int pindexSelected;
    int pindexActive[PROCESSINFOLISTSIZE];
    int NBpindexActive;




    // Create / read process list
    processinfo_shm_list_create();

    NBcpus = GetNumberCPUs();
    GetCPUloads();

    // INITIALIZE ncurses
    initncurses();


    int NBpinfodisp = wrow-5;
    pinfodisp = (PROCESSINFODISP*) malloc(sizeof(PROCESSINFODISP)*NBpinfodisp);
    for(pindex=0; pindex<NBpinfodisp; pindex++)
    {
        pinfodisp[pindex].updatecnt = 0;
        pinfodisp[pindex].NBsubprocesses = 0;
    }

    // Get number of cpus on system
    // getconf _NPROCESSORS_ONLN



    int loopOK = 1;
    int freeze = 0;
    long cnt = 0;
    int MonMode = 0;
    int TimeSorted = 1;  // by default, sort processes by start time (most recent at top)
    int dispindexMax = 0;


    pindexActiveSelected = 0;

    int DisplayMode = 1;
    // display modes:
    // 1: overview
    // 2: CPU affinity

    while( loopOK == 1 )
    {
        int pid;


        usleep((long) (1000000.0/frequ));
        int ch = getch();


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
            loopOK=0;
            break;

        case ' ':     // Mark current PID as selected (if none selected, other commands only apply to highlighted process)
            pindex = pindexSelected;
            if(selectedarray[pindex] == 1)
                selectedarray[pindex] = 0;
            else
                selectedarray[pindex] = 1;
            break;

        case 'u':    // undelect all
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                selectedarray[pindex] = 0;
            }
            break;


        case KEY_UP:
            pindexActiveSelected --;
            if(pindexActiveSelected<0)
                pindexActiveSelected = 0;
            if(TimeSorted == 0)
                pindexSelected = pindexActive[pindexActiveSelected];
            else
                pindexSelected = sorted_pindex_time[pindexActiveSelected];
            break;

        case KEY_DOWN:
            pindexActiveSelected ++;
            if(pindexActiveSelected>NBpindexActive-1)
                pindexActiveSelected = NBpindexActive-1;
            if(TimeSorted == 0)
                pindexSelected = pindexActive[pindexActiveSelected];
            else
                pindexSelected = sorted_pindex_time[pindexActiveSelected];
            break;

        case 'T':
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
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
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
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
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
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
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
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
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(pinfolist->active[pindex]!=1)
                {
                    char SM_fname[200];
                    sprintf(SM_fname, "%s/proc.%06d.shm", SHAREDMEMDIR, (int) pinfolist->PIDarray[pindex]);
                    remove(SM_fname);
                }
            }
            break;

        // loop controls
        case 'p': // pause
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    if(pinfoarray[pindex]->CTRLval == 0)
                        pinfoarray[pindex]->CTRLval = 1;
                    else
                        pinfoarray[pindex]->CTRLval = 0;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                if(pinfoarray[pindex]->CTRLval == 0)
                    pinfoarray[pindex]->CTRLval = 1;
                else
                    pinfoarray[pindex]->CTRLval = 0;
            }
            break;

        case 's': // step
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    pinfoarray[pindex]->CTRLval = 2;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                pinfoarray[pindex]->CTRLval = 2;
            }
            break;

        case 'e': // exit
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    pinfoarray[pindex]->CTRLval = 3;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                pinfoarray[pindex]->CTRLval = 3;
            }
            break;

        case 'z': // apply current value as offset (zero loop counter)
            selectedOK = 0;
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    loopcntoffsetarray[pindex] = pinfoarray[pindex]->loopcnt;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                loopcntoffsetarray[pindex] = pinfoarray[pindex]->loopcnt;
            }
            break;

        case 'Z': // revert to original counter value
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(selectedarray[pindex] == 1)
                {
                    selectedOK = 1;
                    loopcntoffsetarray[pindex] = 0;
                }
            }
            if(selectedOK == 0)
            {
                pindex = pindexSelected;
                loopcntoffsetarray[pindex] = 0;
            }
            break;

        case 't':
            endwin();
            sprintf(syscommand, "tmux a -t %s", pinfoarray[pindexSelected]->tmuxname);
            system(syscommand);
            initncurses();
            break;

        case 'a':
            pindex = pindexSelected;
            if(pinfolist->active[pindex]==1)
            {
                endwin();
                sprintf(syscommand, "watch -n 0.1 cat /proc/%d/status", (int) pinfolist->PIDarray[pindex]);
                system(syscommand);
                initncurses();
            }
            break;

        case 'd':
            pindex = pindexSelected;
            if(pinfolist->active[pindex]==1)
            {
                endwin();
                sprintf(syscommand, "watch -n 0.1 cat /proc/%d/sched", (int) pinfolist->PIDarray[pindex]);
                system(syscommand);
                initncurses();
            }
            break;


        case 'o':
            if(TimeSorted == 1)
                TimeSorted = 0;
            else
                TimeSorted = 1;
            break;

        // Set Display Mode

        case KEY_F(1):
            DisplayMode = 1;
            break;

        case KEY_F(2):
            DisplayMode = 2;
            break;


        }


        if(freeze==0)
        {
            clear();

            printw("E(x)it (f)reeze *** SIG(T)ERM SIG(K)ILL SIG(I)NT *** (r)emove (R)emoveall *** (t)mux\n");
            printw("time-s(o)rted    st(a)tus sche(d) *** Loop Controls: (p)ause (s)tep (e)xit *** (z)ero or un(Z)ero counter\n");
            printw("(SPACE):select toggle   (u)nselect all\n");
            printw("%2d cpus   %2d processes tracked    Display Mode %d\n", NBcpus, NBpindexActive, DisplayMode);
            printw("\n");



			sprintf(pselected_FILE, "?");
			sprintf(pselected_FUNCTION, "?");
			pselected_LINE = 0;

			 if(pinfommapped[pindexSelected] == 1)
			 {
				
				strcpy(pselected_FILE, pinfoarray[pindexSelected]->source_FILE);
				strcpy(pselected_FUNCTION, pinfoarray[pindexSelected]->source_FUNCTION);
				pselected_LINE = pinfoarray[pindexSelected]->source_LINE;
				
				printw("Source Code: %s line %d (function %s)\n", pselected_FILE,  pselected_LINE, pselected_FUNCTION);
			}
			else
				printw("\n");
				
			printw("\n");


            // LOAD / UPDATE process information

            for(pindex=0; pindex<NBpinfodisp; pindex++)
            {
                // SHOULD WE (RE)LOAD ?
                if(pinfolist->active[pindex] == 0) // inactive
                    updatearray[pindex] = 0;

                if((pinfolist->active[pindex] == 1)||(pinfolist->active[pindex] == 2)) // active or crashed
                {
                    if(pinfolist->PIDarray[pindex] == PIDarray[pindex] ) // don't reload if PID same as before
                        updatearray[pindex] = 0;
                    else
                    {
                        updatearray[pindex] = 1;
                        PIDarray[pindex] = pinfolist->PIDarray[pindex];
                    }
                }
                //    if(pinfolist->active[pindex] == 2) // mmap crashed, file may still be present
                //        updatearray[pindex] = 1;

                if(pinfolist->active[pindex] == 3) // file has gone away
                    updatearray[pindex] = 0;


                char SM_fname[200];


                // check if process info file exists

                struct stat file_stat;
                sprintf(SM_fname, "%s/proc.%06d.shm", SHAREDMEMDIR, (int) pinfolist->PIDarray[pindex]);

                // Does file exist ?
                if(stat(SM_fname, &file_stat) == -1 && errno == ENOENT)
                {
                    // if not, don't (re)load and remove from process info list
                    pinfolist->active[pindex] = 0;
                    updatearray[pindex] = 0;
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





                if((updatearray[pindex] == 1)&&(pindex<NBpinfodisp))
                {
                    // (RE)LOAD
                    struct stat file_stat;

                    // if already mmapped, first unmap
                    if(pinfommapped[pindex] == 1)
                    {
                        fstat(fdarray[pindex], &file_stat);
                        munmap(pinfoarray[pindex], file_stat.st_size);
                        close(fdarray[pindex]);
                        pinfommapped[pindex] == 0;
                    }


                    // COLLECT INFORMATION FROM PROCESSINFO FILE

                    fdarray[pindex] = open(SM_fname, O_RDWR);
                    fstat(fdarray[pindex], &file_stat);
                    pinfoarray[pindex] = (PROCESSINFO*) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fdarray[pindex], 0);
                    if (pinfoarray[pindex] == MAP_FAILED) {
                        close(fdarray[pindex]);
                        endwin();
                        fprintf(stderr, "[%d] Error mmapping file %s\n", __LINE__, SM_fname);
                        pinfolist->active[pindex] = 3;
                    }
                    else
                    {
                        pinfommapped[pindex] = 1;
                        strncpy(pinfodisp[pindex].name, pinfoarray[pindex]->name, 40-1);

                        struct tm *createtm;
                        createtm      = gmtime(&pinfoarray[pindex]->createtime.tv_sec);
                        pinfodisp[pindex].createtime_hr = createtm->tm_hour;
                        pinfodisp[pindex].createtime_min = createtm->tm_min;
                        pinfodisp[pindex].createtime_sec = createtm->tm_sec;
                        pinfodisp[pindex].createtime_ns = pinfoarray[pindex]->createtime.tv_nsec;

                        pinfodisp[pindex].loopcnt = pinfoarray[pindex]->loopcnt;
                    }

                    pinfodisp[pindex].active = pinfolist->active[pindex];
                    pinfodisp[pindex].PID = pinfolist->PIDarray[pindex];
                    pinfodisp[pindex].NBsubprocesses = 0;

                    pinfodisp[pindex].updatecnt ++;

                }
            }



            /** ### Build a time-sorted list of processes
             *
             *
             *
             */
            NBpindexActive = 0;
            for(pindex=0; pindex<PROCESSINFOLISTSIZE; pindex++)
                if(pinfolist->active[pindex] != 0)
                {
                    pindexActive[NBpindexActive] = pindex;
                    NBpindexActive++;
                }
            double *timearray;
            long *indexarray;
            timearray  = (double*) malloc(sizeof(double)*NBpindexActive);
            indexarray = (long*)   malloc(sizeof(long)  *NBpindexActive);
            int listcnt = 0;
            for(index=0; index<NBpindexActive; index++)
            {
                pindex = pindexActive[index];
                if(pinfommapped[pindex] == 1)
                {
                    indexarray[index] = pindex;
                    // minus sign for most recent first
                    //printw("index  %ld  ->  pindex  %ld\n", index, pindex);
                    timearray[index] = -1.0*pinfoarray[pindex]->createtime.tv_sec - 1.0e-9*pinfoarray[pindex]->createtime.tv_nsec;
                    listcnt++;
                }
            }
            NBpindexActive = listcnt;
            quick_sort2l_double(timearray, indexarray, NBpindexActive);

            for(index=0; index<NBpindexActive; index++)
                sorted_pindex_time[index] = indexarray[index];

            free(timearray);
            free(indexarray);




            /** ### Display
             *
             *
             *
             */



            int dispindex;
            //            for(dispindex=0; dispindex<NBpinfodisp; dispindex++)

            if(TimeSorted == 0)
                dispindexMax = wrow-4;
            else
                dispindexMax = NBpindexActive;

            if(DisplayMode == 2)
            {
				NBcpus = GetCPUloads();
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
                printw("                                                                           %2d CPUs  ", NBcpus);
                for(cpu=0; cpu<NBcpus; cpu+=2)
                    printw("|%02d", cpu);                
                printw("|    |");
                for(cpu=1; cpu<NBcpus; cpu+=2)
                    printw("%02d|", cpu);
                printw("\n");
                
				// List CPU # processes
                printw("                                                                         PROCESSES  ", NBcpus);
                for(cpu=0; cpu<NBcpus; cpu+=2)
                {
                    int vint = CPUpcnt[cpu];
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
                printw("|    |");
                for(cpu=1; cpu<NBcpus; cpu+=2)
                {
                    int vint = CPUpcnt[cpu];
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

                    if(ColorCode != 0)
                        attron(COLOR_PAIR(ColorCode));
                    printw("%02d", vint);
                    if(ColorCode != 0)
                        attroff(COLOR_PAIR(ColorCode));
                    printw("|");
                }

                printw("\n");
                
                
                
                
                
                // Print CPU LOAD
                printw("                                                                          CPU LOAD  ", NBcpus);
                for(cpu=0; cpu<NBcpus; cpu+=2)
                {
                    int vint = (int) (100.0*CPUload[cpu]);
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
                printw("|    |");
                for(cpu=1; cpu<NBcpus; cpu+=2)
                {
                    int vint = (int) (100.0*CPUload[cpu]);
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

                    if(ColorCode != 0)
                        attron(COLOR_PAIR(ColorCode));
                    printw("%02d", vint);
                    if(ColorCode != 0)
                        attroff(COLOR_PAIR(ColorCode));
                    printw("|");
                }

                printw("\n");
                printw("\n");
            }


            for(dispindex=0; dispindex<dispindexMax; dispindex++)
            {
                if(TimeSorted == 0)
                    pindex = dispindex;
                else
                    pindex = sorted_pindex_time[dispindex];

                if(pindex<NBpinfodisp)
                {

                    if(pindex == pindexSelected)
                        attron(A_REVERSE);

                    // printw("%d  [%d]  %5ld %3ld  ", dispindex, sorted_pindex_time[dispindex], pindex, pinfodisp[pindex].updatecnt);

                    if(selectedarray[pindex]==1)
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
                        if(pinfoarray[pindex]->loopstat == 3) // clean exit
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











                    //				printw("%5ld %d", pindex, pinfolist->active[pindex]);
                    if(pinfolist->active[pindex] != 0)
                    {
                        if(pindex == pindexSelected)
                            attron(A_REVERSE);

                        printw("  %6d", pinfolist->PIDarray[pindex]);

                        attron(A_BOLD);
                        printw("  %-30s", pinfodisp[pindex].name);
                        attroff(A_BOLD);

                        if( DisplayMode == 1)
                        {
                            switch (pinfoarray[pindex]->loopstat)
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

                            printw(" C%d", pinfoarray[pindex]->CTRLval );

                            printw(" %02d:%02d:%02d.%03d",
                                   pinfodisp[pindex].createtime_hr,
                                   pinfodisp[pindex].createtime_min,
                                   pinfodisp[pindex].createtime_sec,
                                   (int) (0.000001*(pinfodisp[pindex].createtime_ns)));

                            printw(" %26s", pinfoarray[pindex]->tmuxname);


                            if(pinfoarray[pindex]->loopcnt==loopcntarray[pindex])
                            {   // loopcnt has not changed
                                printw("  %10ld", pinfoarray[pindex]->loopcnt-loopcntoffsetarray[pindex]);
                            }
                            else
                            {   // loopcnt has changed
                                attron(COLOR_PAIR(2));
                                printw("  %10ld", pinfoarray[pindex]->loopcnt-loopcntoffsetarray[pindex]);
                                attroff(COLOR_PAIR(2));
                            }

                            loopcntarray[pindex] = pinfoarray[pindex]->loopcnt;

                            if(pinfoarray[pindex]->loopstat == 4) // ERROR
                                attron(COLOR_PAIR(4));
                            printw("  %40s", pinfoarray[pindex]->statusmsg);
                            if(pinfoarray[pindex]->loopstat == 4) // ERROR
                                attroff(COLOR_PAIR(4));
                        }




                        if( DisplayMode == 2)
                        {
                            int cpu;
                            char cpuliststring[200];
                            char cpustring[16];
                            int  psysinfostatus;



                            // collect required info for display
                            psysinfostatus = PIDcollectSystemInfo(pinfodisp[pindex].PID, pindex, pinfodisp, 0);
                            
                            if(psysinfostatus == -1)
                            {
								printw(" no process info available\n");
							}
							else
                            {
                                int spindex; // sub process index
                                for(spindex = 0; spindex < pinfodisp[pindex].NBsubprocesses; spindex++)
                                {

                                    if(spindex>0)
                                    {
                                        printw("               |---%6d                        ", pinfodisp[pindex].subprocPIDarray[spindex]);
                                        PIDcollectSystemInfo(pinfodisp[pindex].subprocPIDarray[spindex], pindex, pinfodisp, 1);
                                    }

                                    printw(" %2d", pinfodisp[pindex].rt_priority);
                                    printw(" %-10s ", pinfodisp[pindex].cpuset);
                                    printw(" %2dx ", pinfodisp[pindex].threads);




									// Context Switches
									
                                    if(pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex] != pinfodisp[pindex].ctxtsw_nonvoluntary)
                                        attron(COLOR_PAIR(4));
                                    else if(pinfodisp[pindex].ctxtsw_voluntary_prev[spindex] != pinfodisp[pindex].ctxtsw_voluntary)
                                        attron(COLOR_PAIR(3));


                                    printw("ctxsw: +%02ld +%02ld",
                                           abs(pinfodisp[pindex].ctxtsw_voluntary    - pinfodisp[pindex].ctxtsw_voluntary_prev[spindex])%100,
                                           abs(pinfodisp[pindex].ctxtsw_nonvoluntary - pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex])%100
                                          );

                                    if(pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex] != pinfodisp[pindex].ctxtsw_nonvoluntary)
                                        attroff(COLOR_PAIR(4));
                                    else if(pinfodisp[pindex].ctxtsw_voluntary_prev[spindex] != pinfodisp[pindex].ctxtsw_voluntary)
                                        attroff(COLOR_PAIR(3));

                                    pinfodisp[pindex].ctxtsw_voluntary_prev[spindex] = pinfodisp[pindex].ctxtsw_voluntary;
                                    pinfodisp[pindex].ctxtsw_nonvoluntary_prev[spindex] = pinfodisp[pindex].ctxtsw_nonvoluntary;





                                    printw(" ");


									// CPU use
									
									int cpuColor = 0;
																		
				//					if(pinfodisp[pindex].subprocCPUloadarray[spindex]>5.0)
									cpuColor = 1;
									if(pinfodisp[pindex].subprocCPUloadarray[spindex]>10.0)
										cpuColor = 2;
									if(pinfodisp[pindex].subprocCPUloadarray[spindex]>20.0)
										cpuColor = 3;
									if(pinfodisp[pindex].subprocCPUloadarray[spindex]>40.0)
										cpuColor = 4;
									if(pinfodisp[pindex].subprocCPUloadarray[spindex]<1.0)
										cpuColor = 5;
									
                                    sprintf(cpuliststring, ",%s,", pinfodisp[pindex].cpusallowed);
                                    
                                    
                                    

                                    // First group of cores (physical CPU 0)
                                    for(cpu=0; cpu<NBcpus; cpu += 2)
                                    {
                                        int cpuOK = 0;
                                        int cpumin, cpumax;

                                        sprintf(cpustring, ",%d,",cpu);
                                        if(strstr(cpuliststring, cpustring) != NULL)
                                            cpuOK = 1;
                                        
                                        
                                        for(cpumin=0;cpumin<=cpu;cpumin++)
											for(cpumax=cpu;cpumax<NBcpus;cpumax++)
											{
												sprintf(cpustring, ",%d-%d,", cpumin, cpumax);
												if(strstr(cpuliststring, cpustring) != NULL)
													cpuOK = 1;
											}
										

										printw("|");
                                        if(cpu == pinfodisp[pindex].processor)
                                            attron(COLOR_PAIR(cpuColor));

                                        if(cpuOK == 1)
                                            printw("%2d", cpu);
                                        else
                                            printw("  ");

                                        if(cpu == pinfodisp[pindex].processor)
                                            attroff(COLOR_PAIR(cpuColor));

                                    }
                                    printw("|    ");


                                    // Second group of cores (physical CPU 0)
                                    for(cpu=1; cpu<NBcpus; cpu += 2)
                                    {
                                        int cpuOK = 0;
                                        int cpumin, cpumax;
                                        
                                        sprintf(cpustring, ",%d,",cpu);
                                        if(strstr(cpuliststring, cpustring) != NULL)
                                            cpuOK = 1;
                                         
                                        for(cpumin=0;cpumin<=cpu;cpumin++)
											for(cpumax=cpu;cpumax<NBcpus;cpumax++)
											{
												sprintf(cpustring, ",%d-%d,", cpumin, cpumax);
												if(strstr(cpuliststring, cpustring) != NULL)
													cpuOK = 1;
											}


										printw("|");
                                        if(cpu == pinfodisp[pindex].processor)
                                            attron(COLOR_PAIR(cpuColor));

                                        if(cpuOK == 1)
                                            printw("%2d", cpu);
                                        else
                                            printw("  ");

                                        if(cpu == pinfodisp[pindex].processor)
                                            attroff(COLOR_PAIR(cpuColor));

                                    }
                                    printw("| ");
                                    
                                    
                                    
                                    attron(COLOR_PAIR(cpuColor));
                                    printw("%4.1f", 
										pinfodisp[pindex].subprocCPUloadarray[spindex]);
                                    attroff(COLOR_PAIR(cpuColor));
                                    
                                    int memColor = 0;
									
									//if(pinfodisp[pindex].subprocMEMloadarray[spindex]>0.5)
									memColor = 1;
									if(pinfodisp[pindex].subprocMEMloadarray[spindex]>1.0)
										memColor = 2;
									if(pinfodisp[pindex].subprocMEMloadarray[spindex]>2.0)
										memColor = 3;
									if(pinfodisp[pindex].subprocMEMloadarray[spindex]>4.0)
										memColor = 4;
									if(pinfodisp[pindex].subprocMEMloadarray[spindex]<0.1)
										memColor = 5;
										
									printw(" ");
									attron(COLOR_PAIR(memColor));
                                    printw("%4.1f", 
										pinfodisp[pindex].subprocMEMloadarray[spindex]);
									attroff(COLOR_PAIR(memColor));



                                    printw("\n");

                                    if(pindex == pindexSelected)
                                        attroff(A_REVERSE);
                                }
                            }
                        
                        
                        }

                        if(pindex == pindexSelected)
                            attroff(A_REVERSE);
						
                    }

                }
                
                if(DisplayMode == 1)
					printw("\n");


            }

            refresh();

            cnt++;

        }

    }
    endwin();


    // cleanup
    for(pindex=0; pindex<NBpinfodisp; pindex++)
    {
        if(pinfommapped[pindex] == 1)
        {
            struct stat file_stat;

            fstat(fdarray[pindex], &file_stat);
            munmap(pinfoarray[pindex], file_stat.st_size);
            pinfommapped[pindex] == 0;
            close(fdarray[pindex]);
        }

    }

    free(pinfodisp);

    return 0;
}
