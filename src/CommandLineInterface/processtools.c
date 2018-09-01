
/**
 * @file processtools.c
 * @brief Tools to manage processes
 * 
 * Manages structure PROCESSINFO
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

#include <unistd.h>    // getpid()
#include <sys/types.h>

#include <sys/stat.h>

#include <ncurses.h>
#include <fcntl.h> 


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
    processinfo->loopstat = 3; // clean exit

    struct timespec tstop;
    struct tm *tstoptm;
    char msgstring[200];

    clock_gettime(CLOCK_REALTIME, &tstop);
    tstoptm = gmtime(&tstop.tv_sec);

    if(processinfo->CTRLval == 3) // loop exit from processinfo control
        sprintf(msgstring, "CTRLexit  %02d:%02d:%02d.%03d", tstoptm->tm_hour, tstoptm->tm_min, tstoptm->tm_sec, (int) (0.000001*(tstop.tv_nsec)));
    else
        sprintf(msgstring, "Loop exit %02d:%02d:%02d.%03d", tstoptm->tm_hour, tstoptm->tm_min, tstoptm->tm_sec, (int) (0.000001*(tstop.tv_nsec)));

    strncpy(processinfo->statusmsg, msgstring, 200);


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

    start_color();
    init_pair(1, COLOR_BLACK, COLOR_WHITE);
    init_pair(2, COLOR_BLACK, COLOR_RED);
    init_pair(3, COLOR_BLACK, COLOR_GREEN);
    init_pair(4, COLOR_BLACK, COLOR_YELLOW);

    init_pair(5, COLOR_GREEN, COLOR_BLACK);
    init_pair(6, COLOR_YELLOW, COLOR_BLACK);
    init_pair(7, COLOR_RED, COLOR_BLACK);
    init_pair(8, COLOR_BLACK, COLOR_RED);

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



    // INITIALIZE ncurses
    initncurses();


    int NBpinfodisp = wrow-5;
    pinfodisp = (PROCESSINFODISP*) malloc(sizeof(PROCESSINFODISP)*NBpinfodisp);
    for(pindex=0; pindex<NBpinfodisp; pindex++)
        pinfodisp[pindex].updatecnt = 0;

    int loopOK = 1;
    int freeze = 0;
    long cnt = 0;
    int MonMode = 0;
    int TimeSorted = 1;  // by default, sort processes by start time (most recent at top)
    int dispindexMax = 0;


    pindexActiveSelected = 0;

	int DisplayMode = 0; 
	// display modes:
	// 0: overview
	// 1: CPU affinity

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
        
        
        case KEY_F(0):
			DisplayMode = 0;
			break;
            
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
            printw("%d processes tracked    Display Mode %d\n", NBpindexActive, DisplayMode);
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
                    
                    
                    
                    // COLLECT INFO FROM SYSTEM
                    FILE *fp;
                    char fname[200];
                    
                    // cpuset
                    sprintf(fname, "/proc/%d/task/%d/cpuset", pinfodisp[pindex].PID, pinfodisp[pindex].PID);
                    fp=fopen(fname, "r");
                    fscanf(fp, "%s", &pinfodisp[pindex].cpuset);
                    fclose(fp);
                    
                    
                    
                    pinfodisp[pindex].updatecnt ++;

                }
            }




            // compute time-sorted list
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





            // DISPLAY


            int dispindex;
            //            for(dispindex=0; dispindex<NBpinfodisp; dispindex++)

            if(TimeSorted == 0)
                dispindexMax = wrow-4;
            else
                dispindexMax = NBpindexActive;

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
                        attron(COLOR_PAIR(3));
                        printw("  ACTIVE");
                        attroff(COLOR_PAIR(3));
                    }

                    if(pinfolist->active[pindex] == 2)  // not active: crashed or terminated
                    {
                        if(pinfoarray[pindex]->loopstat == 3) // clean exit
                        {
                            attron(COLOR_PAIR(4));
                            printw(" STOPPED");
                            attroff(COLOR_PAIR(4));
                        }
                        else
                        {
                            attron(COLOR_PAIR(2));
                            printw(" CRASHED");
                            attroff(COLOR_PAIR(2));
                        }
                    }



                    //				printw("%5ld %d", pindex, pinfolist->active[pindex]);
                    if(pinfolist->active[pindex] != 0)
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

                        printw("  %6d", pinfolist->PIDarray[pindex]);
                        printw(" %16s", pinfoarray[pindex]->tmuxname);

                        attron(A_BOLD);
                        printw("  %40s", pinfodisp[pindex].name);
                        attroff(A_BOLD);

                        if(pinfoarray[pindex]->loopcnt==loopcntarray[pindex])
                        {   // loopcnt has not changed
                            printw("  %10ld", pinfoarray[pindex]->loopcnt-loopcntoffsetarray[pindex]);
                        }
                        else
                        {   // loopcnt has changed
                            attron(COLOR_PAIR(3));
                            printw("  %10ld", pinfoarray[pindex]->loopcnt-loopcntoffsetarray[pindex]);
                            attroff(COLOR_PAIR(3));
                        }

                        loopcntarray[pindex] = pinfoarray[pindex]->loopcnt;

                        if(pinfoarray[pindex]->loopstat == 4) // ERROR
                            attron(COLOR_PAIR(2));
                        printw("  %40s", pinfoarray[pindex]->statusmsg);
                        if(pinfoarray[pindex]->loopstat == 4) // ERROR
                            attroff(COLOR_PAIR(2));
                    }
                    printw("\n");

                    if(pindex == pindexSelected)
                        attroff(A_REVERSE);

                    /*      if(pinfolist->active[pindex] != 0)
                          {
                              pindexActive[NBpindexActive] = pindex;
                              NBpindexActive++;
                          }*/
                }
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
