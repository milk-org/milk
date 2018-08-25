
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
	
	int           createtime_hr;
	int           createtime_min;
	int           createtime_sec;
	long          createtime_ns;
	
	char          statusmsg[200];
	char          tmuxname[80];

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






//
// Create processinfo in shared memory
//
PROCESSINFO* processinfo_shm_create(char *pname)
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

	printf("created processinfe entry at %s\n", SM_fname);
    printf("shared memory space = %ld bytes\n", sharedsize); //TEST

	clock_gettime(CLOCK_REALTIME, &pinfo->createtime);
	strcpy(pinfo->name, pname);

	pinfolist->active[pindex] = 1;

    return pinfo;
}






//
// Remove processinfo in shared memory
//
int processinfo_shm_rm(char *pname)
{
	
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





int_fast8_t processinfo_CTRLscreen()
{
    long pindex;

    PROCESSINFO *pinfoarray[PROCESSINFOLISTSIZE];
    int pinfommapped[PROCESSINFOLISTSIZE];             // 1 if mmapped, 0 otherwise

    pid_t PIDarray[PROCESSINFOLISTSIZE];  // used to track changes
    int updatearray[PROCESSINFOLISTSIZE];   // 0: don't load, 1: (re)load
    int fdarray[PROCESSINFOLISTSIZE];     // file descriptors
	long loopcntarray[PROCESSINFOLISTSIZE];

    // Display fields
    PROCESSINFODISP *pinfodisp;


    for(pindex=0; pindex<PROCESSINFOLISTSIZE; pindex++)
    {
        updatearray[pindex] = 1; // initialize: load all
        pinfommapped[pindex] = 0;
    }


    float frequ = 20.0; // Hz
    char monstring[200];

    // list of active indices
    int pindexActiveSelected;
    int pindexSelected;
    int pindexActive[PROCESSINFOLISTSIZE];
    int NBpindexActive;

    // INITIALIZE ncurses

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
    init_pair(4, COLOR_GREEN, COLOR_BLACK);
    init_pair(5, COLOR_YELLOW, COLOR_BLACK);
    init_pair(6, COLOR_RED, COLOR_BLACK);
    init_pair(7, COLOR_BLACK, COLOR_RED);

    int NBpinfodisp = wrow-2;
    pinfodisp = (PROCESSINFODISP*) malloc(sizeof(PROCESSINFODISP)*NBpinfodisp);
    for(pindex=0; pindex<NBpinfodisp; pindex++)
        pinfodisp[pindex].updatecnt = 0;

    int loopOK = 1;
    int freeze = 0;
    long cnt = 0;
    int MonMode = 0;
    pindexActiveSelected = 0;

    // Create / read process list
    processinfo_shm_list_create();


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

        switch (ch)
        {
        case 'f':
            if(freeze==0)
                freeze = 1;
            else
                freeze = 0;
            break;

        case 'x':
            loopOK=0;
            break;

        case KEY_UP:
            pindexActiveSelected --;
            if(pindexActiveSelected<0)
                pindexActiveSelected = 0;
            pindexSelected = pindexActive[pindexActiveSelected];
            break;

        case KEY_DOWN:
            pindexActiveSelected ++;
            if(pindexActiveSelected>NBpindexActive-1)
                pindexActiveSelected = NBpindexActive-1;
            pindexSelected = pindexActive[pindexActiveSelected];
            break;

        case 'T':
            pid = pinfolist->PIDarray[pindexSelected];
            kill(pid, SIGTERM);
            break;

        case 'K':
            pid = pinfolist->PIDarray[pindexSelected];
            kill(pid, SIGKILL);
            break;

        case 'I':
            pid = pinfolist->PIDarray[pindexSelected];
            kill(pid, SIGINT);
            break;

            break;
        }


        if(freeze==0)
        {
            clear();

            printw("E(x)it   SIG(T)ERM  SIG(K)ILL  SIG(I)NT\n");
            printw("\n");
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
                        // process doesn't exist -> flag as crashed
                        pinfolist->active[pindex] = 2;

                        //						updatearray[pindex] = 0;
                        //						PIDarray[pindex] = 0;
                    }
                }





                if(updatearray[pindex] == 1)
                {
                    // (RE)LOAD
                    struct stat file_stat;

                    fdarray[pindex] = open(SM_fname, O_RDWR);
                    fstat(fdarray[pindex], &file_stat);

                    // if already mmapped, first unmap
                    if(pinfommapped[pindex] == 1)
                    {
                        munmap(pinfoarray[pindex], file_stat.st_size);
                        close(fdarray[pindex]);
                        pinfommapped[pindex] == 0;
                    }


                    pinfoarray[pindex] = (PROCESSINFO*) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fdarray[pindex], 0);
                    if (pinfoarray[pindex] == MAP_FAILED) {
                        close(fdarray[pindex]);
                        endwin();
                        fprintf(stderr, "Error mmapping file %s\n", SM_fname);
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

                    pinfodisp[pindex].updatecnt ++;

                }
            }

            NBpindexActive = 0;
            for(pindex=0; pindex<NBpinfodisp; pindex++)
            {
                if(pindex == pindexSelected)
                    attron(A_REVERSE);

                printw("%5ld %3ld  ", pindex, pinfodisp[pindex].updatecnt);



                if(pinfolist->active[pindex] == 1)
                {
                    attron(COLOR_PAIR(3));
                    printw("  ACTIVE");
                    attroff(COLOR_PAIR(3));
                }

                if(pinfolist->active[pindex] == 2)
                {
                    attron(COLOR_PAIR(2));
                    printw(" CRASHED");
                    attroff(COLOR_PAIR(2));
                }



                //				printw("%5ld %d", pindex, pinfolist->active[pindex]);
                if(pinfolist->active[pindex] != 0)
                {

                    printw(" %02d:%02d:%02d.%09ld",
                           pinfodisp[pindex].createtime_hr,
                           pinfodisp[pindex].createtime_min,
                           pinfodisp[pindex].createtime_sec,
                           pinfodisp[pindex].createtime_ns);

                    printw("  %6d", pinfolist->PIDarray[pindex]);

                    attron(A_BOLD);
                    printw("  %40s", pinfodisp[pindex].name);
                    attroff(A_BOLD);
                    
                    if(pinfoarray[pindex]->loopcnt==loopcntarray[pindex])
                    { // loopcnt has not changed
						printw("  %8ld", pinfoarray[pindex]->loopcnt);
					}
					else
					{ // loopcnt has changed
						attron(COLOR_PAIR(3));
						printw("  %8ld", pinfoarray[pindex]->loopcnt);
						attroff(COLOR_PAIR(3));
					}
                    
                    loopcntarray[pindex] = pinfoarray[pindex]->loopcnt;
                    
                    
                    printw("  %40s", pinfoarray[pindex]->statusmsg);
                }
                printw("\n");

                if(pindex == pindexSelected)
                    attroff(A_REVERSE);

                if(pinfolist->active[pindex] != 0)
                {
                    pindexActive[NBpindexActive] = pindex;
                    NBpindexActive++;
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
            //            char SM_fname[200];
            struct stat file_stat;

            //            sprintf(SM_fname, "%s/proc.%06d.shm", SHAREDMEMDIR, (int) pinfolist->PIDarray[pindex]);
            //            SM_fd = open(SM_fname, O_RDWR);
            fstat(fdarray[pindex], &file_stat);

            munmap(pinfoarray[pindex], file_stat.st_size);
            pinfommapped[pindex] == 0;
            close(fdarray[pindex]);
        }

    }

    free(pinfodisp);

    return 0;
}
