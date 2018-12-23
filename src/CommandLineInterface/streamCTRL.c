
/**
 * @file streamCTRL.c
 * @brief Data streams control panel
 * 
 * Manages data streams
 * 
 * 
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
#include <sys/types.h>


#include <ncurses.h>
#include <fcntl.h> 
#include <ctype.h>

#include <dirent.h>

#include <wchar.h>
#include <locale.h>
#include <errno.h>


#include <00CORE/00CORE.h>
#include <CommandLineInterface/CLIcore.h>
#include "COREMOD_tools/COREMOD_tools.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "info/info.h"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

#define streamNBID_MAX 10000
#define streamOpenNBpid_MAX 50
#define nameNBchar 30




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

	nonl();


    init_color(COLOR_GREEN, 700, 1000, 700);
    init_color(COLOR_YELLOW, 1000, 1000, 700);

    start_color();



    //  colored background
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











int get_process_name_by_pid(const int pid, char *pname)
{
    char* fname = (char*) calloc(1024, sizeof(char));

    sprintf(fname, "/proc/%d/cmdline",pid);
    FILE* fp = fopen(fname,"r");
    if(fp) {
        size_t size;
        size = fread(pname, sizeof(char), 1024, fp);
        if(size>0) {
            if('\n'==pname[size-1])
                pname[size-1]='\0';
        }
        fclose(fp);
    }
    
    free(fname);

    return 0;
}









int streamCTRL_CatchSignals()
{

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

    return 0;
}





static int get_PIDmax()
{
    FILE *fp;
    int PIDmax;

    fp = fopen("/proc/sys/kernel/pid_max", "r");
    fscanf(fp, "%d", &PIDmax);
    fclose(fp);

    return PIDmax;
}








/**
 * ## Purpose
 *
 * Control screen for stream structures
 *
 * ## Description
 *
 * Relies on ncurses for display\n
 *
 *
 */

int_fast8_t streamCTRL_CTRLscreen()
{
    int PIDnameStringLen = 12;

    long sindex;  // scan index
    long sindexscan; // for fuser scan
    long IDscan;
    long dindex;  // display index
    long ssindex[streamNBID_MAX]; // sorted index array

    long index;

    float frequ = 16.0; // Hz
    char  monstring[200];

    long IDmax = streamNBID_MAX;

    int sOK;

    int SORTING = 0;
    int SORT_TOGGLE = 0;

    // timing
    struct timespec t0;
    struct timespec t1;
    double tdiffv;
    struct timespec tdiff;


    // data arrays
    char sname_array[streamNBID_MAX][nameNBchar];
    char linkname_array[streamNBID_MAX][nameNBchar];

    long IDarray[streamNBID_MAX];
    int SymLink_array[streamNBID_MAX];

    pid_t streamOpenPIDarray[streamNBID_MAX][streamOpenNBpid_MAX];
    int streamOpenPIDarray_cnt[streamNBID_MAX];
    int streamOpenPIDarray_cnt1[streamNBID_MAX];                       // number of processes accessing stream
    int streamOpenPIDarray_status[streamNBID_MAX];

    int atype_array[streamNBID_MAX];
    long long cnt0_array[streamNBID_MAX];

    double updatevaluearray[streamNBID_MAX]; // higher value = more actively recent updates [Hz]
    double updatevaluearray_frozen[streamNBID_MAX];

    long long cnt0array[streamNBID_MAX]; // used to check if cnt0 has changed
    long deltacnt0[streamNBID_MAX];



	// display
	int DispName_NBchar = 36;
	int DispSize_NBchar = 20;



    // create PID name table
    char **PIDname_array;
    int PIDmax;

    PIDmax = get_PIDmax();
    PIDname_array = malloc(sizeof(char*)*PIDmax);




	char *homedir = getenv("HOME");
    
    setlocale(LC_ALL, "");


    streamCTRL_CatchSignals();

    // INITIALIZE ncurses
    initncurses();

    int NBsinfodisp = wrow-6;
    int NBsindex = 0;
    int loopOK = 1;
    long cnt = 0;


    int dindexSelected = 0;

    int DisplayMode = 1;
    // display modes:
    // 1: overview

    int fuserUpdate0 = 1; //update on first instance
    int fuserUpdate = 1;
    struct tm *uttime_lastScan;
    time_t rawtime;
    int fuserScan = 0;

    clear();
    clock_gettime(CLOCK_REALTIME, &t0);



    DIR *d;
    struct dirent *dir;

            

    while( loopOK == 1 )
    {
        int pid;
        char command[200];
        
        //if(fuserUpdate != 1) // don't wait if ongoing fuser scan
		
		usleep((long) (1000000.0/frequ));
        int ch = getch();


        // timing measurement
        clock_gettime(CLOCK_REALTIME, &t1);
        tdiff = info_time_diff(t0, t1);
        tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
        clock_gettime(CLOCK_REALTIME, &t0);



        int selectedOK = 0; // goes to 1 if at least one process is selected
        switch (ch)
        {
        case 'x':     // Exit control screen
            loopOK=0;
            break;

        case KEY_UP:
            dindexSelected --;
            if(dindexSelected<0)
                dindexSelected = 0;
            break;

        case KEY_DOWN:
            dindexSelected ++;
            if(dindexSelected > NBsindex-1)
                dindexSelected = NBsindex-1;
            break;

        case KEY_PPAGE:
            dindexSelected -= 10;
            if(dindexSelected<0)
                dindexSelected = 0;
            break;

        case KEY_NPAGE:
            dindexSelected += 10;
            if(dindexSelected > NBsindex-1)
                dindexSelected = NBsindex-1;
            break;


        // Set Display Mode

        case 'h': // help
            DisplayMode = 1;
            break;

        case KEY_F(2): // semvals
            DisplayMode = 2;
            break;

        case KEY_F(3): // write PIDs
            DisplayMode = 3;
            break;

        case KEY_F(4): // read PIDs
            DisplayMode = 4;
            break;

        case KEY_F(5): // read PIDs
            if((DisplayMode == 5)||(fuserUpdate0==1))
            {
                fuserUpdate = 1;
                time(&rawtime);
                uttime_lastScan = gmtime(&rawtime);
                fuserScan = 1;
                sindexscan = 0;
            }

            DisplayMode = 5;
            //erase();
            //printw("SCANNING PROCESSES AND FILESYSTEM: PLEASE WAIT ...\n");
            //refresh();
            break;


        case 'R': // remove stream
            ImageStreamIO_destroyIm( &data.image[IDarray[dindexSelected]]);
            break;


        case '1': // sorting by stream name
            SORTING = 1;
            break;

        case '2': // sorting by update freq
            SORTING = 2;
            SORT_TOGGLE = 1;
            break;

        case '3': // sort by number of processes accessing
            SORTING = 3;
            SORT_TOGGLE = 1;
            break;


        case '+': // faster update
            frequ *= 2.0;
            if(frequ < 1.0)
                frequ = 1.0;
            if(frequ > 64.0)
                frequ = 64.0;
            break;

        case '-': // slower update
            frequ *= 0.5;
            if(frequ < 1.0)
                frequ = 1.0;
            if(frequ > 64.0)
                frequ = 64.0;
            break;

        }

        erase();

        attron(A_BOLD);
        sprintf(monstring, "PRESS x TO STOP MONITOR");
        print_header(monstring, '-');
        attroff(A_BOLD);


        if(DisplayMode == 1) // help
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
            printw("   Display semaphore values\n");

            attron(attrval);
            printw("    F3");
            attroff(attrval);
            printw("   Display semaphore write PIDs\n");

            attron(attrval);
            printw("    F4");
            attroff(attrval);
            printw("   Display semaphore read PIDs\n");

            attron(attrval);
            printw("    F5");
            attroff(attrval);
            printw("   stream open by processes ...\n");

            printw("\n");
            printw("============ ACTIONS \n");

            attron(attrval);
            printw("    R");
            attroff(attrval);
            printw("    Remove stream\n");

            printw("\n");
            printw("============ DISPLAY \n");

            attron(attrval);
            printw("    +");
            attroff(attrval);
            printw("    Increase update frequency\n");

            attron(attrval);
            printw("    -");
            attroff(attrval);
            printw("    Decrease update frequency\n");

            attron(attrval);
            printw("    1");
            attroff(attrval);
            printw("    Sort by stream name (alphabetical)\n");

            attron(attrval);
            printw("    2");
            attroff(attrval);
            printw("    Sort by recently updated\n");

            attron(attrval);
            printw("    3");
            attroff(attrval);
            printw("    Sort by processes access\n");

            printw("\n\n");
        }
        else
        {

            if(DisplayMode==1)
            {
                attron(A_REVERSE);
                printw("[h] Help");
                attroff(A_REVERSE);
            }
            else
                printw("[h] Help");
            printw("   ");

            if(DisplayMode==2)
            {
                attron(A_REVERSE);
                printw("[F2] sem values");
                attroff(A_REVERSE);
            }
            else
                printw("[F2] sem values");
            printw("   ");

            if(DisplayMode==3)
            {
                attron(A_REVERSE);
                printw("[F3] write PIDs");
                attroff(A_REVERSE);
            }
            else
                printw("[F3] write PIDs");
            printw("   ");

            if(DisplayMode==4)
            {
                attron(A_REVERSE);
                printw("[F4] read PIDs");
                attroff(A_REVERSE);
            }
            else
                printw("[F4] read PIDs");
            printw("   ");

            if(DisplayMode==5)
            {
                attron(A_REVERSE);
                printw("[F5] processes access");
                attroff(A_REVERSE);
            }
            else
                printw("[F5] processes access");
            printw("   ");
            printw("\n");


            printw("PIDmax = %d    Update frequ = %d Hz\n", PIDmax, (int) (frequ+0.5));
            if(DisplayMode==5)
            {
                if(fuserScan==1)
                    printw("Last scan on  %02d:%02d:%02d  - Press F5 again to re-scan    C-c to stop scan\n", uttime_lastScan->tm_hour, uttime_lastScan->tm_min,  uttime_lastScan->tm_sec);
                else
                    printw("Last scan on  XX:XX:XX  - Press F5 again to scan             C-c to stop scan\n");
            }
            else
                printw("\n");

            printw("\n");








			


            // COLLECT DATA
            d = opendir("/tmp/");
            if(d)
            {
                sindex = 0;
                sOK = 1;
                while((sOK == 1)&&((dir = readdir(d)) != NULL))
                {
                    char *pch = strstr(dir->d_name, ".im.shm");
				

                    if(pch)
                    {
                        long ID;

                        // is file sym link ?
                        struct stat buf;
                        int retv;
                        char fullname[200];
                        
                        sprintf(fullname, "/tmp/%s", dir->d_name);
                        retv = lstat (fullname, &buf);
                        if (retv == -1 ) {
                            endwin();
                            printf("File \"%s\"", dir->d_name);
                            perror("Error running lstat on file ");
                            exit(0);
                        }



                        // get stream name and ID
                        strncpy(sname_array[sindex], dir->d_name, strlen(dir->d_name)-strlen(".im.shm"));
                        sname_array[sindex][strlen(dir->d_name)-strlen(".im.shm")] = '\0';
                        ID = image_ID(sname_array[sindex]);


                        // connect to stream
                        ID = image_ID(sname_array[sindex]);
                        if(ID == -1)
                        {
                            ID = read_sharedmem_image(sname_array[sindex]);
                            deltacnt0[ID] = 1;
                            updatevaluearray[ID] = 1.0;
                            updatevaluearray_frozen[ID] = 1.0;
                        }
                        else
                        {
                            float gainv = 0.9;
                            deltacnt0[ID] = data.image[ID].md[0].cnt0 - cnt0array[ID];
                            updatevaluearray[ID] = gainv * updatevaluearray[ID] + (1.0-gainv) * (1.0*deltacnt0[ID]/tdiffv);
                        }
                        cnt0array[ID] = data.image[ID].md[0].cnt0; // keep memory of cnt0

                        IDarray[sindex] = ID;

                        atype_array[sindex] = data.image[ID].md[0].atype;

                        if (S_ISLNK(buf.st_mode)) // resolve link name
                        {
							char fullname[200];
                            char linknamefull[200];
                            char linkname[200];
                            int nchar;

                            SymLink_array[ID] = 1;
                            sprintf(fullname, "/tmp/%s", dir->d_name);
                            readlink (fullname, linknamefull, 200-1);

                            strcpy(linkname, basename(linknamefull));

                            int lOK = 1;
                            int ii = 0;
                            while((lOK == 1)&&(ii<strlen(linkname)))
                            {
                                if(linkname[ii] == '.')
                                {
                                    linkname[ii] = '\0';
                                    lOK = 0;
                                }
                                ii++;
                            }

                            strncpy(linkname_array[sindex], linkname, nameNBchar);
                        }
                        else
                            SymLink_array[ID] = 0;

                        sindex++;
                    }                    
                }
                NBsindex = sindex;
            }
            closedir(d);



            // SORT

            // default : no sorting
            for(dindex=0; dindex<NBsindex; dindex++)
                ssindex[dindex] = dindex;

            if(SORTING == 1) // alphabetical sorting
            {
                long *larray;
                larray = (long*) malloc(sizeof(long)*NBsindex);
                for(sindex=0; sindex<NBsindex; sindex++)
                    larray[sindex] = sindex;

                int sindex0, sindex1;
                for(sindex0=0; sindex0<NBsindex-1; sindex0++)
                {
                    for(sindex1=sindex0+1; sindex1<NBsindex; sindex1++)
                    {
                        if( strcmp(sname_array[larray[sindex0]], sname_array[larray[sindex1]]) > 0)
                        {
                            int tmpindex = larray[sindex0];
                            larray[sindex0] = larray[sindex1];
                            larray[sindex1] = tmpindex;
                        }
                    }
                }

                for(dindex=0; dindex<NBsindex; dindex++)
                    ssindex[dindex] = larray[dindex];
                free(larray);
            }


            if((SORTING == 2)||(SORTING == 3)) // recent update and process access
            {
                long *larray;
                double *varray;
                larray = (long*) malloc(sizeof(long)*NBsindex);
                varray = (double*) malloc(sizeof(double)*NBsindex);

                if(SORT_TOGGLE == 1)
                {
                    for(sindex=0; sindex<NBsindex; sindex++)
                        updatevaluearray_frozen[IDarray[sindex]] = updatevaluearray[IDarray[sindex]];

                    if(SORTING==3)
                    {
                        for(sindex=0; sindex<NBsindex; sindex++)
                            updatevaluearray_frozen[IDarray[sindex]] += 10000.0*streamOpenPIDarray_cnt1[IDarray[sindex]];
                    }

                    SORT_TOGGLE = 0;
                }

                for(sindex=0; sindex<NBsindex; sindex++)
                {
                    larray[sindex] = sindex;
                    varray[sindex] = updatevaluearray_frozen[IDarray[sindex]];
                }

                quick_sort2l(varray, larray, NBsindex);

                for(dindex=0; dindex<NBsindex; dindex++)
                    ssindex[NBsindex-dindex-1] = larray[dindex];

                free(larray);
                free(varray);
            }







            // DISPLAY

            sOK = 1;
            for(dindex=0; dindex < NBsindex; dindex++)
            {
                long ID;
                sindex = ssindex[dindex];
                ID = IDarray[sindex];


                if(sOK == 1)
                {
					char line[200];
					char string[200];
					int charcnt = 0; // how many chars are about to be printed
					int linecharcnt = 1; // keeping track of number of characters in line
					
					
					printw("%4d ", sindex);
					
					charcnt = DispName_NBchar+1;
                    if(dindex == dindexSelected)
                        attron(A_REVERSE);

                    if(SymLink_array[ID] == 1)
                    {
                        char namestring[200];
                        sprintf(namestring, "%s->%s", sname_array[sindex], linkname_array[sindex]);

                        attron(COLOR_PAIR(5));
                        printw("%-*.*s", DispName_NBchar, DispName_NBchar, namestring);
                        attroff(COLOR_PAIR(5));
                    }
                    else
                        printw("%-*.*s", DispName_NBchar, DispName_NBchar, sname_array[sindex]);
                    
                    if(strlen(sname_array[sindex]) > DispName_NBchar)
                    {
						attron(COLOR_PAIR(9));
						printw("+");
						attroff(COLOR_PAIR(9));
					}
					else
						printw(" ");        
                    linecharcnt += charcnt;







                    if(DisplayMode < 5)
                    {
                        char str[200];
                        char str1[200];
                        int j;
                        
                        
                        if(atype_array[sindex]==_DATATYPE_UINT8)
                            charcnt = sprintf(string, " UI8");
                        if(atype_array[sindex]==_DATATYPE_INT8)
                            charcnt = sprintf(string, "  I8");

                        if(atype_array[sindex]==_DATATYPE_UINT16)
                            charcnt = sprintf(string, "UI16");
                        if(atype_array[sindex]==_DATATYPE_INT16)
                            charcnt = sprintf(string, " I16");

                        if(atype_array[sindex]==_DATATYPE_UINT32)
                            charcnt = sprintf(string, "UI32");
                        if(atype_array[sindex]==_DATATYPE_INT32)
                            charcnt = sprintf(string, " I32");

                        if(atype_array[sindex]==_DATATYPE_UINT64)
                            charcnt = sprintf(string, "UI64");
                        if(atype_array[sindex]==_DATATYPE_INT64)
                            charcnt = sprintf(string, " I64");

                        if(atype_array[sindex]==_DATATYPE_FLOAT)
                            charcnt = sprintf(string, " FLT");

                        if(atype_array[sindex]==_DATATYPE_DOUBLE)
                            charcnt = sprintf(string, " DBL");

                        if(atype_array[sindex]==_DATATYPE_COMPLEX_FLOAT)
                            charcnt = sprintf(string, "CFLT");

                        if(atype_array[sindex]==_DATATYPE_COMPLEX_DOUBLE)
                            charcnt = sprintf(string, "CDBL");
                        
						linecharcnt += charcnt;
						if(linecharcnt < wcol)
							printw(string);


                        sprintf(str, " [%3ld", (long) data.image[ID].md[0].size[0]);

                        for(j=1; j<data.image[ID].md[0].naxis; j++)
                        {
                            sprintf(str1, "%sx%3ld", str, (long) data.image[ID].md[0].size[j]);
                            strcpy(str, str1);
                        }
                        sprintf(str1, "%s]", str);
                        strcpy(str, str1);
                        
                        
                        
                        charcnt = sprintf(string, "%-*.*s ", DispSize_NBchar, DispSize_NBchar, str);
						linecharcnt += charcnt;
						if(linecharcnt < wcol)
							printw(string);
						
						charcnt = sprintf(string, " %10ld", data.image[ID].md[0].cnt0);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
                        if(deltacnt0[ID] == 0)
                        {
                            printw(string);
                        }
                        else
                        {
                            attron(COLOR_PAIR(2));
                            printw(string);
                            attroff(COLOR_PAIR(2));
                        }
						
                        
                        charcnt = sprintf(string, "  %7.2f Hz", updatevaluearray[ID]);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
							printw(string);
                        
                    }



                    if(DisplayMode == 2) // sem vals
                    {
						
                        charcnt = sprintf(string, " %3d sems ", data.image[ID].md[0].sem);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
							printw(string);
                        
                        int s;
                        for(s=0; s<data.image[ID].md[0].sem; s++)
                        {
                            int semval;
                            sem_getvalue(data.image[ID].semptr[s], &semval);
                            charcnt = sprintf(string, " %7d", semval);
							linecharcnt += charcnt;
							if(linecharcnt < wcol)
								printw(string);
                        }                        
                    }

                    if(DisplayMode == 3) // sem write PIDs
                    {
                        charcnt = sprintf(string, " %3d sems ", data.image[ID].md[0].sem);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
							printw(string);
                        
                        int s;
                        for(s=0; s<data.image[ID].md[0].sem; s++)
                        {
                            pid_t pid = data.image[ID].semWritePID[s];
                            charcnt = sprintf(string, "%7d", pid);
                            linecharcnt += charcnt+1;
                            
                            if(linecharcnt < wcol)
                            {                            
                            if(getpgid(pid) >= 0)
                            {
                                attron(COLOR_PAIR(2));
                                printw(string);
                                attroff(COLOR_PAIR(2));
                            }
                            else
                            {
                                if(pid>0)
                                {
                                    attron(COLOR_PAIR(4));
                                    printw(string);
                                    attroff(COLOR_PAIR(4));
                                }
                                else
                                    printw(string);
                            }                            
                            printw(" ");
							}
                        }
                    }

                    if(DisplayMode == 4) // sem read PIDs
                    {
                        charcnt = sprintf(string, " %3d sems ", data.image[ID].md[0].sem);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
							printw(string);
                        
                        int s;
                        for(s=0; s<data.image[ID].md[0].sem; s++)
                        {
                            pid_t pid = data.image[ID].semReadPID[s];
                            charcnt = sprintf(string, "%7d", pid);
                            linecharcnt += charcnt+1;
                            if(linecharcnt < wcol)
                            {
                            if(getpgid(pid) >= 0)
                            {
                                attron(COLOR_PAIR(2));
                                printw(string);
                                attroff(COLOR_PAIR(2));
                            }
                            else
                            {
                                if(pid>0)
                                {
                                    attron(COLOR_PAIR(4));
                                    printw(string);
                                    attroff(COLOR_PAIR(4));
                                }
                                else
                                    printw(string);
                            }
                            printw(" ");
							}
                        }
                    }

                    if(DisplayMode == 5) // list processes that are accessing streams
                    {
                        if(fuserUpdate==1)
                        {														
                            FILE *fp;
                            char plistoutline[2000];
                            char command[2000];

                            int NBpid = 0;
                            
                            int sindexscan1 = ssindex[sindexscan];
                            IDscan = IDarray[sindexscan1];



                            int PReadMode = 1;

                            if(PReadMode == 0)
                            {
                                // popen option
                                sprintf(command, "/bin/fuser /tmp/%s.im.shm 2>/dev/null", sname_array[sindexscan1]);
                                fp = popen(command, "r");
                                if (fp == NULL) {
                                    streamOpenPIDarray_status[IDscan] = 2; // failed
                                }
                                else
                                {
									streamOpenPIDarray_status[IDscan] = 1;
                                    /* Read the output a line at a time - output it. */
                                    if (fgets(plistoutline, sizeof(plistoutline)-1, fp) != NULL) {
                                    }
                                    pclose(fp);
                                }
                            }
                            else
                            {
                                // filesystem option
                                char plistfname[200];
                                
                                
                                sprintf(plistfname, "/tmp/%s.shmplist", sname_array[sindexscan1]);
                                sprintf(command, "/bin/fuser /tmp/%s.im.shm 2>/dev/null > %s", sname_array[sindexscan1], plistfname);
                                system(command);
                                
                                fp = fopen(plistfname, "r");
                                if (fp == NULL) {
                                    streamOpenPIDarray_status[IDscan] = 2; // failed
                                    
                                    endwin();
                                    printf(" [%s] ", plistfname); //TEST
                                    perror("Error reading fuser output file ");
                                    exit(0);
                                }
                                else
                                {
                                    size_t len = 0;

									if(fgets(plistoutline, 2000, fp) == NULL)
                                        sprintf(plistoutline, " ");


                                    fclose(fp);
                                }                                
                            }



                            if(streamOpenPIDarray_status[IDscan] != 2)
                            {
                                char * pch;

                                pch = strtok (plistoutline," ");

                                while (pch != NULL) {
                                    if(NBpid<streamOpenNBpid_MAX) {
                                        streamOpenPIDarray[IDscan][NBpid] = atoi(pch);
                                        if(getpgid(pid) >= 0)
                                            NBpid++;
                                    }
                                    pch = strtok (NULL, " ");
                                }
                                streamOpenPIDarray_status[IDscan] = 1; // success
                            }

                            streamOpenPIDarray_cnt[IDscan] = NBpid;


                            // Get PID names
                            int pidIndex;
                            for(pidIndex=0; pidIndex<streamOpenPIDarray_cnt[IDscan] ; pidIndex++)
                            {
                                pid_t pid = streamOpenPIDarray[IDscan][pidIndex];
                                if( (getpgid(pid) >= 0) && (pid != getpid()) )
                                {
                                    char* pname = (char*) calloc(1024, sizeof(char));
                                    get_process_name_by_pid(pid, pname);

                                    if(PIDname_array[pid] == NULL)
                                        PIDname_array[pid] = (char*) malloc(sizeof(char)*(PIDnameStringLen+1));
                                    strncpy(PIDname_array[pid], pname, PIDnameStringLen);
                                    free(pname);
                                }
                            }
                            
                            sindexscan++;
							if(sindexscan == NBsindex)
							{
								fuserUpdate = 0;
								fuserUpdate0 = 0;
							}
                        }



                        if(fuserUpdate == 2)
                        {
                            streamOpenPIDarray_status[ID] = 0; // not scanned
                        }



                        
                        int pidIndex;

                        switch (streamOpenPIDarray_status[ID]) {

                        case 1:							
                            streamOpenPIDarray_cnt1[ID] = 0;
                            for(pidIndex=0; pidIndex<streamOpenPIDarray_cnt[ID] ; pidIndex++)
                            {								
                                pid_t pid = streamOpenPIDarray[ID][pidIndex];
                                if( (getpgid(pid) >= 0) && (pid != getpid()) ) {
									
                                    charcnt = sprintf(string, "%6d:%-*.*s", (int) pid, PIDnameStringLen, PIDnameStringLen, PIDname_array[pid]);
                                    linecharcnt += charcnt;
                                    if(linecharcnt < wcol)
										printw(string);
                                    
                                    
                                    streamOpenPIDarray_cnt1[ID]++;
                                }
							}

								//const chtype * lstring1 = "This is a test";
                                //addchstr(lstring1);
                            
                            break;

                        case 2:
                            charcnt = sprintf(string, "FAILED");
                            linecharcnt += charcnt;
                            if(linecharcnt < wcol)
								printw(string);
                            break;

                        default:
                            charcnt = sprintf(string, "NOT SCANNED");
                            linecharcnt += charcnt;
                            if(linecharcnt < wcol)
								printw(string);
                            break;

                        }

                    }

                    if(dindex == dindexSelected)
                        attroff(A_REVERSE);

					if(linecharcnt > wcol)
					{
						attron(COLOR_PAIR(9));
						printw("+");
						attroff(COLOR_PAIR(9));
					}
					printw("\n");
					

                    if(fuserUpdate==1)
                    {
                        refresh();
                        if(data.signal_INT == 1) // stop scan
                        {
                            fuserUpdate = 2;     // complete loop without scan
                            data.signal_INT = 0; // reset
                        }
                    }

                    if(dindex>NBsinfodisp-1)
                        sOK = 0;
                }
            }
            
            
        }

        refresh();

        cnt++;

        if( (data.signal_TERM == 1) || (data.signal_INT == 1) || (data.signal_ABRT == 1) || (data.signal_BUS == 1) || (data.signal_SEGV == 1) || (data.signal_HUP == 1) || (data.signal_PIPE == 1))
            loopOK = 0;
    }

    endwin();



    return 0;
}
