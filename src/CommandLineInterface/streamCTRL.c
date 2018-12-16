
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

#include <ncurses.h>
#include <fcntl.h> 
#include <ctype.h>

#include <dirent.h>

#include <wchar.h>
#include <locale.h>

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












static const char* get_process_name_by_pid(const int pid)
{
    char* name = (char*)calloc(1024,sizeof(char));
    if(name){
        sprintf(name, "/proc/%d/cmdline",pid);
        FILE* f = fopen(name,"r");
        if(f){
            size_t size;
            size = fread(name, sizeof(char), 1024, f);
            if(size>0){
                if('\n'==name[size-1])
                    name[size-1]='\0';
            }
            fclose(f);
        }
    }
    return name;
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
    long sindex, index;

    float frequ = 30.0; // Hz
    char  monstring[200];

    long IDmax = 10000;
    long cnt0array[1000];
    long IDarray[streamNBID_MAX];

    setlocale(LC_ALL, "");



    // INITIALIZE ncurses
    initncurses();

    int NBsinfodisp = wrow-5;
    int NBsindex = 0;
    int loopOK = 1;
    long cnt = 0;


    int sindexSelected = 0;

    int DisplayMode = 1;
    // display modes:
    // 1: overview

    int fuserUpdate = 1;
    pid_t streamOpenPIDarray[streamNBID_MAX][streamOpenNBpid_MAX];
    int streamOpenPIDarray_cnt[streamNBID_MAX];

    clear();

    while( loopOK == 1 )
    {
        int pid;
        char command[200];


        usleep((long) (1000000.0/frequ));
        int ch = getch();



        attron(A_BOLD);
        sprintf(monstring, "PRESS x TO STOP MONITOR");
        print_header(monstring, '-');
        attroff(A_BOLD);

        int selectedOK = 0; // goes to 1 if at least one process is selected
        switch (ch)
        {
        case 'x':     // Exit control screen
            loopOK=0;
            break;

        case KEY_UP:
            sindexSelected --;
            if(sindexSelected<0)
                sindexSelected = 0;
            break;

        case KEY_DOWN:
            sindexSelected ++;
            if(sindexSelected>NBsindex-1)
                sindexSelected = NBsindex-1;
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
            fuserUpdate = 1;
            DisplayMode = 5;
            break;


        case 'R': // remove stream
            ImageStreamIO_destroyIm( &data.image[IDarray[sindexSelected]]);
            break;

        }

        erase();


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

            attron(attrval);
            printw("SPACE");
            attroff(attrval);
            printw("    Select this stream\n");



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


			printw("\n\n");








            DIR *d;
            struct dirent *dir;
            d = opendir("/tmp/");


            sindex = 0;

            int sOK;
            sOK = 1;
            if(d)
            {
                while((sOK == 1)&&((dir = readdir(d)) != NULL))
                {
                    char *pch = strstr(dir->d_name, ".im.shm");

                    if(pch)
                    {
                        if(sindex == sindexSelected)
                            attron(A_REVERSE);

                        long ID;
                        char sname[200];
                        strncpy(sname, dir->d_name, strlen(dir->d_name)-strlen(".im.shm"));
                        sname[strlen(dir->d_name)-strlen(".im.shm")] = '\0';
                        ID = image_ID(sname);
                        if(ID == -1)
                            ID = read_sharedmem_image(sname);
                        printw("%03ld %4ld  %-36s ", sindex, ID, sname);
                        IDarray[sindex] = ID;

                        if(DisplayMode < 5)
                        {
                            int atype = data.image[ID].md[0].atype;
                            char str[200];
                            char str1[200];
                            int j;


                            if(atype==_DATATYPE_UINT8)
                                printw(" UI8");
                            if(atype==_DATATYPE_INT8)
                                printw("  I8");

                            if(atype==_DATATYPE_UINT16)
                                printw("UI16");
                            if(atype==_DATATYPE_INT16)
                                printw(" I16");

                            if(atype==_DATATYPE_UINT32)
                                printw("UI32");
                            if(atype==_DATATYPE_INT32)
                                printw(" I32");

                            if(atype==_DATATYPE_UINT64)
                                printw("UI64");
                            if(atype==_DATATYPE_INT64)
                                printw(" I64");

                            if(atype==_DATATYPE_FLOAT)
                                printw(" FLT");

                            if(atype==_DATATYPE_DOUBLE)
                                printw(" DBL");

                            if(atype==_DATATYPE_COMPLEX_FLOAT)
                                printw("CFLT");

                            if(atype==_DATATYPE_COMPLEX_DOUBLE)
                                printw("CDBL");

                            sprintf(str, " [%3ld", (long) data.image[ID].md[0].size[0]);

                            for(j=1; j<data.image[ID].md[0].naxis; j++)
                            {
                                sprintf(str1, "%sx%3ld", str, (long) data.image[ID].md[0].size[j]);
                                strcpy(str, str1);
                            }
                            sprintf(str1, "%s]", str);
                            strcpy(str, str1);

                            printw("%-20s ", str);


                            // counter and semaphores
                            if(data.image[ID].md[0].cnt0 == cnt0array[ID]) // has not changed
                            {
                                printw(" %8ld", data.image[ID].md[0].cnt0);
                            }
                            else
                            {
                                attron(COLOR_PAIR(2));
                                printw(" %8ld", data.image[ID].md[0].cnt0);
                                attroff(COLOR_PAIR(2));
                            }
                            cnt0array[ID] = data.image[ID].md[0].cnt0;

                        }



                        if(DisplayMode == 2) // sem vals
                        {
                            printw(" [%3ld sems ", data.image[ID].md[0].sem);
                            int s;
                            for(s=0; s<data.image[ID].md[0].sem; s++)
                            {
                                int semval;
                                sem_getvalue(data.image[ID].semptr[s], &semval);
                                printw(" %7d", semval);
                            }
                            printw(" ]");
                        }
                        if(DisplayMode == 3) // sem write PIDs
                        {
                            printw(" [%3ld sems ", data.image[ID].md[0].sem);
                            int s;
                            for(s=0; s<data.image[ID].md[0].sem; s++)
                            {
                                pid_t pid = data.image[ID].semWritePID[s];
                                if(getpgid(pid) >= 0)
                                {
                                    attron(COLOR_PAIR(2));
                                    printw("%7d", pid);
                                    attroff(COLOR_PAIR(2));
                                }
                                else
                                {
                                    if(pid>0)
                                    {
                                        attron(COLOR_PAIR(4));
                                        printw("%7d", pid);
                                        attroff(COLOR_PAIR(4));
                                    }
                                    else
                                        printw("%7d", pid);
                                }
                                printw(" ]");
                            }
                        }
                        if(DisplayMode == 4) // sem read PIDs
                        {
                            printw(" [%3ld sems ", data.image[ID].md[0].sem);
                            int s;
                            for(s=0; s<data.image[ID].md[0].sem; s++)
                            {
                                pid_t pid = data.image[ID].semReadPID[s];
                                if(getpgid(pid) >= 0)
                                {
                                    attron(COLOR_PAIR(2));
                                    printw("%7d", pid);
                                    attroff(COLOR_PAIR(2));
                                }
                                else
                                {
                                    if(pid>0)
                                    {
                                        attron(COLOR_PAIR(4));
                                        printw("%7d", pid);
                                        attroff(COLOR_PAIR(4));
                                    }
                                    else
                                        printw("%7d", pid);
                                }
                                printw(" ]");
                            }
                        }

                        if(DisplayMode == 5) // open by processes...
                        {
                            if(fuserUpdate==1)
                            {
                                FILE *fp;
                                char fuseroutline[1035];
                                char command[2000];

                                /* Open the command for reading. */
                                sprintf(command, "/bin/fuser /tmp/%s.im.shm 2>/dev/null", sname);
                                fp = popen(command, "r");
                                if (fp == NULL) {
                                    printf("Failed to run command\n" );
                                    exit(1);
                                }
                                /* Read the output a line at a time - output it. */
                                if (fgets(fuseroutline, sizeof(fuseroutline)-1, fp) != NULL) {
                                    //printw("  OPEN BY: %-30s", fuseroutline);
                                }
                                pclose(fp);


                                char * pch;
                                int NBpid = 0;

                                pch = strtok (fuseroutline," ");

                                while (pch != NULL) {
                                    if(NBpid<streamOpenNBpid_MAX) {
                                        streamOpenPIDarray[sindex][NBpid] = atoi(pch);
                                        if(getpgid(pid) >= 0)
                                            NBpid++;
                                    }
                                    pch = strtok (NULL, " ");
                                }
                                streamOpenPIDarray_cnt[sindex] = NBpid;


                                fuserUpdate = 0;
                            }

                            printw("  OPENED BY: ");
                            int pidIndex;
                            for(pidIndex=0; pidIndex<streamOpenPIDarray_cnt[sindex] ; pidIndex++)
                            {
                                pid_t pid = streamOpenPIDarray[sindex][pidIndex];
                                if(getpgid(pid) >= 0)
                                    printw(" %s(%d)", get_process_name_by_pid(pid), (int) pid);
                            }
                        }


                        printw("\n");

                        if(sindex == sindexSelected)
                            attroff(A_REVERSE);

                        sindex++;
                        if(sindex>NBsinfodisp-1)
                            sOK = 0;
                    }
                }
                closedir(d);
            }
            NBsindex = sindex;
        }



        refresh();

        cnt++;

    }

    endwin();



    return 0;
}
