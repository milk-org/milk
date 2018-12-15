
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
	long IDarray[200];

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


        case 'R': // remove stream
            ImageStreamIO_destroyIm( &data.image[IDarray[sindex]] );
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
            printw("   Process control screen\n");





            attron(attrval);
            printw("SPACE");
            attroff(attrval);
            printw("    Select this stream\n");



            printw("\n\n");
        }
        else
        {

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
							printw(" %6ld", data.image[ID].md[0].cnt0);
							attroff(COLOR_PAIR(2));
						}
						cnt0array[ID] = data.image[ID].md[0].cnt0;
                        
                        
                        
                        printw(" [%3ld sems ", data.image[ID].md[0].sem);
                        int s;
                        
                        if(DisplayMode == 2) // sem vals
                        {
							for(s=0; s<data.image[ID].md[0].sem; s++)
							{
								int semval;
								sem_getvalue(data.image[ID].semptr[s], &semval);
								printw(" %6d ", semval);
							}
						}
						if(DisplayMode == 3) // sem write PIDs
                        {
							for(s=0; s<data.image[ID].md[0].sem; s++)
							{
								printw(" %6d ", data.image[ID].semWritePID);
							}
						}
						if(DisplayMode == 4) // sem read PIDs
                        {
							for(s=0; s<data.image[ID].md[0].sem; s++)
							{
								printw(" %6d ", data.image[ID].semReadPID);
							}
						}
                        printw("]");


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
