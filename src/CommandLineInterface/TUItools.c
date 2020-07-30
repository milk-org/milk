/**
 * @file    TUItools.c
 * @brief   Text User Interface tools
 */


#include <termios.h>
#include <sys/ioctl.h> // for terminal size

#include <ncurses.h>

#include <CommandLineInterface/CLIcore.h>
#include "TUItools.h"



static struct winsize w;
static short unsigned int wrow, wcol;

/*
 * Defines printfw output
 * 
 * SCREENPRINT_STDIO     printf to stdout
 * SCREENPRINT_NCURSES   printw
 * SCREENPRINT_NONE      don't print (silent)
 */

static int screenprintmode = SCREENPRINT_STDIO;

struct termios orig_termios;
struct termios new_termios;







static int printAEC = 0;

// Foreground color
static int printAECfgcolor = AEC_FGCOLOR_WHITE;

// Background color
static int printAECbgcolor = AEC_BGCOLOR_BLACK;





void TUI_set_screenprintmode(int mode)
{
	screenprintmode = mode;
}



int TUI_get_screenprintmode()
{
	return screenprintmode;
}





void printfw(const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);


    if(screenprintmode == SCREENPRINT_STDIO)
    {
        vfprintf(stdout, fmt, args);
    }

    if(screenprintmode == SCREENPRINT_NCURSES)
    {
        vw_printw(stdscr, fmt, args);
    }

    va_end(args);
}


void screenprint_setcolor( int colorcode )
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attron(COLOR_PAIR(colorcode));
	}
	else
	{
		switch (colorcode) 
		{
			case 1:
			printAECfgcolor = AEC_FGCOLOR_WHITE;
			printAECbgcolor = AEC_BGCOLOR_BLACK;			
			break;

			case 2:
			printAECfgcolor = AEC_FGCOLOR_BLACK;
			printAECbgcolor = AEC_BGCOLOR_GREEN;			
			break;

			case 3:
			printAECfgcolor = AEC_FGCOLOR_BLACK;
			printAECbgcolor = AEC_BGCOLOR_YELLOW;			
			break;

			case 4:
			printAECfgcolor = AEC_FGCOLOR_WHITE;
			printAECbgcolor = AEC_BGCOLOR_RED;			
			break;

			case 5:
			printAECfgcolor = AEC_FGCOLOR_WHITE;
			printAECbgcolor = AEC_BGCOLOR_BLUE;			
			break;

			case 6:
			printAECfgcolor = AEC_FGCOLOR_BLACK;
			printAECbgcolor = AEC_BGCOLOR_GREEN;			
			break;

			case 7:
			printAECfgcolor = AEC_FGCOLOR_WHITE;
			printAECbgcolor = AEC_BGCOLOR_YELLOW;			
			break;

			case 8:
			printAECfgcolor = AEC_FGCOLOR_BLACK;
			printAECbgcolor = AEC_BGCOLOR_RED;			
			break;

			case 9:
			printAECfgcolor = AEC_FGCOLOR_RED;
			printAECbgcolor = AEC_BGCOLOR_BLACK;			
			break;
			
			case 10:
			printAECfgcolor = AEC_FGCOLOR_BLACK;
			printAECbgcolor = AEC_BGCOLOR_BLUE + 60;			
			break;									
		}
		
		printf( "\033[%d;%dm",  printAECfgcolor, printAECbgcolor);
	}
}


void screenprint_unsetcolor( int colorcode )
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attroff(COLOR_PAIR(colorcode));
	}
	else
	{
		printAEC = AEC_NORMAL;
		printAECfgcolor = AEC_FGCOLOR_WHITE;
		printAECbgcolor = AEC_BGCOLOR_BLACK;	
		printf( "\033[%dm", printAEC);//, printAECbgcolor);
	}
}


void screenprint_setbold()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attron(A_BOLD);
	}
	else
	{
		printAEC = AEC_BOLD;
		printf( "\033[%dm",  printAEC);
	}
}


void screenprint_unsetbold()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attroff(A_BOLD);
	}
	else
	{
		printAEC = AEC_NORMAL; //AEC_BOLDOFF;
		printf( "\033[%dm",  printAEC);
	}
}


void screenprint_setblink()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attron(A_BLINK);
	}
	else
	{
		printAEC = AEC_FASTBLINK;
		printf( "\033[%dm",  printAEC);
	}
}


void screenprint_unsetblink()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attroff(A_BLINK);
	}
	else
	{
		printAEC = AEC_NORMAL; //AEC_BLINKOFF;
		printf( "\033[%dm", AEC_NORMAL);
	}
}


void screenprint_setdim()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attron(A_DIM);
	}
	else
	{
		printAEC = AEC_FAINT;
		printf( "\033[%dm",  printAEC);
	}
}


void screenprint_unsetdim()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attroff(A_DIM);
	}
	else
	{
		printAEC = AEC_NORMAL; //AEC_FAINTOFF;
		printf( "\033[%dm",  printAEC);
	}
}


void screenprint_setreverse()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attron(A_REVERSE);
	}
	else
	{
		printAEC = AEC_REVERSE;
		printf( "\033[%dm",  printAEC);
	}
}


void screenprint_unsetreverse()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		attroff(A_REVERSE);
	}
	else
	{
		printAEC = AEC_NORMAL; //AEC_REVERSEOFF;
		printf( "\033[%dm",  printAEC);
	}
}


void screenprint_setnormal()
{
	if(screenprintmode == SCREENPRINT_NCURSES)
	{
		//attron(A_REVERSE);
	}
	else
	{
		printAEC = AEC_NORMAL;
		printAECfgcolor = AEC_FGCOLOR_WHITE;
		printAECbgcolor = AEC_BGCOLOR_BLACK;
		printf( "\033[%d;%d;%dm", printAEC, printAECfgcolor, printAECbgcolor );
	}
}



errno_t TUI_print_header(const char *str, char c)
{
    long n;
    long i;

    screenprint_setbold();
    n = strlen(str);
    for(i = 0; i < (wcol - n) / 2; i++)
    {
        printfw("%c", c);
    }
    printfw("%s", str);
    for(i = 0; i < (wcol - n) / 2 - 1; i++)
    {
        printfw("%c", c);
    }
    printfw("\n");
    screenprint_unsetbold();

    return RETURN_SUCCESS;
}








/** @brief restore terminal settings
 */
void TUI_reset_terminal_mode()
{
    tcsetattr(0, TCSANOW, &orig_termios);
    tcsetattr(0, TCSANOW, &new_termios);
}


errno_t TUI_inittermios(short unsigned int *wrow, short unsigned int *wcol)
{
    tcgetattr(0, &orig_termios);

    memcpy(&new_termios, &orig_termios, sizeof(new_termios));

    //cfmakeraw(&new_termios);
    new_termios.c_lflag &= ~ICANON;
    new_termios.c_lflag &= ~ECHO;
    new_termios.c_lflag &= ~ISIG;
    new_termios.c_cc[VMIN] = 0;
    new_termios.c_cc[VTIME] = 0;

    tcsetattr(0, TCSANOW, &new_termios);

    // get terminal size
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    *wrow = w.ws_row;
    *wcol = w.ws_col;


    atexit(TUI_reset_terminal_mode);

    return RETURN_SUCCESS;
}





void TUI_clearscreen(short unsigned int *wrow, short unsigned int *wcol)
{
    if(screenprintmode == SCREENPRINT_STDIO) // stdio mode
    {
        printf("\e[1;1H\e[2J");
        //printf("[%12lld  %d %d %d ]  ", loopcnt, buffd[0], buffd[1], buffd[2]);

        // update terminal size        
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

        *wrow = w.ws_row;
        *wcol = w.ws_col;
    }
    else
    {
		(void) *wrow;
		(void) *wcol;
	}
}





/** @brief INITIALIZE ncurses
 *
 */
errno_t TUI_initncurses()
{
	DEBUG_TRACEPOINT(" ");
    if( screenprintmode == SCREENPRINT_NCURSES)
    {
		DEBUG_TRACEPOINT(" test abort ");
   
        
        if(initscr() == NULL)
        {
            fprintf(stderr, "Error initialising ncurses.\n");
            exit(EXIT_FAILURE);
        }
        getmaxyx(stdscr, wrow, wcol);		/* get the number of rows and columns */
		
        cbreak();
        // disables line buffering and erase/kill character-processing (interrupt and flow control characters are unaffected),
        // making characters typed by the user immediately available to the program

		DEBUG_TRACEPOINT(" ");
		
        keypad(stdscr, TRUE);
        // enable F1, F2 etc..

		DEBUG_TRACEPOINT(" ");
        nodelay(stdscr, TRUE);
        curs_set(0);

		DEBUG_TRACEPOINT(" ");
        noecho();
        // Don't echo() while we do getch



        //nonl();
        // Do not translates newline into return and line-feed on output

		DEBUG_TRACEPOINT(" ");
        init_color(COLOR_GREEN, 400, 1000, 400);
        start_color();
		DEBUG_TRACEPOINT(" ");
        //  colored background
        init_pair(  1, COLOR_BLACK,  COLOR_WHITE  );
        init_pair(  2, COLOR_BLACK,  COLOR_GREEN  );  // all good
        init_pair(  3, COLOR_BLACK,  COLOR_YELLOW ); // parameter out of sync
        init_pair(  4, COLOR_WHITE,  COLOR_RED    );
        init_pair(  5, COLOR_WHITE,  COLOR_BLUE   ); // DIRECTORY
        init_pair(  6, COLOR_GREEN,  COLOR_BLACK  );
        init_pair(  7, COLOR_YELLOW, COLOR_BLACK  );
        init_pair(  8, COLOR_RED,    COLOR_BLACK  );
        init_pair(  9, COLOR_BLACK,  COLOR_RED    );
        init_pair( 10, COLOR_BLACK,  COLOR_CYAN   );
    }

	DEBUG_TRACEPOINT(" ");

    return RETURN_SUCCESS;
}




errno_t TUI_init_terminal(short unsigned int *wrowptr, short unsigned int *wcolptr)
{

    if( screenprintmode == SCREENPRINT_NCURSES) // ncurses mode
    {
        TUI_initncurses(wrowptr, wcolptr);
        atexit(TUI_atexit);
        clear();
    }
    else
    {
        TUI_inittermios(wrowptr, wcolptr);
    }
    
    return RETURN_SUCCESS;
}



errno_t TUI_exit()
{
    if( screenprintmode == SCREENPRINT_NCURSES) {
        endwin();
    }

    return RETURN_SUCCESS;
}



void TUI_atexit()
{
    //printf("exiting CTRLscreen\n");

    // endwin();
}





errno_t TUI_ncurses_refresh()
{
    if( screenprintmode == SCREENPRINT_NCURSES)
    {
        refresh();
    }
    
    return RETURN_SUCCESS;
}


errno_t TUI_ncurses_erase()
{
    if( screenprintmode == SCREENPRINT_NCURSES)
    {
        erase();
    }

    return RETURN_SUCCESS;
}


errno_t TUI_stdio_clear()
{
    if( screenprintmode == SCREENPRINT_STDIO )
    {
        printf("\e[1;1H\e[2J");
    }

    return RETURN_SUCCESS;
}













int get_singlechar_nonblock()
{
    int ch = -1;

    if(screenprintmode == SCREENPRINT_NCURSES)
    {
        ch = getch();  // ncurses function, non-blocking
    }
    else
    {
        char buff[3];

        int l = read(STDIN_FILENO, buff, 3);

        if(l>0) {
            ch = buff[0];

            if (buff[0] == 13) // enter
            {
                ch = 10; // new line
            }


            if (buff[0] == 27) { // if the first value is esc


                if(buff[1] == 91) {
                    switch (buff[2])
                    {   // the real value
                    case 'A':
                        ch = KEY_UP; // code for arrow up
                        break;
                    case 'B':
                        ch = KEY_DOWN; // code for arrow down
                        break;
                    case 'C':
                        ch = KEY_RIGHT; // code for arrow right
                        break;
                    case 'D':
                        ch = KEY_LEFT; // code for arrow left
                        break;
                    }
                }


                if(buff[1] == 79)
                {
                    switch (buff[2])
                    {
                    case 80:
                        ch = KEY_F(1);
                        break;
                    case 81:
                        ch = KEY_F(2);
                        break;
                    case 82:
                        ch = KEY_F(3);
                        break;
                    }
                }
            }

        }
    }

    return ch;
}



int get_singlechar_block()
{
	int ch;
	
    if(screenprintmode == SCREENPRINT_NCURSES)
    {
        ch = getchar();
    }
    else
    {
        int getchardt_us = 100000;

        ch = -1;
        while(ch == -1)
        {
            usleep(getchardt_us); // kHz
            ch = get_singlechar_nonblock();
        }
    }
    return ch;
}




