/**
 * @file    TUItools.c
 * @brief   Text User Interface tools
 */

#include <sys/ioctl.h> // for terminal size
#include <termios.h>

#include <ncurses.h>

#include <locale.h>
#include <wchar.h>

#include <CommandLineInterface/CLIcore.h>

#include "TUItools.h"

static struct winsize     w;
static short unsigned int wrow, wcol;
static int                wresizecnt = 0;

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

/**
 * @brief print to stdout or through ncurses
 *
 * If screenprintmode :\n
 * is SCREENPRINT_STDIO, use stdio\n
 * is SCREENPRINT_NCURSES, use ncurses\n
 *
 * @param fmt
 * @param ...
 */
void TUI_printfw(const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);

    if (screenprintmode == SCREENPRINT_STDIO)
    {
        vfprintf(stdout, fmt, args);
    }

    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        int  x, y;
        int  MAXLINELEN = 512;
        char prtstring[MAXLINELEN];

        getyx(stdscr, y, x);
        (void) x;
        (void) y;

        vsnprintf(prtstring, MAXLINELEN, fmt, args);
        printw("%s", prtstring);
    }

    va_end(args);
}

/*
void TUI_wprintfw(const wchar_t *wstr)
{
    if (screenprintmode == SCREENPRINT_STDIO)
    {
        printf("%ls", wstr);
    }

    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        int  x, y;
        int  MAXLINELEN = 512;
        //wchar_t prtstring[MAXLINELEN];

        getyx(stdscr, y, x);
        (void) x;
        (void) y;

        //vswprintf(prtstring, MAXLINELEN, fmt, args);
//        printw("%ls", wstr);
        //mvaddwstr(y, x, wstr);
    }
}
*/



void TUI_newline()
{
    if (screenprintmode == SCREENPRINT_STDIO)
    {
        printf("\n");
    }
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        printw("\n");
    }
}




void screenprint_setcolor(int colorcode)
{
    if (screenprintmode == SCREENPRINT_NCURSES)
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

        printf("\033[%d;%dm", printAECfgcolor, printAECbgcolor);
    }
}

void screenprint_unsetcolor(int colorcode)
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attroff(COLOR_PAIR(colorcode));
    }
    else
    {
        printAEC        = AEC_NORMAL;
        printAECfgcolor = AEC_FGCOLOR_WHITE;
        printAECbgcolor = AEC_BGCOLOR_BLACK;
        printf("\033[%dm", printAEC); //, printAECbgcolor);
    }
}

void screenprint_setbold()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attron(A_BOLD);
    }
    else
    {
        printAEC = AEC_BOLD;
        printf("\033[%dm", printAEC);
    }
}

void screenprint_unsetbold()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attroff(A_BOLD);
    }
    else
    {
        printAEC = AEC_NORMAL; //AEC_BOLDOFF;
        printf("\033[%dm", printAEC);
    }
}

void screenprint_setblink()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attron(A_BLINK);
    }
    else
    {
        printAEC = AEC_FASTBLINK;
        printf("\033[%dm", printAEC);
    }
}

void screenprint_unsetblink()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attroff(A_BLINK);
    }
    else
    {
        printAEC = AEC_NORMAL; //AEC_BLINKOFF;
        printf("\033[%dm", AEC_NORMAL);
    }
}

void screenprint_setdim()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attron(A_DIM);
    }
    else
    {
        printAEC = AEC_FAINT;
        printf("\033[%dm", printAEC);
    }
}

void screenprint_unsetdim()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attroff(A_DIM);
    }
    else
    {
        printAEC = AEC_NORMAL; //AEC_FAINTOFF;
        printf("\033[%dm", printAEC);
    }
}

void screenprint_setreverse()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attron(A_REVERSE);
    }
    else
    {
        printAEC = AEC_REVERSE;
        printf("\033[%dm", printAEC);
    }
}

void screenprint_unsetreverse()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        attroff(A_REVERSE);
    }
    else
    {
        printAEC = AEC_NORMAL; //AEC_REVERSEOFF;
        printf("\033[%dm", printAEC);
    }
}

void screenprint_setnormal()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        //attron(A_REVERSE);
    }
    else
    {
        printAEC        = AEC_NORMAL;
        printAECfgcolor = AEC_FGCOLOR_WHITE;
        printAECbgcolor = AEC_BGCOLOR_BLACK;
        printf("\033[%d;%d;%dm", printAEC, printAECfgcolor, printAECbgcolor);
    }
}




/**
 * @brief Print header line
 *
 * @param str     content string
 * @param c       filler character to be printed on either side of content
 * @return errno_t
 */
errno_t TUI_print_header(const char *str, char c)
{
    long n = strlen(str);

    screenprint_setbold();

    int strl = wcol - 1;
    if (n > wcol)
    {
        strl = n + 1;
    }
    char linestring[strl];
    int  spos = 0;

    for (long i = 0; i < (wcol - n) / 2; i++)
    {
        linestring[spos] = c;
        spos++;
    }

    for (size_t i = 0; i < strlen(str); i++)
    {
        linestring[spos] = str[i];
        spos++;
    }

    for (long i = 0; i < (wcol - n) / 2 - 1; i++)
    {
        linestring[spos] = c;
        spos++;
    }

    linestring[spos] = '\0';
    TUI_printfw("%s", linestring);

    TUI_newline();
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

errno_t TUI_inittermios(short unsigned int *wrowptr,
                        short unsigned int *wcolptr)
{
    tcgetattr(0, &orig_termios);

    memcpy(&new_termios, &orig_termios, sizeof(new_termios));

    //cfmakeraw(&new_termios);
    new_termios.c_lflag &= ~ICANON;
    new_termios.c_lflag &= ~ECHO;
    new_termios.c_lflag &= ~ISIG;
    new_termios.c_cc[VMIN]  = 0;
    new_termios.c_cc[VTIME] = 0;

    tcsetattr(0, TCSANOW, &new_termios);

    // get terminal size
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    *wrowptr = w.ws_row;
    *wcolptr = w.ws_col;

    atexit(TUI_reset_terminal_mode);

    return RETURN_SUCCESS;
}

void TUI_clearscreen(short unsigned int *wrowptr, short unsigned int *wcolptr)
{
    if (screenprintmode == SCREENPRINT_STDIO) // stdio mode
    {
        printf("\e[1;1H\e[2J");
        //printf("[%12lld  %d %d %d ]  ", loopcnt, buffd[0], buffd[1], buffd[2]);

        // update terminal size
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

        *wrowptr = w.ws_row;
        *wcolptr = w.ws_col;
    }
    else
    {
        (void) *wrowptr;
        (void) *wcolptr;
    }
}

void TUI_handle_winch(int sig)
{
    wresizecnt++;
    DEBUG_TRACEPOINT("wresizecnt = %d", wresizecnt);
    (void) sig;

    endwin();

    // Needs to be called after an endwin() so ncurses will initialize
    // itself with the new terminal dimensions.
    refresh();

    clear();
    wrow = LINES;
    wcol = COLS;

    DEBUG_TRACEPOINT("window size %d %d", wrow, wcol);

    refresh();
}




/** @brief INITIALIZE ncurses
 *
 */
errno_t TUI_initncurses(short unsigned int *wrowptr,
                        short unsigned int *wcolptr)
{
    DEBUG_TRACE_FSTART();

    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        DEBUG_TRACEPOINT("Initializing TUI ncurses ");

        setlocale(LC_ALL, "");
        if (initscr() == NULL)
        {
            fprintf(stderr, "Error initialising ncurses.\n");
            exit(EXIT_FAILURE);
        }
        DEBUG_TRACEPOINT("Initializing TUI ncurses ");

        getmaxyx(stdscr, wrow, wcol); /* get the number of rows and columns */

        DEBUG_TRACEPOINT("wrow wcol = %d %d", wrow, wcol);

        *wrowptr = wrow;
        *wcolptr = wcol;
        DEBUG_TRACEPOINT("wrow wcol = %d %d", wrow, wcol);

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
        //init_color(COLOR_GREEN, 400, 1000, 400);
        //init_color(COLOR_GREEN, 700, 1000, 700);
        //init_color(COLOR_YELLOW, 1000, 1000, 700);
        start_color();
        DEBUG_TRACEPOINT(" ");

        //  colored background
        init_pair(1, COLOR_BLACK, COLOR_WHITE);
        init_pair(2, COLOR_BLACK, COLOR_GREEN);  // all good
        init_pair(3, COLOR_BLACK, COLOR_YELLOW); // parameter out of sync
        init_pair(4, COLOR_WHITE, COLOR_RED);
        init_pair(5, COLOR_WHITE, COLOR_BLUE); // DIRECTORY
        init_pair(6, COLOR_GREEN, COLOR_BLACK);
        init_pair(7, COLOR_YELLOW, COLOR_BLACK);
        init_pair(8, COLOR_RED, COLOR_BLACK);
        init_pair(9, COLOR_BLACK, COLOR_RED);
        init_pair(10, COLOR_BLACK, COLOR_CYAN);
        init_pair(12, COLOR_GREEN,
                  COLOR_WHITE); // highlighted version of #2

        // handle window resize
        struct sigaction sa;
        memset(&sa, 0, sizeof(struct sigaction));
        sa.sa_handler = TUI_handle_winch;
        sigaction(SIGWINCH, &sa, NULL);
    }

    DEBUG_TRACE_FEXIT();

    return RETURN_SUCCESS;
}




errno_t TUI_init_terminal(short unsigned int *wrowptr,
                          short unsigned int *wcolptr)
{
    DEBUG_TRACE_FSTART();
    if (screenprintmode == SCREENPRINT_NCURSES) // ncurses mode
    {
        TUI_initncurses(wrowptr, wcolptr);
        DEBUG_TRACEPOINT("init terminal ncurses mode %d %d",
                         *wrowptr,
                         *wcolptr);
        atexit(TUI_atexit);
        clear();
    }
    else
    {
        TUI_inittermios(wrowptr, wcolptr);
        DEBUG_TRACEPOINT("init terminal stdio mode %d %d", *wrowptr, *wcolptr);
    }
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


errno_t TUI_get_terminal_size(short unsigned int *wrowptr,
                              short unsigned int *wcolptr)
{
    *wrowptr = wrow;
    *wcolptr = wcol;

    return RETURN_SUCCESS;
}

errno_t TUI_exit()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
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
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        refresh();
    }

    return RETURN_SUCCESS;
}

errno_t TUI_ncurses_erase()
{
    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        erase();
    }

    return RETURN_SUCCESS;
}

errno_t TUI_stdio_clear()
{
    if (screenprintmode == SCREENPRINT_STDIO)
    {
        printf("\e[1;1H\e[2J");
    }

    return RETURN_SUCCESS;
}

int get_singlechar_nonblock()
{
    int ch = -1;

    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        ch = getch(); // ncurses function, non-blocking
    }
    else
    {
        char buff[3];

        int l = read(STDIN_FILENO, buff, 3);

        if (l > 0)
        {
            ch = buff[0];

            if (buff[0] == 13) // enter
            {
                ch = 10; // new line
            }

            if (buff[0] == 27) // if the first value is esc
            {

                if (buff[1] == 91)
                {
                    switch (buff[2])
                    {
                    // the real value
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

                if (buff[1] == 79)
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

    if (screenprintmode == SCREENPRINT_NCURSES)
    {
        ch = getchar();
    }
    else
    {
        int getchardt_us = 100000;

        ch = -1;
        while (ch == -1)
        {
            usleep(getchardt_us); // kHz
            ch = get_singlechar_nonblock();
        }
    }
    return ch;
}
