/**
 * @file print_header.c
 * @brief Print header module
 */

/** @file print_header.c
 */

#ifdef USE_NCURSES
#include <ncurses.h>
#else
#define printw(...) printf(__VA_ARGS__)
#define attron(a)
#define attroff(a)
#define A_BOLD 0
#endif

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

extern int infoscreen_wcol;
extern int infoscreen_wrow;

errno_t print_header(const char *str, char c)
{
    long n;
    long i;

    attron(A_BOLD);
    n = strlen(str);
    for(i = 0; i < (infoscreen_wcol - n) / 2; i++)
    {
        printw("%c", c);
    }
    printw("%s", str);
    for(i = 0; i < (infoscreen_wcol - n) / 2 - 1; i++)
    {
        printw("%c", c);
    }
    printw("\n");
    attroff(A_BOLD);

    return RETURN_SUCCESS;
}
