// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file print_header.c
 */

#include <ncurses.h>

#include "CommandLineInterface/CLIcore.h"

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
