// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    file_exists.c
 */

#include <stdio.h>

int file_exists(const char *restrict file_name)
{
    FILE *fp;
    int   exists = 1;

    if ((fp = fopen(file_name, "r")) == NULL)
    {
        exists = 0;
        /*      printf("file %s does not exist\n",file_name);*/
    }
    else
    {
        fclose(fp);
    }

    return (exists);
}
