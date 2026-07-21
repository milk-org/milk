// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file stringutils.c
 * @brief Stringutils module
 */

/**
 * @file stringutils.c
 */

#include <string.h>

int replace_char(char *content, char cin, char cout)
{
    for (unsigned i = 0; i < strlen(content); i++)
    {
        if (content[i] == cin)
        {
            content[i] = cout;
        }
    }

    return (0);
}
