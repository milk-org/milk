// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_script_expand.c
 *
 * @brief Brace expansion {N..M} and {N..M..S}
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"

/**
 * @brief Emit string into output buffer
 */
static void emit_str(char *out, int *opos, int maxlen, const char *s)
{
    while (*s != '\0' && *opos < maxlen - 1)
    {
        out[(*opos)++] = *s++;
    }
}

/**
 * @brief Expand {N..M} and {N..M..S} brace ranges
 *
 * Replaces tokens like {1..5} with "1 2 3 4 5"
 * and {0..10..2} with "0 2 4 6 8 10".
 *
 * @param line    Buffer to expand in-place
 * @param maxlen  Buffer capacity
 */
void cli_expand_braces(char *line, int maxlen)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i    = 0;

    while (line[i] != '\0' && opos < maxlen - 1)
    {
        if (line[i] == '{')
        {
            /* Try {N..M} or {N..M..S} */
            char *endp = NULL;
            long  sv   = strtol(line + i + 1, &endp, 10);
            if (endp != NULL && endp[0] == '.' && endp[1] == '.')
            {
                char *endp2 = NULL;
                long  ev    = strtol(endp + 2, &endp2, 10);
                long  step  = 1;
                if (endp2 != NULL && endp2[0] == '.' && endp2[1] == '.')
                {
                    char *endp3 = NULL;
                    step        = strtol(endp2 + 2, &endp3, 10);
                    endp2       = endp3;
                }
                if (endp2 != NULL && *endp2 == '}' && step != 0)
                {
                    int first = 1;
                    if (sv <= ev)
                    {
                        if (step < 0)
                        {
                            step = -step;
                        }
                        for (long v = sv; v <= ev; v += step)
                        {
                            if (opos >= maxlen - 1)
                            {
                                break;
                            }
                            char nb[32];
                            snprintf(nb, sizeof(nb), "%s%ld", first ? "" : " ", v);
                            first = 0;
                            emit_str(out, &opos, maxlen, nb);
                        }
                    }
                    else
                    {
                        if (step > 0)
                        {
                            step = -step;
                        }
                        for (long v = sv; v >= ev; v += step)
                        {
                            if (opos >= maxlen - 1)
                            {
                                break;
                            }
                            char nb[32];
                            snprintf(nb, sizeof(nb), "%s%ld", first ? "" : " ", v);
                            first = 0;
                            emit_str(out, &opos, maxlen, nb);
                        }
                    }
                    i = (int) (endp2 - line) + 1;
                    continue;
                }
            }
            if (opos < maxlen - 1)
            {
                out[opos++] = line[i++];
            }
            else
            {
                break;
            }
        }
        else
        {
            if (opos < maxlen - 1)
            {
                out[opos++] = line[i++];
            }
            else
            {
                break;
            }
        }
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}
