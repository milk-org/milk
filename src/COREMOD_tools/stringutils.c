/**
 * @file stringutils.c
 */

#include <string.h>

int replace_char(char *content, char cin, char cout)
{
    unsigned long i;

    for (i = 0; i < strlen(content); i++)
        if (content[i] == cin)
        {
            content[i] = cout;
        }

    return (0);
}
