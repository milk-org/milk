/**
 * @file    multiselect_parse.h
 * @brief   Parse multi-selection input (numbers, ranges)
 *
 * Accepts input like "1 3 5-7 all" and sets the
 * corresponding flags in a boolean array.
 *
 * Supported syntax:
 *   - Single numbers: "3"
 *   - Ranges: "5-7"  (inclusive)
 *   - Mixed: "1 3 5-7 10"
 *   - Comma or space separated: "1,3,5-7"
 *   - "all" or "a": select everything
 *   - "0" or empty: cancel
 *
 * Returns: number of selected items, or -1 to cancel.
 */

#ifndef MULTISELECT_PARSE_H
#define MULTISELECT_PARSE_H

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/**
 * parse_multiselect() - Parse selection string
 * @input:    user input string
 * @selected: boolean array [0..count-1], set to 1
 *            for selected items
 * @count:    total number of items
 *
 * Returns number of items selected, or -1 on cancel.
 */
static inline int parse_multiselect(
    const char *input,
    int        *selected,
    int        count)
{
    /* Clear selection array */
    memset(selected, 0,
           count * sizeof(int));

    /* Skip leading whitespace */
    while(*input && isspace(*input))
    {
        input++;
    }

    /* Empty input => cancel */
    if(*input == '\0')
    {
        return -1;
    }

    /* "0" => cancel */
    if(strcmp(input, "0") == 0)
    {
        return -1;
    }

    /* "all" or "a" => select everything */
    if(strcasecmp(input, "all") == 0 ||
            strcasecmp(input, "a") == 0)
    {
        for(int i = 0; i < count; i++)
        {
            selected[i] = 1;
        }
        return count;
    }

    /* Parse tokens separated by spaces/commas */
    char buf[512];

    strncpy(buf, input, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    int nsel = 0;
    char *saveptr = NULL;
    char *tok = strtok_r(buf, " ,\t", &saveptr);

    while(tok != NULL)
    {
        /* Check for range: "N-M" */
        char *dash = strchr(tok, '-');

        if(dash != NULL && dash != tok)
        {
            *dash = '\0';
            int lo = atoi(tok);
            int hi = atoi(dash + 1);

            if(lo < 1)
            {
                lo = 1;
            }
            if(hi > count)
            {
                hi = count;
            }
            for(int n = lo; n <= hi; n++)
            {
                if(!selected[n - 1])
                {
                    selected[n - 1] = 1;
                    nsel++;
                }
            }
        }
        else
        {
            int n = atoi(tok);

            if(n >= 1 && n <= count)
            {
                if(!selected[n - 1])
                {
                    selected[n - 1] = 1;
                    nsel++;
                }
            }
        }

        tok = strtok_r(NULL, " ,\t", &saveptr);
    }

    if(nsel == 0)
    {
        return -1;
    }

    return nsel;
}

#endif /* MULTISELECT_PARSE_H */
