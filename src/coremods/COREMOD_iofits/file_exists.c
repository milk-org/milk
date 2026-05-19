/**
 * @file file_exists.c
 * @brief File exists module
 */

/**
 * @file    file_exists.c
 */

#include <stdio.h>

/**
 * @brief Check if a file exists on disk.
 *
 * Returns 1 if accessible, 0 otherwise.
 */
int file_exists(const char *restrict file_name)
{
    FILE *fp;
    int   exists = 1;

    if((fp = fopen(file_name, "r")) == NULL)
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
