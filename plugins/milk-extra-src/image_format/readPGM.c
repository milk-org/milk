/** @file readPGM.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

/**
 * ## Purpose
 *
 *  reads PGM images (16 bit only)
 *
 * @note written to read output of "dcraw -t 0 -D -4 xxx.CR2" into FITS
 */
imageID read_PGMimage(const char *__restrict fname,
                      const char *__restrict ID_name)
{
    FILE   *fp;
    imageID ID;

    if((fp = fopen(fname, "r")) == NULL)
    {
        fprintf(stderr, "ERROR: cannot open file \"%s\"\n", fname);
        ID = -1;
    }
    else
    {
        char   line1[100];
        long   xsize, ysize;
        long   maxval;
        long   ii, jj;
        double val;

        {
            int fscanfcnt = fscanf(fp, "%s", line1);
            if(fscanfcnt == EOF)
            {
                if(ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr,
                            "Error: fscanf reached end of file, no matching "
                            "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if(fscanfcnt != 1)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i "
                        "input items, 1 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }
        }

        if(strcmp(line1, "P5") != 0)
        {
            fprintf(stderr, "ERROR: File is not PGM image\n");
        }
        else
        {
            int fscanfcnt = fscanf(fp, "%ld %ld", &xsize, &ysize);
            if(fscanfcnt == EOF)
            {
                if(ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr,
                            "Error: fscanf reached end of file, no matching "
                            "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if(fscanfcnt != 2)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i "
                        "input items, 2 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }

            printf("PGM image size: %ld x %ld\n", xsize, ysize);

            fscanfcnt = fscanf(fp, "%ld", &maxval);
            if(fscanfcnt == EOF)
            {
                if(ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr,
                            "Error: fscanf reached end of file, no matching "
                            "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if(fscanfcnt != 1)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i "
                        "input items, 1 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }

            if(maxval != 65535)
            {
                fprintf(stderr, "Not 16-bit image. Cannot read\n");
            }
            else
            {
                printf("Reading PGM image\n");
                create_2Dimage_ID(ID_name, xsize, ysize, &ID);
                fgetc(fp);
                for(jj = 0; jj < ysize; jj++)
                {
                    for(ii = 0; ii < xsize; ii++)
                    {
                        val =
                            256.0 * ((int) fgetc(fp)) + 1.0 * ((int) fgetc(fp));
                        data.image[ID].array.F[(ysize - jj - 1) * xsize + ii] =
                            val;
                    }
                }
            }
        }
        fclose(fp);
    }

    return ID;
}
