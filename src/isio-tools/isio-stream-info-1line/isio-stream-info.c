/*
 * Open stream(s), print basic info within single line
 *
 */


#define _GNU_SOURCE
#include <string.h>

#include <sys/stat.h>
#include <dirent.h>


#include "ImageStreamIO/ImageStreamIO.h"



int main(int argc, char *argv[])
{
    char *shmdirname = getenv("MILK_SHM_DIR");
    if(shmdirname != NULL)
    {
        // does this direcory exist ?
        DIR *tmpdir;
        tmpdir = opendir(shmdirname);
        if(tmpdir)  // directory exits
        {
            closedir(tmpdir);
        }
        else
        {
            printf("ERROR: cannot open shared memory directory \"%s\"\n", shmdirname);
            return 1;
        }
    }



    for (int i=1; i< argc; i++)
    {
        if(strlen(argv[i]) != 0)
        {
            IMAGE image;


            {
                struct stat buf;

                char fname[STRINGMAXLEN_FILE_NAME];
                snprintf(fname, STRINGMAXLEN_FILE_NAME, "%s/%s.im.shm", shmdirname, argv[i]);
                int retv = lstat(fname, &buf);
                if(retv == -1)
                {
                    printf("ERROR: Cannot read file \"%s\"\n", fname);
                    return 1;
                }

                if(S_ISLNK(buf.st_mode))  // resolve link name
                {
                    char *linknamefull;
                    char  linkname[STRINGMAXLEN_FILE_NAME];
                    int   pathOK = 1;

                    linknamefull = realpath(fname, NULL);

                    if(linknamefull == NULL)
                    {
                        pathOK = 0;
                    }
                    else if(access(linknamefull, R_OK))
                    {
                        // file cannot be read
                        pathOK = 0;
                    }

                    if(pathOK == 0)
                    {
                        // file cannot be read
                        printf("%16s   ERROR: Cannot read link target\n", argv[i]);
                        return 0;
                    }

                    {
                        strcpy(linkname, basename(linknamefull));

                        int          lOK = 1;
                        unsigned int ii  = 0;
                        while((lOK == 1) && (ii < strlen(linkname)))
                        {
                            if(linkname[ii] == '.')
                            {
                                linkname[ii] = '\0';
                                lOK          = 0;
                            }
                            ii++;
                        }
                        printf("%16s  LINK to %s\n", argv[i], linkname);
                        return 0;
                    }

                    if(linknamefull != NULL)
                    {
                        free(linknamefull);
                    }
                }

            }


            if(ImageStreamIO_read_sharedmem_image_toIMAGE(argv[i], &image) ==
                    IMAGESTREAMIO_SUCCESS)
            {
                printf("%16s  %12ld  %d   %d [ %4d %4d %4d ]\n",
                       argv[i],
                       image.md->cnt0,
                       image.md->datatype,
                       image.md->naxis,
                       image.md->size[0],
                       image.md->size[1],
                       image.md->size[2]
                      );

                ImageStreamIO_closeIm(&image);
            }
            else
            {
                printf("\033[1;31m");
                printf("%16s         READ ERROR\n", argv[i]);
                printf("\033[0m");
            }
        }
    }

    return 0;
}
