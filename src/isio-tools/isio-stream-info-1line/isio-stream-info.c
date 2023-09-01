/*
 * Open stream(s), print basic info within single line
 *
 */


#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"



int main(int argc, char *argv[])
{

    for (int i=1; i< argc; i++)
    {
        if(strlen(argv[i]) != 0)
        {
            IMAGE image;

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
