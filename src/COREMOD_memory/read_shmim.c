/**
 * @file    read_shmim.c
 * @brief   read shared memory stream
 */

#include <sys/stat.h>
#include <fcntl.h> // open
#include <unistd.h> // close
#include <sys/mman.h> // mmap

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "list_image.h"


/**
 *  ## Purpose
 *
 *  Read shared memory image size
 *
 *
 * ## Arguments
 *
 * @param[in]
 * name		char*
 * -		stream name
 *
 * @param[in]
 * fname	char*
 * 			file name to write image name
 *
 */
imageID read_sharedmem_image_size(
    const char *name,
    const char *fname
)
{
    int             SM_fd;
    struct          stat file_stat;
    char            SM_fname[200];
    IMAGE_METADATA *map;
    int             i;
    FILE           *fp;
    imageID         ID = -1;


    if((ID = image_ID(name)) == -1)
    {
        sprintf(SM_fname, "%s/%s.im.shm", data.shmdir, name);

        SM_fd = open(SM_fname, O_RDWR);
        if(SM_fd == -1)
        {
            printf("Cannot import file - continuing\n");
        }
        else
        {
            fstat(SM_fd, &file_stat);
            //        printf("File %s size: %zd\n", SM_fname, file_stat.st_size);

            map = (IMAGE_METADATA *) mmap(0, sizeof(IMAGE_METADATA), PROT_READ | PROT_WRITE,
                                          MAP_SHARED, SM_fd, 0);
            if(map == MAP_FAILED)
            {
                close(SM_fd);
                perror("Error mmapping the file");
                exit(0);
            }

            fp = fopen(fname, "w");
            for(i = 0; i < map[0].naxis; i++)
            {
                fprintf(fp, "%ld ", (long) map[0].size[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);


            if(munmap(map, sizeof(IMAGE_METADATA)) == -1)
            {
                printf("unmapping %s\n", SM_fname);
                perror("Error un-mmapping the file");
            }
            close(SM_fd);
        }
    }
    else
    {
        fp = fopen(fname, "w");
        for(i = 0; i < data.image[ID].md[0].naxis; i++)
        {
            fprintf(fp, "%ld ", (long) data.image[ID].md[0].size[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }

    return ID;
}






imageID read_sharedmem_image(
    const char *name
)
{
    imageID ID = -1;
    imageID IDmem = 0;
    IMAGE *image;

    IDmem = next_avail_image_ID();

    image = &data.image[IDmem];
    if(ImageStreamIO_read_sharedmem_image_toIMAGE(name, image) == EXIT_FAILURE)
    {
        printf("read shared mem image failed -> ID = -1\n");
        fflush(stdout);
        ID = -1;
    }
    else
    {
        ID = image_ID(name);
        printf("read shared mem image success -> ID = %ld\n", ID);
        fflush(stdout);
    }

    if(data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }

    return ID;
}


