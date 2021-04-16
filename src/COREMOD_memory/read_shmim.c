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





// ==========================================
// forward declaration
// ==========================================

imageID read_sharedmem_image_size(
    const char *name,
    const char *fname
);

imageID read_sharedmem_image(
    const char *name
);


// ==========================================
// command line interface wrapper functions
// ==========================================



static errno_t read_sharedmem_image_size__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            == 0)
    {

        read_sharedmem_image_size(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




static errno_t read_sharedmem_image__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            == 0)
    {

        read_sharedmem_image(
            data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}








// ==========================================
// Register CLI command(s)
// ==========================================

errno_t read_shmim_addCLIcmd()
{
    RegisterCLIcommand(
        "readshmimsize",
        __FILE__,
        read_sharedmem_image_size__cli,
        "read shared memory image size",
        "<name> <output file>",
        "readshmimsize im1 imsize.txt",
        "read_sharedmem_image_size(const char *name, const char *fname)");

    RegisterCLIcommand(
        "readshmim",
        __FILE__, read_sharedmem_image__cli,
        "read shared memory image",
        "<name>",
        "readshmim im1",
        "read_sharedmem_image(const char *name)");

    return RETURN_SUCCESS;
}







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

        printf("%d keywords\n", (int) data.image[ID].md[0].NBkw);
        for(int kw = 0; kw < data.image[ID].md[0].NBkw; kw++)
        {
            if(data.image[ID].kw[kw].type != 'N')
            {
                printf("%3d %c %16s  ",
                       kw,
                       data.image[ID].kw[kw].type,
                       data.image[ID].kw[kw].name
                      );

                switch(data.image[ID].kw[kw].type)
                {
                    case 'S' : // string
                        printf("%16s", data.image[ID].kw[kw].value.valstr);
                        break;
                    case 'L' : // string
                        printf("%16ld", data.image[ID].kw[kw].value.numl);
                        break;
                    case 'D' : // string
                        printf("%16f", data.image[ID].kw[kw].value.numf);
                        break;
                    default : // unknown
                        printf("== DATA TYPE NOT RECOGNIZED --");
                    break;
                }

                printf("   %s\n", data.image[ID].kw[kw].comment);
            }
        }
    }

    if(data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }

    return ID;
}


