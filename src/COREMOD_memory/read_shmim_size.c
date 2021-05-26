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




// Local variables pointers
static char *insname;
static char *outfname;


// List of arguments to function
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STR, ".in_sname", "input stream", "ims1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &insname
    },
    {
        CLIARG_STR_NOT_IMG, ".outfname", "output file name", "outsize.dat",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outfname
    }
};




// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata =
{
    "readshmimsize",
    "read shared memory image size",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
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
    char            SM_fname[STRINGMAXLEN_FULLFILENAME];
    IMAGE_METADATA *map;
    int             i;
    FILE           *fp;
    imageID         ID = -1;


    if((ID = image_ID(name)) == -1)
    {
        WRITE_FULLFILENAME(SM_fname, "%s/%s.im.shm", data.shmdir, name);

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









// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    read_sharedmem_image_size(
        insname,
        outfname
    );

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_COREMOD_memory__read_sharedmem_image_size()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
