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


// List of arguments to function
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STR_NOT_IMG, ".in_sname", "input stream", "ims1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &insname
    }
};




// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata =
{
    "readshmim",
    "read shared memory image",
    CLICMD_FIELDS_DEFAULTS
};






imageID read_sharedmem_image(
    const char *sname
)
{
    imageID ID = -1;
    imageID IDmem = 0;
    IMAGE *image;

    IDmem = next_avail_image_ID();

    image = &data.image[IDmem];
    if(ImageStreamIO_read_sharedmem_image_toIMAGE(sname, image) == EXIT_FAILURE)
    {
        printf("read shared mem image failed -> ID = -1\n");
        fflush(stdout);
        ID = -1;
    }
    else
    {
        ID = image_ID(sname);
        printf("read shared mem image success -> ID = %ld\n", ID);
        fflush(stdout);

        if(ID != -1)
        {
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
    }

    if(data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }

    return ID;
}



// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    read_sharedmem_image(insname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions

errno_t CLIADDCMD_COREMOD_memory__read_sharedmem_image()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}



