/**
 * @file    delete_image.c
 * @brief   delete image(s)
 */

#include <malloc.h>
#include <sys/mman.h>


#include "CommandLineInterface/CLIcore.h"

#include "image_ID.h"
#include "list_image.h"



// CLI function arguments and parameters
static char *imname;




// CLI function arguments and parameters
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, ".imname", "image name", "im",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imname
    }
};



// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "rmshmim",
    "remove shared image and files",
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg,
    CLICMDFLAG_FPS,
    NULL
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




errno_t destroy_shared_image_ID(
    const char *__restrict__ imname
)
{
    imageID ID;

    ID = image_ID(imname);
    if((ID != -1) && (data.image[ID].md[0].shared == 1))
    {
        ImageStreamIO_destroyIm(&data.image[ID]);
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm WARNING: shared image %s does not exist [ %s  %s  %d ] %c[%d;m\n",
                (char) 27, 1, 31, imname, __FILE__, __func__, __LINE__, (char) 27, 0);
    }

    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    errno_t ret = 0;

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    ret = destroy_shared_image_ID(
              imname
          );

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return ret;
}





INSERT_STD_FPSCLIfunctions


// Register function in CLI
errno_t CLIADDCMD_COREMOD_memory__delete_sharedmem_image()
{
    //INSERT_STD_FPSCLIREGISTERFUNC

    int cmdi = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
