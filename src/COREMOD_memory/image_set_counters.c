/**
 * @file    image_set_couters.c
 * @brief   SET IMAGE FLAGS / COUNTERS        
 */

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"



errno_t COREMOD_MEMORY_image_set_status(
    const char *IDname,
    int         status
)
{
    imageID ID;

    ID = image_ID(IDname);
    data.image[ID].md[0].status = status;

    return RETURN_SUCCESS;
}


errno_t COREMOD_MEMORY_image_set_cnt0(
    const char *IDname,
    int         cnt0
)
{
    imageID ID;

    ID = image_ID(IDname);
    data.image[ID].md[0].cnt0 = cnt0;

    return RETURN_SUCCESS;
}


errno_t COREMOD_MEMORY_image_set_cnt1(
    const char *IDname,
    int         cnt1
)
{
    imageID ID;

    ID = image_ID(IDname);
    data.image[ID].md[0].cnt1 = cnt1;

    return RETURN_SUCCESS;
}



