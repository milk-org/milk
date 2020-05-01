/**
 * @file    image_copy.h
 */




errno_t image_copy_addCLIcmd();


imageID copy_image_ID(
    const char *name,
    const char *newname,
    int         shared
);

imageID chname_image_ID(
    const char *ID_name,
    const char *new_name
);

errno_t COREMOD_MEMORY_cp2shm(
    const char *IDname,
    const char *IDshmname
);
