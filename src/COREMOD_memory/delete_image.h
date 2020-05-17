/**
 * @file    delete_image.h
 */


errno_t delete_image_addCLIcmd();



errno_t    delete_image_ID(
    const char *imname
);

errno_t    delete_image_ID_prefix(
    const char *prefix
);

errno_t destroy_shared_image_ID(
    const char *__restrict__ imname
);
