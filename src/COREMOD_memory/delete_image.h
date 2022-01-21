#ifndef MILK_COREMOD_MEMORY_DELETE_IMAGE_H
#define MILK_COREMOD_MEMORY_DELETE_IMAGE_H

#define DELETE_IMAGE_ERRMODE_IGNORE  0
#define DELETE_IMAGE_ERRMODE_WARNING 1
#define DELETE_IMAGE_ERRMODE_ERROR   2
#define DELETE_IMAGE_ERRMODE_EXIT    3

errno_t CLIADDCMD_COREMOD_memory__delete_image();

errno_t delete_image_ID(const char *imname, int errmode);

errno_t delete_image_ID_prefix(const char *prefix);

#endif
