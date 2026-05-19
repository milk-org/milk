/**
 * @file image_copy.h
 * @brief Image copy, rename, copy to shm
 */

errno_t
CLIADDCMD_COREMOD_memory__image_copy();

imageID copy_image_ID(
    const char *name,
    const char *newname,
    int        shared);
imageID copy_image_ID_IMGID(
    IMGID *imgin,
    IMGID *imgout,
    int   shared);

imageID chname_image_ID(
    const char *ID_name,
    const char *new_name);
imageID chname_image_ID_IMGID(
    IMGID      *imgin,
    const char *new_name);

errno_t COREMOD_MEMORY_cp2shm(
    const char *IDname,
    const char *IDshmname);
errno_t COREMOD_MEMORY_cp2shm_IMGID(
    IMGID *imgin,
    IMGID *imgout);
