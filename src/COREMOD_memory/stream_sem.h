/**
 * @file    stream_sem.h
 */

errno_t stream_sem_addCLIcmd();

imageID COREMOD_MEMORY_image_set_createsem(const char *IDname, long NBsem);

imageID COREMOD_MEMORY_image_seminfo(const char *IDname);

imageID COREMOD_MEMORY_image_set_sempost(const char *IDname, long index);

imageID COREMOD_MEMORY_image_set_sempost_byID(imageID ID, long index);

imageID COREMOD_MEMORY_image_set_sempost_excl_byID(imageID ID, long index);

imageID COREMOD_MEMORY_image_set_sempost_loop(const char *IDname, long index, long dtus);

imageID COREMOD_MEMORY_image_set_semwait(const char *IDname, long index);

void *waitforsemID(void *ID);

errno_t COREMOD_MEMORY_image_set_semwait_OR_IDarray(imageID *IDarray, long NB_ID);

errno_t COREMOD_MEMORY_image_set_semflush_IDarray(imageID *IDarray, long NB_ID);

imageID COREMOD_MEMORY_image_set_semflush(const char *IDname, long index);
