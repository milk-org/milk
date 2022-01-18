/** @file saveall.h
 */

errno_t saveall_addCLIcmd();

errno_t COREMOD_MEMORY_SaveAll_snapshot(const char *dirname);

errno_t COREMOD_MEMORY_SaveAll_sequ(const char *dirname, const char *IDtrig_name, long semtrig, long NBframes);
