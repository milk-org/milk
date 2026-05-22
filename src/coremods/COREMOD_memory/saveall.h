/**
 * @file saveall.h
 * @brief Save all images/streams to disk
 */

errno_t CLIADDCMD_COREMOD_memory__saveall();

errno_t COREMOD_MEMORY_SaveAll_snapshot(const char *dirname);

errno_t COREMOD_MEMORY_SaveAll_sequ(const char *dirname,
                                    const char *IDtrig_name,
                                    long        semtrig,
                                    long        NBframes);
