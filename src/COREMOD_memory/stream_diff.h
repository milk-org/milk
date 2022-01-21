/**
 * @file    stream_diff.h
 */

errno_t stream_diff_addCLIcmd();

imageID COREMOD_MEMORY_streamDiff(const char *IDstream0_name,
                                  const char *IDstream1_name,
                                  const char *IDstreammask_name,
                                  const char *IDstreamout_name,
                                  long        semtrig);
