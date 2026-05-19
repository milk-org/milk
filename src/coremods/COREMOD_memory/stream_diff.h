/**
 * @file stream_diff.h
 * @brief Compute stream difference
 */

errno_t CLIADDCMD_COREMOD_memory__stream_diff();

imageID COREMOD_MEMORY_streamDiff(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreammask_name,
    const char *IDstreamout_name,
    long       semtrig);
