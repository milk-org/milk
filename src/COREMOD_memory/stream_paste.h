/**
 * @file stream_paste.h
 */

errno_t stream_paste_addCLIcmd();

imageID COREMOD_MEMORY_streamPaste(const char *IDstream0_name, const char *IDstream1_name, const char *IDstreamout_name,
                                   long semtrig0, long semtrig1, int master);
