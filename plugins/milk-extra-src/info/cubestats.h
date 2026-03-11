/**
 * @file cubestats.h
 * @brief Image cube stats
 */

errno_t CLIADDCMD_info__cubestats();

imageID info_cubestats(
    const char *ID_name,
    const char *IDmask_name,
    const char *outfname);
