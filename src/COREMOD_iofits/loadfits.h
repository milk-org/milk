/**
 * @file    loadfits.h
 */


errno_t loadfits_addCLIcmd();


imageID load_fits(
    const char *restrict file_name,
    const char *restrict ID_name,
    int         errcode
);
