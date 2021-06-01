/**
 * @file    loadfits.h
 */


#ifndef MILK_COREMOD_IOFIT_LOADFITS_H
#define MILK_COREMOD_IOFIT_LOADFITS_H

#define LOADFITS_ERRCODE_IGNORE  0
#define LOADFITS_ERRCODE_WARNING 1
#define LOADFITS_ERRCODE_EXIT    2
#define LOADFITS_ERRCODE_RETRY   3

errno_t CLIADDCMD_COREMOD_iofits__loadfits();

imageID load_fits(
    const char *restrict file_name,
    const char *restrict ID_name,
    int         errcode
);

#endif
