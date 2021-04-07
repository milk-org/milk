/**
 * @file    loadfits.h
 */


#ifndef MILK_COREMOD_IOFIT_LOADFITS_H
#define MILK_COREMOD_IOFIT_LOADFITS_H

errno_t CLIADDCMD_loadfits();


imageID load_fits(
    const char *restrict file_name,
    const char *restrict ID_name,
    int         errcode
);

#endif
