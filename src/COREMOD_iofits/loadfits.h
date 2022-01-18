/**
 * @file    loadfits.h
 */

#ifndef MILK_COREMOD_IOFIT_LOADFITS_H
#define MILK_COREMOD_IOFIT_LOADFITS_H

#define LOADFITS_ERRMODE_IGNORE 0
#define LOADFITS_ERRMODE_WARNING 1
#define LOADFITS_ERRMODE_ERROR 2
#define LOADFITS_ERRMODE_EXIT 3

errno_t CLIADDCMD_COREMOD_iofits__loadfits();

errno_t load_fits(const char *restrict file_name, const char *restrict ID_name, int errmode, imageID *ID);

#endif
