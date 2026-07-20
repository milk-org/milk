// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    loadfits.h
 */

#ifndef MILK_COREMOD_IOFIT_LOADFITS_H
#define MILK_COREMOD_IOFIT_LOADFITS_H

#define LOADFITS_ERRMODE_IGNORE  0
#define LOADFITS_ERRMODE_WARNING 1
#define LOADFITS_ERRMODE_ERROR   2
#define LOADFITS_ERRMODE_EXIT    3

errno_t CLIADDCMD_COREMOD_iofits__loadfits();

errno_t load_fits(const char *restrict file_name,
                  const char *restrict ID_name,
                  int      errmode,
                  imageID *ID);

errno_t load_fits_IMGID(const char *restrict file_name,
                        IMGID *imgout,
                        int errmode);

#endif
