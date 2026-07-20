// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file CR2toFITS.h
 */

errno_t CR2toFITS_addCLIcmd();

imageID CR2toFITS(const char *__restrict fnameCR2, const char *__restrict fnameFITS);
