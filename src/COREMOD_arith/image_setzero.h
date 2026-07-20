// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef COREMOD_MODULE_ARITH_IMSETZERO_H
#define COREMOD_MODULE_ARITH_IMSETZERO_H


errno_t image_setzero(IMGID inimg);
errno_t image_setzero_IMGID(IMGID *inimg);

errno_t CLIADDCMD_COREMOD_arith__imsetzero();

#endif
