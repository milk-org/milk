// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef COREMOD_ARITH_IMAGE_NORM_H
#define COREMOD_ARITH_IMAGE_NORM_H

errno_t CLIADDCMD_COREMOD_arith__image_normslice();

errno_t image_slicenorm(const char *inname, const char *outname, uint8_t sliceaxis);
errno_t image_slicenorm_IMGID(IMGID *inimg, IMGID *outimg, uint8_t sliceaxis);

#endif
