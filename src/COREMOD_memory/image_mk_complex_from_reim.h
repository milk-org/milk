// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_REIM_H
#define COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_REIM_H

errno_t mk_complex_from_reim(const char *re_name,
                             const char *im_name,
                             const char *out_name,
                             int         sharedmem);

errno_t mk_complex_from_reim_IMGID(IMGID *imgre,
                                   IMGID *imgim,
                                   IMGID *imgout);

errno_t CLIADDCMD_COREMOD__mk_complex_from_reim();

#endif
