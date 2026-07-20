// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef COREMOD_MEMORY_IMAGE_KEYWORD_ADDS_H
#define COREMOD_MEMORY_IMAGE_KEYWORD_ADDS_H

errno_t image_keyword_addS(IMGID img, char *kwname, char *kwval, char *comment);

errno_t CLIADDCMD_COREMOD_memory__image_keyword_addS();

#endif
