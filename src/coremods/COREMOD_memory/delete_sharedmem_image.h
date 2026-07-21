// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file delete_sharedmem_image.h
 * @brief Delete sharedmem image module
 */

#ifndef MILK_COREMOD_MEMORY_DELETE_SHAREDMEM_IMAGE_H
#define MILK_COREMOD_MEMORY_DELETE_SHAREDMEM_IMAGE_H

errno_t CLIADDCMD_COREMOD_memory__delete_sharedmem_image();

errno_t destroy_shared_image_ID(const char *__restrict imname);

#endif
