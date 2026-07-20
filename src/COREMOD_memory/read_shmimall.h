// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    read_shmimall.h
 */

#ifndef _READ_SHMIMALL_H
#define _READ_SHMIMALL_H

errno_t read_shmimall_addCLIcmd();

imageID read_sharedmem_image_all(const char *strfilter);

#endif
