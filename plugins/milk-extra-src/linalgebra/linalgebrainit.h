// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file linalgebrainit.h
 */

#ifdef HAVE_CUDA

errno_t linalgebrainit_addCLIcmd();

int LINALGEBRA_init();

void *GPU_scanDevices(void *deviceCount_void_ptr);

#endif
