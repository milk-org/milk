// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file linalgebratest.h
 */

#ifdef HAVE_CUDA

errno_t linalgebratest_addCLIcmd();

errno_t GPUcomp_test(__attribute__((unused)) long NBact, long NBmodes, long WFSsize, long GPUcnt);

#endif
