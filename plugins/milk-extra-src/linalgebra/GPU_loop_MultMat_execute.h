// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file GPU_loop_MultMat_execute.h
 */

#ifdef HAVE_CUDA

int GPU_loop_MultMat_execute(int   index,
                             int  *status,
                             int  *GPUstatus,
                             float alpha,
                             float beta,
                             int   timing,
                             int   TimerOffsetIndex);

#endif
