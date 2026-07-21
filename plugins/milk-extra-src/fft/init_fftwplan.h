// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file init_fftwplan.h
 * @brief Initialize FFTW plans
 */

errno_t CLIADDCMD_milkfft__init_fftwplan();

errno_t init_fftw_plans(int mode);
errno_t init_fftw_plans0();
