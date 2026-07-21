// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imstretch.h
 * @brief Imstretch module
 */

/** @file imstretch.h
 */

imageID basic_stretch(const char *__restrict name_in,
                      const char *__restrict name_out,
                      float coeff,
                      long  Xcenter,
                      long  Ycenter);

imageID basic_stretch_range(const char *__restrict name_in,
                            const char *__restrict name_out,
                            float coeff1,
                            float coeff2,
                            long  Xcenter,
                            long  Ycenter,
                            long  NBstep,
                            float ApoCoeff);

imageID basic_stretchc(const char *__restrict name_in,
                       const char *__restrict name_out,
                       float coeff);
