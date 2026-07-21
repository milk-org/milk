// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tableto2Dim.h
 * @brief Tableto2dim module
 */

/** @file tableto2Dim.h
 */

imageID basic_tableto2Dim(const char *__restrict fname,
                          float xmin,
                          float xmax,
                          float ymin,
                          float ymax,
                          long  xsize,
                          long  ysize,
                          const char *__restrict ID_name,
                          float convsize);
