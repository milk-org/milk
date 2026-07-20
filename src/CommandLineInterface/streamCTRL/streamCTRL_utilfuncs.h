// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef _STREAMCTRL_UTILS_H
#define _STREAMCTRL_UTILS_H


imageID image_ID_from_images(IMAGE *images, const char *__restrict name);


imageID image_get_first_ID_available_from_images(IMAGE *images);


errno_t get_process_name_by_pid(const int pid, char *pname);


int get_PIDmax();


#endif
