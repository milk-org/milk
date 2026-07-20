// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef _PROCESSINFO_UPDATE_OUTPUT_STREAM_H
#define _PROCESSINFO_UPDATE_OUTPUT_STREAM_H

errno_t processinfo_update_output_stream(PROCESSINFO *processinfo, imageID outstreamID);

errno_t processinfo_update_output_stream_atime(PROCESSINFO     *processinfo,
                                               imageID          outstreamID,
                                               struct timespec *atime);

#endif
