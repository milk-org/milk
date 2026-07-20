// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file stream_updateloop.h
 */

errno_t stream_updateloop_addCLIcmd();

imageID COREMOD_MEMORY_image_streamupdateloop(const char *IDinname,
        const char *IDoutname,
        long        usperiod,
        long        NBcubes,
        long        period,
        long        offsetus,
        const char *IDsync_name,
        int         semtrig,
        int         timingmode);

imageID COREMOD_MEMORY_image_streamupdateloop_semtrig(const char *IDinname,
        const char *IDoutname,
        long        period,
        long        offsetus,
        const char *IDsync_name,
        int         semtrig,
        int         timingmode);
