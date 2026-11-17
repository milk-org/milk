// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_processinfo.c
 * @brief   ProcessInfo integration helpers for FPS
 */

#include <sys/mman.h>

#include "computeunit.h"

/**
 * @brief Write CTRLval @signal_val into the processinfo of the FPS's run process.
 *
 * Only that process's proc.<name>.<pid>.shm, in the directory it uses (run.info), is written:
 * other processes whose name starts like the FPS's are not touched.
 */
errno_t functionparameter_FPS_processinfo_signal(const char *fps_name, int signal_val)
{
    FPS_UNIT_INFO info;
    char          path[PATH_MAX + 256];
    int           fd;

    if (fps_unit_info_read(fps_name, FPS_UNIT_RUN, &info) != 0 || !milkdetach_id_alive(&info.id) ||
        fps_unit_proc_path(fps_name, &info, path, sizeof(path)) != 0)
    {
        return RETURN_FAILURE;
    }
    PROCESSINFO *pinfo = processinfo_shm_link(path, &fd);
    if (pinfo == (PROCESSINFO *) MAP_FAILED)
    {
        return RETURN_FAILURE;
    }
    pinfo->CTRLval = signal_val;
    processinfo_shm_close(pinfo, fd);
    return RETURN_SUCCESS;
}
