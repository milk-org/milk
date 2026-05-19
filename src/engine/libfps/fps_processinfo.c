/**
 * @file    fps_processinfo.c
 * @brief   ProcessInfo integration helpers for FPS
 */

#include <dirent.h>
#include <sys/mman.h>

#include "fps.h"

// Prototypes for functions in libprocessinfo if not in processinfo.h
// Usually they are in separate headers or processinfo.h
// Using implicit declaration or assuming headers are included via processinfo.h
// Checking prior includes in processor.c:
// #include "processinfo_shm_link.h"
// #include "processinfo_procdirname.h"
// I should include them here too if available in include path.

#include "processinfo_procdirname.h"

/**
 * @brief Send a signal to the process managing an FPS.
 *
 * Looks up the FPS by name, reads its run PID,
 * and sends the specified signal.
 */
errno_t functionparameter_FPS_processinfo_signal(const char *fps_name, int signal_val) {
    char procdname[STRINGMAXLEN_DIR_NAME];
    processinfo_procdirname(procdname);

    DIR *d = opendir(procdname);
    if (!d) return RETURN_FAILURE;

    struct dirent *dir;
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "proc.%s.", fps_name);

    while ((dir = readdir(d)) != NULL) {
        if (strncmp(dir->d_name, prefix, strlen(prefix)) == 0) {
            char fullpath[2048];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", procdname, dir->d_name);
            int fd;
            PROCESSINFO *pinfo = processinfo_shm_link(fullpath, &fd);
            if (pinfo != (PROCESSINFO *)MAP_FAILED) {
                pinfo->CTRLval = signal_val;
                munmap(pinfo, sizeof(PROCESSINFO));
                close(fd);
            }
        }
    }
    closedir(d);
    return RETURN_SUCCESS;
}
