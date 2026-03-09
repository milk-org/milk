/**
 * @file    fps_isvalid.c
 * @brief   Check if FPS is valid
 */

#include <sys/stat.h>

#include "fps.h"
#include "fps_isvalid.h"

/**
 * @brief Check if a connected FPS struct is still valid
 * 
 * An FPS might become invalid if it is removed externally (e.g., by milk-fps-rm).
 * This function quickly checks if the shared memory file descriptor is still valid
 * and if the file still exists on disk.
 * 
 * @param fps Pointer to FUNCTION_PARAMETER_STRUCT
 * @return int 1 if valid, 0 if invalid
 */
int function_parameter_struct_isvalid(FUNCTION_PARAMETER_STRUCT *fps)
{
    if (fps == NULL) {
        return 0;
    }

    if (fps->SMfd < 0) {
        return 0; // Not connected
    }

    if (fps->md == NULL) {
        return 0; // No mapped memory
    }

    // Check if the file still exists by calling fstat
    // If the file was unlinked, fstat on the open fd succeeds,
    // but st_nlink will be 0.
    struct stat file_stat;
    if (fstat(fps->SMfd, &file_stat) == 0) {
        if (file_stat.st_nlink == 0) {
            return 0; // File was deleted
        }
    } else {
        return 0; // fstat failed, assume invalid
    }

    return 1;
}
