/**
 * @file    fps_print_info.h
 * @brief   Print content of a Function Parameter Structure (FPS)
 */

#ifndef _FPS_PRINT_INFO_H
#define _FPS_PRINT_INFO_H

#include "fps.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Print content of an FPS to stdout
 * 
 * @param fps Pointer to the FPS structure
 * @param verbose Verbose mode (shows workdir, source file, keywords)
 * @param show_info Show detailed stream information
 * @return int 0 on success, -1 on error
 */
int function_parameter_print_info(
    FUNCTION_PARAMETER_STRUCT *fps,
    int verbose,
    int show_info
);

#ifdef __cplusplus
}
#endif

#endif /* _FPS_PRINT_INFO_H */
