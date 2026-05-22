/**
 * @file scheduler_display.h
 * @brief Scheduler display module
 */

#ifndef FPS_CTRLSCREEN_SCHEDULER_DISPLAY_H
#define FPS_CTRLSCREEN_SCHEDULER_DISPLAY_H

#include <errno.h>
#include "fps_types.h"

errno_t fpsCTRL_scheduler_display(FPSCTRL_PROCESS_VARS *fpsCTRLvar, int wrow, int *wrowstart);

#endif
