/**
 * @file fps_read_fpsCMD_fifo.h
 * @brief Fps read fpscmd fifo module
 */

/**
 * @file    fps_read_fpsCMD_fifo.h
 */

#ifndef FPS_READ_FPSCMD_FIFO_H
#define FPS_READ_FPSCMD_FIFO_H

#include "fps.h"

int functionparameter_read_fpsCMD_fifo(int                 fpsCTRLfifofd,
                                       FPSCTRL_TASK_ENTRY *fpsctrltasklist,
                                       FPSCTRL_TASK_QUEUE *fpsctrlqueuelist);

#endif
