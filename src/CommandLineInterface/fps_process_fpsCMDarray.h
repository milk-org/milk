/**
 * @file fps_process_fpsCMDarray.h
 */

#ifndef FPS_PROCESS_FPSCMDARRAY_H
#define FPS_PROCESS_FPSCMDARRAY_H

#include "function_parameters.h"

int function_parameter_process_fpsCMDarray(FPSCTRL_TASK_ENTRY *fpsctrltasklist, FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
                                           KEYWORD_TREE_NODE *keywnode, FPSCTRL_PROCESS_VARS *fpsCTRLvar,
                                           FUNCTION_PARAMETER_STRUCT *fps);

#endif
