/**
 * @file    fps_processcmdline.h
 * @brief   FPS process command line
 */

#ifndef FPS_PROCESSCMDLINE_H
#define FPS_PROCESSCMDLINE_H

#include "function_parameters.h"

int functionparameter_FPSprocess_cmdline(char                 *FPScmdline,
                                         FPSCTRL_TASK_QUEUE   *fpsctrlqueuelist,
                                         KEYWORD_TREE_NODE    *keywnode,
                                         FPSCTRL_PROCESS_VARS *fpsCTRLvar,
                                         FUNCTION_PARAMETER_STRUCT *fps,
                                         uint64_t                  *taskstatus);

int functionparameter_FPSprocess_cmdfile(char                      *infname,
                                         FUNCTION_PARAMETER_STRUCT *fps,
                                         KEYWORD_TREE_NODE         *keywnode,
                                         FPSCTRL_TASK_QUEUE   *fpsctrlqueuelist,
                                         FPSCTRL_PROCESS_VARS *fpsCTRLvar);

#endif
