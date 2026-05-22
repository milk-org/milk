/**
 * @file    fps_processcmdline.h
 * @brief   FPS process command line
 */

#ifndef FPS_PROCESSCMDLINE_H
#define FPS_PROCESSCMDLINE_H

#include "fps.h"

int functionparameter_FPSprocess_cmdline(char                 *FPScmdline,
                                         FPSCTRL_TASK_QUEUE   *fpsctrlqueuelist,
                                         KEYWORD_TREE_NODE    *keywnode,
                                         FPSCTRL_PROCESS_VARS *fpsCTRLvar,
                                         FPS                  *fps,
                                         uint64_t             *taskstatus);

int functionparameter_FPSprocess_cmdfile(char                 *infname,
                                         FPS                  *fps,
                                         KEYWORD_TREE_NODE    *keywnode,
                                         FPSCTRL_TASK_QUEUE   *fpsctrlqueuelist,
                                         FPSCTRL_PROCESS_VARS *fpsCTRLvar);

#endif
