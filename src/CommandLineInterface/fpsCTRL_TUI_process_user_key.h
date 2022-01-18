/**
 * @file    fpsCTRL_TUI_process_user_key.h
 * @brief   TUI key input processing
 */

#ifndef FPS_PROCESS_USER_KEY_H
#define FPS_PROCESS_USER_KEY_H

#include "function_parameters.h"

int fpsCTRL_TUI_process_user_key(int ch, FUNCTION_PARAMETER_STRUCT *fps, KEYWORD_TREE_NODE *keywnode,
                                 FPSCTRL_TASK_ENTRY *fpsctrltasklist, FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
                                 FPSCTRL_PROCESS_VARS *fpsCTRLvar);

#endif
