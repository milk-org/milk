/**
 * @file    fps_process_user_key.h
 * @brief   TUI key input processing
 */
 
 
 
int fpsCTRLscreen_process_user_key(
    int ch,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_TASK_ENTRY *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar
);

