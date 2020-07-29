/**
 * @file    fps_processcmdline.h
 * @brief   FPS process command line
 */



int functionparameter_FPSprocess_cmdline(
    char *FPScmdline,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar,
    FUNCTION_PARAMETER_STRUCT *fps,
    uint64_t *taskstatus
);
