/**
 * @file    fps_scan.h
 * @brief   scan and load FPSs
 */


errno_t functionparameter_scan_fps(
    uint32_t mode,
    char *fpsnamemask,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    int *ptr_NBkwn,
    int *ptr_fpsindex,
    long *ptr_pindex,
    int verbose
);
