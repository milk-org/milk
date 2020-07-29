/**
 * @file    fps_add_entry.h
 * @brief   add parameter entry to FPS
 */



int function_parameter_add_entry(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *keywordstring,
    const char                *descriptionstring,
    uint64_t             type,
    uint64_t             fpflag,
    void                *valueptr
);

