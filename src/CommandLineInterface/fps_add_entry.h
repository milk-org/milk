
/**
 * @file    fps_add_entry.h
 * @brief   add parameter entry to FPS
 */


#ifndef FPS_ADD_ENTRY_H
#define FPS_ADD_ENTRY_H

int function_parameter_add_entry(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *keywordstring,
    const char                *descriptionstring,
    uint64_t             type,
    uint64_t             fpflag,
    void                *valueptr
);

#endif
