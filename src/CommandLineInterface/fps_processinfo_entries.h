/**
 * @file    fps_add_RTsetting_entries.h
 * @brief   Add parameters to FPS for real-time process settings
 */
 
 
errno_t fps_add_processinfo_entries(FUNCTION_PARAMETER_STRUCT *fps);

errno_t fps_to_processinfo(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *procinfo);
