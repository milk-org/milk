
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


// =====================================================================
// INPUT 
// =====================================================================

/** @brief Add 32-bit float parameter entry
 * 
 * Default setting for input parameter\n
 * Also creates function parameter index (fp_##key), type long
 * 
 * (void) statement suppresses compiler unused parameter warning
 */
#define FPS_ADDPARAM_FLT32_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_FLOAT32, FPFLAG_DEFAULT_INPUT, (dflt));\
  (void) fp_##key;\
} while(0)


/** @brief Add 64-bit float parameter entry
 */
#define FPS_ADDPARAM_FLT64_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_FLOAT64, FPFLAG_DEFAULT_INPUT, (dflt));\
  (void) fp_##key;\
} while(0)





/** @brief Add INT32 input parameter entry
 */
#define FPS_ADDPARAM_INT32_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_INT32, FPFLAG_DEFAULT_INPUT, (dflt));\
  (void) fp_##key;\
} while(0)


/** @brief Add INT64 input parameter entry
 */
#define FPS_ADDPARAM_INT64_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, (dflt));\
  (void) fp_##key;\
} while(0)




/** @brief Add filename input parameter entry
 */
#define FPS_ADDPARAM_FILENAME_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_FILENAME, FPFLAG_DEFAULT_INPUT_STREAM, (dflt));\
  (void) fp_##key;\
} while(0)



/** @brief Add stream input parameter entry
 */
#define FPS_ADDPARAM_STREAM_IN(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_STREAMNAME, FPFLAG_DEFAULT_INPUT_STREAM, (dflt));\
  (void) fp_##key;\
} while(0)



// =====================================================================
// ON/OFF
// =====================================================================

/** @brief Add ON/OFF parameter entry
 */
#define FPS_ADDPARAM_ONOFF(key, pname, pdescr, dflt) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, (dflt));\
  (void) fp_##key;\
} while(0)





// =====================================================================
// OUTPUT 
// =====================================================================

/** @brief Add FLT32 output parameter entry
 */
#define FPS_ADDPARAM_FLT32_OUT(key, pname, pdescr) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_FLOAT32, FPFLAG_DEFAULT_OUTPUT, NULL);\
  (void) fp_##key;\
} while(0)


/** @brief Add FLT64 output parameter entry
 */
#define FPS_ADDPARAM_FLT64_OUT(key, pname, pdescr) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_FLOAT64, FPFLAG_DEFAULT_OUTPUT, NULL);\
  (void) fp_##key;\
} while(0)


/** @brief Add IN32 output parameter entry
 */
#define FPS_ADDPARAM_INT32_OUT(key, pname, pdescr) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_INT64, FPFLAG_DEFAULT_OUTPUT, NULL);\
  (void) fp_##key;\
} while(0)



/** @brief Add INT64 output parameter entry
 */
#define FPS_ADDPARAM_INT64_OUT(key, pname, pdescr) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_INT64, FPFLAG_DEFAULT_OUTPUT, NULL);\
  (void) fp_##key;\
} while(0)


/** @brief Add stream output parameter entry
 */
#define FPS_ADDPARAM_STREAM_OUT(key, pname, pdescr) \
long fp_##key = 0; \
do{ \
  fp_##key = function_parameter_add_entry(&fps, (pname), (pdescr), FPTYPE_STREAMNAME, FPFLAG_DEFAULT_OUTPUT_STREAM, NULL);\
  (void) fp_##key;\
} while(0)






#endif
