/**
 * @file    fps_paramvalue.h
 * @brief   set and get parameter values
 */

#ifndef FPS_PARAMVALUE_H
#define FPS_PARAMVALUE_H



// =====================================================================
// INT32
// =====================================================================

int functionparameter_SetParamValue_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    long value
);

long functionparameter_GetParamValue_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);

long *functionparameter_GetParamPtr_INT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


// Creates variable "_varname"
//
#define FPS_GETPARAM_INT32(varname, pname) \
int32_t _##varname = 0; \
do{ \
   _##varname = functionparameter_GetParamValue_INT32(&fps, pname);\
  (void) _##varname;\
} while(0)






// =====================================================================
// INT64
// =====================================================================

int functionparameter_SetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    long value
);

long functionparameter_GetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);

long *functionparameter_GetParamPtr_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


// Creates variable "_varname"
//
#define FPS_GETPARAM_INT64(varname, pname) \
int64_t _##varname = 0; \
do{ \
   _##varname = functionparameter_GetParamValue_INT64(&fps, pname);\
  (void) _##varname;\
} while(0)





// =====================================================================
// FLOAT32
// =====================================================================

float functionparameter_GetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    float value
);


float *functionparameter_GetParamPtr_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


// Creates variable "_varname"
//
#define FPS_GETPARAM_FLOAT32(varname, pname) \
float _##varname = 0; \
do{ \
   _##varname = functionparameter_GetParamValue_FLOAT32(&fps, pname);\
  (void) _##varname;\
} while(0)







// =====================================================================
// FLOAT64
// =====================================================================

double functionparameter_GetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    double value
);


double *functionparameter_GetParamPtr_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


// Creates variable "_varname"
//
#define FPS_GETPARAM_FLOAT64(varname, pname) \
double _##varname = 0; \
do{ \
   _##varname = functionparameter_GetParamValue_FLOAT64(&fps, pname);\
  (void) _##varname;\
} while(0)






// =====================================================================
// STRING
// =====================================================================

char *functionparameter_GetParamPtr_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    const char *stringvalue
);

// Creates variable "_varname"
//
#define FPS_GETPARAM_STRING(varname, pname) \
char *_##varname; \
do{ \
   _##varname = functionparameter_GetParamPtr_STRING(&fps, pname);\
  (void) _##varname;\
} while(0)



// =====================================================================
// ON/OFF
// =====================================================================


int functionparameter_GetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);


int functionparameter_SetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int ONOFFvalue
);




uint64_t *functionparameter_GetParamPtr_fpflag(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
);





#endif
