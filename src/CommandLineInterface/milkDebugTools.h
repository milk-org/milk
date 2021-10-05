/**
 * @file    milkDebugTools.h
 *
 * Error handling and checking
 *
 */

#ifndef _MILKDEBUGTOOLS_H
#define _MILKDEBUGTOOLS_H

#include <errno.h>

// define (custom) types for function return value

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#include "CommandLineInterface/CLIcore_signals.h"

// error mode
// defines function behavior on error
#define ERRMODE_NULL   0  // ignore error
#define ERRMODE_WARN   1  // issue warning and continue
#define ERRMODE_FAIL   2  // fail nicely if possible
#define ERRMODE_ABORT  3  // abort process


#define STRINGMAXLEN_DEFAULT       1000
#define STRINGMAXLEN_ERRORMSG      1000
#define STRINGMAXLEN_CLICMDLINE    1000  // CPU input command line
#define STRINGMAXLEN_COMMAND       1000
#define STRINGMAXLEN_STREAMNAME     100
#define STRINGMAXLEN_IMGNAME        100
#define STRINGMAXLEN_FILENAME       200  // without directory, includes extension
#define STRINGMAXLEN_DIRNAME        800
#define STRINGMAXLEN_FULLFILENAME  1000  // includes directory name
#define STRINGMAXLEN_FUNCTIONNAME   200
#define STRINGMAXLEN_FUNCTIONARGS  1000
#define STRINGMAXLEN_SHMDIRNAME     200

#define STRINGMAXLEN_FPSPROCESSTYPE 64


#define xstr(a) zstr(a)
#define zstr(a) #a



//
// ************ ERROR HANDLING **********************************
//

#define STRINGMAXLEN_FUNCERRORMSG  2000

#define FUNC_RETURN_FAILURE(...) do { \
char errmsg_funcretfailure[STRINGMAXLEN_FUNCERRORMSG]; \
int errormsg_slen = snprintf(errmsg_funcretfailure, STRINGMAXLEN_FUNCERRORMSG, __VA_ARGS__); \
if(errormsg_slen<1) {                                              \
    printf("snprintf in FUNC_RETURN_FAILURE: wrote <1 char");      \
    abort();                                                       \
}                                                                  \
if(errormsg_slen >= STRINGMAXLEN_FUNCERRORMSG) {                   \
    printf("snprintf in FUNC_RETURN_FAILURE: string truncation");  \
    abort();                                                       \
}                                                                  \
DEBUG_TRACEPOINT("%c[%d;%dm FERR %c[%dm %s", (char) 27, 1, 31, (char) 27, 0, errmsg_funcretfailure); \
printf("\n");                                                      \
printf("%c[%d;%dm ERROR %c[%dm [ %s %s %d ]\n", (char) 27, 1, 31, (char) 27, 0, __FILE__, __func__, __LINE__);     \
printf("%c[%d;%dm ***** %c[%d;m %s\n", (char) 27, 1, 31, (char) 27, 0, data.testpoint.msg); \
printf("%c[%d;%dm ***** %c[%d;m -> Function %s returns RETURN_FAILURE\n", (char) 27, 1, 31, (char) 27, 0, __func__); \
DEBUG_TRACE_FEXIT();\
return RETURN_FAILURE; \
} while(0)



// Check function call return value.
// If not RETURN_SUCCESS, fail current (caller) function
//
#define FUNC_CHECK_RETURN(errval) do {                        \
data.testpoint.linestack[data.testpoint.funclevel] = __LINE__; \
errno_t retcheckvalue = errval;                                    \
if(retcheckvalue != RETURN_SUCCESS) {                          \
    char errmsg_funcretfailure[STRINGMAXLEN_FUNCERRORMSG]; \
    int errormsg_slen = snprintf(errmsg_funcretfailure, STRINGMAXLEN_FUNCERRORMSG, "[%d] %s", retcheckvalue, xstr(errval)); \
    if(errormsg_slen<1) {                                              \
        printf("snprintf in FUNC_RETURN_FAILURE: wrote <1 char");      \
        abort();                                                       \
    }                                                                  \
    if(errormsg_slen >= STRINGMAXLEN_FUNCERRORMSG) {                   \
        printf("snprintf in FUNC_RETURN_FAILURE: string truncation");  \
        abort();                                                       \
    }                                                                    \
    DEBUG_TRACEPOINT("%c[%d;%dm FCALLERR %c[%dm %s", (char) 27, 1, 31, (char) 27, 0, errmsg_funcretfailure); \
    printf("\n");                                                      \
printf("%c[%d;%dm > > > FCALLERR %c[%dm [ %s %s %d ]\n", (char) 27, 1, 31, (char) 27, 0, __FILE__, __func__, __LINE__); \
    printf("%c[%d;%dm ***** %c[%d;m [rval = %d] %s\n", (char) 27, 1, 31, (char) 27, 0, retcheckvalue, xstr(errval)); \
    printf("%c[%d;%dm ***** %c[%d;m -> Function %s returns RETURN_FAILURE\n", (char) 27, 1, 31, (char) 27, 0, __func__); \
    DEBUG_TRACE_FEXIT();                                               \
    return RETURN_FAILURE;                                             \
}                                                                  \
} while(0)




/** @brief Print error (in red) and continue
 *  @ingroup errcheckmacro
 */
#define PRINT_ERROR(...) do { \
int print_error_slen = snprintf(data.testpoint.msg, STRINGMAXLEN_FUNCTIONARGS, __VA_ARGS__); \
if(print_error_slen<1) {                                                    \
    printf("snprintf in PRINT_ERROR: wrote <1 char");           \
    abort();                                                    \
}                                                               \
if(print_error_slen >= STRINGMAXLEN_FUNCTIONARGS) {                         \
    printf("snprintf in PRINT_ERROR: string truncation");       \
    abort();                                                    \
}\
printf("ERROR: %c[%d;%dm %s %c[%d;m\n", (char) 27, 1, 31, data.testpoint.msg, (char) 27, 0); \
print_error_slen = snprintf(data.testpoint.file, STRINGMAXLEN_FILENAME, "%s", __FILE__); \
if(print_error_slen<1) {                                                    \
    printf("snprintf in PRINT_ERROR: wrote <1 char");           \
    abort();                                                    \
}                                                               \
if(print_error_slen >= STRINGMAXLEN_FILENAME) {                             \
    printf("snprintf in PRINT_ERROR: string truncation");       \
    abort();                                                    \
}\
print_error_slen = snprintf(data.testpoint.func, STRINGMAXLEN_FUNCTIONNAME, "%s", __func__); \
if(print_error_slen<1) {                                                    \
    printf("snprintf in PRINT_ERROR: wrote <1 char");           \
    abort();                                                    \
}                                                               \
if(print_error_slen >= STRINGMAXLEN_FUNCTIONNAME) {                         \
    printf("snprintf in PRINT_ERROR: string truncation");       \
    abort();                                                    \
}\
data.testpoint.line = __LINE__; \
clock_gettime(CLOCK_REALTIME, &data.testpoint.time); \
} while(0)



/**
 * @brief Print warning and continue
 * @ingroup errcheckmacro
 */
#define PRINT_WARNING(...) do { \
char warnmessage[1000]; \
snprintf(warnmessage, 1000, __VA_ARGS__); \
fprintf(stderr, \
"%c[%d;%dm WARNING [ FILE: %s   FUNCTION: %s  LINE: %d ]  %c[%d;m\n", \
(char) 27, 1, 35, __FILE__, __func__, __LINE__, (char) 27, 0); \
if(C_ERRNO != 0) \
{ \
char buff[256]; \
if( strerror_r( errno, buff, 256 ) == 0 ) { \
fprintf(stderr,"C Error: %s\n", buff ); \
} else { \
fprintf(stderr,"Unknown C Error\n"); \
} \
} else { \
fprintf(stderr,"No C error (errno = 0)\n"); } \
fprintf(stderr, "%c[%d;%dm ", (char) 27, 1, 35); \
fprintf(stderr, "%s", warnmessage); \
fprintf(stderr, " %c[%d;m\n", (char) 27, 0); \
C_ERRNO = 0; \
} while(0)









// Enter function
#if defined NDEBUG
#define DEBUG_TRACE_FSTART(...)
#else
#define DEBUG_TRACE_FSTART(...) \
int funclevel = data.testpoint.funclevel; \
static int funccallcnt; \
do { \
if(data.testpoint.funclevel < MAXNB_FUNCSTACK) { \
    strncpy(data.testpoint.funcstack[data.testpoint.funclevel], __func__, STRINGMAXLEN_FUNCSTAK_FUNCNAME-1);\
    data.testpoint.fcntstack[data.testpoint.funclevel] = funccallcnt; \
    data.testpoint.funccallcnt = funccallcnt; \
}\
funccallcnt++; \
data.testpoint.funclevel++; \
} while(0)
#endif

// Exit function
#if defined NDEBUG
#define DEBUG_TRACE_FEXIT(...)
#else
#define DEBUG_TRACE_FEXIT(...) do { \
data.testpoint.funclevel--; \
data.testpoint.funccallcnt = data.testpoint.fcntstack[data.testpoint.funclevel];\
if(data.testpoint.funclevel>0) {\
    if (data.testpoint.funclevel != funclevel) { \
    PRINT_ERROR("function level mismatch - check source code"); \
    abort(); \
} \
}\
} while(0)
#endif





/**
 * @ingroup debugmacro
 * @brief register trace point
 */
#define DEBUG_TRACEPOINTRAW(...) do {                    \
int slen = snprintf(data.testpoint.file, STRINGMAXLEN_FULLFILENAME, "%s", __FILE__);\
if(slen<1) {                                                               \
    PRINT_ERROR("snprintf wrote <1 char");                                 \
    abort();                                                               \
}                                                                          \
if(slen >= STRINGMAXLEN_FULLFILENAME) {                                    \
    PRINT_ERROR("snprintf string truncation");                             \
    abort();                                                               \
}                                                                          \
slen = snprintf(data.testpoint.func, STRINGMAXLEN_FUNCTIONNAME, "%s", __func__);\
if(slen<1) {                                                               \
    PRINT_ERROR("snprintf wrote <1 char");                                 \
    abort();                                                               \
}                                                                          \
if(slen >= STRINGMAXLEN_FUNCTIONNAME) {                                    \
    PRINT_ERROR("snprintf string truncation");                             \
    abort();                                                               \
}                                                                          \
data.testpoint.line = __LINE__;                       \
clock_gettime(CLOCK_REALTIME, &data.testpoint.time);  \
slen = snprintf(data.testpoint.msg, STRINGMAXLEN_FUNCTIONARGS, __VA_ARGS__);\
if(slen<1) {                                                               \
    PRINT_ERROR("snprintf wrote <1 char");                                 \
    abort();                                                               \
}                                                                          \
if(slen >= STRINGMAXLEN_FUNCTIONARGS) {                                    \
    PRINT_ERROR("snprintf string truncation");                             \
    abort();                                                               \
}                                                                          \
data.testpoint.loopcnt = data.testpointloopcnt;                            \
if(data.testpointarrayinit == 1) {                                         \
memcpy(&data.testpointarray[data.testpointcnt], &data.testpoint, sizeof(CODETESTPOINT));\
data.testpointcnt++;                                                       \
if(data.testpointcnt == CODETESTPOINTARRAY_NBCNT) {                        \
data.testpointcnt = 0;                                                     \
data.testpointloopcnt++;                                                   \
}\
}\
} while(0)



#if defined NDEBUG
#define DEBUG_TRACEPOINT_PRINT(...)
#else
#define DEBUG_TRACEPOINT_PRINT(...) do {                    \
DEBUG_TRACEPOINTRAW(__VA_ARGS__);                              \
printf("DEBUG MSG [%s %s  %d]: %s\n", data.testpoint.file, data.testpoint.func, data.testpoint.line, data.testpoint.msg);   \
} while(0)
#endif


#if defined NDEBUG
#define DEBUG_TRACEPOINT_LOG(...)
#else
#define DEBUG_TRACEPOINT_LOG(...) do {  \
DEBUG_TRACEPOINTRAW(__VA_ARGS__);          \
write_process_log();                    \
} while(0)
#endif



#if defined NDEBUG
#define DEBUG_TRACEPOINT(...)
#else
#if defined DEBUGLOG
#define DEBUG_TRACEPOINT(...) do {                    \
DEBUG_TRACEPOINT_LOG(__VA_ARGS__);                              \
} while(0)
#else
#define DEBUG_TRACEPOINT(...) do {                    \
DEBUG_TRACEPOINTRAW(__VA_ARGS__);                              \
} while(0)
#endif
#endif


/*
#if defined DEBUGLOG \
write_process_log(); \
#endif \
#if defined DEBUGPRINT \
printf("DEBUG MSG [%s %s  %d]: %s\n", data.testpoint.file, data.testpoint.func, data.testpoint.line, data.testpoint.msg);   \
#endif \
*/



//
// ************ ERROR-CHECKING FUNCTIONS **********************************
//




/**
 * @ingroup errcheckmacro
 * @brief system call with error checking and handling
 *
 */
#define EXECUTE_SYSTEM_COMMAND(...) do {                                   \
char syscommandstring[STRINGMAXLEN_COMMAND];                               \
int slen = snprintf(syscommandstring, STRINGMAXLEN_COMMAND, __VA_ARGS__);  \
if(slen<1) {                                                               \
    PRINT_ERROR("snprintf wrote <1 char");                                 \
    abort();                                                               \
}                                                                          \
if(slen >= STRINGMAXLEN_COMMAND) {                                         \
    PRINT_ERROR("snprintf string truncation");                             \
    abort();                                                               \
}                                                                          \
data.retvalue = system(syscommandstring);                                  \
} while(0)


#define EXECUTE_SYSTEM_COMMAND_ERRCHECK(...) do {                          \
char syscommandstring[STRINGMAXLEN_COMMAND];                               \
int slen = snprintf(syscommandstring, STRINGMAXLEN_COMMAND, __VA_ARGS__);  \
if(slen<1) {                                                               \
    PRINT_ERROR("snprintf wrote <1 char");                                 \
    abort();                                                               \
}                                                                          \
if(slen >= STRINGMAXLEN_COMMAND) {                                         \
    PRINT_ERROR("snprintf string truncation");                             \
    abort();                                                               \
}                                                                          \
data.retvalue = system(syscommandstring);                                  \
if(data.retvalue != 0)                                                     \
{                                                                          \
    PRINT_ERROR("system() error %d %s", data.retvalue, strerror(data.retvalue));\
    abort();                                                               \
}                                                                          \
} while(0)




/**
 * @ingroup errcheckmacro
 * @brief snprintf with error checking and handling
 *
 */
#define SNPRINTF_CHECK(string, maxlen, ...) do { \
int slen = snprintf(string, maxlen, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= maxlen) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)



/**
 * @ingroup errcheckmacro
 * @brief Write image name to string
 *
 * Requires existing image string of len #STRINGMAXLEN_IMGNAME
 *
 * Example use:
 * @code
 * char imname[STRINGMAXLEN_IMGNAME];
 * char name[]="im";
 * int imindex = 34;
 * WRITE_FULLFILENAME(imname, "%s_%04d", name, imindex);
 * @endcode
 *
 *
 */
#define WRITE_IMAGENAME(imname, ...) do { \
int slen = snprintf(imname, STRINGMAXLEN_IMGNAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_IMGNAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)


#define CREATE_IMAGENAME(imname, ...) \
char imname[STRINGMAXLEN_IMGNAME]; \
do { \
int slen = snprintf(imname, STRINGMAXLEN_IMGNAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_IMGNAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)




/**
 * @ingroup errcheckmacro
 * @brief Write directory name to string
 *
 * Requires existing image string of len #STRINGMAXLEN_DIRNAME
 *
 * Example use:
 * @code
 * char fname[STRINGMAXLEN_FILENAME];
 * char name[]="imlog";
 * WRITE_FULLFILENAME(fname, "%s.txt", name);
 * @endcode
 *
 */
#define WRITE_DIRNAME(dirname, ...) do { \
int slen = snprintf(dirname, STRINGMAXLEN_DIRNAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_DIRNAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)




/**
 * @ingroup errcheckmacro
 * @brief Write filename to string
 *
 * Requires existing image string of len #STRINGMAXLEN_FILENAME
 *
 * Example use:
 * @code
 * char fname[STRINGMAXLEN_FILENAME];
 * char name[]="imlog";
 * WRITE_FULLFILENAME(fname, "%s.txt", name);
 * @endcode
 *
 */
#define WRITE_FILENAME(fname, ...) do { \
int slen = snprintf(fname, STRINGMAXLEN_FILENAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_FILENAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)




/**
 * @ingroup errcheckmacro
 * @brief Write full path filename to string
 *
 * Requires existing image string of len #STRINGMAXLEN_FULLFILENAME
 *
 * Example use:
 * @code
 * char ffname[STRINGMAXLEN_FULLFILENAME];
 * char directory[]="/tmp/";
 * char name[]="imlog";
 * WRITE_FULLFILENAME(ffname, "%s/%s.txt", directory, name);
 * @endcode
 *
 */
#define WRITE_FULLFILENAME(ffname, ...) do { \
int slen = snprintf(ffname, STRINGMAXLEN_FULLFILENAME, __VA_ARGS__); \
if(slen<1) {                                                    \
    PRINT_ERROR("snprintf wrote <1 char");                      \
    abort();                                                    \
}                                                               \
if(slen >= STRINGMAXLEN_FULLFILENAME) {                              \
    PRINT_ERROR("snprintf string truncation");                  \
    abort();                                                    \
}                                                               \
} while(0)




/**
 * @ingroup errcheckmacro
 * @brief Write a string to file
 *
 * Creates file, writes string, and closes file.
 *
 * Example use:
 * @code
 * float piapprox = 3.14;
 * WRITE_STRING_TO_FILE("logfile.txt", "pi is approximately %f\n", piapprox);
 * @endcode
 *
 */
#define WRITE_STRING_TO_FILE(fname, ...) do { \
FILE *fptmp;                                                                \
fptmp = fopen(fname, "w");                                                  \
if (fptmp == NULL) {                                                        \
int errnum = errno;                                                         \
PRINT_ERROR("fopen() returns NULL");                                        \
fprintf(stderr, "Error opening file %s: %s\n", fname, strerror( errnum ));  \
abort();                                                                    \
} else {                                                                    \
fprintf(fptmp, __VA_ARGS__);                                                \
fclose(fptmp);                                                              \
}                                                                           \
} while(0)







// *************************** FUNCTION RETURN VALUE *********************************************
// For function returning type errno_t (= int)
//
#define RETURN_SUCCESS       0
#define RETURN_FAILURE       1   // generic error code
#define RETURN_MISSINGFILE   2

#define RETURN_OTHER         3



#endif
