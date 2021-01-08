/**
 * @file    milkDebugTools.h
 *
 * Error handling and checking
 *
 */

#ifndef _MILKDEBUGTOOLS_H
#define _MILKDEBUGTOOLS_H


// define (custom) types for function return value

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif


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






//
// ************ ERROR HANDLING **********************************
//

/** @brief Print error (in red) and continue
 *  @ingroup errcheckmacro
 */
#ifndef STANDALONE
#define PRINT_ERROR(...) do { \
sprintf(data.testpoint_msg, __VA_ARGS__); \
printf("ERROR: %c[%d;%dm %s %c[%d;m\n", (char) 27, 1, 31, data.testpoint_msg, (char) 27, 0); \
sprintf(data.testpoint_file, "%s", __FILE__); \
sprintf(data.testpoint_func, "%s", __func__); \
data.testpoint_line = __LINE__; \
clock_gettime(CLOCK_REALTIME, &data.testpoint_time); \
} while(0)
#else
#define PRINT_ERROR(...) printf("ERROR: %c[%d;%dm %s %c[%d;m\n", (char) 27, 1, 31, __VA_ARGS__, (char) 27, 0)
#endif



/**
 * @brief Print warning and continue
 * @ingroup errcheckmacro
 */
#define PRINT_WARNING(...) do { \
char warnmessage[1000]; \
sprintf(warnmessage, __VA_ARGS__); \
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




/**
 * @ingroup debugmacro
 * @brief register trace point
 */
#if defined NDEBUG || defined STANDALONE
#define DEBUG_TRACEPOINT(...)
#else
#define DEBUG_TRACEPOINT(...) do {                    \
sprintf(data.testpoint_file, "%s", __FILE__);         \
sprintf(data.testpoint_func, "%s", __func__);         \
data.testpoint_line = __LINE__;                       \
clock_gettime(CLOCK_REALTIME, &data.testpoint_time);  \
sprintf(data.testpoint_msg, __VA_ARGS__);             \
} while(0)
#endif


/**
 * @ingroup debugmacro
 * @brief register and log trace point
 */
#if defined NDEBUG || defined STANDALONE
#define DEBUG_TRACEPOINTLOG(...)
#else
#define DEBUG_TRACEPOINTLOG(...) do {                \
sprintf(data.testpoint_file, "%s", __FILE__);        \
sprintf(data.testpoint_func, "%s", __func__);        \
data.testpoint_line = __LINE__;                      \
clock_gettime(CLOCK_REALTIME, &data.testpoint_time); \
sprintf(data.testpoint_msg, __VA_ARGS__);            \
write_process_log();                                 \
} while(0)
#endif



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





#endif
