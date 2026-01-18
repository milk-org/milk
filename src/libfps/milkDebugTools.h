#ifndef _MILKDEBUGTOOLS_H
#define _MILKDEBUGTOOLS_H

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>

#define STRINGMAXLEN_DEFAULT      1000
#define STRINGMAXLEN_ERRORMSG     1000
#define STRINGMAXLEN_CLICMDLINE   1000
#define STRINGMAXLEN_COMMAND      1000
#define STRINGMAXLEN_STREAMNAME   100
#define STRINGMAXLEN_IMGNAME      STRINGMAXLEN_STREAMNAME
#define STRINGMAXLEN_FILENAME     200
#define STRINGMAXLEN_DIRNAME      800
#define STRINGMAXLEN_FULLFILENAME 1000
#define STRINGMAXLEN_FUNCTIONNAME 200
#define STRINGMAXLEN_FUNCTIONARGS 1000
#define STRINGMAXLEN_SHMDIRNAME   200

#define STRINGMAXLEN_FPSPROCESSTYPE 64

#define xstr(a) zstr(a)
#define zstr(a) #a

// Simple macros to avoid dependency on global 'data' struct
#ifndef PRINT_ERROR
#define PRINT_ERROR(...) fprintf(stderr, "ERROR: " __VA_ARGS__)
#endif

#ifndef DEBUG_TRACEPOINT
#define DEBUG_TRACEPOINT(...)
#endif

#ifndef DEBUG_TRACE_FSTART
#define DEBUG_TRACE_FSTART(...)
#endif

#ifndef DEBUG_TRACE_FEXIT
#define DEBUG_TRACE_FEXIT(...)
#endif

#ifndef SNPRINTF_CHECK
#define SNPRINTF_CHECK(str, size, format, ...) snprintf(str, size, format, ##__VA_ARGS__)
#endif

#ifndef WRITE_FULLFILENAME
#define WRITE_FULLFILENAME(ffname, ...) snprintf(ffname, STRINGMAXLEN_FULLFILENAME, __VA_ARGS__)
#endif

#ifndef WRITE_DIRNAME
#define WRITE_DIRNAME(dname, ...) snprintf(dname, STRINGMAXLEN_DIRNAME, __VA_ARGS__)
#endif

#ifndef WRITE_IMAGENAME
#define WRITE_IMAGENAME(iname, ...) snprintf(iname, STRINGMAXLEN_IMGNAME, __VA_ARGS__)
#endif

#ifndef WRITE_FILENAME
#define WRITE_FILENAME(fname, ...) snprintf(fname, STRINGMAXLEN_FILENAME, __VA_ARGS__)
#endif

#ifndef EXECUTE_SYSTEM_COMMAND
#define EXECUTE_SYSTEM_COMMAND(...) do { \
    char cmd[STRINGMAXLEN_COMMAND]; \
    snprintf(cmd, STRINGMAXLEN_COMMAND, __VA_ARGS__); \
    int __attribute__((unused)) ret = system(cmd); \
} while(0)
#endif

#endif