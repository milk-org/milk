/**
 * @file logfunc.c
 */

#include <stdio.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/**
 * ## Purpose
 *
 * Log function call (for testing / debugging only).
 *
 * Function calls are logged if to file .FileName.funccalls.log
 *
 * Variable AOLOOPCONTROL_logfunc_level keeps track of function depth: \n
 * it is incremented when entering a function \n
 * decremented when exiting a function
 *
 * Variable AOLOOPCONTROL_logfunc_level_max sets the max depth of logging
 *
 *
 * At the beginning of each function, insert this code:
 * @code
 * #ifdef TEST
 * CORE_logFunctionCall( logfunc_level, logfunc_level_max, 1, __FILE__, __func__, __LINE__, "");
 * #endif
 * @endcode
 * and at the end of each function:
 * @code
 * #ifdef TEST
 * CORE_logFunctionCall( logfunc_level, logfunc_level_max, 1, __FILE__, __func__, __LINE__, "");
 * #endif
 * @endcode
 *
 *
 * ## Arguments
 *
 * @param[in]
 * funclevel		INT
 * 					Function level (0: top level, always log)
 *
 * @param[in]
 * loglevel			INT
 * 					Log level: log all function with level =< loglevel
 *
 * logfuncMODE		INT
 * 					Log mode, 0:entering function, 1:exiting function
 *
 * @param[in]
 * FileName			char*
 * 					Name of source file, usually __FILE__ so that preprocessor fills this parameter.
 *
 * @param[in]
 * FunctionName		char*
 * 					Name of function, usually __FUNCTION__ so that preprocessor fills this parameter.
 *
 * @param[in]
 * line				char*
 * 					Line in cource code, usually __LINE__ so that preprocessor fills this parameter.
 *
 * @param[in]
 * comments			char*
 * 					comment string
 *
 * @return void
 *
 * @note Carefully set depth value to avoid large output file.
 * @warning May slow down code. Only use for debugging. Output file may grow very quickly.
 */

void CORE_logFunctionCall(const int                           funclevel,
                          const int                           loglevel,
                          const int                           logfuncMODE,
                          __attribute__((unused)) const char *FileName,
                          const char                         *FunctionName,
                          const long                          line,
                          char                               *comments)
{
    time_t          tnow;
    struct timespec timenow;
    pid_t           tid;
    char            modechar;

    modechar = '?';

    if (logfuncMODE == 0)
    {
        modechar = '>';
    }
    else if (logfuncMODE == 1)
    {
        modechar = '<';
    }
    else
    {
        modechar = '?';
    }

    if (funclevel <= loglevel)
    {
        char fname[500];

        FILE *fp;

        sprintf(fname, ".%s.funccalls.log", FunctionName);

        struct tm *uttime;
        tnow   = time(NULL);
        uttime = gmtime(&tnow);
        clock_gettime(CLOCK_REALTIME, &timenow);
        tid = syscall(SYS_gettid);

        // add custom parameter into string (optional)

        fp = fopen(fname, "a");
        fprintf(fp,
                "%02d:%02d:%02ld.%09ld  %10d  %10d  %c %40s %6ld   %s\n",
                uttime->tm_hour,
                uttime->tm_min,
                timenow.tv_sec % 60,
                timenow.tv_nsec,
                getpid(),
                (int) tid,
                modechar,
                FunctionName,
                line,
                comments);
        fclose(fp);
    }
}
