/**
 * @file    COREMOD_tools.c
 * @brief   Frequently used tools
 *
 * Includes basic file I/O
 *
 *
 */


/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT ""

// Module short description
#define MODULE_DESCRIPTION       "misc tools"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <ncurses.h>

#ifdef __MACH__
#include <mach/mach_time.h>long AOloopControl_ComputeOpenLoopModes(long loop)
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
static int clock_gettime(int clk_id, struct mach_timespec *t)
{
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t time;
    time = mach_absolute_time();
    double nseconds = ((double)time * (double)timebase.numer) / ((
                          double)timebase.denom);
    double seconds = ((double)time * (double)timebase.numer) / ((
                         double)timebase.denom * 1e9);
    t->tv_sec = seconds;
    t->tv_nsec = nseconds;
    return 0;
}
#else
#include <time.h>
#endif


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"



/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

#define SBUFFERSIZE 1000





/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */


static FILE *fpgnuplot;







/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(COREMOD_tools)


/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

/** @name CLI bindings */


errno_t COREMOD_TOOLS_mvProcCPUset_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            == 0)
    {
        COREMOD_TOOLS_mvProcCPUset(
            data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


errno_t write_flot_file_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(2, CLIARG_FLOAT)
            == 0)
    {
        write_float_file(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numf);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


errno_t COREMOD_TOOLS_imgdisplay3D_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_TOOLS_imgdisplay3D(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


errno_t COREMOD_TOOLS_statusStat_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_TOOLS_statusStat(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




/* =============================================================================================== */
/* =============================================================================================== */
/*                                    MODULE INITIALIZATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name Module initialization */



static errno_t init_module_CLI()
{
    strcpy(data.cmd[data.NBcmd].key, "csetpmove");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = COREMOD_TOOLS_mvProcCPUset_cli;
    strcpy(data.cmd[data.NBcmd].info, "move current process to CPU set");
    strcpy(data.cmd[data.NBcmd].syntax, "<CPU set name>");
    strcpy(data.cmd[data.NBcmd].example, "csetpmove realtime");
    strcpy(data.cmd[data.NBcmd].Ccall,
           "int COREMOD_TOOLS_mvProcCPUset(const char *csetname)");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "writef2file");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = write_flot_file_cli;
    strcpy(data.cmd[data.NBcmd].info, "write float to file");
    strcpy(data.cmd[data.NBcmd].syntax, "<filename> <float variable>");
    strcpy(data.cmd[data.NBcmd].example, "writef2file val.txt a");
    strcpy(data.cmd[data.NBcmd].Ccall,
           "int write_float_file(const char *fname, float value)");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "dispim3d");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = COREMOD_TOOLS_imgdisplay3D_cli;
    strcpy(data.cmd[data.NBcmd].info,
           "display 2D image as 3D surface using gnuplot");
    strcpy(data.cmd[data.NBcmd].syntax, "<imname> <step>");
    strcpy(data.cmd[data.NBcmd].example, "dispim3d im1 5");
    strcpy(data.cmd[data.NBcmd].Ccall,
           "int COREMOD_TOOLS_imgdisplay3D(const char *IDname, long step)");
    data.NBcmd++;

    strcpy(data.cmd[data.NBcmd].key, "ctsmstats");
    strcpy(data.cmd[data.NBcmd].module, __FILE__);
    data.cmd[data.NBcmd].fp = COREMOD_TOOLS_statusStat_cli;
    strcpy(data.cmd[data.NBcmd].info, "monitors shared variable status");
    strcpy(data.cmd[data.NBcmd].syntax, "<imname> <NBstep>");
    strcpy(data.cmd[data.NBcmd].example, "ctsmstats imst 100000");
    strcpy(data.cmd[data.NBcmd].Ccall,
           "long COREMOD_TOOLS_statusStat(const char *IDstat_name, long indexmax)");
    data.NBcmd++;


    return RETURN_SUCCESS;
}










/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name TEMPLATEMODULE functions */








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

void CORE_logFunctionCall(
    const int funclevel,
    const int loglevel,
    const int logfuncMODE,
    __attribute__((unused)) const char *FileName,
    const char *FunctionName,
    const long line,
    char *comments
)
{
    time_t tnow;
    struct timespec timenow;
    pid_t tid;
    char modechar;

    modechar = '?';

    if(logfuncMODE == 0)
    {
        modechar = '>';
    }
    else if(logfuncMODE == 1)
    {
        modechar = '<';
    }
    else
    {
        modechar = '?';
    }

    if(funclevel <= loglevel)
    {
        char  fname[500];

        FILE *fp;


        sprintf(fname, ".%s.funccalls.log", FunctionName);

        struct tm *uttime;
        tnow = time(NULL);
        uttime = gmtime(&tnow);
        clock_gettime(CLOCK_REALTIME, &timenow);
        tid = syscall(SYS_gettid);

        // add custom parameter into string (optional)

        fp = fopen(fname, "a");
        fprintf(fp, "%02d:%02d:%02ld.%09ld  %10d  %10d  %c %40s %6ld   %s\n",
                uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec,
                getpid(), (int) tid,
                modechar, FunctionName, line, comments);
        fclose(fp);
    }


}





struct timespec timespec_diff(struct timespec start, struct timespec end)
{
    struct timespec temp;

    if((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}






double timespec_diff_double(struct timespec start, struct timespec end)
{
    struct timespec temp;
    double val;

    if((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }

    val = temp.tv_sec;
    val += 0.000000001 * temp.tv_nsec;

    return val;
}





int file_exist(char *filename)
{
    struct stat   buffer;
    return (stat(filename, &buffer) == 0);
}













int COREMOD_TOOLS_mvProcCPUset(
    const char *csetname
)
{
    int pid;
    char command[200];

    pid = getpid();

#ifndef __MACH__

    if(seteuid(data.euid) != 0)     //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }

    sprintf(command, "sudo -n cset proc -m -p %d -t %s\n", pid, csetname);
    printf("Executing command: %s\n", command);

    if(system(command) != 0)
    {
        PRINT_ERROR("system() returns non-zero value");
    }

    if(seteuid(data.ruid) != 0)     //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }
#endif

    return(0);
}




int create_counter_file(
    const char *fname,
    unsigned long NBpts
)
{
    unsigned long i;
    FILE *fp;

    if((fp = fopen(fname, "w")) == NULL)
    {
        PRINT_ERROR("cannot create file \"%s\"", fname);
        abort();
    }

    for(i = 0; i < NBpts; i++)
    {
        fprintf(fp, "%ld %f\n", i, (double)(1.0 * i / NBpts));
    }

    fclose(fp);

    return(0);
}




int bubble_sort(
    double *array,
    unsigned long count
)
{
    unsigned long a, b;
    double t;

    for(a = 1; a < count; a++)
        for(b = count - 1; b >= a; b--)
            if(array[b - 1] > array[b])
            {
                t = array[b - 1];
                array[b - 1] = array[b];
                array[b] = t;
            }

    return(0);
}



void qs_float(
    float *array,
    unsigned long left, 
    unsigned long right
)
{
    unsigned long i, j;
    float x, y;

    i = left;
    j = right;
    x = array[(left + right) / 2];


    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;
            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs_float(array, left, j);
    }
    if(i < right)
    {
        qs_float(array, i, right);
    }
}




void qs_long(
    long *array,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    long x, y;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;
            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs_long(array, left, j);
    }
    if(i < right)
    {
        qs_long(array, i, right);
    }
}


void qs_double(
    double *array,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs_double(array, left, j);
    }
    if(i < right)
    {
        qs_double(array, i, right);
    }
}



void qs_ushort(
    unsigned short *array,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    unsigned short x, y;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs_ushort(array, left, j);
    }
    if(i < right)
    {
        qs_ushort(array, i, right);
    }
}



void qs3(
    double *array,
    double *array1,
    double *array2,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;
    double y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            y1 = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2 = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3(array, array1, array2, left, j);
    }
    if(i < right)
    {
        qs3(array, array1, array2, i, right);
    }
}



void qs3_float(
    float *array,
    float *array1,
    float *array2,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    float x, y;
    float y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            y1 = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2 = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3_float(array, array1, array2, left, j);
    }
    if(i < right)
    {
        qs3_float(array, array1, array2, i, right);
    }
}



void qs3_double(
    double *array,
    double *array1,
    double *array2,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;
    double y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            y1 = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2 = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3_double(array, array1, array2, left, j);
    }
    if(i < right)
    {
        qs3_double(array, array1, array2, i, right);
    }
}



void qs2l(
    double *array,
    long *array1,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;
    long l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2l(array, array1, left, j);
    }
    if(i < right)
    {
        qs2l(array, array1, i, right);
    }
}



void qs2ul(
    double *array,
    unsigned long *array1,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;
    unsigned long l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;
            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2ul(array, array1, left, j);
    }
    if(i < right)
    {
        qs2ul(array, array1, i, right);
    }
}


void qs2l_double(
    double *array,
    long *array1,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;
    long l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2l_double(array, array1, left, j);
    }
    if(i < right)
    {
        qs2l_double(array, array1, i, right);
    }
}


void qs2ul_double(
    double *array,
    unsigned long *array1,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;
    unsigned long l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2ul_double(array, array1, left, j);
    }
    if(i < right)
    {
        qs2ul_double(array, array1, i, right);
    }
}



void qs3ll_double(
    double *array,
    long *array1,
    long *array2,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;
    long l1, l2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            l2 = array2[i];
            array2[i] = array2[j];
            array2[j] = l2;

            i++;
            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3ll_double(array, array1, array2, left, j);
    }
    if(i < right)
    {
        qs3ll_double(array, array1, array2, i, right);
    }
}



void qs3ulul_double(
    double *array,
    unsigned long *array1,
    unsigned long *array2,
    unsigned long left,
    unsigned long right
)
{
    register unsigned long i, j;
    double x, y;
    unsigned long l1, l2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            l2 = array2[i];
            array2[i] = array2[j];
            array2[j] = l2;

            i++;
            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3ulul_double(array, array1, array2, left, j);
    }

    if(i < right)
    {
        qs3ulul_double(array, array1, array2, i, right);
    }
}




void quick_sort_float(
    float *array,
    unsigned long count
)
{
    qs_float(array, 0, count - 1);
}

void quick_sort_long(
    long *array,
    unsigned long count
)
{
    qs_long(array, 0, count - 1);
}

void quick_sort_double(
    double *array,
    unsigned long count
)
{
    qs_double(array, 0, count - 1);
}

void quick_sort_ushort(
    unsigned short *array,
    unsigned long count
)
{
    qs_ushort(array, 0, count - 1);
}

void quick_sort3(
    double *array,
    double *array1,
    double *array2,
    unsigned long count
)
{
    qs3(array, array1, array2, 0, count - 1);
}

void quick_sort3_float(
    float *array,
    float *array1,
    float *array2,
    unsigned long count
)
{
    qs3_float(array, array1, array2, 0, count - 1);
}

void quick_sort3_double(
    double *array,
    double *array1,
    double *array2,
    unsigned long count
)
{
    qs3_double(array, array1, array2, 0, count - 1);
}

void quick_sort2l(
    double *array,
    long *array1,
    unsigned long count
)
{
    qs2l(array, array1, 0, count - 1);
}

void quick_sort2ul(
    double *array,
    unsigned long *array1,
    unsigned long count
)
{
    qs2ul(array, array1, 0, count - 1);
}

void quick_sort2l_double(
    double *array,
    long *array1,
    unsigned long count
)
{
    qs2l_double(array, array1, 0, count - 1);
}

void quick_sort2ul_double(
    double *array,
    unsigned long *array1,
    unsigned long count
)
{
    qs2ul_double(array, array1, 0, count - 1);
}

void quick_sort3ll_double(
    double *array,
    long *array1,
    long *array2,
    unsigned long count
)
{
    qs3ll_double(array, array1, array2, 0, count - 1);
}

void quick_sort3ulul_double(
    double *array,
    unsigned long *array1,
    unsigned long *array2,
    unsigned long count
)
{
    qs3ulul_double(array, array1, array2, 0, count - 1);
}



errno_t lin_regress(
    double *a,
    double *b,
    double *Xi2,
    double *x,
    double *y,
    double *sig,
    unsigned int nb_points
)
{
    double S, Sx, Sy, Sxx, Sxy, Syy;
    unsigned int i;
    double delta;

    S = 0;
    Sx = 0;
    Sy = 0;
    Sxx = 0;
    Syy = 0;
    Sxy = 0;

    for(i = 0; i < nb_points; i++)
    {
        S += 1.0 / sig[i] / sig[i];
        Sx += x[i] / sig[i] / sig[i];
        Sy += y[i] / sig[i] / sig[i];
        Sxx += x[i] * x[i] / sig[i] / sig[i];
        Syy += y[i] * y[i] / sig[i] / sig[i];
        Sxy += x[i] * y[i] / sig[i] / sig[i];
    }

    delta = S * Sxx - Sx * Sx;
    *a = (Sxx * Sy - Sx * Sxy) / delta;
    *b = (S * Sxy - Sx * Sy) / delta;
    *Xi2 = Syy - 2 * (*a) * Sy - 2 * (*a) * (*b) * Sx + (*a) * (*a) * S + 2 *
           (*a) * (*b) * Sx - (*b) * (*b) * Sxx;

    return RETURN_SUCCESS;
}



int replace_char(
    char *content,
    char cin,
    char cout
)
{
    unsigned long i;

    for(i = 0; i < strlen(content); i++)
        if(content[i] == cin)
        {
            content[i] = cout;
        }

    return(0);
}



int read_config_parameter_exists(
    const char *config_file,
    const char *keyword
)
{
    FILE *fp;
    char line[1000];
    char keyw[200];
    int read;

    read = 0;
    if((fp = fopen(config_file, "r")) == NULL)
    {
        PRINT_ERROR("cannot open file \"%s\"", config_file);
        abort();
    }

    while((fgets(line, 1000, fp) != NULL) && (read == 0))
    {
        sscanf(line, " %20s", keyw);
        if(strcmp(keyw, keyword) == 0)
        {
            read = 1;
        }
    }
    if(read == 0)
    {
        PRINT_WARNING("parameter \"%s\" does not exist in file \"%s\"", keyword,
                      config_file);
    }

    fclose(fp);

    return(read);
}




int read_config_parameter(
    const char *config_file,
    const char *keyword,
    char *content
)
{
    FILE *fp;
    char line[1000];
    char keyw[200];
    char cont[200];
    int read;

    read = 0;
    if((fp = fopen(config_file, "r")) == NULL)
    {
        PRINT_ERROR("cannot open file \"%s\"", config_file);
        abort();
    }

    strcpy(content, "---");
    while(fgets(line, 1000, fp) != NULL)
    {
        sscanf(line, "%100s %100s", keyw, cont);
        if(strcmp(keyw, keyword) == 0)
        {
            strcpy(content, cont);
            read = 1;
        }
        /*      printf("KEYWORD : \"%s\"   CONTENT : \"%s\"\n",keyw,cont);*/
    }
    if(read == 0)
    {
        PRINT_ERROR("parameter \"%s\" does not exist in file \"%s\"", keyword,
                config_file);
        sprintf(content, "-");
        //  exit(0);
    }

    fclose(fp);

    return(read);
}




float read_config_parameter_float(const char *config_file, const char *keyword)
{
    float value;
    char content[SBUFFERSIZE];

    read_config_parameter(config_file, keyword, content);
    //printf("content = \"%s\"\n",content);
    value = atof(content);
    //printf("Value = %g\n",value);

    return(value);
}

long read_config_parameter_long(const char *config_file, const char *keyword)
{
    long value;
    char content[SBUFFERSIZE];

    read_config_parameter(config_file, keyword, content);
    value = atol(content);

    return(value);
}



int read_config_parameter_int(const char *config_file, const char *keyword)
{
    int value;
    char content[SBUFFERSIZE];

    read_config_parameter(config_file, keyword, content);
    value = atoi(content);

    return(value);
}





long file_number_lines(const char *file_name)
{
    long cnt;
    int c;
    FILE *fp;

    if((fp = fopen(file_name, "r")) == NULL)
    {
        PRINT_ERROR("cannot open file \"%s\"", file_name);
        abort();
    }

    cnt = 0;
    while((c = fgetc(fp)) != EOF)
        if(c == '\n')
        {
            cnt++;
        }
    fclose(fp);

    return(cnt);
}


FILE *open_file_w(const char *filename)
{
    FILE *fp;

    if((fp = fopen(filename, "w")) == NULL)
    {
        PRINT_ERROR("cannot create file \"%s\"", filename);
        abort();
    }

    return(fp);
}


FILE *open_file_r(const char *filename)
{
    FILE *fp;

    if((fp = fopen(filename, "r")) == NULL)
    {
        PRINT_ERROR("cannot read file \"%s\"", filename);
        abort();
    }

    return(fp);
}


errno_t write_1D_array(
    double *array,
    long nbpoints,
    const char *filename
)
{
    FILE *fp;
    long ii;

    fp = open_file_w(filename);
    for(ii = 0; ii < nbpoints; ii++)
    {
        fprintf(fp, "%ld\t%f\n", ii, array[ii]);
    }
    fclose(fp);

    return RETURN_SUCCESS;
}



errno_t read_1D_array(
    double *array,
    long nbpoints,
    const char *filename
)
{
    FILE *fp;
    long ii;
    long tmpl;

    fp = open_file_r(filename);
    for(ii = 0; ii < nbpoints; ii++)
    {
        if(fscanf(fp, "%ld\t%lf\n", &tmpl, &array[ii]) != 2)
        {
            PRINT_ERROR("fscanf error");
            exit(0);
        }
    }
    fclose(fp);

    return RETURN_SUCCESS;
}



/* test point */
errno_t tp(
    const char *word
)
{
    printf("---- Test point %s ----\n", word);
    fflush(stdout);

    return RETURN_SUCCESS;
}


int read_int_file(
    const char *fname
)
{
    int value;
    FILE *fp;

    if((fp = fopen(fname, "r")) == NULL)
    {
        value = 0;
    }
    else
    {
        if(fscanf(fp, "%d", &value) != 1)
        {
            PRINT_ERROR("fscanf error");
            exit(0);
        }
        fclose(fp);
    }

    return(value);
}



errno_t write_int_file(
    const char *fname,
    int         value
)
{
    FILE *fp;

    if((fp = fopen(fname, "w")) == NULL)
    {
        PRINT_ERROR("cannot create file \"%s\"\n", fname);
        abort();
    }

    fprintf(fp, "%d\n", value);
    fclose(fp);

    return RETURN_SUCCESS;
}



errno_t write_float_file(
    const char *fname,
    float       value
)
{
    FILE *fp;
    int mode = 0; // default, create single file

    if(variable_ID("WRITE2FILE_APPEND") != -1)
    {
        mode = 1;
    }

    if(mode == 0)
    {
        if((fp = fopen(fname, "w")) == NULL)
        {
            PRINT_ERROR("cannot create file \"%s\"\n", fname);
            abort();
        }
        fprintf(fp, "%g\n", value);
        fclose(fp);
    }

    if(mode == 1)
    {
        if((fp = fopen(fname, "a")) == NULL)
        {
            PRINT_ERROR("cannot create file \"%s\"\n", fname);
            abort();
        }
        fprintf(fp, " %g", value);
        fclose(fp);
    }

    return RETURN_SUCCESS;
}


// displays 2D image in 3D using gnuplot
//
errno_t COREMOD_TOOLS_imgdisplay3D(
    const char *IDname,
    long        step
)
{
    imageID ID;
    long xsize, ysize;
    long ii, jj;
    char cmd[512];
    FILE *fp;

    ID = image_ID(IDname);
    xsize = data.image[ID].md[0].size[0];
    ysize = data.image[ID].md[0].size[1];

    snprintf(cmd, 512, "gnuplot");

    if((fpgnuplot = popen(cmd, "w")) == NULL)
    {
        fprintf(stderr, "could not connect to gnuplot\n");
        return -1;
    }

    printf("image: %s [%ld x %ld], step = %ld\n", IDname, xsize, ysize, step);

    fprintf(fpgnuplot, "set pm3d\n");
    fprintf(fpgnuplot, "set hidden3d\n");
    fprintf(fpgnuplot, "set palette\n");
    //fprintf(gnuplot, "set xrange [0:%li]\n", image.md[0].size[0]);
    //fprintf(gnuplot, "set yrange [0:1e-5]\n");
    //fprintf(gnuplot, "set xlabel \"Mode #\"\n");
    //fprintf(gnuplot, "set ylabel \"Mode RMS\"\n");
    fflush(fpgnuplot);

    fp = fopen("pts.dat", "w");
    fprintf(fpgnuplot, "splot \"-\" w d notitle\n");
    for(ii = 0; ii < xsize; ii += step)
    {
        for(jj = 0; jj < xsize; jj += step)
        {
            fprintf(fpgnuplot, "%ld %ld %f\n", ii, jj,
                    data.image[ID].array.F[jj * xsize + ii]);
            fprintf(fp, "%ld %ld %f\n", ii, jj, data.image[ID].array.F[jj * xsize + ii]);
        }
        fprintf(fpgnuplot, "\n");
        fprintf(fp, "\n");
    }
    fprintf(fpgnuplot, "e\n");
    fflush(fpgnuplot);
    fclose(fp);


    return RETURN_SUCCESS;
}




//
// watch shared memory status image and perform timing statistics
//
imageID COREMOD_TOOLS_statusStat(
    const char *IDstat_name,
    long        indexmax
)
{
    imageID  IDout;
    int      RT_priority = 91; //any number from 0-99
    struct   sched_param schedpar;
    float    usec0 = 50.0;
    float    usec1 = 150.0;
    long long k;
    long long NBkiter = 2000000000;
    imageID  IDstat;

    unsigned short st;

    struct timespec t1;
    struct timespec t2;
    struct timespec tdiff;
    double tdisplay = 1.0; // interval
    double tdiffv1 = 0.0;
    uint32_t *sizearray;

    long cnttot;



    IDstat = image_ID(IDstat_name);

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    sizearray[0] = indexmax;
    sizearray[1] = 1;
    IDout = create_image_ID("statout", 2, sizearray, _DATATYPE_INT64, 0, 0);
    free(sizearray);

    for(st = 0; st < indexmax; st++)
    {
        data.image[IDout].array.SI64[st] = 0;
    }

    schedpar.sched_priority = RT_priority;
#ifndef __MACH__
    sched_setscheduler(0, SCHED_FIFO, &schedpar);
#endif


    printf("Measuring status distribution \n");
    fflush(stdout);

    clock_gettime(CLOCK_REALTIME, &t1);
    for(k = 0; k < NBkiter; k++)
    {
        double tdiffv;

        usleep((long)(usec0 + usec1 * (1.0 * k / NBkiter)));
        st = data.image[IDstat].array.UI16[0];
        if(st < indexmax)
        {
            data.image[IDout].array.SI64[st]++;
        }


        clock_gettime(CLOCK_REALTIME, &t2);
        tdiff = timespec_diff(t1, t2);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        if(tdiffv > tdiffv1)
        {
            tdiffv1 += tdisplay;
            printf("\n");
            printf("============== %10lld  %d  ==================\n", k, st);
            printf("\n");
            cnttot = 0;
            for(st = 0; st < indexmax; st++)
            {
                cnttot += data.image[IDout].array.SI64[st];
            }

            for(st = 0; st < indexmax; st++)
            {
                printf("STATUS  %5d    %20ld   %6.3f  \n", st, data.image[IDout].array.SI64[st],
                       100.0 * data.image[IDout].array.SI64[st] / cnttot);
            }
        }
    }


    printf("\n");
    for(st = 0; st < indexmax; st++)
    {
        printf("STATUS  %5d    %10ld\n", st, data.image[IDout].array.SI64[st]);
    }

    printf("\n");


    return(IDout);
}



