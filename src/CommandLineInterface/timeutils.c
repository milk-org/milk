/**
 * @file timeutils.c
 */

#include "CommandLineInterface/CLIcore.h"
#include <time.h>

#define CLOCK_MILK CLOCK_TAI
// handles leap seconds better than CLOCK_REALTIME

errno_t mkUTtimestring_nanosec(char *timestring, struct timespec tnow)
{
    struct tm *uttime;
    time_t     tvsec0;

    tvsec0 = tnow.tv_sec;
    uttime = gmtime(&tvsec0);

    sprintf(timestring,
            "%04d-%02d-%02dT%02d:%02d:%02d.%09ldZ",
            1900 + uttime->tm_year,
            1 + uttime->tm_mon,
            uttime->tm_mday,
            uttime->tm_hour,
            uttime->tm_min,
            uttime->tm_sec,
            tnow.tv_nsec);

    return RETURN_SUCCESS;
}

errno_t mkUTtimestring_nanosec_now(char *timestring)
{
    struct timespec tnow;

    clock_gettime(CLOCK_MILK, &tnow);
    mkUTtimestring_nanosec(timestring, tnow);

    return RETURN_SUCCESS;
}

errno_t mkUTtimestring_microsec(char *timestring, struct timespec tnow)
{
    struct tm *uttime;
    time_t     tvsec0;

    tvsec0 = tnow.tv_sec;
    uttime = gmtime(&tvsec0);

    sprintf(timestring,
            "%04d-%02d-%02dT%02d:%02d:%02d.%06ldZ",
            1900 + uttime->tm_year,
            1 + uttime->tm_mon,
            uttime->tm_mday,
            uttime->tm_hour,
            uttime->tm_min,
            uttime->tm_sec,
            (long)(tnow.tv_nsec / 1000));

    return RETURN_SUCCESS;
}

errno_t mkUTtimestring_microsec_now(char *timestring)
{
    struct timespec tnow;

    clock_gettime(CLOCK_MILK, &tnow);
    mkUTtimestring_microsec(timestring, tnow);

    return RETURN_SUCCESS;
}

errno_t mkUTtimestring_millisec(char *timestring, struct timespec tnow)
{
    struct tm *uttime;
    time_t     tvsec0;

    tvsec0 = tnow.tv_sec;
    uttime = gmtime(&tvsec0);

    sprintf(timestring,
            "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
            1900 + uttime->tm_year,
            1 + uttime->tm_mon,
            uttime->tm_mday,
            uttime->tm_hour,
            uttime->tm_min,
            uttime->tm_sec,
            (long)(tnow.tv_nsec / 1000000));

    return RETURN_SUCCESS;
}

errno_t mkUTtimestring_millisec_now(char *timestring)
{
    struct timespec tnow;

    clock_gettime(CLOCK_MILK, &tnow);
    mkUTtimestring_millisec(timestring, tnow);

    return RETURN_SUCCESS;
}

errno_t mkUTtimestring_sec(char *timestring, struct timespec tnow)
{
    struct tm *uttime;
    time_t     tvsec0;

    tvsec0 = tnow.tv_sec;
    uttime = gmtime(&tvsec0);

    sprintf(timestring,
            "%04d-%02d-%02dT%02d:%02d:%02dZ",
            1900 + uttime->tm_year,
            1 + uttime->tm_mon,
            uttime->tm_mday,
            uttime->tm_hour,
            uttime->tm_min,
            uttime->tm_sec);

    return RETURN_SUCCESS;
}

errno_t mkUTtimestring_sec_now(char *timestring)
{
    struct timespec tnow;

    clock_gettime(CLOCK_MILK, &tnow);
    mkUTtimestring_sec(timestring, tnow);

    return RETURN_SUCCESS;
}




struct timespec timespec_diff(struct timespec start, struct timespec end)
{
    struct timespec temp;

    if((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec  = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec  = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}



double timespec_diff_double(struct timespec start, struct timespec end)
{
    struct timespec temp;
    double          val;

    if((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec  = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec  = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }

    val = temp.tv_sec;
    val += 0.000000001 * temp.tv_nsec;

    return val;
}




/**
 * @brief Returns time string in form of HH:MM:SS.SS
 *
 * @param timedouble Unix time
 * @return char*
 */
char *timedouble_to_UTC_timeofdaystring(double timedouble)
{
    char *tstring = malloc(12);

    time_t     timet  = (time_t) timedouble;
    struct tm *timetm = gmtime(&timet);

    float sec = 1.0 * timetm->tm_sec + timedouble - (long) timedouble;

    printf("TIME double     : %lf\n", timedouble);

    struct timespec tsnow;
    clock_gettime(CLOCK_REALTIME, &tsnow);
    double tdoublenow = 1.0 * tsnow.tv_sec + 1.0e-9 * tsnow.tv_nsec;
    printf("TIME double NOW : %lf\n", tdoublenow);

    printf("DATE: %04d-%02d-%02d  %02d:%02d:%02d  %05.2f\n",
           1900 + timetm->tm_year,
           1 + timetm->tm_mon,
           1 + timetm->tm_mday,
           timetm->tm_hour,
           timetm->tm_min,
           timetm->tm_sec,
           sec);

    sprintf(tstring, "%02d:%02d:%05.2f", timetm->tm_hour, timetm->tm_min, sec);

    return tstring;
}
