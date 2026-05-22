/**
 * @file timeutils.c
 */


#include "timeutils.h"

#include "milkDebugTools.h"

#ifndef RETURN_SUCCESS
#    define RETURN_SUCCESS 0
#endif


/**
 * @brief Gets the current time using the standard milk clock.
 */
errno_t milk_clock_gettime(struct timespec *tnow_p)
{
    return clock_gettime(CLOCK_MILK, tnow_p);
}


/**
 * @brief Format a timespec as an ISO 8601 UTC string with
 *        nanosecond precision.
 *
 * Output format: "YYYY-MM-DDThh:mm:ss.nnnnnnnnnZ"
 *
 * @param timestring  Buffer of at least TIMESTRINGLEN bytes
 * @param tnow        Timestamp to format
 * @return RETURN_SUCCESS
 */
errno_t mkUTtimestring_nanosec(char *timestring, struct timespec tnow)
{
    struct tm *uttime;
    time_t     tvsec0;

    tvsec0 = tnow.tv_sec;
    uttime = gmtime(&tvsec0);


    {
        int slen = snprintf(timestring, TIMESTRINGLEN, "%04d-%02d-%02dT%02d:%02d:%02d.%09ldZ",
                            1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday,
                            uttime->tm_hour, uttime->tm_min, uttime->tm_sec, tnow.tv_nsec);
        if (slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if (slen >= TIMESTRINGLEN)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    return RETURN_SUCCESS;
}


/**
 * @brief Formats the current time as a UT string with nanosecond precision.
 */
errno_t mkUTtimestring_nanosec_now(char *timestring)
{
    struct timespec tnow;

    clock_gettime(CLOCK_MILK, &tnow);
    mkUTtimestring_nanosec(timestring, tnow);

    return RETURN_SUCCESS;
}


/**
 * @brief Format a timespec as an ISO 8601 UTC string with
 *        microsecond precision.
 *
 * Output format: "YYYY-MM-DDThh:mm:ss.uuuuuuZ"
 *
 * @param timestring  Buffer of at least TIMESTRINGLEN bytes
 * @param tnow        Timestamp to format
 * @return RETURN_SUCCESS
 */
errno_t mkUTtimestring_microsec(char *timestring, struct timespec tnow)
{
    struct tm *uttime;
    time_t     tvsec0;

    tvsec0 = tnow.tv_sec;
    uttime = gmtime(&tvsec0);

    {
        int slen =
            snprintf(timestring, TIMESTRINGLEN, "%04d-%02d-%02dT%02d:%02d:%02d.%06ldZ",
                     1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday, uttime->tm_hour,
                     uttime->tm_min, uttime->tm_sec, (long) (tnow.tv_nsec / 1000));
        if (slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if (slen >= TIMESTRINGLEN)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }


    return RETURN_SUCCESS;
}


/**
 * @brief Formats the current time as a UT string with microsecond precision.
 */
errno_t mkUTtimestring_microsec_now(char *timestring)
{
    struct timespec tnow;

    clock_gettime(CLOCK_MILK, &tnow);
    mkUTtimestring_microsec(timestring, tnow);

    return RETURN_SUCCESS;
}


/**
 * @brief Format a timespec as an ISO 8601 UTC string with
 *        millisecond precision.
 *
 * Output format: "YYYY-MM-DDThh:mm:ss.mmmZ"
 *
 * @param timestring  Buffer of at least TIMESTRINGLEN bytes
 * @param tnow        Timestamp to format
 * @return RETURN_SUCCESS
 */
errno_t mkUTtimestring_millisec(char *timestring, struct timespec tnow)
{
    struct tm *uttime;
    time_t     tvsec0;

    tvsec0 = tnow.tv_sec;
    uttime = gmtime(&tvsec0);


    {
        int slen =
            snprintf(timestring, TIMESTRINGLEN, "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
                     1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday, uttime->tm_hour,
                     uttime->tm_min, uttime->tm_sec, (long) (tnow.tv_nsec / 1000000));
        if (slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if (slen >= TIMESTRINGLEN)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Formats the current time as a UT string with millisecond precision.
 */
errno_t mkUTtimestring_millisec_now(char *timestring)
{
    struct timespec tnow;

    clock_gettime(CLOCK_MILK, &tnow);
    mkUTtimestring_millisec(timestring, tnow);

    return RETURN_SUCCESS;
}


/**
 * @brief Format a timespec as an ISO 8601 UTC string with
 *        second precision.
 *
 * Output format: "YYYY-MM-DDThh:mm:ssZ"
 *
 * @param timestring  Buffer of at least TIMESTRINGLEN bytes
 * @param tnow        Timestamp to format
 * @return RETURN_SUCCESS
 */
errno_t mkUTtimestring_sec(char *timestring, struct timespec tnow)
{
    struct tm *uttime;
    time_t     tvsec0;

    tvsec0 = tnow.tv_sec;
    uttime = gmtime(&tvsec0);


    {
        int slen = snprintf(timestring, TIMESTRINGLEN, "%04d-%02d-%02dT%02d:%02d:%02dZ",
                            1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday,
                            uttime->tm_hour, uttime->tm_min, uttime->tm_sec);
        if (slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if (slen >= TIMESTRINGLEN)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    return RETURN_SUCCESS;
}


/**
 * @brief Formats the current time as a UT string with second precision.
 */
errno_t mkUTtimestring_sec_now(char *timestring)
{
    struct timespec tnow;

    clock_gettime(CLOCK_MILK, &tnow);
    mkUTtimestring_sec(timestring, tnow);

    return RETURN_SUCCESS;
}


/**
 * @brief Compute the difference between two timespecs
 *        as a timespec.
 *
 * Returns (end - start), handling nanosecond borrow.
 *
 * @param start  Earlier timestamp
 * @param end    Later timestamp
 * @return Difference as struct timespec
 */
struct timespec timespec_diff(struct timespec start, struct timespec end)
{
    struct timespec temp;

    if ((end.tv_nsec - start.tv_nsec) < 0)
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


/**
 * @brief Compute the difference between two timespecs
 *        as a double (seconds).
 *
 * @param start  Earlier timestamp
 * @param end    Later timestamp
 * @return Difference in seconds
 */
double timespec_diff_double(struct timespec start, struct timespec end)
{
    struct timespec temp;
    double          val;

    if ((end.tv_nsec - start.tv_nsec) < 0)
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
    clock_gettime(CLOCK_MILK, &tsnow);
    double tdoublenow = 1.0 * tsnow.tv_sec + 1.0e-9 * tsnow.tv_nsec;
    printf("TIME double NOW : %lf\n", tdoublenow);

    printf("DATE from timedouble_to_UTC_timeofdaystring: %04d-%02d-%02d  %02d:%02d:%02d  %05.2f\n",
           1900 + timetm->tm_year, 1 + timetm->tm_mon, 1 + timetm->tm_mday, timetm->tm_hour,
           timetm->tm_min, timetm->tm_sec, sec);

    snprintf(tstring, 12, "%02d:%02d:%05.2f", timetm->tm_hour, timetm->tm_min, sec);

    return tstring;
}
