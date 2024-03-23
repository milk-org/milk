/**
 * @file timeutils.h
 */

#ifndef _CLICORE_TIMEUTILS_H
#define _CLICORE_TIMEUTILS_H

//#include <errno.h> // errno_t

// holds "%04d-%02d-%02dT%02d:%02d:%02d.%09ldZ" + \0 + 1 char extra
#define TIMESTRINGLEN 32

// handles leap seconds better than CLOCK_REALTIME
// Really we should go get CLOCK_ISIO here
#ifdef CLOCK_TAI
    #define CLOCK_MILK CLOCK_TAI
#else
    #define CLOCK_MILK CLOCK_REALTIME
#endif
#define TZ_MILK_STR "HST" // Name of timezone to use in FITS headers.
#define TZ_MILK_UTC_OFF -36000.0 // Offset east of UTC in seconds for TZ_MILK_STR

// Don't move above #definitions, this an unresolved circular import of hell
#include "CommandLineInterface/CLIcore.h" // errno_t

errno_t milk_clock_gettime(struct timespec *tnow_p);

errno_t mkUTtimestring_nanosec(char *timestring, struct timespec tnow);
errno_t mkUTtimestring_nanosec_now(char *timestring);

errno_t mkUTtimestring_microsec(char *timestring, struct timespec tnow);
errno_t mkUTtimestring_microsec_now(char *timestring);

errno_t mkUTtimestring_millisec(char *timestring, struct timespec tnow);
errno_t mkUTtimestring_millisec_now(char *timestring);

errno_t mkUTtimestring_sec(char *timestring, struct timespec tnow);
errno_t mkUTtimestring_sec_now(char *timestring);

struct timespec timespec_diff(struct timespec start, struct timespec end);

double timespec_diff_double(struct timespec start, struct timespec end);

char *timedouble_to_UTC_timeofdaystring(double timedouble);

#endif
