/**
 * @file timeutils.h
 */

#ifndef _CLICORE_TIMEUTILS_H
#define _CLICORE_TIMEUTILS_H

// holds "%04d-%02d-%02dT%02d:%02d:%02d.%09ldZ" + \0 + 1 char extra
#define TIMESTRINGLEN 32


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
