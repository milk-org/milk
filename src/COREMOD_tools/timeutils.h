/**
 * @file timeutils.h
 */

struct timespec timespec_diff(
    struct timespec start,
    struct timespec end
);

double timespec_diff_double(
    struct timespec start,
    struct timespec end
);
