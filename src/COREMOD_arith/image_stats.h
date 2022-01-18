/**
 * @file    image_stats.h
 *
 *
 */

double arith_image_mean(const char *ID_name);

double arith_image_min(const char *ID_name);

double arith_image_max(const char *ID_name);

double arith_image_percentile(const char *ID_name, double fraction);

double arith_image_median(const char *ID_name);
