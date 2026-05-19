/**
 * @file image_stats.h
 * @brief Image stats module
 */

/**
 * @file    image_stats.h
 *
 *
 */

#include <libfps/IMGID.h>

double arith_image_mean(const char *ID_name);
double arith_image_mean_IMGID(IMGID *imgin);

double arith_image_min(const char *ID_name);
double arith_image_min_IMGID(IMGID *imgin);

double arith_image_max(const char *ID_name);
double arith_image_max_IMGID(IMGID *imgin);

double arith_image_percentile(
    const char *ID_name,
    double     fraction);
double arith_image_percentile_IMGID(
    IMGID  *imgin,
    double fraction);

double arith_image_median(const char *ID_name);
double arith_image_median_IMGID(IMGID *imgin);

double arith_image_dot(
    const char *ID1_name,
    const char *ID2_name);
double arith_image_dot_IMGID(
    IMGID *imgin1,
    IMGID *imgin2);

double arith_image_norm(const char *ID_name);
double arith_image_norm_IMGID(IMGID *imgin);
