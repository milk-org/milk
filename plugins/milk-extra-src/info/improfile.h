/**
 * @file improfile.h
 * @brief Radial profile
 */

errno_t CLIADDCMD_info__improfile();

errno_t profile(
    const char *ID_name,
    const char *outfile,
    double      xcenter,
    double      ycenter,
    double      step,
    long        nb_step);

errno_t profile2im(
    const char   *profile_name,
    long          nbpoints,
    unsigned long size,
    double        xcenter,
    double        ycenter,
    double        radius,
    const char   *out);
