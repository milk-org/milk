/**
 * @file    image_keyword.h
 */

errno_t image_keyword_addCLIcmd();

long image_write_keyword_L(const char *IDname, const char *kname, long value, const char *comment);

long image_write_keyword_D(const char *IDname, const char *kname, double value, const char *comment);

long image_write_keyword_S(const char *IDname, const char *kname, const char *value, const char *comment);

long image_list_keywords(const char *IDname);

long image_read_keyword_D(const char *IDname, const char *kname, double *val);

long image_read_keyword_L(const char *IDname, const char *kname, long *val);
