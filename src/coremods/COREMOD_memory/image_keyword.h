/**
 * @file image_keyword.h
 * @brief Image keyword read/write
 */

errno_t
CLIADDCMD_COREMOD_memory__image_keyword();

long image_write_keyword_L(
    const char *IDname,
    const char *kname,
    long       value,
    const char *comment);

long image_write_keyword_D(
    const char *IDname,
    const char *kname,
    double     value,
    const char *comment);

long image_write_keyword_S(
    const char *IDname,
    const char *kname,
    const char *value,
    const char *comment);

errno_t image_keyword_addL(
    IMGID      img,
    const char *kwname,
    long       kwval,
    const char *comment);
errno_t image_keyword_addD(
    IMGID      img,
    const char *kwname,
    double     kwval,
    const char *comment);
errno_t image_keyword_addS(
    IMGID      img,
    const char *kwname,
    const char *kwval,
    const char *comment);

imageID image_list_keywords(
    const char *restrict IDname);

long image_read_keyword_D(
    const char *IDname,
    const char *kname,
    double     *val);

long image_read_keyword_L(
    const char *IDname,
    const char *kname,
    long       *val);
