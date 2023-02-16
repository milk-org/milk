/** @file imresize.h
 */

errno_t imresize_addCLIcmd();

long basic_resizeim(const char *imname_in,
                    const char *imname_out,
                    long        xsizeout,
                    long        ysizeout);
