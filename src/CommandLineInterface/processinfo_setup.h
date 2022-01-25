#ifndef _PROCESSINFO_SETUP_H
#define _PROCESSINFO_SETUP_H

PROCESSINFO *processinfo_setup(char       *pinfoname,
                               const char *descriptionstring,
                               const char *msgstring,
                               const char *functionname,
                               const char *filename,
                               int         linenumber);

errno_t processinfo_error(PROCESSINFO *processinfo, char *errmsgstring);

errno_t processinfo_loopstart(PROCESSINFO *processinfo);

#endif
