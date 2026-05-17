#ifndef PIAACMCSIMUL_LOADPIAACMCCONF_H
#define PIAACMCSIMUL_LOADPIAACMCCONF_H

errno_t PIAACMCsimul_loadpiaacmcconf(const char *dname);

errno_t PIAACMCsimul_update_fnamedescr_conf();

errno_t PIAACMCsimul_update_fnamedescr();

errno_t PIAACMCsimul_savepiaacmcconf(const char *__restrict dname);

#endif
