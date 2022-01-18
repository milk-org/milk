/**
 * @file    logshmim.h
 */

#ifndef CLICORE_MEMORY_LOGSHMIM_H
#define CLICORE_MEMORY_LOGSHMIM_H

errno_t logshmim_addCLIcmd();

void *save_fits_function(void *ptr);

errno_t COREMOD_MEMORY_logshim_printstatus(const char *IDname);

errno_t COREMOD_MEMORY_logshim_set_on(const char *IDname, int setv);

errno_t COREMOD_MEMORY_logshim_set_logexit(const char *IDname, int setv);

errno_t COREMOD_MEMORY_sharedMem_2Dim_log(const char *IDname, uint32_t zsize, const char *logdir,
                                          const char *IDlogdata_name);

#endif
