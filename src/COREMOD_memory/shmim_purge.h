#ifndef COREMOD_MEMORY_SHMIM_PURGE_H
#define COREMOD_MEMORY_SHMIM_PURGE_H

errno_t CLIADDCMD_COREMOD_memory__shmim_purge();

errno_t shmim_purge(const char *strfilter);

#endif
