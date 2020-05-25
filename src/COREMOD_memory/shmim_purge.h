/**
 * @file    shmim_purge.h
 */


#ifndef _SHMIM_PURGE_H
#define _SHMIM_PURGE_H



errno_t shmim_purge_addCLIcmd();


errno_t    shmim_purge(
    const char *strfilter
);

#endif
