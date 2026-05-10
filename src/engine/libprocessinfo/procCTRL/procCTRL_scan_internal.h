/**
 * @file procCTRL_scan_internal.h
 * @brief Internal headers for procCTRL scanner modules
 */

#ifndef procCTRL_scan_internal_H
#define procCTRL_scan_internal_H

#include "processinfo.h"
#include "processinfo_scan_shm.h"
#include "milkDebugTools.h"
#include <sys/mman.h>

// Global mappings for the scanner to avoid repeated map/unmap
extern PROCESSINFO *pinfo_mappings[PROCESSINFOLISTSIZE];
extern int          pinfo_fds[PROCESSINFOLISTSIZE];

// Extracted helpers
void Scan_GetCPUloads(PROCSCAN_SHM *scan_shm);
void rebuild_process_list(const char *procdname, PROCSCAN_SHM *scan_shm);
void scan_update_pinfolist_status(const char *procdname, PROCSCAN_SHM *scan_shm, int *active_count, int *stopped_count, int *crashed_count, double *timearray, long *indexarray, int *listcnt);
void scan_update_process_details(const char *procdname, PROCSCAN_SHM *scan_shm, int *serviced_count);

#endif // procCTRL_scan_internal_H
