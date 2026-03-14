/**
 * @file processinfo_scan_shm.h
 * @brief Shared memory structure for scanner daemon
 */

#ifndef _PROCESSINFO_SCAN_SHM_H
#define _PROCESSINFO_SCAN_SHM_H

#include "processtools.h" // for PROCESSINFODISP

#define PROCESSINFO_SCAN_SHM_NAME "processinfo.scan.shm"

// Shared memory structure for scanner daemon
// This holds the results of the scan (CPU loads, process details)
// And flags from readers (TUIs) to request specific data.

typedef struct
{
    // System-wide CPU loads
    int NBcpus;
    float CPUload[MAXNBCPU];
    long long CPUcnt0[MAXNBCPU]; // Needed for diff calculation?
    // The current GetCPUloads uses previous values to compute load.
    // We need to store previous values here or in local memory of scanner.
    // If multiple scanners? No, one scanner.
    // So scanner can keep history local. But if we restart scanner?
    // Let's keep history in SHM so it persists? Or just local static in scanner.
    // TUI expects 'float CPUload' (0.0 to 1.0).
    
    // Process Scan Results & Requests
    // Indexed by pindex (same as processinfo.list.shm)
    
    uint8_t request_scan[PROCESSINFOLISTSIZE]; // 1 if any reader wants details for this index
    
    // Aggregated stats for display
    // We assume PROCESSINFODISP is sufficient for display
    PROCESSINFODISP pinfodisp[PROCESSINFOLISTSIZE];

    int sorted_pindex[PROCESSINFOLISTSIZE]; // Pre-sorted list of indices (newest first)
    int NBactive;                           // Total number of entries in sorted_pindex (active+stopped+crashed)

    int NBreaders; // Number of active TUI instances

} PROCSCAN_SHM;

#endif
