
#ifndef CMDSETTINGS_H
#define CMDSETTINGS_H

// command supports FPS mode
#define CLICMDFLAG_FPS 0x00000001

// processinfo enabled
#define CLICMDFLAG_PROCINFO 0x00000002

// Function attributes
// These values are copied to processinfo upon function startup
typedef struct
{
    uint64_t flags;

    long procinfo_loopcntMax;
    int64_t *procinfo_loopcntMax_ptr;

    // triggering
    int             triggermode;
    int64_t        *triggermodeptr;

    char            triggerstreamname[STRINGMAXLEN_IMAGE_NAME];

    struct timespec triggerdelay;
    struct timespec *triggerdelayptr;

    struct timespec triggertimeout;
    struct timespec *triggertimeoutptr;


    int             semindexrequested;

    int             RT_priority; // -1 if unused. 0-99 for higher priority
    cpu_set_t       CPUmask;

    int             procinfo_MeasureTiming;

} CMDSETTINGS;

#endif
