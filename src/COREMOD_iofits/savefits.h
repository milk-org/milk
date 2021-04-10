/**
 * @file    savefits.h
 */


#ifndef MILK_COREMOD_IOFITS_SAVEFITS_H
#define MILK_COREMOD_IOFITS_SAVEFITS_H

//errno_t savefits_addCLIcmd();

errno_t CLIADDCMD_COREMOD_iofits__saveFITS();


errno_t saveFITS(
    const char *restrict inputimname,
    const char *restrict outputFITSname,
    int outputbitpix,
    const char *restrict importheaderfile
);

errno_t save_fits(
    const char *restrict inputimname,
    const char *restrict outputFITSname
);

errno_t save_fl_fits(
    const char *restrict inputimname,
    const char *restrict outputFITSname
);


#endif
