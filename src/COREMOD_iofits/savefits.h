/**
 * @file    savefits.h
 */


#ifndef MILK_COREMOD_IOFITS_SAVEFITS_H
#define MILK_COREMOD_IOFITS_SAVEFITS_H

//errno_t savefits_addCLIcmd();

errno_t CLIADDCMD_COREMOD_iofits__saveFITS();


errno_t saveFITS(
    const char *__restrict inputimname,
    const char *__restrict outputFITSname,
    int outputbitpix,
    const char *__restrict importheaderfile,
    IMAGE_KEYWORD *kwarray,
    int kwarraysize
);

errno_t save_fits(
    const char *__restrict inputimname,
    const char *__restrict outputFITSname
);

errno_t save_fl_fits(
    const char *__restrict inputimname,
    const char *__restrict outputFITSname
);

errno_t save_db_fits(
    const char *__restrict inputimname,
    const char *__restrict outputFITSname
);

#endif
