/** @file loadCR2toFITSRGB.h
 */

errno_t loadCR2toFITSRGB_addCLIcmd();

errno_t loadCR2toFITSRGB(const char *__restrict fnameCR2,
                         const char *__restrict fnameFITSr,
                         const char *__restrict fnameFITSg,
                         const char *__restrict fnameFITSb);
