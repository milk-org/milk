/**
 * @file CR2toFITS.h
 * @brief Cr2tofits module
 */

/** @file CR2toFITS.h
 */

errno_t CR2toFITS_addCLIcmd();

imageID CR2toFITS(const char *__restrict fnameCR2,
                  const char *__restrict fnameFITS);
