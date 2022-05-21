/**
 * @file    fps_outlog.h
 * @brief   output log functions for FPS
 */

#ifndef FPS_OUTLOG_H
#define FPS_OUTLOG_H

errno_t getFPSlogfname(char *logfname);

errno_t functionparameter_outlog_file(char *keyw, char *msgstring, FILE *fpout);

errno_t functionparameter_outlog(char *keyw, const char *fmt, ...);

errno_t functionparameter_outlog_namelink();

#endif
