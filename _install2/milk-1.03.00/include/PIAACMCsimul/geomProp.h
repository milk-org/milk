#ifndef PIAACMCSIMUL_LYOTSTOP_GEOMPROP_H
#define PIAACMCSIMUL_LYOTSTOP_GEOMPROP_H

errno_t CLIADDCMD_PIAACMCsimul__LyotStop__PIAACMCsimul_geomProp();

errno_t PIAACMCsimul_geomProp(const char *__restrict__ IDin_name,
                              const char *__restrict__ IDsag_name,
                              const char *__restrict__ IDout_name,
                              const char *__restrict__ IDoutcnt_name,
                              double   drindex,
                              double   pscale,
                              double   zprop,
                              double   krad,
                              double   kstep,
                              double   rlim,
                              imageID *outID);

#endif
