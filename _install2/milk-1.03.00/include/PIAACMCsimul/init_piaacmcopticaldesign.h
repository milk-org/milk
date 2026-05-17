#ifndef PIAACMCSIMUL_INIT_PIAACMCOPTICALDESIGN_H
#define PIAACMCSIMUL_INIT_PIAACMCOPTICALDESIGN_H

#define INIT_PIAACMCOPTICALDESIGN_MODE__DEFAULT 0x0000000000000000

#define INIT_PIAACMCOPTICALDESIGN_MODE__READCONF        0x0000000000000001
#define INIT_PIAACMCOPTICALDESIGN_MODE__LOADPIAACMCCONF 0x0000000000000002

#define INIT_PIAACMCOPTICALDESIGN_MODE__FPMPHYSICAL 0x0000000000000004

#define INIT_PIAACMCOPTICALDESIGN_MODE__WSCMODE 0x0000000000000008

errno_t init_piaacmcopticaldesign(double    fpmradld,
                                  double    centobs0,
                                  double    centobs1,
                                  uint64_t  flags,
                                  uint64_t *status);

#endif
