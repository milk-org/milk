#ifndef MKPIAASHAPES_FROM_RADSAG_H
#define MKPIAASHAPES_FROM_RADSAG_H

errno_t mkPIAAMshapes_from_RadSag(const char *__restrict__ piaa1Dsagfname,
                                  double   PIAAsep,
                                  double   beamrad,
                                  double   r0limfact,
                                  double   r1limfact,
                                  uint32_t size,
                                  double   beamradpix,
                                  long     NBradpts,
                                  const char *__restrict__ ID_PIAAM0_name,
                                  const char *__restrict__ ID_PIAAM1_name);

#endif
