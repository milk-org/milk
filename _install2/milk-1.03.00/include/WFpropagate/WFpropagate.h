#ifndef _WFPROPAGATEMODULE_H
#define _WFPROPAGATEMODULE_H

int Init_Fresnel_propagate_wavefront(char  *Cim,
                                     long   size,
                                     double PUPIL_SCALE,
                                     double z,
                                     double lambda,
                                     double FPMASKRAD,
                                     int    Precision);

errno_t Fresnel_propagate_wavefront(const char *restrict in,
                                    const char *restrict out,
                                    double PUPIL_SCALE,
                                    double z,
                                    double lambda);

int Fresnel_propagate_wavefront1(char *in, char *out, char *Cin);

//long WFpropagate_run();

#endif
