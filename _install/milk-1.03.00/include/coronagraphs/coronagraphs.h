
#ifndef _CORONAGRAPHS_H
#define _CORONAGRAPHS_H

#define CORONAGRAPHS_ARRAYSIZE 4096

/** @brief Initialize module. */
void __attribute__((constructor)) libinit_coronagraphs();

/** @brief Initialize command line interface. */
errno_t init_coronagraphs();

errno_t coronagraph_make_2Dprolate(double      masksizepix,
                                   double      beamradpix,
                                   double      centralObs,
                                   const char *outname,
                                   long        size,
                                   const char *pupmask_name,
                                   double     *transmval);

errno_t coronagraph_make_2Dprolateld(double      masksizeld,
                                     double      beamradpix,
                                     double      centralObs,
                                     const char *outname,
                                     long        size,
                                     const char *pupmask_name);

errno_t coronagraph_update_2Dprolate(double masksizeld,
                                     double beamradpix,
                                     double centralObs,
                                     double zfactor);

errno_t coronagraph_make_2Dprolate_CS(double      masksize,
                                      double      centralObs,
                                      const char *outname);

errno_t coronagraph_APLCapo_compile();

errno_t coronagraph_init_PIAA();

errno_t coronagraphs_make_SUBARU_pupil();

errno_t coronagraphs_PIAA_apodize_beam(const char *ampl1,
                                       const char *opd1,
                                       const char *ampl2,
                                       const char *opd2);

errno_t coronagraph_simul_AIC(double xld, double yld, const char *psfname);

errno_t coronagraph_simul_4QPM(double xld, double yld, const char *psfname);

errno_t coronagraph_simul_BL8(double xld, double yld, const char *psfname);

errno_t coronagraph_simul_RRPM(double xld, double yld, const char *psfname);

errno_t coronagraph_simul_OVC(double xld, double yld, const char *psfname);

errno_t coronagraph_simul_CPA(double xld, double yld, const char *psfname);

errno_t coronagraph_simul_PIAA(double xld, double yld, const char *psfname);

errno_t coronagraph_simul_PIAAC(double xld, double yld, const char *psfname);

errno_t
coronagraph_simul_AIC_PIAAC(double xld, double yld, const char *psfname);

errno_t
coronagraph_simul_MULTISTEP_APLC(double xld, double yld, const char *psfname);

errno_t coronagraph_simulPSF(double      xld,
                             double      yld,
                             const char *psfname,
                             long        coronagraph_type,
                             const char *options);

errno_t coronagraph_transm(const char *fname,
                           long        coronagraph_type,
                           double      logcontrast,
                           const char *options);

errno_t coronagraph_userfunc();

errno_t coronagraph_compute_limitcoeff();

errno_t CORONAGRAPHS_scanPIAACMC_centObs_perf(double obs0input);

#endif
