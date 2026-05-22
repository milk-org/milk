#ifndef LINARFILTERPRED_INTERNAL_H
#define LINARFILTERPRED_INTERNAL_H

#include <assert.h>
#include <ctype.h>
#include <malloc.h>
#include <math.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <fitsio.h>


#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "timeutils.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "info/info.h"
#include "linopt_imtools/linopt_imtools.h"
#include "statistic/statistic.h"

#include "linARfilterPred/linARfilterPred.h"
#include "build_linPF.h"
#include "applyPF.h"

#ifdef HAVE_CUDA
#    include "linalgebra/linalgebra.h"
#endif

// Shared Data prototypes
int NBwords(const char sentence[]);

long LINARFILTERPRED_LoadASCIIfiles(double      tstart,
                                    double      dt,
                                    long        NBpt,
                                    long        NBfr,
                                    const char *IDoutname);

imageID LINARFILTERPRED_SelectBlock(const char *IDin_name,
                                    const char *IDblknb_name,
                                    long        blkNB,
                                    const char *IDout_name);

imageID linARfilterPred_repeat_shift_X(const char *IDin_name, long NBstep, const char *IDout_name);

// Shared Build prototypes
imageID LINARFILTERPRED_Build_LinPredictor(const char *IDin_name,
                                           long        PForder,
                                           float       PFlag,
                                           double      SVDeps,
                                           double      RegLambda,
                                           const char *IDoutPF_name,
                                           int         outMode,
                                           int         LOOPmode,
                                           float       LOOPgain,
                                           int         testmode);

float LINARFILTERPRED_ScanGain(char *IDin_name, float multfact, float framelag);

// Shared Apply prototypes
imageID LINARFILTERPRED_Apply_LinPredictor_RT(const char *IDfilt_name,
                                              const char *IDin_name,
                                              const char *IDout_name);

imageID LINARFILTERPRED_Apply_LinPredictor(const char *IDfilt_name,
                                           const char *IDin_name,
                                           float       PFlag,
                                           const char *IDout_name);

// Shared PF prototypes
imageID LINARFILTERPRED_PF_updatePFmatrix(const char *IDPF_name,
                                          const char *IDPFM_name,
                                          float       alpha);

imageID LINARFILTERPRED_PF_RealTimeApply(const char *IDmodevalIN_name,
                                         long        IndexOffset,
                                         int         semtrig,
                                         const char *IDPFM_name,
                                         long        NBPFstep,
                                         const char *IDPFout_name,
                                         int         nbGPU,
                                         long        loop,
                                         long        NBiter,
                                         int         SAVEMODE,
                                         float       tlag,
                                         long        PFindex);

#endif // LINARFILTERPRED_INTERNAL_H
