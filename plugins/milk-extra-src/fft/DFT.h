#ifndef FFT_DFT_H
#define FFT_DFT_H

errno_t fft_DFT(const char *IDin_name,
                const char *IDinmask_name,
                const char *IDout_name,
                const char *IDoutmask_name,
                double      Zfactor,
                int         dir,
                long        kin,
                imageID    *outID);

errno_t fft_DFTinsertFPM(const char *pupin_name,
                         const char *fpmz_name,
                         double      zfactor,
                         const char *pupout_name,
                         imageID    *outID);

errno_t fft_DFTinsertFPM_re(const char *pupin_name,
                            const char *fpmz_name,
                            double      zfactor,
                            const char *pupout_name,
                            imageID    *outID);

#endif
