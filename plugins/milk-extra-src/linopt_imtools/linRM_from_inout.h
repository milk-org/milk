#ifndef LINOPT_IMTOOLS__LINRM_FROM_INOUT_H
#define LINOPT_IMTOOLS__LINRM_FROM_INOUT_H

errno_t CLIADDCMD_linopt_imtools__linRM_from_inout();

errno_t linopt_compute_linRM_from_inout(const char *IDinput_name,
                                        const char *IDinmask_name,
                                        const char *IDoutput_name,
                                        const char *IDRM_name,
                                        imageID    *outID);

#endif
