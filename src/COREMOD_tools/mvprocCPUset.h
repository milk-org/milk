/**
 * @file mvprocCPUset.h
 */

errno_t cpuset_utils_addCLIcmd();

int COREMOD_TOOLS_mvProcRTPrio(const int rtprio);
int COREMOD_TOOLS_mvProcTset(const char *tsetspec);
int COREMOD_TOOLS_mvProcTsetExt(const int pid, const char *tsetspec);
int COREMOD_TOOLS_mvProcCPUset(const char *csetname);
int COREMOD_TOOLS_mvProcCPUsetExt(const int   pid,
                                  const char *csetname,
                                  const int   rtprio);
