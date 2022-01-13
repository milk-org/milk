/**
 * @file mvprocCPUset.h
 */

errno_t mvprocTset_addCLIcmd();
errno_t mvprocTsetExt_addCLIcmd();
errno_t mvprocCPUset_addCLIcmd();
errno_t mvprocCPUsetExt_addCLIcmd();

int COREMOD_TOOLS_mvProcTset(const char *tsetspec);
int COREMOD_TOOLS_mvProcTsetExt(const int pid, const char *tsetspec);
int COREMOD_TOOLS_mvProcCPUset(const char *csetname);
int COREMOD_TOOLS_mvProcCPUsetExt(const int pid, const char *csetname, const int rtprio);
