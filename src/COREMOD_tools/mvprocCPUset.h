/**
 * @file mvprocCPUset.h
 */

errno_t mvprocCPUset_addCLIcmd();
errno_t mvprocCPUsetExt_addCLIcmd();

int COREMOD_TOOLS_mvProcCPUset(const char *csetname);
int COREMOD_TOOLS_mvProcCPUsetExt(int pid, const char *csetname, int rtprio);
