/**
 * @file    milkdata_macros.h
 * @brief   Shorthand macros for MILK_DATA fields
 *
 * Prefix: dc (data core)
 * Avoids underscore for quick typing.
 */

#ifndef MILKDATA_MACROS_H
#define MILKDATA_MACROS_H

/* Image array */
#define dcimg (milk_data.image)
#define dcnimg (milk_data.NB_MAX_IMAGE)

/* Variable array */
#define dcvar (milk_data.variable)
#define dcnvar (milk_data.NB_MAX_VARIABLE)

/* FPS */
#define dcfpsarr (milk_data.fpsarray)
#define dcnfps (milk_data.NB_MAX_FPS)
#define dcfpsptr (milk_data.fpsptr)
#define dcfpscode (milk_data.FPS_CMDCODE)
#define dcfpsname (milk_data.FPS_name)
#define dcfpststamp (milk_data.FPS_TIMESTAMP)

/* SHM */
#define dcshmdir (milk_data.shmdir)
#define dcshmsemdir (milk_data.shmsemdirname)

/* Runtime config */
#define dcquiet (milk_data.quiet)
#define dcdebug (milk_data.Debug)
#define dcexitcode (milk_data.exitcode)
#define dcprecision (milk_data.precision)
#define dcshareddft (milk_data.SHARED_DFT)
#define dcoverwrite (milk_data.overwrite)
#define dcrmshm (milk_data.rmSHMfile)
#define dcerrorexit (milk_data.errorexit)

/* Signals */
#define dcsigact (milk_data.sigact)
#define dcsigUSR1 (milk_data.signal_USR1)
#define dcsigUSR2 (milk_data.signal_USR2)
#define dcsigTERM (milk_data.signal_TERM)
#define dcsigINT (milk_data.signal_INT)
#define dcsigSEGV (milk_data.signal_SEGV)
#define dcsigABRT (milk_data.signal_ABRT)
#define dcsigBUS (milk_data.signal_BUS)
#define dcsigHUP (milk_data.signal_HUP)
#define dcsigPIPE (milk_data.signal_PIPE)

/**
 * True if any fatal signal has been received.
 *
 * Use in loop-exit checks instead of repeating
 * a multi-line OR chain of dcsig* flags.
 */
#define DCSIG_ANY_SET() \
    (dcsigINT || dcsigTERM || dcsigABRT || dcsigBUS || dcsigSEGV || dcsigHUP || dcsigPIPE)

/* Process info */

#define dcpinfo (milk_data.pinfo)
#define dcprocinfo (milk_data.processinfo)
#define dcprocinfoact (milk_data.processinfoActive)

/* UIDs */
#define dcruid (milk_data.ruid)
#define dceuid (milk_data.euid)
#define dcsuid (milk_data.suid)

/* Package info */
#define dcpkgname (milk_data.package_name)
#define dcpkgver (milk_data.package_version)
#define dcpkgmajor (milk_data.package_version_major)
#define dcpkgminor (milk_data.package_version_minor)
#define dcpkgpatch (milk_data.package_version_patch)

/* Misc */
#define dcrndgen (milk_data.rndgen)
#define dcmemmon (milk_data.MEM_MONITOR)
#define dcretval (milk_data.retvalue)
#define dcstatus0 (milk_data.status0)
#define dcstatus1 (milk_data.status1)
#define dcprogstatus (milk_data.progStatus)

/* Test points */
#define dctestpoint (milk_data.testpoint)
#define dctestptarr (milk_data.testpointarray)
#define dctestptinit (milk_data.testpointarrayinit)
#define dctestptlcnt (milk_data.testpointloopcnt)
#define dctestptcnt (milk_data.testpointcnt)

/* Convenience arrays */
#define dcfloatarr (milk_data.FLOATARRAY)
#define dcdoublearr (milk_data.DOUBLEARRAY)
#define dcsavedir (milk_data.SAVEDIR)

/* Config/source/install dirs */
#define dcconfigdir (milk_data.configdir)
#define dcsourcedir (milk_data.sourcedir)
#define dcinstalldir (milk_data.installdir)

/* INVRANDMAX */
#define dcinvrandmax (milk_data.INVRANDMAX)

#endif /* MILKDATA_MACROS_H */
