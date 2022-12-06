#include "CLIcore.h"
#include <processtools.h>

#include "CommandLineInterface/timeutils.h"

#include "processinfo/processinfo_procdirname.h"



static struct timespec t1;
static struct timespec t2;
static struct timespec tdiff;

static double scantime_CPUload;
static double scantime_CPUpcnt;


int GetCPUloads(PROCINFOPROC *pinfop)
{
    char     *line      = NULL;
    size_t    maxstrlen = 256;
    FILE     *fp;
    ssize_t   read;
    int       cpu;
    long long vall0, vall1, vall2, vall3, vall4, vall5, vall6, vall7, vall8;
    long long v0, v1, v2, v3, v4, v5, v6, v7, v8;
    char      string0[80];

    static int cnt = 0;

    clock_gettime(CLOCK_REALTIME, &t1);

    line = (char *) malloc(sizeof(char) * maxstrlen);
    if(line == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    fp = fopen("/proc/stat", "r");
    if(fp == NULL)
    {
        exit(EXIT_FAILURE);
    }

    cpu = 0;

    read = getline(&line, &maxstrlen, fp);
    if(read == -1)
    {
        printf("[%s][%d]  ERROR: cannot read file\n", __FILE__, __LINE__);
        exit(EXIT_SUCCESS);
    }

    while(((read = getline(&line, &maxstrlen, fp)) != -1) &&
            (cpu < pinfop->NBcpus))
    {

        sscanf(line,
               "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld",
               string0,
               &vall0,
               &vall1,
               &vall2,
               &vall3,
               &vall4,
               &vall5,
               &vall6,
               &vall7,
               &vall8);

        v0 = vall0 - pinfop->CPUcnt0[cpu];
        v1 = vall1 - pinfop->CPUcnt1[cpu];
        v2 = vall2 - pinfop->CPUcnt2[cpu];
        v3 = vall3 - pinfop->CPUcnt3[cpu];
        v4 = vall4 - pinfop->CPUcnt4[cpu];
        v5 = vall5 - pinfop->CPUcnt5[cpu];
        v6 = vall6 - pinfop->CPUcnt6[cpu];
        v7 = vall7 - pinfop->CPUcnt7[cpu];
        v8 = vall8 - pinfop->CPUcnt8[cpu];

        pinfop->CPUcnt0[cpu] = vall0;
        pinfop->CPUcnt1[cpu] = vall1;
        pinfop->CPUcnt2[cpu] = vall2;
        pinfop->CPUcnt3[cpu] = vall3;
        pinfop->CPUcnt4[cpu] = vall4;
        pinfop->CPUcnt5[cpu] = vall5;
        pinfop->CPUcnt6[cpu] = vall6;
        pinfop->CPUcnt7[cpu] = vall7;
        pinfop->CPUcnt8[cpu] = vall8;

        pinfop->CPUload[cpu] = (1.0 * v0 + v1 + v2 + v4 + v5 + v6) /
                               (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8);
        cpu++;
    }
    free(line);
    fclose(fp);
    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = timespec_diff(t1, t2);
    scantime_CPUload += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    clock_gettime(CLOCK_REALTIME, &t1);

    // number of process per CPU -> we can get that from ps
    char command[STRINGMAXLEN_COMMAND];
    char psoutfname[STRINGMAXLEN_FULLFILENAME];
    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    WRITE_FULLFILENAME(psoutfname, "%s/_psoutput.txt", procdname);

    // use ps command to scan processes, store result in file psoutfname

    EXECUTE_SYSTEM_COMMAND(
        "{ if [ ! -f %s/_psOKlock ]; then touch %s/_psOKlock; ps -e -o "
        "pid,psr,cpu,cmd > %s; fi; rm "
        "%s/_psOKlock &> /dev/null; }",
        procdname,
        procdname,
        psoutfname,
        procdname);


    // read and process psoutfname file

    if(access(psoutfname, F_OK) != -1)
    {

        for(cpu = 0; cpu < pinfop->NBcpus; cpu++)
        {
            char  outstring[STRINGMAXLEN_DEFAULT];
            FILE *fpout;
            {
                int slen = snprintf(command,
                                    STRINGMAXLEN_COMMAND,
                                    "CORENUM=%d; cat %s | grep -E  "
                                    "\"^[[:space:]][[:digit:]]+[[:"
                                    "space:]]+${CORENUM}\"|wc -l",
                                    cpu,
                                    psoutfname);
                if(slen < 1)
                {
                    PRINT_ERROR("snprintf wrote <1 char");
                    abort(); // can't handle this error any other way
                }
                if(slen >= STRINGMAXLEN_COMMAND)
                {
                    PRINT_ERROR("snprintf string truncation");
                    abort(); // can't handle this error any other way
                }
            }
            fpout = popen(command, "r");
            if(fpout == NULL)
            {
                printf("WARNING: cannot run command \"%s\"\n", command);
            }
            else
            {
                if(fgets(outstring, STRINGMAXLEN_DEFAULT, fpout) == NULL)
                {
                    printf("WARNING: fgets error\n");
                }
                pclose(fpout);
                pinfop->CPUpcnt[cpu] = atoi(outstring);
            }
        }
        remove(psoutfname);
    }
    cnt++;

    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = timespec_diff(t1, t2);
    scantime_CPUpcnt += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    return (cpu);
}
