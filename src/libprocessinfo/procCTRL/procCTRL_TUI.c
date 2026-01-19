#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ncurses.h>
#include <pthread.h>
#include <signal.h>

#include "processinfo.h"
#include "processtools.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_list_create.h"
#include "processinfo_signals.h"
#include "procCTRL_GetNumberCPUs.h"
#include "procCTRL_GetCPUloads.h"
#include "procCTRL_processinfo_scan.h"
#include "procCTRL_TUI.h"

extern PROCESSINFOLIST *pinfolist;
extern pid_t CLIPID;

static short unsigned int wrow, wcol;

errno_t processinfo_CTRLscreen() {
    PROCINFOPROC procinfoproc = {0};
    pthread_t threadscan;

    CLIPID = getpid();
    processinfo_CatchSignals();

    // Init procinfoproc
    procinfoproc.loop = 1;
    procinfoproc.twaitus = 1000000;
    procinfoproc.NBcpus = GetNumberCPUs(&procinfoproc);

    processinfo_shm_list_create();
    procinfoproc.pinfolist = pinfolist;

    initscr();
    noecho();
    cbreak();
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);
    getmaxyx(stdscr, wrow, wcol);

    pthread_create(&threadscan, NULL, processinfo_scan, (void *) &procinfoproc);

    while (procinfoproc.loop) {
        int ch = getch();
        if (ch == 'x') procinfoproc.loop = 0;

        procinfoproc.SCANBLOCK_OK = 1;
        while(procinfoproc.SCANBLOCK_OK == 1) usleep(100);

        erase();
        mvprintw(0, 0, "MILK PROCESS CONTROL - PID %d - %d processes", (int)CLIPID, procinfoproc.NBpindexActive);
        mvprintw(1, 0, "Press 'x' to exit");

        for (int i = 0; i < procinfoproc.NBpindexActive && i < wrow - 4; i++) {
            long idx = procinfoproc.pindexActive[i];
            if (procinfoproc.pinfommapped[idx]) {
                PROCESSINFO *p = procinfoproc.pinfoarray[idx];
                mvprintw(i + 3, 0, "%-20s PID %6d STAT %d LOOP %ld MSG: %s", 
                         p->name, (int)p->PID, p->loopstat, p->loopcnt, p->statusmsg);
            }
        }
        refresh();
        usleep(100000);
    }

    procinfoproc.loop = 0;
    pthread_join(threadscan, NULL);
    endwin();

    return 0;
}