#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "processinfo.h"
#include "processinfo_shm_list_create.h"

int main(int argc, char *argv[])
{
    if(argc > 1)
    {
        if(strcmp(argv[1], "-h") == 0)
        {
            printf("Usage: %s\n", argv[0]);
            printf("List active processinfo instances.\n");
            return 0;
        }
    }

    if (processinfo_shm_list_create() == -1) {
        fprintf(stderr, "Error connecting to process list shared memory\n");
        return 1;
    }

    printf("%-30s %-10s %-10s\n", "Process Name", "PID", "Status");
    printf("------------------------------------------------------------\n");

    if (pinfolist != NULL) {
        for (long i = 0; i < PROCESSINFOLISTSIZE; i++) {
            if (pinfolist->active[i] != 0) {
                char status_str[32];
                switch(pinfolist->active[i]) {
                    case 1: strcpy(status_str, "ACTIVE"); break;
                    case 2: strcpy(status_str, "STOPPED"); break;
                    case 3: strcpy(status_str, "CRASHED"); break;
                    default: strcpy(status_str, "UNKNOWN"); break;
                }
                
                printf("%-30s %-10ld %s\n", 
                    pinfolist->pnamearray[i], 
                    (long)pinfolist->PIDarray[i], 
                    status_str);
            }
        }
    }

    return 0;
}
