#ifndef FPS_CTRLSCREEN_SCHEDULER_DISPLAY_H
#define FPS_CTRLSCREEN_SCHEDULER_DISPLAY_H

errno_t fpsCTRL_scheduler_display(FPSCTRL_TASK_ENTRY *fpsctrltasklist,
                                  FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
                                  int                 wrow,
                                  int                *wrowstart);

#endif
