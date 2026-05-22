#include "streamCTRL_TUI_render_internal.h"

/**
 * @brief Render the stream list rows.
 */
void streamCTRL__render_stream_rows(streamCTRLarg_struct        *streamCTRLdata,
                                    struct streamCTRL_TUI_state *state,
                                    int                          NBsinfodisp,
                                    double                       frame_t_sec,
                                    int                          frame_color_level)
{
    int       DisplayFlag    = 0;
    int       print_pid_mode = PRINT_PID_DEFAULT;
    const int stringmaxlen   = 300;
    for (int dindex = 0; dindex < sTUIparam.NBsindex; dindex++)
    {
        imageID ID;
        int     sindex = sTUIparam.ssindex[dindex];
        ID             = streaminfo[sindex].ID;

        int downstreammin = NO_DOWNSTREAM_INDEX;
        // minumum downstream index
        // looks for inodeselected in the list of upstream inodes
        // picks the smallest corresponding index
        // for example, if equal to 3, the current inode is a 3-rd gen children of selected inode
        // default initial value 100 is a placeholder indicating it is not a child

        DEBUG_TRACEPOINT(" ");

        if ((dindex >= doffsetindex) && (dindex < NBsinfodisp + doffsetindex))
        {
            DisplayFlag = 1;
        }
        else
        {
            DisplayFlag = 0;
        }

        if (sTUIparam.DisplayDetailLevel == 1)
        {
            if (dindex == sTUIparam.dindexSelected)
            {
                DisplayFlag = 1;
            }
            else
            {
                DisplayFlag = 0;
            }
        }

        DEBUG_TRACEPOINT(" ");

        /* Stream name discovered but SHM not yet opened. Show
         * just the name in dim style until connection is ready. */
        if (ID < 0)
        {
            if (DisplayFlag == 1)
            {
                if (dindex == sTUIparam.dindexSelected)
                {
                    screenprint_setreverse();
                }

                screenprint_setcolor(4);
                TUI_printfw("          %-*.*s  ...", DispName_NBchar, DispName_NBchar,
                            streaminfo[sindex].sname);
                screenprint_unsetcolor(4);

                if (dindex == sTUIparam.dindexSelected)
                {
                    screenprint_unsetreverse();
                }

                TUI_newline();
            }
            continue;
        }

        // Stream is guaranteed active and not erased


        if (streaminfo[sindex].ISIOretval != IMAGESTREAMIO_SUCCESS)
        {
            if (DisplayFlag == 1)
            {
                TUI_printfw("          ");


                if ((dindex == sTUIparam.dindexSelected) && (sTUIparam.DisplayDetailLevel == 0))
                {
                    screenprint_setreverse();
                }

                TUI_printfw("%-*.*s", DispName_NBchar, DispName_NBchar, streaminfo[sindex].sname);


                screenprint_setcolor(4);
                TUI_printfw("ERROR:");
                screenprint_unsetcolor(4);
                TUI_printfw("  ");

                switch (streaminfo[sindex].ISIOretval)
                {
                case IMAGESTREAMIO_FILEOPEN:
                    TUI_printfw("cannot open file");
                    break;

                case IMAGESTREAMIO_VERSION:
                    TUI_printfw("incompatible ISIO version");
                    break;

                case IMAGESTREAMIO_FAILURE:
                    TUI_printfw("failed verification");
                    break;
                }


                if (dindex == sTUIparam.dindexSelected)
                {
                    screenprint_unsetreverse();
                }

                TUI_newline();
            }
        }
        else
        {
            if (dindex == sTUIparam.dindexSelected)
            {
                DEBUG_TRACEPOINT("dindex %d %d", dindex,
                                 streamCTRLimages[streaminfo[sindex].ID].used);

                // currently selected inode
                inodeselected = streamCTRLimages[streaminfo[sindex].ID].md->inode;

                DEBUG_TRACEPOINT("inode %lu %s", inodeselected,
                                 streamCTRLimages[streaminfo[sindex].ID].md->name);

                // identify upstream inodes
                NBupstreaminode = 0;
                for (int spti = 0; spti < streamCTRLimages[ID].md[0].NBproctrace; spti++)
                {
                    if (NBupstreaminode < NBupstreaminodeMAX)
                    {
                        ino_t inode = streamCTRLimages[ID].streamproctrace[spti].trigger_inode;
                        if (inode != 0)
                        {
                            upstreaminode[NBupstreaminode] = inode;
                            NBupstreaminode++;
                        }
                    }
                }

                DEBUG_TRACEPOINT(" ");

                // identify upstream processes
                print_pid_mode = PRINT_PID_FORCE_NOUPSTREAM;
                NBupstreamproc = 0;
                for (int spti = 0; spti < streamCTRLimages[ID].md[0].NBproctrace; spti++)
                {
                    if (NBupstreamproc < NBupstreamprocMAX)
                    {
                        ino_t procpid = streamCTRLimages[ID].streamproctrace[spti].procwrite_PID;
                        if (procpid > 0)
                        {
                            upstreamproc[NBupstreamproc] = procpid;
                            NBupstreamproc++;
                        }
                    }

                    DEBUG_TRACEPOINT(" ");
                }
            }
            else
            {
                if (DisplayFlag == 1)
                {
                    DEBUG_TRACEPOINT(
                        "%d, %s, ID = %ld, used = %d, name= %s, ISIOcode= %d (OK = %d)", sindex,
                        streaminfo[sindex].sname, streaminfo[sindex].ID, streamCTRLimages[ID].used,
                        streamCTRLimages[ID].name, streaminfo[sindex].ISIOretval,
                        IMAGESTREAMIO_SUCCESS);

                    print_pid_mode = PRINT_PID_DEFAULT;
                    if (streamCTRLimages[ID].used == 1)
                    {
                        for (int spti = 0; spti < streamCTRLimages[ID].md->NBproctrace; spti++)
                        {
                            ino_t inode = streamCTRLimages[ID].streamproctrace[spti].trigger_inode;
                            if (inode == inodeselected)
                            {
                                if (spti < downstreammin)
                                {
                                    downstreammin = spti;
                                }
                            }
                        }
                    }
                    DEBUG_TRACEPOINT(" ");
                }
            }

            DEBUG_TRACEPOINT(" ");

            int  stringlen = 200;
            char string[stringlen];

            if (DisplayFlag == 1)
            {
                // print file inode
                if (streamCTRLimages[ID].used == 1)
                {
                    streamCTRL_print_inode(streamCTRLimages[ID].md[0].inode, upstreaminode,
                                           NBupstreaminode, downstreammin);
                }
                TUI_printfw(" ");
            }

            if ((dindex == sTUIparam.dindexSelected) && (sTUIparam.DisplayDetailLevel == 0))
            {
                screenprint_setreverse();
            }

            DEBUG_TRACEPOINT(" ");

            if (DisplayFlag == 1)
            {
                if (streaminfo[sindex].SymLink == 1)
                {
                    char namestring[stringmaxlen];

                    snprintf(namestring, stringmaxlen, "%s->%s", streaminfo[sindex].sname,
                             streaminfo[sindex].linkname);

                    screenprint_setbold();
                    screenprint_setcolor(5);
                    TUI_printfw("%-*.*s", DispName_NBchar, DispName_NBchar, namestring);
                    screenprint_unsetcolor(5);
                    screenprint_unsetbold();
                }
                else
                {
                    screenprint_setbold();
                    TUI_printfw("%-*.*s", DispName_NBchar, DispName_NBchar,
                                streaminfo[sindex].sname);
                    screenprint_unsetbold();
                }

                /*if((int) strlen(streaminfo[sindex].sname) > DispName_NBchar)
                {
                    attron(COLOR_PAIR(9));
                    TUI_printfw("+");
                    attroff(COLOR_PAIR(9));
                }
                else
                {
                    TUI_printfw(" ");
                }*/
            }

            DEBUG_TRACEPOINT(" ");

            if ((sTUIparam.DisplayMode < DISPLAY_MODE_FUSER) && (DisplayFlag == 1))
            {
                char str[STRINGMAXLEN_DEFAULT];
                char str1[STRINGMAXLEN_DEFAULT];


                if (streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                {
                    snprintf(string, stringlen, " ???");
                }
                else
                {
                    snprintf(string, stringlen, "%s",
                             ImageStreamIO_typename_short(streaminfo[sindex].datatype));
                }
                TUI_printfw("%s", string);

                DEBUG_TRACEPOINT(" ");
                if (streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                {
                    snprintf(str, stringlen, "???");
                }
                else
                {
                    snprintf(str, stringlen, " [%3ld", (long) streamCTRLimages[ID].md[0].size[0]);

                    for (int j = 1; j < streamCTRLimages[ID].md[0].naxis; j++)
                    {
                        {
                            int slen = snprintf(str1, STRINGMAXLEN_DEFAULT, "%sx%3ld", str,
                                                (long) streamCTRLimages[ID].md[0].size[j]);
                            if (slen < 1)
                            {
                                PRINT_ERROR("snprintf "
                                            "wrote <1 "
                                            "char");
                                abort(); // can't handle this error any other way
                            }
                            if (slen >= STRINGMAXLEN_DEFAULT)
                            {
                                PRINT_ERROR("snprintf "
                                            "string "
                                            "truncatio"
                                            "n");
                                abort(); // can't handle this error any other way
                            }
                        }
                        snprintf(str, STRINGMAXLEN_DEFAULT, "%s", str1);
                    }
                    {
                        int slen = snprintf(str1, STRINGMAXLEN_DEFAULT, "%s]", str);
                        if (slen < 1)
                        {
                            PRINT_ERROR("snprintf wrote <1 "
                                        "char");
                            abort(); // can't handle this error any other way
                        }
                        if (slen >= STRINGMAXLEN_DEFAULT)
                        {
                            PRINT_ERROR("snprintf string "
                                        "truncation");
                            abort(); // can't handle this error any other way
                        }
                    }

                    snprintf(str, STRINGMAXLEN_DEFAULT, "%s", str1);
                }

                DEBUG_TRACEPOINT(" ");

                snprintf(string, stringlen, "%-*.*s ", DispSize_NBchar, DispSize_NBchar, str);
                TUI_printfw("%s", string);

                if (streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                {
                    snprintf(string, stringlen, " %*s ", Dispcnt0_NBchar, "???");
                }
                else
                {
                    snprintf(string, stringlen, " %*ld ", Dispcnt0_NBchar,
                             streamCTRLimages[ID].md[0].cnt0);
                }

                double t_sec = frame_t_sec;

                /* Update holdoff timestamp whenever the stream is active. */
                if (streaminfo[sindex].deltacnt0 != 0)
                {
                    streaminfo[sindex].last_wave_t = t_sec;
                }

                double wave_age = t_sec - streaminfo[sindex].last_wave_t;

                /* 1-second block average of cnt0 update frequency.
                 * When the 1-s window expires, compute freq from the
                 * accumulated Δcnt0 and reset the window. */
                if (streamCTRLimages[ID].md != NULL)
                {
                    uint64_t cnt0now = streamCTRLimages[ID].md[0].cnt0;
                    double   dt_avg  = t_sec - streaminfo[sindex].t_avg_start;

                    if (dt_avg >= 1.0)
                    {
                        uint64_t dcnt                 = cnt0now - streaminfo[sindex].cnt0_avg_start;
                        streaminfo[sindex].frequ_disp = (double) dcnt / dt_avg;
                        streaminfo[sindex].cnt0_avg_start = cnt0now;
                        streaminfo[sindex].t_avg_start    = t_sec;
                    }
                }

                /* Highlight cnt0 field when counter is
                 * actively changing (within 1s holdoff). */
                if (wave_age <= 1.0 && frame_color_level >= 2)
                {
                    int len_cnt = strlen(string);
                    screenprint_setcolor(2);
                    streamCTRL_render_active_bg(string, len_cnt, frame_color_level);
                    SC_APPEND("\033[0m");

                    if ((dindex == sTUIparam.dindexSelected) && (sTUIparam.DisplayDetailLevel == 0))
                    {
                        screenprint_setreverse();
                    }
                }
                else
                {
                    TUI_printfw("%s", string);
                }


                // creatorPID
                // ownerPID
                if (streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                {
                    snprintf(string, stringlen, "???");
                }
                else
                {
                    pid_t cpid; // creator PID
                    pid_t opid; // owner PID

                    cpid = streamCTRLimages[ID].md[0].creatorPID;
                    opid = streamCTRLimages[ID].md[0].ownerPID;

                    streamCTRL_print_procpid(8, cpid, upstreamproc, NBupstreamproc, print_pid_mode);
                    TUI_printfw(" ");
                    streamCTRL_print_procpid(8, opid, upstreamproc, NBupstreamproc, print_pid_mode);
                    TUI_printfw(" ");
                }

                // stream update frequency
                //
                if (streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                {
                    snprintf(string, stringlen, "???");
                    TUI_printfw("%s", string);
                }
                else
                {
                    streamCTRL_print_frequ_field(streaminfo[sindex].frequ_disp, wave_age,
                                                 frame_color_level);
                }
            }

            DEBUG_TRACEPOINT(" ");

            if (streamCTRLimages[streaminfo[sindex].ID].md != NULL)
            {
                if ((sTUIparam.DisplayMode == DISPLAY_MODE_SUMMARY) &&
                    (DisplayFlag == 1)) // sem vals
                {
                    int max_s = sTUIparam.DISPLAY_ALL_SEMS ? streamCTRLimages[ID].md[0].sem : 3;
                    TUI_printfw(" ");
                    for (int s = 0; s < max_s; s++)
                    {
                        int semval = ImageStreamIO_semvalue(streamCTRLimages + ID, s);
                        if (s > 0)
                        {
                            TUI_printfw(":");
                        }
                        streamCTRL_set_sem_color(semval);
                        snprintf(string, stringlen, "%02d", semval);
                        TUI_printfw("%s", string);
                        screenprint_unsetcolor(0);
                    }
                }
            }

            DEBUG_TRACEPOINT(" ");

            if (streamCTRLimages[streaminfo[sindex].ID].md != NULL)
            {
                if ((sTUIparam.DisplayMode == DISPLAY_MODE_WRITE) &&
                    (DisplayFlag == 1)) // sem write PIDs
                {
                    {
                        pid_t pid = streamCTRLimages[ID].semWritePID[0];
                        TUI_printfw(" ");
                        streamCTRL_print_procpid(8, pid, upstreamproc, NBupstreamproc,
                                                 print_pid_mode);
                    }

                    if (sTUIparam.DisplayDetailLevel == 1)
                    {
#ifdef IMAGESTRUCT_WRITEHISTORY
                        TUI_newline();
                        TUI_printfw("WRITE timings :");
                        TUI_newline();
                        int    windexref = streamCTRLimages[ID].md->wCBindex;
                        double tdouble0  = 0.0;

                        double *dtarray =
                            (double *) malloc(sizeof(double) * (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2));

                        double tdoubleprev = 0.0;
                        double deltatsum   = 0.0;
                        double deltatsum2  = 0.0;
                        for (int wioffset = 0; wioffset < IMAGESTRUCT_FRAMEWRITEMDSIZE - 1;
                             wioffset++)
                        {
                            int windex = windexref - wioffset;
                            if (windex < 0)
                            {
                                windex += IMAGESTRUCT_FRAMEWRITEMDSIZE;
                            }
                            double tdouble =
                                1.0 * streamCTRLimages[ID].writehist[windex].writetime.tv_sec +
                                1.0e-9 * streamCTRLimages[ID].writehist[windex].writetime.tv_nsec;
                            double deltat = 0.0;

                            if (wioffset == 0)
                            {
                                tdouble0 = tdouble;
                                deltat   = 0.0;
                            }
                            else
                            {
                                deltat                = tdoubleprev - tdouble;
                                dtarray[wioffset - 1] = deltat;
                                deltatsum += deltat;
                                deltatsum2 += deltat * deltat;
                            }

                            if (wioffset < 10)
                            {
                                TUI_printfw(
                                    "%4d  cnt0 %8d  PID %6d  ts %9ld.%09ld   %.9f s ago  delta = "
                                    "%9.3f us",
                                    wioffset, streamCTRLimages[ID].writehist[windex].cnt0,
                                    streamCTRLimages[ID].writehist[windex].wpid,
                                    streamCTRLimages[ID].writehist[windex].writetime.tv_sec,
                                    streamCTRLimages[ID].writehist[windex].writetime.tv_nsec,
                                    tdouble0 - tdouble, 1.0e6 * (deltat));
                                TUI_newline();
                            }
                            tdoubleprev = tdouble;
                        }

                        quick_sort_double(dtarray, IMAGESTRUCT_FRAMEWRITEMDSIZE - 2);

                        TUI_newline();

                        TUI_printfw("delta time (nbsample = %d):", IMAGESTRUCT_FRAMEWRITEMDSIZE);
                        TUI_newline();

                        double tave = 1.0e6 * deltatsum / (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2);
                        TUI_printfw("AVERAGE =        %9.3f us", tave);
                        TUI_newline();

                        double trms =
                            deltatsum2 - deltatsum * deltatsum / (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2);
                        trms = 1.0e6 * sqrt(trms / (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2));
                        TUI_printfw("RMS     =        %9.3f us  ( %8.3f %% )", trms,
                                    100.0 * trms / tave);
                        TUI_newline();

                        double p0us = 1.0e6 * dtarray[0];
                        TUI_printfw("  min          : %9.3f us    %9.3f us", p0us, p0us - tave);
                        TUI_newline();

                        double p10us =
                            1.0e6 * dtarray[(int) (0.1 * (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2))];
                        TUI_printfw("  p10          : %9.3f us    %9.3f us", p10us, p10us - tave);
                        TUI_newline();

                        double p50us = 1.0e6 * dtarray[(IMAGESTRUCT_FRAMEWRITEMDSIZE - 2) / 2];
                        TUI_printfw("  p50 (median) : %9.3f us    %9.3f us", p50us, p50us - tave);
                        TUI_newline();

                        double p90us =
                            1.0e6 * dtarray[(int) (0.9 * (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2))];
                        TUI_printfw("  p90          : %9.3f us    %9.3f us", p90us, p90us - tave);
                        TUI_newline();

                        double p100us = 1.0e6 * dtarray[IMAGESTRUCT_FRAMEWRITEMDSIZE - 3];
                        TUI_printfw("  max          : %9.3f us    %9.3f us", p100us, p100us - tave);
                        TUI_newline();


                        free(dtarray);
#endif
                    }
                }
            }

            DEBUG_TRACEPOINT(" ");

            if (streamCTRLimages[streaminfo[sindex].ID].md != NULL)
            {
                if ((sTUIparam.DisplayMode == DISPLAY_MODE_READ) &&
                    (DisplayFlag == 1)) // sem read PIDs
                {
                    int max_s = sTUIparam.DISPLAY_ALL_SEMS ? streamCTRLimages[ID].md[0].sem : 3;
                    TUI_printfw(" ");
                    for (int s = 0; s < max_s; s++)
                    {
                        pid_t pid = streamCTRLimages[ID].semReadPID[s];
                        if (s > 0)
                        {
                            TUI_printfw(":");
                        }
                        streamCTRL_print_procpid(0, // 0 for minimal width
                                                 pid, upstreamproc, NBupstreamproc, print_pid_mode);
                    }
                }
            }

            DEBUG_TRACEPOINT(" ");

            if (streamCTRLimages[streaminfo[sindex].ID].md != NULL)
            {
                if ((sTUIparam.DisplayMode == DISPLAY_MODE_SPTRACE) && (DisplayFlag == 1))
                {
                    DEBUG_TRACEPOINT("show stream process trace");
                    DEBUG_TRACEPOINT("NBproctrace = %d", streamCTRLimages[ID].md->NBproctrace);

                    snprintf(string, stringlen, " %2d ", streamCTRLimages[ID].md->NBproctrace);
                    TUI_printfw("%s", string);

                    for (int spti = 0; spti < streamCTRLimages[ID].md->NBproctrace; spti++)
                    {
                        DEBUG_TRACEPOINT("stream process trace step %d", spti);
                        ino_t inode = streamCTRLimages[ID].streamproctrace[spti].trigger_inode;
                        int   sem   = streamCTRLimages[ID].streamproctrace[spti].trigsemindex;
                        pid_t pid   = streamCTRLimages[ID].streamproctrace[spti].procwrite_PID;


                        DEBUG_TRACEPOINT("stream process trace step %d: triggermode", spti);

                        switch (streamCTRLimages[ID].streamproctrace[spti].triggermode)
                        {
                        case PROCESSINFO_TRIGGERMODE_IMMEDIATE:
                            snprintf(string, stringlen, "(%7lu IM ", inode);
                            break;

                        case PROCESSINFO_TRIGGERMODE_CNT0:
                            snprintf(string, stringlen, "(%7lu C0 ", inode);
                            break;

                        case PROCESSINFO_TRIGGERMODE_CNT1:
                            snprintf(string, stringlen, "(%7lu C1 ", inode);
                            break;

                        case PROCESSINFO_TRIGGERMODE_CNT2:
                            snprintf(string, stringlen, "(%7lu C2 ", inode);
                            break;

                        case PROCESSINFO_TRIGGERMODE_SEMAPHORE:
                            snprintf(string, stringlen, "(%7lu %02d ", inode, sem);
                            break;

                        case PROCESSINFO_TRIGGERMODE_DELAY:
                            snprintf(string, stringlen, "(%7lu DL ", inode);
                            break;

                        default:
                            snprintf(string, stringlen, "(%7lu ?? ", inode);
                            break;
                        }
                        TUI_printfw("%s", string);

                        DEBUG_TRACEPOINT(" ");

                        streamCTRL_print_procpid(8, pid, upstreamproc, NBupstreamproc,
                                                 print_pid_mode);
                        TUI_printfw(")> ");
                        DEBUG_TRACEPOINT(" ");
                    }

                    if (sTUIparam.DisplayDetailLevel == 1)
                    {
                        DEBUG_TRACEPOINT(" ");
                        TUI_newline();
                        streamCTRL_print_SPTRACE_details(streamCTRLimages, ID, upstreamproc,
                                                         NBupstreamproc, PRINT_PID_DEFAULT);
                        DEBUG_TRACEPOINT(" ");
                    }
                }


                DEBUG_TRACEPOINT(" ");
                if ((sTUIparam.DisplayMode == DISPLAY_MODE_SUMMARY) && (DisplayFlag == 1))
                {
                    if (sTUIparam.DisplayDetailLevel == 1)
                    {
                        TUI_newline();
                        TUI_newline();
                        TUI_printfw("name            %10s", streamCTRLimages[ID].name);
                        TUI_newline();
                        TUI_printfw("createcnt       %10ld", streamCTRLimages[ID].createcnt);
                        TUI_newline();
                        TUI_printfw("shmfd           %10d", streamCTRLimages[ID].shmfd);
                        TUI_newline();
                        TUI_printfw("memsize         %10lu", streamCTRLimages[ID].memsize);
                        TUI_newline();
                        TUI_printfw("md.version      %10s", streamCTRLimages[ID].md->version);
                        TUI_newline();
                        TUI_printfw("md.name         %10s", streamCTRLimages[ID].md->name);
                        TUI_newline();
                        TUI_printfw("md.naxis        %10d", (int) streamCTRLimages[ID].md->naxis);
                        TUI_newline();
                        for (int axis = 0; axis < streamCTRLimages[ID].md->naxis; axis++)
                        {
                            TUI_printfw("   md.size[%d]   %10d", axis,
                                        (int) streamCTRLimages[ID].md->size[axis]);
                            TUI_newline();
                        }
                        TUI_printfw("md.nelement         %10lu", streamCTRLimages[ID].md->nelement);
                        TUI_newline();
                        TUI_printfw("md.datatype         %10d",
                                    (int) streamCTRLimages[ID].md->datatype);
                        TUI_newline();
                        TUI_printfw("md.creationtime     %10ld.%09ld",
                                    streamCTRLimages[ID].md->creationtime.tv_sec,
                                    (long) streamCTRLimages[ID].md->creationtime.tv_nsec);
                        TUI_newline();
                        TUI_printfw("md.lastaccesstime   %10ld.%09ld",
                                    streamCTRLimages[ID].md->lastaccesstime.tv_sec,
                                    (long) streamCTRLimages[ID].md->lastaccesstime.tv_nsec);
                        TUI_newline();
                        TUI_printfw("md.atime            %10ld.%09ld",
                                    streamCTRLimages[ID].md->atime.tv_sec,
                                    (long) streamCTRLimages[ID].md->atime.tv_nsec);
                        TUI_newline();
                        TUI_printfw("md.writetime        %10ld.%09ld",
                                    streamCTRLimages[ID].md->writetime.tv_sec,
                                    (long) streamCTRLimages[ID].md->writetime.tv_nsec);
                        TUI_newline();
                        TUI_printfw("md.creatorPID       %10ld",
                                    (long) streamCTRLimages[ID].md->creatorPID);
                        TUI_newline();
                        TUI_printfw("md.ownerPID         %10ld",
                                    (long) streamCTRLimages[ID].md->ownerPID);
                        TUI_newline();
                        TUI_printfw("md.shared           %10d",
                                    (int) streamCTRLimages[ID].md->shared);
                        TUI_newline();
                        TUI_printfw("md.inode            %10lu",
                                    (unsigned long) streamCTRLimages[ID].md->inode);
                        TUI_newline();
                        TUI_newline();
                        TUI_printfw("md.sem              %10d", (int) streamCTRLimages[ID].md->sem);
                        TUI_newline();
                    }
                }
                DEBUG_TRACEPOINT(" ");
            }

            DEBUG_TRACEPOINT(" ");

            if ((sTUIparam.DisplayMode == DISPLAY_MODE_FUSER) &&
                (DisplayFlag == 1)) // list processes that are accessing streams
            {
                if (streaminfoproc.fuserUpdate == 2)
                {
                    streaminfo[sindex].streamOpenPID_status = 0; // not scanned
                }

                DEBUG_TRACEPOINT(" ");


                switch (streaminfo[sindex].streamOpenPID_status)
                {
                case 1:
                    streaminfo[sindex].streamOpenPID_cnt1 = 0;
                    for (int pidIndex = 0; pidIndex < streaminfo[sindex].streamOpenPID_cnt;
                         pidIndex++)
                    {
                        pid_t pid = streaminfo[sindex].streamOpenPID[pidIndex];
                        streamCTRL_print_procpid(8, pid, upstreamproc, NBupstreamproc,
                                                 print_pid_mode);

                        if ((getpgid(pid) >= 0) && (pid != getpid()))
                        {
                            snprintf(string, stringlen, ":%-*.*s", PIDnameStringLen,
                                     PIDnameStringLen, PIDname_array[pid]);
                            TUI_printfw("%s", string);

                            streaminfo[sindex].streamOpenPID_cnt1++;
                        }
                    }
                    break;

                case 2:
                    snprintf(string, stringlen, "FAILED");
                    TUI_printfw("%s", string);
                    break;

                default:
                    snprintf(string, stringlen, "NOT SCANNED");
                    TUI_printfw("%s", string);
                    break;
                }
            }

            DEBUG_TRACEPOINT(" ");

            if (DisplayFlag == 1)
            {
                if (dindex == sTUIparam.dindexSelected)
                {
                    screenprint_unsetreverse();
                }

                TUI_newline();
            }
        }

        DEBUG_TRACEPOINT(" ");

        if (streaminfoproc.fuserUpdate == 1)
        {
            //      refresh();
            if (sc_sigINT == 1) // stop scan
            {
                // complete loop without scan
                streaminfoproc.fuserUpdate = 2;

                sc_sigINT = 0; // reset
            } // complete loop without scan
        }

        DEBUG_TRACEPOINT(" ");
    }
}
