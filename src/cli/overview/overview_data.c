/**
 * @file overview_data.c
 * @brief Scan and graph-building for milkCTRL
 *
 * Implements the three scanners (streams, FPS, processes)
 * and the cross-referencing graph builder.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include "overview_defs.h"
#include "overview_data.h"

#include "ImageStreamIO/ImageStreamIO.h"

/* fps_shmdirname.h for function_parameter_struct_shmdirname */
#include "fps_shmdirname.h"

/* Forward-declare FPS connect/disconnect to avoid
 * pulling in fps.h (which conflicts with our defs) */
long function_parameter_struct_connect(
    const char *name,
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsconnectmode);
int function_parameter_struct_disconnect(
    FUNCTION_PARAMETER_STRUCT *fps);

/* STRINGMAXLEN_FPS_DIRNAME defined in fps_types.h */
#define OV_SHMDIR_MAXLEN STRINGMAXLEN_FPS_DIRNAME


/* =========================================================
 * Helpers
 * ========================================================= */

/**
 * pid_is_alive - check if a PID is currently alive.
 */
static int pid_is_alive(pid_t pid)
{
    if (pid <= 0)
    {
        return 0;
    }
    return (kill(pid, 0) == 0);
}


/**
 * ov_datatype_name - short name for data type code.
 */
static const char *ov_datatype_name(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "UI8";
    case _DATATYPE_INT8:
        return "SI8";
    case _DATATYPE_UINT16:
        return "U16";
    case _DATATYPE_INT16:
        return "S16";
    case _DATATYPE_UINT32:
        return "U32";
    case _DATATYPE_INT32:
        return "S32";
    case _DATATYPE_UINT64:
        return "U64";
    case _DATATYPE_INT64:
        return "S64";
    case _DATATYPE_FLOAT:
        return "F32";
    case _DATATYPE_DOUBLE:
        return "F64";
    case _DATATYPE_COMPLEX_FLOAT:
        return "CF";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "CD";
    default:
        return "???";
    }
}
/* suppress unused-function warning when header is
 * included but ov_datatype_name is only used by
 * the render code */
__attribute__((unused))
static const char *ov_datatype_name_ref =
    (const char *)(uintptr_t) ov_datatype_name;


/* =========================================================
 * Stream scanning
 * ========================================================= */

void ov_scan_streams(OV_MODEL *model)
{
    const char *shmdir = SHAREDSHMDIR;

    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        model->nb_streams = 0;
        return;
    }

    int idx = 0;
    struct dirent *ep;

    while ((ep = readdir(dp)) != NULL
           && idx < OV_MAX_STREAMS)
    {
        /* Match *.im.shm */
        int namelen = (int) strlen(ep->d_name);
        if (namelen < 8)
        {
            continue;
        }
        if (strcmp(ep->d_name + namelen - 7,
                  ".im.shm") != 0)
        {
            continue;
        }

        /* Extract stream name */
        char sname[STRINGMAXLEN_IMAGE_NAME];
        int  snlen = namelen - 7;
        if (snlen >= (int) sizeof(sname))
        {
            snlen = (int) sizeof(sname) - 1;
        }
        memcpy(sname, ep->d_name, (size_t) snlen);
        sname[snlen] = '\0';

        /* Get inode */
        char fpath[1024];
        snprintf(fpath, sizeof(fpath),
                 "%s/%s", shmdir, ep->d_name);
        struct stat st;
        if (stat(fpath, &st) != 0)
        {
            continue;
        }

        /* Try to connect to the stream */
        IMAGE img;
        memset(&img, 0, sizeof(IMAGE));
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(
                sname, &img) != IMAGESTREAMIO_SUCCESS)
        {
            continue;
        }

        OV_STREAM *s = &model->streams[idx];
        memset(s, 0, sizeof(OV_STREAM));
        strncpy(s->name, sname,
                sizeof(s->name) - 1);
        s->valid = 1;
        s->inode = st.st_ino;

        if (img.md != NULL)
        {
            s->datatype  = img.md->datatype;
            s->naxis     = img.md->naxis;
            s->size[0]   = img.md->size[0];
            s->size[1]   = img.md->size[1];
            s->size[2]   = img.md->size[2];
            s->nelement  = img.md->nelement;
            s->creatorPID = img.md->creatorPID;
            s->ownerPID   = img.md->ownerPID;

            /* Update rate */
            uint64_t cnt_new = img.md->cnt0;
            if (s->cnt0_prev > 0
                && cnt_new > s->cnt0_prev)
            {
                /* rate estimated by caller
                 * on successive scans */
            }
            s->cnt0 = cnt_new;

            /* Read process trace entries */
            int npt = img.md->NBproctrace;
            if (npt > IMAGE_NB_PROCTRACE)
            {
                npt = IMAGE_NB_PROCTRACE;
            }
            s->nb_proctrace = 0;

            if (img.streamproctrace != NULL)
            {
                for (int t = 0; t < npt; t++)
                {
                    STREAM_PROC_TRACE *spt =
                        &img.streamproctrace[t];
                    if (spt->procwrite_PID > 0)
                    {
                        int ti = s->nb_proctrace;
                        s->proctrace_pid[ti] =
                            spt->procwrite_PID;
                        s->proctrace_inode[ti] =
                            spt->trigger_inode;
                        s->proctrace_trigmode[ti] =
                            spt->triggermode;
                        s->proctrace_status[ti] =
                            spt->triggerstatus;
                        s->nb_proctrace++;
                    }
                }
            }

            s->active = pid_is_alive(s->ownerPID)
                        || pid_is_alive(s->creatorPID);
        }

        s->node_idx = -1;
        idx++;

        ImageStreamIO_closeIm(&img);
    }

    closedir(dp);
    model->nb_streams = idx;
}


/* =========================================================
 * FPS scanning
 * ========================================================= */

void ov_scan_fps(OV_MODEL *model)
{
    char shmdir[OV_SHMDIR_MAXLEN];
    function_parameter_struct_shmdirname(shmdir);

    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        model->nb_fps = 0;
        return;
    }

    int idx = 0;
    struct dirent *ep;

    while ((ep = readdir(dp)) != NULL
           && idx < OV_MAX_FPS)
    {
        /* Match *.fps.shm */
        int namelen = (int) strlen(ep->d_name);
        if (namelen < 9)
        {
            continue;
        }
        if (strcmp(ep->d_name + namelen - 8,
                  ".fps.shm") != 0)
        {
            continue;
        }

        /* Extract FPS name */
        char fname[STRINGMAXLEN_FPS_NAME];
        int fnlen = namelen - 8;
        if (fnlen >= (int) sizeof(fname))
        {
            fnlen = (int) sizeof(fname) - 1;
        }
        memcpy(fname, ep->d_name, (size_t) fnlen);
        fname[fnlen] = '\0';

        /* Try to connect */
        FUNCTION_PARAMETER_STRUCT fps;
        memset(&fps, 0, sizeof(fps));
        fps.SMfd = -1;

        long fpsID = function_parameter_struct_connect(
                         fname, &fps, FPSCONNECT_SIMPLE);
        if (fpsID < 0)
        {
            continue;
        }

        OV_FPS *f = &model->fps[idx];
        memset(f, 0, sizeof(OV_FPS));
        strncpy(f->name, fname,
                sizeof(f->name) - 1);
        f->valid = 1;

        if (fps.md != NULL)
        {
            f->md_status = fps.md->status;
            f->confpid   = fps.md->confpid;
            f->runpid    = fps.md->runpid;
            f->conf_alive = pid_is_alive(f->confpid);
            f->run_alive  = pid_is_alive(f->runpid);

            /* Scan for stream-type parameters */
            int sp = 0;
            int nb_params = fps.md->NBparamMAX;
            if (nb_params > 10000)
            {
                nb_params = 10000;
            }
            for (int p = 0;
                 p < nb_params
                 && sp < OV_FPS_MAX_STREAM_PARAMS;
                 p++)
            {
                FUNCTION_PARAMETER *fp =
                    &fps.parray[p];
                if (fp->fpflag
                    & FPFLAG_ACTIVE)
                {
                    if (fp->type == FPTYPE_STREAMNAME)
                    {
                        /* keyword[0] is the FPS instance name.
                         * Build the full parameter key by joining all
                         * non-empty keyword levels from index 1 onwards
                         * with dots, e.g. "procinfo.triggersname". */
                        char kbuf[FUNCTION_PARAMETER_STRMAXLEN];
                        kbuf[0] = '\0';
                        {
                            int klen = 0;
                            for (int kl = 1;
                                 kl < FUNCTION_PARAMETER_KEYWORD_MAXLEVEL;
                                 kl++)
                            {
                                if (fp->keyword[kl][0] == '\0')
                                {
                                    break;
                                }
                                if (klen > 0 && klen < (int) sizeof(kbuf) - 1)
                                {
                                    kbuf[klen++] = '.';
                                    kbuf[klen]   = '\0';
                                }
                                int rem = (int) sizeof(kbuf) - klen - 1;
                                if (rem > 0)
                                {
                                    strncat(kbuf + klen,
                                            fp->keyword[kl],
                                            (size_t) rem);
                                    klen = (int) strlen(kbuf);
                                }
                            } /* for kl */
                            /* Fall back to keyword[0] if nothing found */
                            if (kbuf[0] == '\0')
                            {
                                strncpy(kbuf, fp->keyword[0],
                                        sizeof(kbuf) - 1);
                            }
                        }
                        strncpy(
                            f->stream_param_name[sp],
                            kbuf,
                            FUNCTION_PARAMETER_STRMAXLEN
                            - 1);
                        strncpy(
                            f->stream_param_value[sp],
                            fp->val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN
                            - 1);
                        f->stream_param_flags[sp] =
                            fp->fpflag;
                        sp++;
                    }
                }
            }
            f->nb_stream_params = sp;
        }

        /* Read description from md if available */
        if (fps.md != NULL
            && fps.md->description[0] != '\0')
        {
            strncpy(f->description,
                    fps.md->description,
                    sizeof(f->description) - 1);
        }

        f->node_idx = -1;
        idx++;

        function_parameter_struct_disconnect(&fps);
    }

    closedir(dp);
    model->nb_fps = idx;
}


/* =========================================================
 * Process scanning
 * ========================================================= */

void ov_scan_procs(OV_MODEL *model)
{
    char shmdir[OV_SHMDIR_MAXLEN];
    function_parameter_struct_shmdirname(shmdir);

    /* Map the processinfo list */
    char pilistfname[1024];
    snprintf(pilistfname, sizeof(pilistfname),
             "%s/processinfo.list.shm", shmdir);

    int fd = open(pilistfname, O_RDONLY);
    if (fd == -1)
    {
        model->nb_procs = 0;
        return;
    }

    PROCESSINFOLIST *pilist =
        (PROCESSINFOLIST *) mmap(
            NULL,
            sizeof(PROCESSINFOLIST),
            PROT_READ,
            MAP_SHARED,
            fd,
            0);

    if (pilist == MAP_FAILED)
    {
        close(fd);
        model->nb_procs = 0;
        return;
    }

    int idx = 0;

    for (int i = 0;
         i < PROCESSINFOLISTSIZE
         && idx < OV_MAX_PROCS;
         i++)
    {
        if (pilist->active[i] == 0)
        {
            continue;
        }

        pid_t pid = pilist->PIDarray[i];
        if (!pid_is_alive(pid))
        {
            continue;
        }

        /* Map the individual processinfo SHM */
        char pinfofname[1024];
        snprintf(pinfofname, sizeof(pinfofname),
                 "%s/proc.%s.%d.shm",
                 shmdir,
                 pilist->pnamearray[i],
                 (int) pid);

        int pfd = open(pinfofname, O_RDONLY);
        if (pfd == -1)
        {
            continue;
        }

        PROCESSINFO *pinfo =
            (PROCESSINFO *) mmap(
                NULL,
                sizeof(PROCESSINFO),
                PROT_READ,
                MAP_SHARED,
                pfd,
                0);

        if (pinfo == MAP_FAILED)
        {
            close(pfd);
            continue;
        }

        OV_PROC *p = &model->procs[idx];
        memset(p, 0, sizeof(OV_PROC));
        strncpy(p->name,
                pilist->pnamearray[i],
                sizeof(p->name) - 1);
        p->PID    = pid;
        p->valid  = 1;
        p->active = 1;

        p->loopstat = pinfo->loopstat;
        p->CTRLval  = pinfo->CTRLval;
        p->loopcnt  = pinfo->loopcnt;

        /* Timing stats */
        p->dtmedian_iter_ns =
            pinfo->dtmedian_iter_ns;
        p->dtmedian_exec_ns =
            pinfo->dtmedian_exec_ns;
        if (p->dtmedian_iter_ns > 0)
        {
            p->loop_hz =
                1.0e9
                / (double) p->dtmedian_iter_ns;
        }

        /* Trigger info */
        strncpy(p->trigstreamname,
                pinfo->triggerstreamname,
                sizeof(p->trigstreamname) - 1);
        p->triggermode = pinfo->triggermode;
        p->triggersem  = pinfo->triggersem;
        p->triggermissed =
            pinfo->triggermissedframe;
        p->triggermissed_cumul =
            pinfo->triggermissedframe_cumul;
        p->MeasureTiming = pinfo->MeasureTiming;

        p->rt_priority = pinfo->RT_priority;

        p->node_idx = -1;
        idx++;

        munmap(pinfo, sizeof(PROCESSINFO));
        close(pfd);
    }

    munmap(pilist, sizeof(PROCESSINFOLIST));
    close(fd);

    model->nb_procs = idx;
}


/* =========================================================
 * Lookup helpers
 * ========================================================= */

int ov_find_stream_by_inode(
    const OV_MODEL *model,
    ino_t inode)
{
    if (inode == 0)
    {
        return -1;
    }
    for (int i = 0; i < model->nb_streams; i++)
    {
        if (model->streams[i].valid
            && model->streams[i].inode == inode)
        {
            return i;
        }
    }
    return -1;
}

int ov_find_stream_by_name(
    const OV_MODEL *model,
    const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }
    for (int i = 0; i < model->nb_streams; i++)
    {
        if (model->streams[i].valid
            && strcmp(model->streams[i].name,
                     name) == 0)
        {
            return i;
        }
    }
    return -1;
}

int ov_find_proc_by_pid(
    const OV_MODEL *model,
    pid_t pid)
{
    if (pid <= 0)
    {
        return -1;
    }
    for (int i = 0; i < model->nb_procs; i++)
    {
        if (model->procs[i].valid
            && model->procs[i].PID == pid)
        {
            return i;
        }
    }
    return -1;
}


/* =========================================================
 * Edge management
 * ========================================================= */

void ov_add_edge(
    OV_MODEL *model,
    int src,
    int tgt,
    ov_edge_type_t type,
    const char *label)
{
    if (src < 0 || tgt < 0 || src == tgt)
    {
        return;
    }
    if (model->nb_edges >= OV_MAX_EDGES)
    {
        return;
    }

    /* Check for duplicate */
    for (int i = 0; i < model->nb_edges; i++)
    {
        if (model->edges[i].src_node == src
            && model->edges[i].tgt_node == tgt
            && model->edges[i].type == type)
        {
            return;
        }
    }

    OV_EDGE *e = &model->edges[model->nb_edges];
    e->src_node = src;
    e->tgt_node = tgt;
    e->type     = type;
    e->active   = 1;
    strncpy(e->label, label,
            sizeof(e->label) - 1);
    e->label[sizeof(e->label) - 1] = '\0';
    model->nb_edges++;
}


/* =========================================================
 * Graph builder
 * ========================================================= */

/**
 * add_node - add a node to the graph.
 * @model: model
 * @type:  node type
 * @index: type-specific index
 * @name:  display name
 * @active: whether the node is active
 *
 * Return: node index in model->nodes[].
 */
static int add_node(
    OV_MODEL *model,
    ov_node_type_t type,
    int index,
    const char *name,
    int active)
{
    if (model->nb_nodes >= OV_MAX_NODES)
    {
        return -1;
    }
    int ni = model->nb_nodes;
    OV_NODE *n = &model->nodes[ni];
    n->type   = type;
    n->index  = index;
    n->active = active;
    n->gx     = 0;
    n->gy     = 0;
    strncpy(n->name, name,
            sizeof(n->name) - 1);
    n->name[sizeof(n->name) - 1] = '\0';
    model->nb_nodes++;
    return ni;
}


void ov_build_graph(OV_MODEL *model)
{
    model->nb_nodes = 0;
    model->nb_edges = 0;

    /* ---- Create nodes for all entities ---- */

    for (int i = 0; i < model->nb_streams; i++)
    {
        if (!model->streams[i].valid)
        {
            continue;
        }
        model->streams[i].node_idx = add_node(
                                         model, OV_NODE_STREAM, i,
                                         model->streams[i].name,
                                         model->streams[i].active);
    }

    for (int i = 0; i < model->nb_fps; i++)
    {
        if (!model->fps[i].valid)
        {
            continue;
        }
        model->fps[i].node_idx = add_node(
                                     model, OV_NODE_FPS, i,
                                     model->fps[i].name,
                                     model->fps[i].run_alive);
    }

    for (int i = 0; i < model->nb_procs; i++)
    {
        if (!model->procs[i].valid)
        {
            continue;
        }
        model->procs[i].node_idx = add_node(
                                       model, OV_NODE_PROC, i,
                                       model->procs[i].name,
                                       model->procs[i].active);
    }

    /* ---- Build edges from process trace ---- */

    for (int si = 0; si < model->nb_streams; si++)
    {
        OV_STREAM *s = &model->streams[si];
        if (!s->valid || s->node_idx < 0)
        {
            continue;
        }

        for (int t = 0; t < s->nb_proctrace; t++)
        {
            pid_t wpid = s->proctrace_pid[t];
            if (wpid <= 0)
            {
                continue;
            }

            /* Find the process that writes
             * this stream */
            int pi = ov_find_proc_by_pid(model, wpid);
            if (pi >= 0 && model->procs[pi].node_idx >= 0)
            {
                /* proc → stream (writes) */
                ov_add_edge(
                    model,
                    model->procs[pi].node_idx,
                    s->node_idx,
                    OV_EDGE_PROC_WRITES_STREAM,
                    "writes");
            }

            /* Find the trigger stream */
            ino_t trig_inode =
                s->proctrace_inode[t];
            if (trig_inode != 0
                && pi >= 0
                && model->procs[pi].node_idx >= 0)
            {
                int tsi = ov_find_stream_by_inode(
                              model, trig_inode);
                if (tsi >= 0
                    && model->streams[tsi].node_idx
                    >= 0)
                {
                    /* stream → proc (triggers) */
                    ov_add_edge(
                        model,
                        model->streams[tsi].node_idx,
                        model->procs[pi].node_idx,
                        OV_EDGE_STREAM_TRIGGERS_PROC,
                        "triggers");
                }
            }
        }
    }

    /* ---- Build edges from FPS ---- */

    for (int fi = 0; fi < model->nb_fps; fi++)
    {
        OV_FPS *f = &model->fps[fi];
        if (!f->valid || f->node_idx < 0)
        {
            continue;
        }

        /* FPS → Process (via runpid match) */
        if (f->run_alive && f->runpid > 0)
        {
            int pi = ov_find_proc_by_pid(
                         model, f->runpid);
            if (pi >= 0
                && model->procs[pi].node_idx >= 0)
            {
                ov_add_edge(
                    model,
                    f->node_idx,
                    model->procs[pi].node_idx,
                    OV_EDGE_FPS_RUNS_PROC,
                    "runs");
            }
        }

        /* FPS stream params → Stream nodes */
        for (int sp = 0;
             sp < f->nb_stream_params;
             sp++)
        {
            const char *sval =
                f->stream_param_value[sp];
            if (sval[0] == '\0')
            {
                continue;
            }

            int si = ov_find_stream_by_name(
                         model, sval);
            if (si < 0
                || model->streams[si].node_idx < 0)
            {
                continue;
            }

            uint64_t flags =
                f->stream_param_flags[sp];
            const char *pname =
                f->stream_param_name[sp];

            int is_out = 0;
            if (flags & FPFLAG_WRITE)
            {
                is_out = 1;
            }
            else if (strstr(pname, "out") || strstr(pname, "OUT"))
            {
                is_out = 1;
            }

            if (is_out)
            {
                /* FPS → stream (output) */
                ov_add_edge(
                    model,
                    f->node_idx,
                    model->streams[si].node_idx,
                    OV_EDGE_FPS_OUTPUT_STREAM,
                    "output");
            }
            else
            {
                /* stream → FPS (input) */
                ov_add_edge(
                    model,
                    model->streams[si].node_idx,
                    f->node_idx,
                    OV_EDGE_FPS_INPUT_STREAM,
                    "input");
            }
        }
    }

    /* ---- Build edges from process trigger ---- */

    for (int pi = 0; pi < model->nb_procs; pi++)
    {
        OV_PROC *p = &model->procs[pi];
        if (!p->valid || p->node_idx < 0)
        {
            continue;
        }

        if (p->trigstreamname[0] != '\0')
        {
            int si = ov_find_stream_by_name(
                         model, p->trigstreamname);
            if (si >= 0
                && model->streams[si].node_idx >= 0)
            {
                ov_add_edge(
                    model,
                    model->streams[si].node_idx,
                    p->node_idx,
                    OV_EDGE_PROC_TRIGGER_STREAM,
                    "trigger");
            }
        }
    }
}


/* =========================================================
 * Full scan
 * ========================================================= */

void ov_model_full_scan(OV_MODEL *model)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Preserve old cnt0 for rate estimation */
    OV_STREAM old_streams[OV_MAX_STREAMS];
    int old_nb = model->nb_streams;
    if (old_nb > 0)
    {
        memcpy(old_streams, model->streams,
               (size_t) old_nb * sizeof(OV_STREAM));
    }

    /* Clear dynamic parts */
    model->nb_streams = 0;
    model->nb_fps     = 0;
    model->nb_procs   = 0;
    model->nb_nodes   = 0;
    model->nb_edges   = 0;

    ov_scan_streams(model);
    ov_scan_fps(model);
    ov_scan_procs(model);

    /* Compute stream update rates from cnt0 delta */
    if (old_nb > 0 && model->scan_count > 0)
    {
        struct timespec dt_ts = ov_timespec_diff(
                                    model->last_scan_time, t0);
        double dt_sec =
            (double) dt_ts.tv_sec
            + (double) dt_ts.tv_nsec * 1.0e-9;

        if (dt_sec > 0.01)
        {
            for (int i = 0;
                 i < model->nb_streams; i++)
            {
                OV_STREAM *s = &model->streams[i];
                /* Find matching old stream */
                for (int j = 0; j < old_nb; j++)
                {
                    if (old_streams[j].valid
                        && strcmp(old_streams[j].name,
                                 s->name) == 0)
                    {
                        uint64_t dc =
                            s->cnt0
                            - old_streams[j].cnt0;
                        s->update_hz =
                            (double) dc / dt_sec;

                        /* Update sparkline */
                        float sv =
                            (float) s->update_hz;
                        float maxhz = 10000.0f;
                        float norm =
                            sv / maxhz;
                        if (norm > 1.0f)
                        {
                            norm = 1.0f;
                        }
                        s->spark_rate[
                            s->spark_idx
                            % OV_SPARKLINE_LEN]
                            = norm;
                        s->spark_idx++;

                        break;
                    }
                }
            }
        }
    }

    ov_build_graph(model);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    struct timespec dt = ov_timespec_diff(t0, t1);
    model->scan_time_ms =
        (double) dt.tv_sec * 1000.0
        + (double) dt.tv_nsec * 1.0e-6;
    model->last_scan_time = t0;
    model->scan_count++;
}


/* =========================================================
 * Sorting helpers
 * ========================================================= */

static int sort_stream_by_name(
    const void *a, const void *b)
{
    return strcmp(
        ((const OV_STREAM *) a)->name,
        ((const OV_STREAM *) b)->name);
}

static int sort_stream_by_hz(
    const void *a, const void *b)
{
    double ha = ((const OV_STREAM *) a)->update_hz;
    double hb = ((const OV_STREAM *) b)->update_hz;
    if (hb > ha) { return 1; }
    if (hb < ha) { return -1; }
    return 0;
}

static int sort_stream_by_type(
    const void *a, const void *b)
{
    int ta = ((const OV_STREAM *) a)->datatype;
    int tb = ((const OV_STREAM *) b)->datatype;
    return ta - tb;
}

void ov_sort_streams(OV_MODEL *model, int key)
{
    if (model->nb_streams < 2)
    {
        return;
    }
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:  cmp = sort_stream_by_hz;   break;
    case 2:  cmp = sort_stream_by_type; break;
    default: cmp = sort_stream_by_name; break;
    }
    qsort(model->streams,
          (size_t) model->nb_streams,
          sizeof(OV_STREAM), cmp);
}


static int sort_proc_by_name(
    const void *a, const void *b)
{
    return strcmp(
        ((const OV_PROC *) a)->name,
        ((const OV_PROC *) b)->name);
}

static int sort_proc_by_pid(
    const void *a, const void *b)
{
    pid_t pa = ((const OV_PROC *) a)->PID;
    pid_t pb = ((const OV_PROC *) b)->PID;
    if (pa < pb) { return -1; }
    if (pa > pb) { return 1; }
    return 0;
}

static int sort_proc_by_hz(
    const void *a, const void *b)
{
    double ha = ((const OV_PROC *) a)->loop_hz;
    double hb = ((const OV_PROC *) b)->loop_hz;
    if (hb > ha) { return 1; }
    if (hb < ha) { return -1; }
    return 0;
}

static int sort_proc_by_stat(
    const void *a, const void *b)
{
    int sa = ((const OV_PROC *) a)->loopstat;
    int sb = ((const OV_PROC *) b)->loopstat;
    return sa - sb;
}

void ov_sort_procs(OV_MODEL *model, int key)
{
    if (model->nb_procs < 2)
    {
        return;
    }
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:  cmp = sort_proc_by_pid;  break;
    case 2:  cmp = sort_proc_by_hz;   break;
    case 3:  cmp = sort_proc_by_stat; break;
    default: cmp = sort_proc_by_name; break;
    }
    qsort(model->procs,
          (size_t) model->nb_procs,
          sizeof(OV_PROC), cmp);
}


static int sort_fps_by_name(
    const void *a, const void *b)
{
    return strcmp(
        ((const OV_FPS *) a)->name,
        ((const OV_FPS *) b)->name);
}

static int sort_fps_by_alive(
    const void *a, const void *b)
{
    const OV_FPS *fa = (const OV_FPS *) a;
    const OV_FPS *fb = (const OV_FPS *) b;
    /* Active (conf+run alive) first */
    int aa = fa->conf_alive + fa->run_alive;
    int ab = fb->conf_alive + fb->run_alive;
    return ab - aa;
}

void ov_sort_fps(OV_MODEL *model, int key)
{
    if (model->nb_fps < 2)
    {
        return;
    }
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:  cmp = sort_fps_by_alive; break;
    default: cmp = sort_fps_by_name;  break;
    }
    qsort(model->fps,
          (size_t) model->nb_fps,
          sizeof(OV_FPS), cmp);
}
