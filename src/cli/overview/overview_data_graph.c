#include "overview_data_internal.h"

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
             * this stream.
             * Only proctrace[0] is the direct
             * writer; entries [1..N] are upstream
             * triggers in the causal chain and
             * must NOT get write-edges. */
            int pi = ov_find_proc_by_pid(model, wpid);
            if (t == 0
                && pi >= 0
                && model->procs[pi].node_idx >= 0)
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

        /* Create edges for input-based reading (via read_pids) */
        for (int r = 0; r < s->nb_read_pids; r++)
        {
            pid_t rpid = s->read_pids[r];
            if (rpid > 0)
            {
                int rpi = ov_find_proc_by_pid(model, rpid);
                if (rpi >= 0 && model->procs[rpi].node_idx >= 0)
                {
                    /* stream → proc (reads/inputs) */
                    ov_add_edge(
                        model,
                        s->node_idx,
                        model->procs[rpi].node_idx,
                        OV_EDGE_STREAM_READ_BY_PROC,
                        "reads");
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

            const char *pname =
                f->stream_param_name[sp];

            /* Skip procinfo trigger stream —
             * already handled by
             * PROC_TRIGGER_STREAM edges */
            if (strstr(pname, "procinfo.trigger"))
            {
                continue;
            }

            /* Determine direction by param name.
             * FPFLAG_WRITE cannot be used — it
             * means "user-editable", which is
             * set for both inputs and outputs. */
            int is_out = 0;
            if (strstr(pname, "out")
                || strstr(pname, "OUT"))
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


