// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"

/* =========================================================
 * Lookup helpers: O(1) through the model indexes once the graph is built, linear before
 * ========================================================= */

/** stream_inode_match - stream @si has the inode in @ctx. */
static int stream_inode_match(const void *ctx, int32_t si)
{
    const OV_MODEL *m = ((const void *const *) ctx)[0];
    return m->streams[si].valid &&
           m->streams[si].inode == *(const ino_t *) ((const void *const *) ctx)[1];
}

/** stream_name_match - stream @si is named as in @ctx. */
static int stream_name_match(const void *ctx, int32_t si)
{
    const OV_MODEL *m = ((const void *const *) ctx)[0];
    return m->streams[si].valid && strcmp(m->streams[si].name, ((const void *const *) ctx)[1]) == 0;
}

/** proc_pid_match - process @pi has the PID in @ctx. */
static int proc_pid_match(const void *ctx, int32_t pi)
{
    const OV_MODEL *m = ((const void *const *) ctx)[0];
    return m->procs[pi].PID == *(const pid_t *) ((const void *const *) ctx)[1];
}

/** proc_better - process @a should stand for its PID rather than @b: live first, then the newest. */
static int proc_better(const OV_MODEL *m, int a, int b)
{
    const OV_PROC *pa = &m->procs[a], *pb = &m->procs[b];
    return (pa->active && !pb->active) ||
           (pa->active == pb->active && pa->createtime > pb->createtime);
}

/** ov_model_index - (re)build the lookup indexes of @m from its stream and process arrays. */
void ov_model_index(OV_MODEL *m)
{
    ov_ix_clear(&m->ix_stream_name);
    ov_ix_clear(&m->ix_stream_inode);
    ov_ix_clear(&m->ix_proc_pid);
    for (int i = 0; i < m->nb_streams; i++)
    {
        if (m->streams[i].valid)
        {
            ov_ix_put(&m->ix_stream_name, ov_hash_str(m->streams[i].name), i);
            ov_ix_put(&m->ix_stream_inode, ov_hash_u64((uint64_t) m->streams[i].inode), i);
        }
    }
    for (int i = 0; i < m->nb_procs; i++)
    {
        pid_t       pid    = m->procs[i].PID;
        const void *ctx[2] = { m, &pid };
        if (!m->procs[i].valid || pid <= 0)
        {
            continue;
        }
        int slot = ov_ix_slot(&m->ix_proc_pid, ov_hash_u64((uint64_t) pid), proc_pid_match, ctx);
        if (slot < 0)
        {
            ov_ix_put(&m->ix_proc_pid, ov_hash_u64((uint64_t) pid), i);
        }
        else if (proc_better(m, i, m->ix_proc_pid.val[slot]))
        {
            m->ix_proc_pid.val[slot] = i;
        }
    }
    m->indexed = 1;
}

/**
 * ov_find_stream_by_inode - Find stream index by inode number
 * @model: Pointer to data model
 * @inode: Inode number to match
 *
 * Return: Stream index in @model->streams, or -1 if not found.
 */
int ov_find_stream_by_inode(const OV_MODEL *model, ino_t inode)
{
    const void *ctx[2] = { model, &inode };
    if (inode == 0)
    {
        return -1;
    }
    if (model->indexed)
    {
        return ov_ix_find(&model->ix_stream_inode, ov_hash_u64((uint64_t) inode),
                          stream_inode_match, ctx);
    }
    for (int i = 0; i < model->nb_streams; i++)
    {
        if (stream_inode_match(ctx, i))
        {
            return i;
        }
    }
    return -1;
}

/**
 * ov_find_stream_by_name - Find stream index by name
 * @model: Pointer to data model
 * @name:  Stream name to match
 *
 * Return: Stream index in @model->streams, or -1 if not found.
 */
int ov_find_stream_by_name(const OV_MODEL *model, const char *name)
{
    const void *ctx[2] = { model, name };
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }
    if (model->indexed)
    {
        return ov_ix_find(&model->ix_stream_name, ov_hash_str(name), stream_name_match, ctx);
    }
    for (int i = 0; i < model->nb_streams; i++)
    {
        if (stream_name_match(ctx, i))
        {
            return i;
        }
    }
    return -1;
}

/**
 * ov_find_proc_by_pid - Find the process entry for a PID: the live one, else the newest.
 * @model: Pointer to data model
 * @pid:   Process ID to match
 *
 * A dead registry entry may share its PID with a later process; the live entry wins.
 * Return: Process index in @model->procs, or -1 if not found.
 */
int ov_find_proc_by_pid(const OV_MODEL *model, pid_t pid)
{
    const void *ctx[2] = { model, &pid };
    int         best   = -1;
    if (pid <= 0)
    {
        return -1;
    }
    if (model->indexed)
    {
        return ov_ix_find(&model->ix_proc_pid, ov_hash_u64((uint64_t) pid), proc_pid_match, ctx);
    }
    for (int i = 0; i < model->nb_procs; i++)
    {
        if (model->procs[i].valid && model->procs[i].PID == pid &&
            (best < 0 || proc_better(model, i, best)))
        {
            best = i;
        }
    }
    return best;
}


/* =========================================================
 * Edge management
 * ========================================================= */

/* (src, tgt, type) -> edge, for the duplicate check; follows one model's edge array */
static OV_INDEX_DEF(16384) s_edge_ix;
static const OV_MODEL *s_edge_model;
static int             s_edge_n; /* edges indexed */

/** edge_hash - hash of an edge's endpoints and type. */
static uint64_t edge_hash(int src, int tgt, int type)
{
    return ov_hash_u64(((uint64_t) (uint32_t) src << 32) ^ ((uint64_t) (uint32_t) tgt << 4) ^
                       (uint64_t) type);
}

/** edge_match - edge @ei joins the endpoints and type in @ctx (an int[3]). */
static int edge_match(const void *ctx, int32_t ei)
{
    const int     *k = ctx;
    const OV_EDGE *e = &s_edge_model->edges[ei];
    return e->src_node == k[0] && e->tgt_node == k[1] && (int) e->type == k[2];
}

/**
 * ov_add_edge - Add a directed relationship edge between two graph nodes
 * @model: Pointer to data model
 * @src:   Source node index
 * @tgt:   Target node index
 * @type:  Relationship edge type
 * @evidence: where it was seen (OV_EV_*); seeing an existing edge again adds to its evidence
 */
void ov_add_edge(OV_MODEL *model, int src, int tgt, ov_edge_type_t type, unsigned evidence)
{
    const int k[3] = { src, tgt, (int) type };
    if (src < 0 || tgt < 0 || src == tgt)
    {
        return;
    }
    /* Duplicate check: (re)index this model's edges when the array is new or was reset */
    if (model != s_edge_model || s_edge_n > model->nb_edges)
    {
        ov_ix_clear(&s_edge_ix);
        s_edge_model = model;
        s_edge_n     = 0;
    }
    for (; s_edge_n < model->nb_edges; s_edge_n++)
    {
        const OV_EDGE *x = &model->edges[s_edge_n];
        ov_ix_put(&s_edge_ix, edge_hash(x->src_node, x->tgt_node, (int) x->type), s_edge_n);
    }
    int dup = ov_ix_find(&s_edge_ix, edge_hash(src, tgt, (int) type), edge_match, k);
    if (dup >= 0)
    {
        model->edges[dup].evidence |= evidence;
        return;
    }
    if (model->nb_edges >= OV_MAX_EDGES)
    {
        return;
    }

    OV_EDGE *e  = &model->edges[model->nb_edges];
    e->src_node = src;
    e->tgt_node = tgt;
    e->type     = type;
    e->evidence = evidence;
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
static int add_node(OV_MODEL *model, ov_node_type_t type, int index, const char *name, int active)
{
    if (model->nb_nodes >= OV_MAX_NODES)
    {
        return -1;
    }
    int      ni = model->nb_nodes;
    OV_NODE *n  = &model->nodes[ni];
    n->type     = type;
    n->index    = index;
    n->active   = active;
    n->gx       = 0;
    n->gy       = 0;
    strncpy(n->name, name, sizeof(n->name) - 1);
    n->name[sizeof(n->name) - 1] = '\0';
    model->nb_nodes++;
    return ni;
}


/**
 * fps_param_is_output - direction of FPS stream parameter @sp (stream @si). A live run process decides
 * by what it does (writes: output, reads a semaphore: input); otherwise the trigger/run-required flags
 * mark inputs, and the "out" name heuristic is the last resort (FPFLAG_WRITE only means editable).
 */
static int fps_param_is_output(const OV_MODEL *m, const OV_FPS *f, int sp, int si)
{
    const OV_STREAM *s = &m->streams[si];
    if (f->run_alive && f->runpid > 0)
    {
        if (s->write_pid == f->runpid)
        {
            return 1;
        }
        for (int k = 0; k < s->nb_read_pids; k++)
        {
            if (s->read_pids[k] == f->runpid)
            {
                return 0;
            }
        }
    }
    if (f->stream_param_flags[sp] & (FPFLAG_TRIGGER_STREAM | FPFLAG_STREAM_RUN_REQUIRED))
    {
        return 0;
    }
    const char *pname = f->stream_param_name[sp];
    return strstr(pname, "out") != NULL || strstr(pname, "OUT") != NULL;
}

/**
 * @brief Build the stream-process connectivity graph.
 *
 * Analyzes processinfo triggers and stream
 * semaphore chains to construct the DAG.
 */
void ov_build_graph(OV_MODEL *model)
{
    model->nb_nodes = 0;
    model->nb_edges = 0;
    ov_model_index(model);

    /* ---- Create nodes for all entities ---- */

    for (int i = 0; i < model->nb_streams; i++)
    {
        if (!model->streams[i].valid)
        {
            continue;
        }
        model->streams[i].node_idx =
            add_node(model, OV_NODE_STREAM, i, model->streams[i].name, model->streams[i].active);
    }

    for (int i = 0; i < model->nb_fps; i++)
    {
        if (!model->fps[i].valid)
        {
            continue;
        }
        model->fps[i].node_idx =
            add_node(model, OV_NODE_FPS, i, model->fps[i].name, model->fps[i].run_alive);
    }

    for (int i = 0; i < model->nb_procs; i++)
    {
        if (!model->procs[i].valid)
        {
            continue;
        }
        model->procs[i].node_idx =
            add_node(model, OV_NODE_PROC, i, model->procs[i].name, model->procs[i].active);
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
            if (t == 0 && pi >= 0 && model->procs[pi].node_idx >= 0)
            {
                /* proc → stream (writes) */
                ov_add_edge(model, model->procs[pi].node_idx, s->node_idx,
                            OV_EDGE_PROC_WRITES_STREAM, OV_EV_PROCTRACE);
            }

            /* Find the trigger stream */
            ino_t trig_inode = s->proctrace_inode[t];
            if (trig_inode != 0 && pi >= 0 && model->procs[pi].node_idx >= 0)
            {
                int tsi = ov_find_stream_by_inode(model, trig_inode);
                if (tsi >= 0 && model->streams[tsi].node_idx >= 0)
                {
                    /* stream → proc (triggers) */
                    ov_add_edge(model, model->streams[tsi].node_idx, model->procs[pi].node_idx,
                                OV_EDGE_STREAM_TRIGGERS_PROC, OV_EV_PROCTRACE);
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
                    ov_add_edge(model, s->node_idx, model->procs[rpi].node_idx,
                                OV_EDGE_STREAM_READ_BY_PROC, OV_EV_SEM_READ);
                }
            }
        }

        /* Writers that posted the stream's semaphores (semWritePID) */
        for (int w = 0; w < s->nb_write_pids; w++)
        {
            int wpi = ov_find_proc_by_pid(model, s->write_pids[w]);
            if (wpi >= 0 && model->procs[wpi].node_idx >= 0)
            {
                ov_add_edge(model, model->procs[wpi].node_idx, s->node_idx,
                            OV_EDGE_PROC_WRITES_STREAM, OV_EV_SEM_WRITE);
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
            int pi = ov_find_proc_by_pid(model, f->runpid);
            if (pi >= 0 && model->procs[pi].node_idx >= 0)
            {
                ov_add_edge(model, f->node_idx, model->procs[pi].node_idx, OV_EDGE_FPS_RUNS_PROC,
                            OV_EV_FPS_RUNPID);
            }
        }

        /* FPS stream params → Stream nodes */
        for (int sp = 0; sp < f->nb_stream_params; sp++)
        {
            const char *sval = f->stream_param_value[sp];
            if (sval[0] == '\0')
            {
                continue;
            }

            int si = ov_find_stream_by_name(model, sval);
            if (si < 0 || model->streams[si].node_idx < 0)
            {
                continue;
            }

            const char *pname = f->stream_param_name[sp];

            /* Skip the procinfo trigger stream parameter: the process trigger edge covers it */
            if (strstr(pname, "procinfo.trigger"))
            {
                continue;
            }

            if (fps_param_is_output(model, f, sp, si))
            {
                /* FPS → stream (output) */
                ov_add_edge(model, f->node_idx, model->streams[si].node_idx,
                            OV_EDGE_FPS_OUTPUT_STREAM, OV_EV_FPS_PARAM);
            }
            else
            {
                /* stream → FPS (input) */
                ov_add_edge(model, model->streams[si].node_idx, f->node_idx,
                            OV_EDGE_FPS_INPUT_STREAM, OV_EV_FPS_PARAM);
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
            int si = ov_find_stream_by_name(model, p->trigstreamname);
            if (si >= 0 && model->streams[si].node_idx >= 0)
            {
                ov_add_edge(model, model->streams[si].node_idx, p->node_idx,
                            OV_EDGE_STREAM_TRIGGERS_PROC, OV_EV_PINFO_TRIGGER);
            }
        }
    }
}
