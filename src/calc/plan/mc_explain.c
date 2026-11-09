// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_explain.c
 * @brief `--explain`: how the planner schedules each statement of a program (`mcalc_prog_explain`).
 *
 * Compiles a program without running it, then prints, for each statement, whether it is inlined
 * (`def`), fused into an elementwise pass (shape, work type, operands, pre-stages), run as a
 * constructor step, or run on the scalar tape.
 */
#include <stdio.h>

#include "front/mc_ast.h"
#include "plan/mc_plan.h"
#include "mcalc.h"

/**
 * count_scalar_ops_reds - Count scalar operations and image reductions inside expression @e
 * @e:    AST expression to inspect recursively
 * @ops:  In/out counter incremented for each operator and function call
 * @reds: In/out counter incremented for each scalar reduction call (`imean`, `istats`, etc.)
 *
 * Used by `mcalc_prog_explain()` to summarize `MC_STEP_SCALAR` steps.
 *
 * Return: void
 */
static void count_scalar_ops_reds(const mc_expr_t *e, int *ops, int *reds)
{
    if (e == NULL)
    {
        return;
    }
    if (e->kind == MC_EXPR_ASSIGN)
    {
        count_scalar_ops_reds(e->u.assign.rhs, ops, reds);
        return;
    }
    if (e->kind == MC_EXPR_CALL)
    {
        const mc_func_t *fn = e->u.call.fn;
        if (fn != NULL && (fn->kind == MC_FN_REDUCE_D || fn->kind == MC_FN_REDUCE_L ||
                           fn->kind == MC_FN_DOT || fn->kind == MC_FN_EAGER_RED ||
                           fn->kind == MC_FN_ISTATS || fn->kind == MC_FN_ICENTROID))
        {
            (*reds)++;
            (*ops)++;
            return;
        }
        (*ops)++;
        for (int i = 0; i < e->u.call.nargs; i++)
        {
            count_scalar_ops_reds(e->u.call.args[i], ops, reds);
        }
        return;
    }
    if (e->kind == MC_EXPR_UNARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.unary.operand, ops, reds);
    }
    else if (e->kind == MC_EXPR_BINARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.binary.left, ops, reds);
        count_scalar_ops_reds(e->u.binary.right, ops, reds);
    }
    else if (e->kind == MC_EXPR_TERNARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.ternary.cond, ops, reds);
        count_scalar_ops_reds(e->u.ternary.t_expr, ops, reds);
        count_scalar_ops_reds(e->u.ternary.f_expr, ops, reds);
    }
    else if (e->kind == MC_EXPR_CHAIN)
    {
        (*ops) += e->u.chain.nops;
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            count_scalar_ops_reds(e->u.chain.operands[i], ops, reds);
        }
    }
}

/**
 * mcalc_prog_explain - Compile @text and print its execution plan to @out, without running it
 * @text: Program source text to analyze
 * @out:  Output `FILE *` stream (e.g. `stdout`)
 *
 * Prints each user statement (skipping synthesized `hist()` tail statements) followed by how the
 * planner scheduled it:
 *   - `def, inlined where it is called`
 *   - `kept, merged into the fused pass of statement K`
 *   - `kept: fused pass over NxMxK (<dtype>, 1D|ND, P operands[, Q pre-stages]), writes ...`, or
 *     `over a shape set at run time (<dtype>, P operands...)` when planning cannot know it (an
 *     operand is a target the program creates, or the operand shapes do not match)
 *   - `kept: constructor/library step`
 *   - `kept: scalar tape: N ops, M reductions, 0 pre-stages`
 *
 * Return: 0 on success, 1 on compilation error
 */
int mcalc_prog_explain(const char *text, FILE *out)
{
    char err[512] = "";
    mc_udef_save(); /* its defs stay out of the session */
    mcalc_prog *p = mcalc_prog_new(text, NULL, 0, err, sizeof(err));
    mc_udef_restore();
    if (p == NULL)
    {
        if (out != NULL)
        {
            fprintf(out, "milk: calc error: %s\n", err[0] != '\0' ? err : "compilation error");
        }
        return 1;
    }
    int              nkept = 0;
    const mc_stmt_t *cur_s = p->ast_head;
    for (int i = 0; i < p->nstmts; i++, cur_s = cur_s ? cur_s->next : NULL)
    {
        if (cur_s != NULL && cur_s->is_hist_stmt)
        {
            continue;
        }
        const char *st_txt =
            (p->stmt_texts != NULL && p->stmt_texts[i] != NULL) ? p->stmt_texts[i] : "";
        int  sidx = (p->stmt_to_step != NULL) ? p->stmt_to_step[i] : -1;
        char how[512];
        if (sidx < 0)
        {
            snprintf(how, sizeof(how), "def, inlined where it is called");
        }
        else
        {
            const mc_step_t *st = &p->steps[sidx];
            nkept++;
            if (st->kind == MC_STEP_PASS)
            {
                const char *wtype =
                    st->pass.is_complex
                        ? (st->pass.work_double ? "complex double" : "complex float32")
                        : (st->pass.work_double ? "double" : "float32");
                if (i > st->first_stmt_idx)
                {
                    snprintf(how, sizeof(how), "kept, merged into the fused pass of statement %d",
                             st->first_stmt_idx + 1);
                }
                else
                {
                    char   ps_buf[64]  = "";
                    char   wr_buf[256] = "";
                    size_t wlen        = 0;
                    if (st->pass.prestage_cnt > 0)
                    {
                        snprintf(ps_buf, sizeof(ps_buf), ", %d pre-stage%s", st->pass.prestage_cnt,
                                 (st->pass.prestage_cnt == 1) ? "" : "s");
                    }
                    int nin_ops = 0;
                    for (int k = 0; k < st->pass.nops; k++)
                    {
                        nin_ops += (!st->pass.ops[k].is_out && !st->pass.ops[k].is_scalar);
                    }
                    for (int k = 0; k < st->pass.nouts; k++)
                    {
                        if (!st->pass.outs[k].is_elided && wlen + 32 < sizeof(wr_buf))
                        {
                            wlen += (size_t) snprintf(wr_buf + wlen, sizeof(wr_buf) - wlen, " %s",
                                                      p->slots[st->pass.outs[k].slot_idx].bare);
                        }
                    }
                    const char *lim_note  = (cur_s != NULL && cur_s->kind == MC_STMT_MULTI_ASSIGN &&
                                             cur_s->u.multi.nlhs > 4)
                                                ? " (more than 4 targets (FUSE_MAXDST))"
                                                : "";
                    char        shape[48] = "a shape set at run time";
                    if (st->pass.vnx != 0)
                    {
                        snprintf(shape, sizeof(shape), "%ux%ux%u", st->pass.vnx,
                                 st->pass.vny ? st->pass.vny : 1, st->pass.vnz ? st->pass.vnz : 1);
                    }
                    snprintf(how, sizeof(how),
                             "kept: fused pass over %s (%s, %s%d operand%s%s)%s, writes%s", shape,
                             wtype,
                             (st->pass.vnx == 0)     ? ""
                             : st->pass.is_1d_contig ? "1D, "
                                                     : "ND, ",
                             nin_ops, (nin_ops == 1) ? "" : "s", ps_buf, lim_note, wr_buf);
                }
            }
            else if (st->kind == MC_STEP_CTOR)
            {
                const char *lim_note = (cur_s != NULL && cur_s->kind == MC_STMT_MULTI_ASSIGN &&
                                        cur_s->u.multi.nlhs > 4)
                                           ? " (more than 4 targets (FUSE_MAXDST))"
                                           : "";
                snprintf(how, sizeof(how), "kept: constructor/library step%s", lim_note);
            }
            else
            {
                int nops  = 0;
                int nreds = 0;
                if (st->scalar_stmt != NULL && st->scalar_stmt->kind == MC_STMT_EXPR)
                {
                    count_scalar_ops_reds(st->scalar_stmt->u.expr, &nops, &nreds);
                }
                else if (st->scalar_stmt != NULL && st->scalar_stmt->kind == MC_STMT_MULTI_ASSIGN)
                {
                    for (int j = 0; j < st->scalar_stmt->u.multi.nrhs; j++)
                    {
                        count_scalar_ops_reds(st->scalar_stmt->u.multi.rhs[j], &nops, &nreds);
                    }
                }
                if (nops == 0)
                {
                    nops = 1;
                }
                snprintf(how, sizeof(how),
                         "kept: scalar tape: %d op%s, %d reduction%s, 0 pre-stages", nops,
                         (nops == 1) ? "" : "s", nreds, (nreds == 1) ? "" : "s");
            }
        }
        if (out != NULL)
        {
            fprintf(out, "%3d  %s\n       %s\n", i + 1, st_txt, how);
        }
    }
    if (out != NULL)
    {
        fprintf(out, "%d statement%s kept, 0 evaluated as text\n", nkept, (nkept == 1) ? "" : "s");
    }
    mcalc_prog_free(p);
    return 0;
}
