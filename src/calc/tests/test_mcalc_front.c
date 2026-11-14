// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest: the parts of the calculator that need no session: the tokens of a text, the tree of a
 * statement, the scalar operators, the shell's own evaluator and the function catalog. */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "front/mc_ast.h"

static int g_fail;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            g_fail++;                                                    \
        }                                                                \
    } while (0)

static mc_tok_list_t g_tl;

/* The statement that @src is, or NULL on a lexical or syntax error */
static mc_stmt_t *parse(const char *src)
{
    char err[256] = "";
    return (mc_lex(src, &g_tl) == 0) ? mc_parse_stmt(&g_tl, err, sizeof(err)) : NULL;
}

static void test_tokens(void)
{
    CHECK(mc_lex("a1 = 0x1F + 2.5e1 # comment", &g_tl) == 0 && g_tl.ntok == 5);
    CHECK(g_tl.toks[0].type == MC_TOK_IDENT && strcmp(g_tl.toks[0].sval, "a1") == 0);
    CHECK(g_tl.toks[1].type == MC_TOK_ASSIGN && g_tl.toks[3].type == MC_TOK_PLUS);
    CHECK(g_tl.toks[2].type == MC_TOK_LONG && g_tl.toks[2].lval == 31);
    CHECK(g_tl.toks[4].type == MC_TOK_DOUBLE && g_tl.toks[4].dval == 25.0);
    CHECK(g_tl.toks[5].type == MC_TOK_EOF);

    /* a stream name may hold '-': one identifier, then the operator */
    CHECK(mc_lex("s.my-cam.cnt0 - 1", &g_tl) == 0 && g_tl.ntok == 3);
    CHECK(strcmp(g_tl.toks[0].sval, "s.my-cam.cnt0") == 0 && g_tl.toks[1].type == MC_TOK_MINUS);

    CHECK(mc_lex("A @ B", &g_tl) == 0 && g_tl.toks[1].type == MC_TOK_MATMUL);
    CHECK(mc_lex("a.T", &g_tl) == 0 && g_tl.toks[1].type == MC_TOK_DOT_T);
    CHECK(mc_lex("6 ^^ 3", &g_tl) == 0 && g_tl.toks[1].type == MC_TOK_BXOR);
    CHECK(mc_lex("\"two words\"", &g_tl) == 0 && g_tl.toks[0].type == MC_TOK_STRING &&
          strcmp(g_tl.toks[0].sval, "two words") == 0);

    CHECK(mc_lex("a ^ b", &g_tl) != 0 && strstr(g_tl.errmsg, "pow(a, b)") != NULL);
    CHECK(mc_lex("\"open", &g_tl) != 0 && strstr(g_tl.errmsg, "unterminated") != NULL);
    CHECK(mc_lex("1.5.2", &g_tl) != 0 && strstr(g_tl.errmsg, "numeric") != NULL);
}

static void test_trees(void)
{
    mc_stmt_t *s = parse("1 + 2 * 3"); /* * binds tighter than + */
    CHECK(s != NULL && s->kind == MC_STMT_EXPR && s->u.expr->kind == MC_EXPR_BINARY);
    if (s != NULL)
    {
        const mc_expr_t *e = s->u.expr;
        CHECK(e->u.binary.op == MC_TOK_PLUS && e->u.binary.left->kind == MC_EXPR_LONG);
        CHECK(e->u.binary.right->kind == MC_EXPR_BINARY &&
              e->u.binary.right->u.binary.op == MC_TOK_MUL);
    }
    mc_stmt_free(s);

    s = parse("1 < x <= 3"); /* one chained comparison, not (1 < x) <= 3 */
    CHECK(s != NULL && s->u.expr->kind == MC_EXPR_CHAIN && s->u.expr->u.chain.nops == 2);
    mc_stmt_free(s);

    s = parse("c ? a : b");
    CHECK(s != NULL && s->u.expr->kind == MC_EXPR_TERNARY);
    mc_stmt_free(s);

    s = parse("y = sin(x) + f(x, 2)"); /* a call keeps its name: the resolver binds it */
    CHECK(s != NULL && s->u.expr->kind == MC_EXPR_ASSIGN &&
          s->u.expr->u.assign.op == MC_TOK_ASSIGN);
    if (s != NULL)
    {
        const mc_expr_t *rhs = s->u.expr->u.assign.rhs;
        const mc_expr_t *l   = rhs->u.binary.left;
        const mc_expr_t *r   = rhs->u.binary.right;
        CHECK(l->kind == MC_EXPR_CALL && strcmp(l->u.call.name, "sin") == 0 &&
              l->u.call.nargs == 1);
        CHECK(r->kind == MC_EXPR_CALL && strcmp(r->u.call.name, "f") == 0 && r->u.call.nargs == 2);
    }
    mc_stmt_free(s);

    s = parse("A @ B.T"); /* the operators @ and .T are the functions matmul and transpose */
    CHECK(s != NULL && s->u.expr->kind == MC_EXPR_CALL && s->u.expr->u.call.fn != NULL &&
          s->u.expr->u.call.fn->id == MC_FID_matmul && s->u.expr->u.call.nargs == 2);
    CHECK(s != NULL && s->u.expr->u.call.args[1]->kind == MC_EXPR_CALL &&
          s->u.expr->u.call.args[1]->u.call.fn->id == MC_FID_transpose);
    mc_stmt_free(s);

    s = parse("im[1:3, *]");
    CHECK(s != NULL && s->u.expr->kind == MC_EXPR_SLICE && s->u.expr->u.slice.naxis == 2);
    CHECK(s != NULL && s->u.expr->u.slice.axes[0].has_colon && s->u.expr->u.slice.axes[1].is_full);
    mc_stmt_free(s);

    s = parse("a, b = istats(im)");
    CHECK(s != NULL && s->kind == MC_STMT_MULTI_ASSIGN && s->u.multi.nlhs == 2);
    mc_stmt_free(s);

    s = parse("def f(x, y) = x + y");
    CHECK(s != NULL && s->kind == MC_STMT_DEF && s->def.nargs == 2 &&
          strcmp(s->def.name, "f") == 0);
    mc_stmt_free(s);

    CHECK(parse("1 +") == NULL);
    CHECK(parse("(1") == NULL);
    CHECK(parse("x = ") == NULL);

    char err[256] = "";
    CHECK(mc_lex("a = 1; b = 2\nc = 3", &g_tl) == 0);
    s     = mc_parse_prog(&g_tl, err, sizeof(err));
    int n = 0;
    for (const mc_stmt_t *q = s; q != NULL; q = q->next)
    {
        n++;
    }
    CHECK(n == 3);
    mc_stmt_free(s);
}

static mc_val_t vl(long l)
{
    mc_val_t v = { .type = MC_VAL_LONG, .lval = l };
    return v;
}

static mc_val_t vd(double d)
{
    mc_val_t v = { .type = MC_VAL_DOUBLE, .dval = d };
    return v;
}

static void test_scalar_operators(void)
{
    mc_val_t a   = vl(7);
    mc_val_t b   = vl(2);
    mc_val_t big = vl(LONG_MAX);
    mc_val_t nan = vd(NAN);
    mc_val_t z   = vl(0);
    mc_val_t r;

    /* two integers stay exact; '/' always gives a double; an overflow goes to double */
    CHECK(mc_scalar_binop(MC_TOK_PLUS, &a, &b, &r) == NULL && r.type == MC_VAL_LONG && r.lval == 9);
    CHECK(mc_scalar_binop(MC_TOK_MOD, &a, &b, &r) == NULL && r.type == MC_VAL_LONG && r.lval == 1);
    CHECK(mc_scalar_binop(MC_TOK_DIV, &a, &b, &r) == NULL && r.type == MC_VAL_DOUBLE &&
          r.dval == 3.5);
    CHECK(mc_scalar_binop(MC_TOK_MUL, &big, &b, &r) == NULL && r.type == MC_VAL_DOUBLE &&
          r.dval == 2.0 * (double) LONG_MAX);
    CHECK(mc_scalar_binop(MC_TOK_DIV, &a, &z, &r) != NULL); /* a zero divisor is an error */
    CHECK(mc_scalar_binop(MC_TOK_EQ, &nan, &nan, &r) == NULL && mc_truth(&r) == 0);
    CHECK(mc_scalar_binop(MC_TOK_LT, &b, &a, &r) == NULL && mc_truth(&r) == 1);

    mc_val_t e10 = vl(10);
    mc_val_t e64 = vl(64);
    mc_val_t em1 = vl(-1);
    mc_scalar_pow(&b, &e10, &r);
    CHECK(r.type == MC_VAL_LONG && r.lval == 1024);
    mc_scalar_pow(&b, &e64, &r);
    CHECK(r.type == MC_VAL_DOUBLE && r.dval == 0x1p64);
    mc_scalar_pow(&b, &em1, &r);
    CHECK(r.type == MC_VAL_DOUBLE && r.dval == 0.5);

    CHECK(mc_fmodd(7.5, 2.0) == 1.5 && mc_fmodd(-7.0, 3.0) == -1.0);
    CHECK(fabs(mc_wrap_pi(4.0) - (4.0 - 2.0 * M_PI)) < 1e-15 && mc_wrap_pi(1.0) == 1.0);
    CHECK(mc_clamp(5.0, 0.0, 3.0) == 3.0 && mc_sign(-2.0) == -1.0 && mc_sinc(0.0) == 1.0);
    CHECK(mc_isnan(NAN) == 1.0 && mc_isnan(1.0) == 0.0 && mc_isfinite(INFINITY) == 0.0);

    mc_val_t str = { .type = MC_VAL_STRING, .sval = "x" };
    CHECK(mc_truth(&z) == 0 && mc_truth(&nan) == 0 && mc_truth(&a) == 1 && mc_truth(&str) == -1);
}

static void test_shell_evaluator(void)
{
    static const char *const ok[][2] = {
        { "1+2*3", "7" },         { "7 / 2", "3.5" },           { "-7 % 3", "-1" },
        { "1 < 2 < 3", "1" },     { "1 < 3 < 2", "0" },         { "0 && (1 / 0)", "0" },
        { "pow(2, 10)", "1024" }, { "0 ? 2 : 0 ? 4 : 5", "5" }, { "6 ^^ 3", "5" },
        { "1 # comment", "1" },   { "max(1, min(5, 3))", "3" }, { "!nan", "1" },
    };
    static const char *const declined[] = { "x + 1",   "x = 3", "\"a\"", "1 +",  "3x",
                                            "sin (1)", "1 / 0", "",      "a ^ b" };
    char                     out[256];
    int                      truth;

    for (size_t i = 0; i < sizeof(ok) / sizeof(ok[0]); i++)
    {
        int rc = mcalc_local(ok[i][0], 15, out, sizeof(out), &truth);
        if (rc != 1 || strcmp(out, ok[i][1]) != 0)
        {
            fprintf(stderr, "FAIL mcalc_local(\"%s\") = %d \"%s\", expected \"%s\"\n", ok[i][0], rc,
                    rc ? out : "", ok[i][1]);
            g_fail++;
        }
    }
    for (size_t i = 0; i < sizeof(declined) / sizeof(declined[0]); i++)
    {
        if (mcalc_local(declined[i], 15, out, sizeof(out), &truth) != 0)
        {
            fprintf(stderr, "FAIL mcalc_local(\"%s\") evaluated: the session's\n", declined[i]);
            g_fail++;
        }
    }
    CHECK(mcalc_local("0", 15, out, sizeof(out), &truth) == 1 && truth == 0);
    CHECK(mcalc_local("2 > 1", 15, out, sizeof(out), &truth) == 1 && truth == 1);
}

static void test_catalog(void)
{
    int              n                  = 0;
    const mc_func_t *f                  = mc_func_table(&n);
    char             seen[MC_FID_COUNT] = { 0 };

    CHECK(n == MC_FID_COUNT - 1);
    for (int i = 0; i < n; i++)
    {
        size_t len   = strlen(f[i].name);
        int    depth = 0;

        CHECK(f[i].id > MC_FID_NONE && f[i].id < MC_FID_COUNT && !seen[f[i].id]);
        seen[f[i].id] = 1;
        CHECK(mc_func_find(f[i].name) == &f[i] && strcmp(mc_func_name(f[i].id), f[i].name) == 0);
        CHECK(f[i].min_args >= 0 && (f[i].max_args < 0 || f[i].max_args >= f[i].min_args));
        CHECK(f[i].signature != NULL && strncmp(f[i].signature, f[i].name, len) == 0 &&
              f[i].signature[len] == '(' && f[i].help != NULL && f[i].help[0] != '\0');
        for (const char *p = f[i].args; p != NULL && *p != '\0'; p++) /* the argument kinds */
        {
            CHECK(strchr("InvxS.[]*|", *p) != NULL);
            depth += (*p == '[') - (*p == ']');
            CHECK(depth >= 0 && (*p != '|' || depth == 0));
        }
        CHECK(depth == 0);
    }
    CHECK(mc_func_find("nosuchfunction") == NULL && mc_func_find(NULL) == NULL);
    CHECK(mc_func_name(MC_FID_NONE)[0] == '\0');
    CHECK(mc_func_replacement("ln") != NULL && strcmp(mc_func_replacement("ln"), "log") == 0);
    CHECK(mc_is_reserved_const("pi") && !mc_is_reserved_const("pie"));
}

int main(void)
{
    test_tokens();
    test_trees();
    test_scalar_operators();
    test_shell_evaluator();
    test_catalog();
    mc_tok_list_free(&g_tl);
    if (g_fail > 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
    }
    return (g_fail > 0) ? 1 : 0;
}
