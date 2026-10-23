// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_kernels.h
 * @brief Block kernels of the fused evaluator, written once and instantiated per element type.
 *
 * Internal to cli_calc_fuse.c, which includes this file once per element type after defining:
 *   FT        element type of the stack (double or float)
 *   FK(name)  name of an instantiated function (e.g. name##_d)
 *   FM(fn)    the libm function of that type (fn for double, fn##f for float)
 *   FK_LOAD   load_block-like function: (const IMAGE *, uint64_t off, int len, FT *dst)
 *   FK_STORE  store_block-like function: (IMAGE *, uint64_t off, int len, const FT *src)
 *
 * Math functions are called directly (never through the instruction's function pointer, except for
 * functions without a fast form), so that GCC can vectorize the loops with libmvec under -Ofast.
 * No include guard: included once per element type.
 */

/* apply EXPR(a, b) with a/b each either a vector slot or the constant c */
#define BIN_LOOPS(EXPR)                  \
    if (in->cpos == 0)                   \
    {                                    \
        for (int j = 0; j < len; j++)    \
        {                                \
            const FT a = l[j], b = r[j]; \
            l[j] = (EXPR);               \
        }                                \
    }                                    \
    else if (in->cpos == 2)              \
    {                                    \
        const FT b = (FT) in->c;         \
        for (int j = 0; j < len; j++)    \
        {                                \
            const FT a = l[j];           \
            l[j]       = (EXPR);         \
        }                                \
    }                                    \
    else                                 \
    {                                    \
        const FT a = (FT) in->c;         \
        for (int j = 0; j < len; j++)    \
        {                                \
            const FT b = l[j];           \
            l[j]       = (EXPR);         \
        }                                \
    }

/* l[j] = EXPR(x) with x = l[j] */
#define MAP_LOOP(EXPR)            \
    for (int j = 0; j < len; j++) \
    {                             \
        const FT x = l[j];        \
        l[j]       = (EXPR);      \
    }

/**
 * FK(run_pow_const) - l := l ^ e for a constant exponent: exact algebraic forms where they exist
 * @l:   Vector stack slot, updated in place
 * @e:   Constant exponent
 * @len: Number of elements
 *
 * Integer exponents up to 4 in magnitude become multiplications; halves become square roots.
 * Every form gives what pow() gives, including for negative, zero and NaN bases. Other exponents
 * use pow().
 */
static void FK(run_pow_const)(FT *restrict l, double e, int len)
{
    if (e == 0.0)
    {
        MAP_LOOP((FT) 1);
    }
    else if (e == 1.0)
    {
        return;
    }
    else if (e == 2.0)
    {
        MAP_LOOP(x * x);
    }
    else if (e == 3.0)
    {
        MAP_LOOP(x * x * x);
    }
    else if (e == 4.0)
    {
        MAP_LOOP((x * x) * (x * x));
    }
    else if (e == -1.0)
    {
        MAP_LOOP((FT) 1 / x);
    }
    else if (e == -2.0)
    {
        MAP_LOOP((FT) 1 / (x * x));
    }
    else if (e == 0.5)
    {
        MAP_LOOP(FM(sqrt)(x));
    }
    else if (e == -0.5)
    {
        MAP_LOOP((FT) 1 / FM(sqrt)(x));
    }
    else if (e == 1.5)
    {
        MAP_LOOP(x * FM(sqrt)(x));
    }
    else if (e == 2.5)
    {
        MAP_LOOP(x * x * FM(sqrt)(x));
    }
    else
    {
        const FT b = (FT) e;
        MAP_LOOP(FM(pow)(x, b));
    }
}

/**
 * FK(run_bin) - Execute a binary operator instruction (`FI_BIN`) over a pixel block
 * @in:  Instruction: operator, operand layout (`cpos`) and constant operand (`c`)
 * @l:   Left / in-place vector stack slot of length @len
 * @r:   Right vector stack slot (used when `in->cpos == 0`)
 * @len: Number of elements in the block
 *
 * Return: void
 */
static void FK(run_bin)(const fins *in, FT *restrict l, const FT *restrict r, int len)
{
    switch (in->op)
    {
    case TOK_OP_PLUS:
        BIN_LOOPS(a + b);
        break;
    case TOK_OP_MINUS:
        BIN_LOOPS(a - b);
        break;
    case TOK_OP_STAR:
        BIN_LOOPS(a * b);
        break;
    case TOK_OP_SLASH:
        BIN_LOOPS(a / b);
        break;
    case TOK_OP_MOD:
        BIN_LOOPS(FM(fmod)(a, b));
        break;
    case TOK_OP_CARET:
        if (in->cpos == 2)
        {
            FK(run_pow_const)(l, in->c, len);
        }
        else
        {
            BIN_LOOPS(FM(pow)(a, b));
        }
        break;
    case TOK_OP_LT:
        BIN_LOOPS((FT) (a < b));
        break;
    case TOK_OP_LE:
        BIN_LOOPS((FT) (a <= b));
        break;
    case TOK_OP_GT:
        BIN_LOOPS((FT) (a > b));
        break;
    case TOK_OP_GE:
        BIN_LOOPS((FT) (a >= b));
        break;
    case TOK_OP_EQ:
        BIN_LOOPS((FT) (a == b));
        break;
    case TOK_OP_NEQ:
        BIN_LOOPS((FT) (a != b));
        break;
    case TOK_OP_AND:
        BIN_LOOPS((FT) ((a != (FT) 0) & (b != (FT) 0)));
        break;
    default:
        BIN_LOOPS((FT) ((a != (FT) 0) | (b != (FT) 0)));
        break;
    }
}

/**
 * FK(run_fn2) - Execute a two-argument function instruction (`FI_FN2`) over a pixel block
 * @in:  Instruction: function (`in->fn`), operand layout (`cpos`) and constant operand (`c`)
 * @l:   Left / in-place vector stack slot of length @len
 * @r:   Right vector stack slot (used when `in->cpos == 0`)
 * @len: Number of elements in the block
 *
 * Return: void
 */
static void FK(run_fn2)(const fins *in, FT *restrict l, const FT *restrict r, int len)
{
    double (*f)(double, double) = (double (*)(double, double)) in->fn;
    if (f == fmin)
    {
        BIN_LOOPS((a < b) ? a : b);
    }
    else if (f == fmax)
    {
        BIN_LOOPS((a > b) ? a : b);
    }
    else if (f == fmod)
    {
        BIN_LOOPS(FM(fmod)(a, b));
    }
    else if (f == atan2)
    {
        BIN_LOOPS(FM(atan2)(a, b));
    }
    else if (f == hypot)
    {
        BIN_LOOPS(FM(sqrt)(a * a + b * b));
    }
    else if (f == pow)
    {
        if (in->cpos == 2)
        {
            FK(run_pow_const)(l, in->c, len);
        }
        else
        {
            BIN_LOOPS(FM(pow)(a, b));
        }
    }
    else
    {
        BIN_LOOPS((FT) f((double) a, (double) b));
    }
}

/**
 * FK(run_fn1) - Execute a one-argument function instruction (`FI_FN1`) over a pixel block
 * @fn:  The function, as stored in the instruction
 * @l:   Vector stack slot, updated in place
 * @len: Number of elements in the block
 *
 * Return: void
 */
static void FK(run_fn1)(double (*fn)(), FT *restrict l, int len)
{
    double (*f)(double) = (double (*)(double)) fn;
#define FN1_CASE(NAME)         \
    if (f == NAME)             \
    {                          \
        MAP_LOOP(FM(NAME)(x)); \
        return;                \
    }
    FN1_CASE(fabs)
    FN1_CASE(sqrt)
    FN1_CASE(floor)
    FN1_CASE(ceil)
    FN1_CASE(round)
    FN1_CASE(sin)
    FN1_CASE(cos)
    FN1_CASE(tan)
    FN1_CASE(sinh)
    FN1_CASE(cosh)
    FN1_CASE(tanh)
    FN1_CASE(exp)
    FN1_CASE(exp2)
    FN1_CASE(log)
    FN1_CASE(log10)
    FN1_CASE(log2)
    FN1_CASE(erf)
    FN1_CASE(erfc)
    FN1_CASE(atan)
    FN1_CASE(asin)
    FN1_CASE(acos)
    FN1_CASE(cbrt)
#undef FN1_CASE
    if (f == cli_calc_posi)
    {
        MAP_LOOP((FT) (x > (FT) 0));
        return;
    }
    if (f == cli_calc_sign)
    {
        MAP_LOOP((FT) ((x > (FT) 0) - (x < (FT) 0)));
        return;
    }
    /* bit-test NaN checks: -Ofast would fold isnan() away */
    if (f == cli_calc_isnan || f == cli_calc_isfinite)
    {
        MAP_LOOP((FT) f((double) x));
        return;
    }
    MAP_LOOP((FT) f((double) x)); /* casts: through the pointer, in double */
}

#define SEL_LOOP(T, F)                       \
    for (int j = 0; j < len; j++)            \
    {                                        \
        c[j] = (c[j] != (FT) 0) ? (T) : (F); \
    }

/** FK(run_sel) - c := c ? t : f, with t or f the constant in->c / in->c2 when not on the stack. */
static void FK(
    run_sel)(const fins *in, FT *restrict c, const FT *restrict tv, const FT *restrict fv, int len)
{
    const FT tc = (FT) in->c;
    const FT fc = (FT) in->c2;
    switch (in->cpos)
    {
    case 0:
        SEL_LOOP(tc, fc);
        break;
    case 1:
        SEL_LOOP(tv[j], fc);
        break;
    case 2:
        SEL_LOOP(tc, fv[j]);
        break;
    default:
        SEL_LOOP(tv[j], fv[j]);
        break;
    }
}
#undef SEL_LOOP

/**
 * FK(eval_block) - Evaluate program @p on pixels [@off, @off + @len) leaving result in `stk[0]`
 * @p:   Compiled program (scalar constants already evaluated)
 * @off: First pixel of the block
 * @len: Number of pixels (`<= FUSE_BLK`)
 * @stk: Per-thread vector stack, FUSE_MAXSTK slots of FUSE_BLK elements
 *
 * Return: void
 */
static void FK(eval_block)(const fprog *p, uint64_t off, int len, FT (*stk)[FUSE_BLK])
{
    int sp = 0;
    for (int i = 0; i < p->n; i++)
    {
        const fins *in = &p->ins[i];
        FT         *t  = stk[(sp > 0) ? sp - 1 : 0];
        switch (in->k)
        {
        case FI_IMG:
            FK_LOAD(p->img[in->img], off, len, stk[sp++]);
            break;
        case FI_BIN:
            if (in->cpos == 0)
            {
                FK(run_bin)(in, stk[sp - 2], t, len);
                sp--;
            }
            else
            {
                FK(run_bin)(in, t, NULL, len);
            }
            break;
        case FI_FN2:
            if (in->cpos == 0)
            {
                FK(run_fn2)(in, stk[sp - 2], t, len);
                sp--;
            }
            else
            {
                FK(run_fn2)(in, t, NULL, len);
            }
            break;
        case FI_NEG:
            for (int j = 0; j < len; j++)
            {
                t[j] = (FT) 0 - t[j];
            }
            break;
        case FI_NOT:
            for (int j = 0; j < len; j++)
            {
                t[j] = (FT) (t[j] == (FT) 0);
            }
            break;
        case FI_FN1:
            FK(run_fn1)(in->fn, t, len);
            break;
        case FI_FN3:
        {
            const FT lo = (FT) in->c;
            const FT hi = (FT) in->c2;
            for (int j = 0; j < len; j++)
            {
                const FT x = t[j];
                t[j]       = (x < lo) ? lo : (x > hi) ? hi : x;
            }
            break;
        }
        case FI_SEL:
        {
            int nv = (in->cpos & 1) + (in->cpos >> 1);
            FK(run_sel)(in, stk[sp - 1 - nv], (in->cpos & 1) ? stk[sp - nv] : NULL,
                        (in->cpos & 2) ? stk[sp - 1] : NULL, len);
            sp -= nv;
            break;
        }
        case FI_CONST:
        {
            const FT x = (FT) in->c;
            FT      *d = stk[sp++];
            for (int j = 0; j < len; j++)
            {
                d[j] = x;
            }
            break;
        }
        default:
        {
            const IMAGE *im = p->img[in->img];
            FK(coord_block)(in->cpos, im->md[0].size[0],
                            (im->md[0].naxis > 1) ? im->md[0].size[1] : 1, in->c, in->c2, off, len,
                            stk[sp++]);
            break;
        }
        } // switch (in->k)
    } // for (int i = 0; i < p->n; i++)
}

/**
 * FK(run_block) - Evaluate program @p on pixels [@off, @off + @len) and store the result into @out
 * @p:   Compiled program (scalar constants already evaluated)
 * @out: Destination image
 * @off: First pixel of the block
 * @len: Number of pixels (`<= FUSE_BLK`)
 * @stk: Per-thread vector stack, FUSE_MAXSTK slots of FUSE_BLK elements
 *
 * Return: void
 */
static void FK(run_block)(const fprog *p, IMAGE *out, uint64_t off, int len, FT (*stk)[FUSE_BLK])
{
    FK(eval_block)(p, off, len, stk);
    FK_STORE(out, off, len, stk[0]);
}

/**
 * FK(reduce_block) - Reduce a block of evaluated pixels @v (with optional mask/second vector @m)
 * @op:  Reduction operation (`RED_SUM`..`RED_DOT`)
 * @v:   Evaluated primary block of length @len
 * @m:   Optional mask block (for `RED_SUM`..`RED_VAR`) or second operand block (for `RED_DOT`)
 * @len: Number of elements in the block (`<= FUSE_BLK`)
 * @acc: Thread-local reduction accumulator updated in place
 *
 * Return: void
 */
static void FK(
    reduce_block)(reduce_op op, const FT *restrict v, const FT *restrict m, int len, fred_acc *acc)
{
    if (op == RED_DOT)
    {
        double s = 0.0;
        for (int j = 0; j < len; j++)
        {
            s += (double) v[j] * (double) m[j];
        }
        acc->sum += s;
        acc->n += (uint64_t) len;
        return;
    }
    if (m != NULL)
    {
        uint64_t nb = 0;
        double   sb = 0.0;
        double   mn = acc->min;
        double   mx = acc->max;
        for (int j = 0; j < len; j++)
        {
            if (m[j] != (FT) 0)
            {
                double x = (double) v[j];
                sb += (op == RED_ASUM) ? fabs(x) : (op == RED_NORM) ? x * x : x;
                mn = (nb == 0 || x < mn) ? x : mn;
                mx = (nb == 0 || x > mx) ? x : mx;
                nb++;
            }
        }
        if (nb == 0)
        {
            return;
        }
        double bm = sb / (double) nb;
        double m2 = 0.0;
        if (op == RED_STD || op == RED_VAR)
        {
            for (int j = 0; j < len; j++)
            {
                if (m[j] != (FT) 0)
                {
                    double d = (double) v[j] - bm;
                    m2 += d * d;
                }
            }
        }
        fred_acc blk = { nb, sb, bm, m2, mn, mx };
        fred_merge(acc, &blk);
        return;
    }
    switch (op)
    {
    case RED_SUM:
    case RED_MEAN:
    {
        double s = 0.0;
        for (int j = 0; j < len; j++)
        {
            s += (double) v[j];
        }
        acc->sum += s;
        acc->n += (uint64_t) len;
        break;
    }
    case RED_ASUM:
    {
        double s = 0.0;
        for (int j = 0; j < len; j++)
        {
            s += fabs((double) v[j]);
        }
        acc->sum += s;
        acc->n += (uint64_t) len;
        break;
    }
    case RED_MIN:
    {
        double mn = acc->min;
        int    j0 = 0;
        if (acc->n == 0 && len > 0)
        {
            mn = (double) v[0];
            j0 = 1;
        }
        for (int j = j0; j < len; j++)
        {
            double x = (double) v[j];
            mn       = (x < mn) ? x : mn;
        }
        acc->min = mn;
        acc->n += (uint64_t) len;
        break;
    }
    case RED_MAX:
    {
        double mx = acc->max;
        int    j0 = 0;
        if (acc->n == 0 && len > 0)
        {
            mx = (double) v[0];
            j0 = 1;
        }
        for (int j = j0; j < len; j++)
        {
            double x = (double) v[j];
            mx       = (x > mx) ? x : mx;
        }
        acc->max = mx;
        acc->n += (uint64_t) len;
        break;
    }
    case RED_STD:
    case RED_VAR:
    {
        double s = 0.0;
        for (int j = 0; j < len; j++)
        {
            s += (double) v[j];
        }
        double bm = s / (double) len;
        double m2 = 0.0;
        for (int j = 0; j < len; j++)
        {
            double d = (double) v[j] - bm;
            m2 += d * d;
        }
        fred_acc blk = { (uint64_t) len, s, bm, m2, acc->min, acc->max };
        fred_merge(acc, &blk);
        break;
    }
    case RED_NORM:
    {
        double s2 = 0.0;
        for (int j = 0; j < len; j++)
        {
            double x = (double) v[j];
            s2 += x * x;
        }
        acc->sum += s2;
        acc->n += (uint64_t) len;
        break;
    }
    case RED_COUNT:
    default:
    {
        uint64_t c = 0;
        for (int j = 0; j < len; j++)
        {
            c += (v[j] != (FT) 0);
        }
        acc->n += c;
        break;
    }
    } // switch (op)
}

#undef MAP_LOOP
#undef BIN_LOOPS
