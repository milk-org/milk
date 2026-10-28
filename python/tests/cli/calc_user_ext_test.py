"""
Regression tests for user-defined C extensions in milk.calc / @calc (`milk_calc_ext.h`)
and standalone compilation via `milk-cc`:
  - Tier 1a: 1D SIMD block kernels (`MILK_CALC_REG_BLK`, `MILK_CALC_REG_BLK2`)
  - Tier 1b: 2D neighborhood stencils (`MILK_CALC_REG_STENCIL`)
  - Tier 1c: Single-function whole-image kernels (`MILK_CALC_REG_IMG`)
  - Tier 2:  Multi-input / multi-output stateful operators (`MILK_CALC_OP_DECLARE`, `MILK_CALC_REG_OP`)
  - Tier 3:  Single-command `MILK_RUNNABLE` compilation (`milk-cc <unit.c>`)
  - FPS parameter exposure rules: unbound params exposed under `.<func>.<param>` or `.<label>.<param>`,
    call-site bound params (`th = 3 * rms`) hidden from FPS, and scalar output ports exposed as
    `FPFLAG_DEFAULT_OUTPUT`.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(90),
]


def test_tier1a_blk_and_fps_param_binding_rules(tmp_path):
    """Tier 1a (`MILK_CALC_REG_BLK` & `MILK_CALC_REG_BLK2`):
    - `.c` file is auto-compiled into a cached `.so` and loaded via `-l`.
    - Unbound parameter (`th`) is exposed in FPS as `.clamp_hi.th` (or `.stage1.th` for `clamp_hi:stage1`).
    - Bound parameter at call site (`clamp_hi:stage2(a, th = 2.0)`) is NOT exposed in FPS.
    - 2-input block kernel (`blend(a, b, alpha = 0.5)`) fuses cleanly.
    """
    c_file = tmp_path / "tier1_blk.c"
    c_file.write_text(r"""
#include <math.h>
#include "milk_calc_ext.h"

#define CLAMP_PARAMS(X) \
    X(".th", th, FPTYPE_FLOAT64, 5.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Clamp ceiling")
MILK_CALC_PARAMS(clamp_hi, CLAMP_PARAMS);

static void clamp_hi_f(
    float *restrict            out,
    const float *restrict      in,
    const struct clamp_hi_par *P,
    int                        len)
{
    const float th = (float) P->th;
    #pragma omp simd
    for (int i = 0; i < len; i++)
    {
        out[i] = (in[i] > th) ? th : in[i];
    }
}

#define BLEND_PARAMS(X) \
    X(".alpha", alpha, FPTYPE_FLOAT64, 0.5, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Blend weight")
MILK_CALC_PARAMS(blend, BLEND_PARAMS);

static void blend_f(
    float *restrict         out,
    const float *restrict   in0,
    const float *restrict   in1,
    const struct blend_par *P,
    int                     len)
{
    const float a = (float) P->alpha;
    #pragma omp simd
    for (int i = 0; i < len; i++)
    {
        out[i] = (1.0f - a) * in0[i] + a * in1[i];
    }
}

MILK_CALC_EXT_EXPORT(
    MILK_CALC_REG_BLK("clamp_hi", clamp_hi_f, clamp_hi, 2),
    MILK_CALC_REG_BLK2("blend", blend_f, blend, 2));
""")

    (tmp_path / "prog_blk.calc").write_text(
        "a = clamp_hi:stage1(raw)\n"
        "b = clamp_hi:stage2(a, th = 3.0)\n"
        "c = blend(b, ref, alpha = 0.5)\n"
    )
    res = milk_bash(
        """
mem.mk2Dim "@S:raw" 4 4 >/dev/null
mem.mk2Dim "@S:ref" 4 4 >/dev/null
@calc "raw = 10; ref = 2"

# Unbound clamp_hi:stage1 exposes .stage1.th (default 5.0); bound clamp_hi:stage2(a, th=3.0) hides .stage2.th
milk-fpsexec-milk-calc -n u_blk -l tier1_blk.c -f prog_blk.calc exec >/dev/null 2>&1
echo "rc=$?"
echo "a=$(@get s.a.mean) b=$(@get s.b.mean) c=$(@get s.c.mean)"
echo "s1_th=$(@get f.u_blk.stage1.th)"
@get f.u_blk.stage2.th >/dev/null 2>&1 && echo "s2_exists=1" || echo "s2_exists=0"

# Override .stage1.th to 2.0 on re-run
@set f.u_blk.stage1.th = 2.0
milk-fpsexec-milk-calc -n u_blk exec >/dev/null 2>&1
echo "a2=$(@get s.a.mean) b2=$(@get s.b.mean)"
""",
        tmp_path,
    )
    assert [value(res.stdout, k) for k in ("rc", "a", "s1_th", "s2_exists", "a2")] == [
        "0",
        "5 b=3 c=2.5",
        "5",
        "0",
        "2 b2=2",
    ], (
        res.stdout + res.stderr
    )


def test_bound_params_stay_out_of_the_fps_in_any_call(tmp_path):
    """A parameter bound at the call site, by name or by position, also inside a nested argument, is
    not in the FPS, and its value is used. The call-site scanner took every name, comma and `)` for a
    nested call, so it never saw a bound parameter."""
    (tmp_path / "clamp.c").write_text(r"""
#include "milk_calc_ext.h"

#define CLAMP_PARAMS(X) \
    X(".th", th, FPTYPE_FLOAT64, 5.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Clamp ceiling")
MILK_CALC_PARAMS(clamp_hi, CLAMP_PARAMS);

static void clamp_hi_f(float *restrict out, const float *restrict in, const struct clamp_hi_par *P, int len)
{
    for (int i = 0; i < len; i++)
    {
        out[i] = (in[i] > (float) P->th) ? (float) P->th : in[i];
    }
}

MILK_CALC_EXT_EXPORT(MILK_CALC_REG_BLK("clamp_hi", clamp_hi_f, clamp_hi, 2));
""")
    calls = {
        "kw": "o1 = clamp_hi(raw, th = 3.0)",
        "pos": "o2 = clamp_hi(raw, 3.0)",
        "nested": "o3 = clamp_hi(raw * 1 + imean(raw) * 0, th = 2 * 1.5)",
        "free": "o4 = clamp_hi(raw)",
    }
    res = milk_bash(
        'mem.mk2Dim "@S:raw" 4 4 >/dev/null; @calc "raw = 10"\n'
        + "".join(
            f'milk-fpsexec-milk-calc -n b{k} -l clamp.c -e "{e}" exec >/dev/null 2>&1; echo "rc_{k}=$?"\n'
            f'@get f.b{k}.clamp_hi.th >/dev/null 2>&1 && echo "{k}=1" || echo "{k}=0"\n'
            for k, e in calls.items()
        )
        + 'echo "vals=$(@get s.o1.mean) $(@get s.o2.mean) $(@get s.o3.mean) $(@get s.o4.mean)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert [value(res.stdout, f"rc_{k}") for k in calls] == ["0"] * 4, out
    assert [value(res.stdout, k) for k in calls] == ["0", "0", "0", "1"], out
    assert value(res.stdout, "vals") == "3 3 3 5", out


def test_tier1b_stencil_3x3(tmp_path):
    """Tier 1b (`MILK_CALC_REG_STENCIL`):
    2D 3x3 neighborhood stencil (`R = 1`) replaces isolated spikes with the 8-neighbor average.
    """
    c_file = tmp_path / "tier1_stencil.c"
    c_file.write_text(r"""
#include <math.h>
#include "milk_calc_ext.h"

#define DESPIKE_PARAMS(X) \
    X(".th", th, FPTYPE_FLOAT64, 5.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Spike threshold")
MILK_CALC_PARAMS(despike, DESPIKE_PARAMS);

static void despike3x3_f(
    float *restrict           out,
    const float *const       *rows,
    const struct despike_par *P,
    int                       len)
{
    const float *a  = rows[0];
    const float *b  = rows[1];
    const float *c  = rows[2];
    const float  th = (float) P->th;

    #pragma omp simd
    for (int i = 0; i < len; i++)
    {
        float m8 = 0.125f * (a[i] + a[i + 1] + a[i + 2] +
                             b[i]            + b[i + 2] +
                             c[i] + c[i + 1] + c[i + 2]);
        float v  = b[i + 1];
        out[i]   = (fabsf(v - m8) > th) ? m8 : v;
    }
}

MILK_CALC_EXT_EXPORT(
    MILK_CALC_REG_STENCIL("despike3x3", despike3x3_f, despike, 1, 4));
""")

    res = milk_bash(
        f"""
mem.mk2Dim "@S:im_sp" 5 5 >/dev/null
@calc "im_sp = 10; im_sp[2,2] = 30"
@calc -l "{c_file}" "clean = despike3x3(im_sp)"
echo "max=$(@get s.clean.max) min=$(@get s.clean.min) mean=$(@get s.clean.mean)"
""",
        tmp_path,
    )
    assert value(res.stdout, "max") == "10 min=10 mean=10", res.stdout + res.stderr


def test_tier1c_whole_image_kernel(tmp_path):
    """Tier 1c (`MILK_CALC_REG_IMG`):
    Single-function whole-image kernel (`col_demean`) subtracts each column's mean
    and composes inside an expression (`out = col_demean(raw) + offset`).
    """
    c_file = tmp_path / "tier1_img.c"
    c_file.write_text(r"""
#include <stdint.h>
#include "milk_calc_ext.h"

#define COL_PARAMS(X) \
    X(".scale", scale, FPTYPE_FLOAT64, 1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Post-demean scale")
MILK_CALC_PARAMS(col_demean, COL_PARAMS);

static int col_demean_img(
    IMAGE                       *out,
    const IMAGE                 *in,
    const struct col_demean_par *P)
{
    uint32_t     nx  = in->md[0].size[0];
    uint32_t     ny  = (in->md[0].naxis > 1) ? in->md[0].size[1] : 1;
    const float *src = in->array.F;
    float       *dst = out->array.F;
    const float  sc  = (float) P->scale;

    for (uint32_t x = 0; x < nx; x++)
    {
        double sum = 0.0;
        for (uint32_t y = 0; y < ny; y++)
        {
            sum += src[(size_t) y * nx + x];
        }
        float mean = (float) (sum / (double) ny);
        for (uint32_t y = 0; y < ny; y++)
        {
            dst[(size_t) y * nx + x] = (src[(size_t) y * nx + x] - mean) * sc;
        }
    }
    return 0;
}

MILK_CALC_EXT_EXPORT(
    MILK_CALC_REG_IMG("col_demean", col_demean_img, col_demean));
""")

    res = milk_bash(
        """
mem.mk2Dim "@S:raw_col" 4 4 >/dev/null
@calc "raw_col = ix(raw_col) * 10 + iy(raw_col)"
milk-fpsexec-milk-calc -n u_col -l tier1_img.c \\
    -e "out_col = col_demean(raw_col) + 10" exec .col_demean.scale=2 >/dev/null 2>&1
echo "rc=$?"
echo "mean=$(@get s.out_col.mean) min=$(@get s.out_col.min) max=$(@get s.out_col.max)"
""",
        tmp_path,
    )
    # iy() in 0..3 has mean 1.5; (iy - 1.5) * 2 + 10 has min=7, max=13, mean=10
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert value(res.stdout, "mean") == "10 min=7 max=13", res.stdout + res.stderr


def test_tier2_stateful_multi_port_operator(tmp_path):
    """Tier 2 (`MILK_CALC_OP_DECLARE` / `MILK_CALC_REG_OP`):
    Multi-input / multi-output stateful operator (`shslopes`) with:
      - `init` / `exec` / `free` lifecycle
      - Output shape `milk_op_set_out_shape` different from input shape
      - Scalar output port (`flux`) auto-exposed as `FPFLAG_DEFAULT_OUTPUT` in the unit's FPS
      - Init-time parameter (`nsub`, no `FPFLAG_WRITERUN`) vs runtime parameter (`thresh`, `FPFLAG_WRITERUN`)
    """
    c_file = tmp_path / "tier2_op.c"
    c_file.write_text(r"""
#include <stdlib.h>
#include "milk_calc_ext.h"

#define SH_PARAMS(X) \
    X(".nsub",   nsub,   FPTYPE_INT64,   2,   FPFLAG_DEFAULT_INPUT,                   "Grid size") \
    X(".thresh", thresh, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Threshold")

#define SH_PORTS(X) \
    X(IN,  frame, MILK_PORT_IMAGE,  MILK_PORT_TRIGGER,                "Calibrated WFS image") \
    X(IN,  flat,  MILK_PORT_IMAGE,  MILK_PORT_LATEST | MILK_PORT_OPT, "Optional flat field") \
    X(OUT, sx,    MILK_PORT_IMAGE,  MILK_POST_EVERY,                  "X slopes (nsub x nsub)") \
    X(OUT, sy,    MILK_PORT_IMAGE,  MILK_POST_EVERY,                  "Y slopes (nsub x nsub)") \
    X(OUT, flux,  MILK_PORT_SCALAR, 0,                                "Total flux")

MILK_CALC_OP_DECLARE(shs, SH_PARAMS, SH_PORTS);

struct sh_state
{
    int nsub;
    int sub_pix;
};

static int sh_init(
    milk_op_ctx          *ctx,
    const struct shs_par *P,
    void                **state)
{
    const IMAGE *frame = MILK_IN(ctx, shs_frame);
    int          nsub  = (P->nsub > 0) ? (int) P->nsub : 2;
    uint32_t     nx    = frame->md[0].size[0];

    struct sh_state *st = calloc(1, sizeof(*st));
    if (st == NULL)
    {
        return -1;
    }
    st->nsub    = nsub;
    st->sub_pix = (int) (nx / (uint32_t) nsub);

    milk_op_set_out_shape(ctx, shs_sx, (uint32_t) nsub, (uint32_t) nsub);
    milk_op_set_out_shape(ctx, shs_sy, (uint32_t) nsub, (uint32_t) nsub);
    *state = st;
    return 0;
}

static int sh_exec(
    milk_op_ctx          *ctx,
    const struct shs_par *P,
    void                 *state)
{
    struct sh_state *st    = (struct sh_state *) state;
    const IMAGE     *frame = MILK_IN(ctx, shs_frame);
    IMAGE           *sx    = MILK_OUT(ctx, shs_sx);
    IMAGE           *sy    = MILK_OUT(ctx, shs_sy);

    const float *in  = frame->array.F;
    uint32_t     nx  = frame->md[0].size[0];
    int          ns  = st->nsub;
    int          sp  = st->sub_pix;
    float        th  = (float) P->thresh;
    double       tot = 0.0;

    for (int j = 0; j < ns; j++)
    {
        for (int i = 0; i < ns; i++)
        {
            double s = 0.0;
            for (int dy = 0; dy < sp; dy++)
            {
                for (int dx = 0; dx < sp; dx++)
                {
                    float v = in[(j * sp + dy) * (int) nx + (i * sp + dx)] - th;
                    if (v > 0.0f)
                    {
                        s += v;
                    }
                }
            }
            sx->array.F[j * ns + i] = (float) s;
            sy->array.F[j * ns + i] = (float) (s * 2.0);
            tot += s;
        }
    }
    MILK_OUT_SCALAR(ctx, shs_flux) = tot;
    return 0;
}

static void sh_free(void *state)
{
    free(state);
}

MILK_CALC_EXT_EXPORT(
    MILK_CALC_REG_OP("shslopes", shs, sh_init, sh_exec, sh_free));
""")

    res = milk_bash(
        """
mem.mk2Dim "@S:wfs_in" 8 8 >/dev/null
@calc "wfs_in = 3"

milk-fpsexec-milk-calc -n u_sh -l tier2_op.c \\
    -e "wfs_sx, wfs_sy, total_flux = shslopes(wfs_in)" exec >/dev/null 2>&1
echo "rc=$?"
echo "shape=$(@get s.wfs_sx.xsize)x$(@get s.wfs_sx.ysize) sx=$(@get s.wfs_sx.mean) sy=$(@get s.wfs_sy.mean)"
echo "flux=$(@get f.u_sh.total_flux)"

# Change runtime threshold parameter (.shslopes.thresh = 1.0 -> each pixel is 3 - 1 = 2, 16 pix/subap -> 32)
@set f.u_sh.shslopes.thresh = 1.0
milk-fpsexec-milk-calc -n u_sh exec >/dev/null 2>&1
echo "sx2=$(@get s.wfs_sx.mean) flux2=$(@get f.u_sh.total_flux)"
""",
        tmp_path,
    )
    assert [value(res.stdout, k) for k in ("rc", "shape", "flux", "sx2")] == [
        "0",
        "2x2 sx=48 sy=96",
        "192",
        "32 flux2=128",
    ], (
        res.stdout + res.stderr
    )


def test_tier3_milk_cc_standalone_runnable(tmp_path):
    """Tier 3 (`milk-cc <unit.c>`):
    Compiles a single-file `MILK_RUNNABLE` `.c` source into `./milk-fpsexec-milk-custom-scale`
    with zero CMake setup and runs it.
    """
    c_file = tmp_path / "custom_scale.c"
    c_file.write_text(r"""
#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

static char   in_sname[FUNCTION_PARAMETER_STRMAXLEN]  = "cscale_in";
static char   out_sname[FUNCTION_PARAMETER_STRMAXLEN] = "cscale_out";
static double factor                                  = 3.0;

#define CSCALE_PARAMS(X) \
    X(".in_sname",  in_sname,  FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT_STREAM, "Input stream") \
    X(".out_sname", out_sname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,        "Output stream") \
    X(".factor",    &factor,   FPTYPE_FLOAT64,    1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Scale")

MILK_PARAMS_DEFINE(cscale, CSCALE_PARAMS);

static errno_t compute_cscale(const MILK_RUNNABLE *self)
{
    IMGID imgin = imgid_make_from_name(in_sname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }
    IMGID imgout = imgid_make_from_name(out_sname);
    imgid_copy(&imgin, &imgout);
    imgout.mdt->shared = 1;
    imcreateIMGID(&imgout);

    uint64_t nelem = imgin.md->nelement;
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        const float f = (float) factor;
        imgout.md->write = 1;
        for (uint64_t i = 0; i < nelem; i++)
        {
            imgout.im->array.F[i] = imgin.im->array.F[i] * f;
        }
        processinfo_update_output_stream(processinfo, imgout.im, imgin.im);
    }
    MILK_PROCINFO_END(self)
    imgid_free(&imgin);
    imgid_free(&imgout);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cscale = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = {
        .fps_name         = "cscale",
        .cmdkey           = "cscale",
        .description      = "Scale a stream by a factor",
        .description_long = "Standalone MILK_RUNNABLE compiled via milk-cc.",
    },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cscale),
    .compute = compute_cscale,
};

EXPORT_FPSEXEC(milk, "custom-scale", runnable_cscale);
""")

    res = milk_bash(
        f"""
milk-cc "{c_file}" >/dev/null 2>&1
echo "cc_rc=$?"
mem.mk2Dim "@S:cscale_in" 4 4 >/dev/null
@calc "cscale_in = 4"
./milk-fpsexec-custom-scale -n cs1 exec cscale_in cscale_out 2.5 >/dev/null 2>&1
echo "exec_rc=$?"
echo "out=$(@get s.cscale_out.mean)"
""",
        tmp_path,
    )
    assert [value(res.stdout, k) for k in ("cc_rc", "exec_rc", "out")] == [
        "0",
        "0",
        "10",
    ], (
        res.stdout + res.stderr
    )


def test_calc_list_engines(tmp_path):
    """Verify `@calc --list` / `--engines` / `--tokens` and `milk-fpsexec-milk-calc --list`
    for built-in bytecode engines, filtered patterns, JSON output, and loaded user C extensions.
    """
    c_file = tmp_path / "list_ext.c"
    c_file.write_text(r"""
#include "milk_calc_ext.h"

#define GAIN_PARAMS(X) \
    X(".gain", gain, FPTYPE_FLOAT32, 2.0f, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Gain")
MILK_CALC_PARAMS(gain_p, GAIN_PARAMS);

static void gain_blk(
    float *restrict         out,
    const float *restrict   in,
    const struct gain_p_par *P,
    int                     len)
{
    for (int i = 0; i < len; i++)
    {
        out[i] = in[i] * P->gain;
    }
}

MILK_CALC_EXT_EXPORT(
    MILK_CALC_REG_BLK("my_gain_op", gain_blk, gain_p, 1));
""")

    res = milk_bash(
        f"""
@calc --list | grep -q "FI_INTERP2" && echo "full=ok"
@calc --engines stencil | grep -q "FI_MEDFILT" && echo "stencil=ok"
@calc --tokens reduce | grep -q "FI_RED (STATS)" && echo "reduce=ok"
@calc --list --json interp2 | grep -q '"engine": "FI_INTERP2"' && echo "json=ok"
@calc -l "{c_file}" --list my_gain_op | grep -q "FI_USER_BLK" && echo "ext_bash=ok"
milk-fpsexec-milk-calc -l "{c_file}" --list my_gain_op | grep -q "my_gain_op(x)" && echo "ext_unit=ok"
""",
        tmp_path,
    )
    assert [
        value(res.stdout, k)
        for k in ("full", "stencil", "reduce", "json", "ext_bash", "ext_unit")
    ] == ["ok", "ok", "ok", "ok", "ok", "ok"], (res.stdout + res.stderr)


def test_builtin_module_exported_kernels(tmp_path):
    """Verify `MILK_CALC_MODULE_EXPORT` kernels (`bblur1pix`, `rotate2d`, `circsym`, `circasym`)
    are pre-registered at module load time (usable without `-l`), expose unbound FPS parameters
    when used in `milk-fpsexec-milk-calc`, and agree with the augmented `image_basic` / `image_filter`
    commands.
    """
    res = milk_bash(
        """
mem.mk2Dim "@S:im_in" 16 16 >/dev/null
@calc "im_in = exp(-0.1 * ((ix(im_in) - 8)^2 + (iy(im_in) - 8)^2))"

# 1. Built-in module kernels usable directly in @calc without -l
@calc "im_rot0 = rotate2d(im_in, angle = 0.0)"
@calc "im_blur = bblur1pix(im_in, amp = 0.1, niter = 2)"
@calc "im_csym = circsym(im_in, cx = 8.0, cy = 8.0)"
@calc "im_casym = circasym(im_in, cx = 8.0, cy = 8.0)"

# 2. Unbound parameter in milk-fpsexec-milk-calc exposes .bblur1pix.amp in FPS
milk-fpsexec-milk-calc -n u_mod_ext -e "im_blur_u = bblur1pix(im_in)" exec >/dev/null 2>&1
echo "u_rc=$?"
echo "u_amp=$(@get f.u_mod_ext.bblur1pix.amp)"

# 3. Augmented image_basic commands (imexpand, resizeim)
imgbasic.imexpand "@S:im_in" "@S:im_exp" 2 2 >/dev/null
imgbasic.resizeim im_in im_res 32 32 >/dev/null
echo "exp_shape=$(@get s.im_exp.xsize)x$(@get s.im_exp.ysize)"
echo "res_shape=$(@get s.im_res.xsize)x$(@get s.im_res.ysize)"

# Check identity rotation and small radial residual on symmetric Gaussian
@calc "rot_err = imax(abs(im_rot0 - im_in)); asym_rms = irms(im_casym)"
echo "rot_ok=$(@calc 'rot_err < 1e-6')"
echo "asym_ok=$(@calc 'asym_rms < 0.05')"
echo "blur_ok=$(@calc 'imax(im_blur) > 0.1 && imax(im_blur) < imax(im_in)')"
""",
        tmp_path,
    )
    assert [
        value(res.stdout, k)
        for k in (
            "u_rc",
            "u_amp",
            "exp_shape",
            "res_shape",
            "rot_ok",
            "asym_ok",
            "blur_ok",
        )
    ] == ["0", "0.1", "32x32", "32x32", "1", "1", "1"], (
        res.stdout + res.stderr
    )
