"""Seeded random calculator expression generator (§6.3)."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Iterator

try:
    from tests.cli.calcdiff.datasets import all_dataset_ids
except ImportError:
    from python.tests.cli.calcdiff.datasets import all_dataset_ids

FALLBACK_FUNCTIONS: list[dict[str, object]] = [
    {"name": "sin", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "cos", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "tan", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "asin", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "acos", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "atan", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "sinh", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "cosh", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "tanh", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "exp", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "exp2", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "ln", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "log", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "log10", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "log2", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "sqrt", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "cbrt", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "abs", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "fabs", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "floor", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "ceil", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "round", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "erf", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "erfc", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "posi", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "sign", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "isnan", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "isfinite", "min_args": 1, "max_args": 1, "kind": "elem"},
    {"name": "f32", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "f64", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "u8", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "i8", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "u16", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "i16", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "u32", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "i32", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "u64", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "i64", "min_args": 1, "max_args": 1, "kind": "cast"},
    {"name": "pow", "min_args": 2, "max_args": 2, "kind": "elem"},
    {"name": "atan2", "min_args": 2, "max_args": 2, "kind": "elem"},
    {"name": "hypot", "min_args": 2, "max_args": 2, "kind": "elem"},
    {"name": "fmod", "min_args": 2, "max_args": 2, "kind": "elem"},
    {"name": "min", "min_args": 2, "max_args": 2, "kind": "elem"},
    {"name": "max", "min_args": 2, "max_args": 2, "kind": "elem"},
    {"name": "clamp", "min_args": 3, "max_args": 3, "kind": "elem"},
    {"name": "trunc", "min_args": 3, "max_args": 3, "kind": "elem"},
    {"name": "where", "min_args": 3, "max_args": 3, "kind": "where"},
    {"name": "ix", "min_args": 1, "max_args": 3, "kind": "coord"},
    {"name": "iy", "min_args": 1, "max_args": 3, "kind": "coord"},
    {"name": "ir", "min_args": 1, "max_args": 3, "kind": "coord"},
    {"name": "itheta", "min_args": 1, "max_args": 3, "kind": "coord"},
    {"name": "rand", "min_args": 1, "max_args": 2, "kind": "rng"},
    {"name": "randn", "min_args": 1, "max_args": 2, "kind": "rng"},
    {"name": "urand", "min_args": 1, "max_args": 2, "kind": "rng"},
    {"name": "grand", "min_args": 1, "max_args": 2, "kind": "rng"},
    {"name": "shift", "min_args": 3, "max_args": 4, "kind": "shift"},
    {"name": "isum", "min_args": 1, "max_args": 2, "kind": "reduce"},
    {"name": "imean", "min_args": 1, "max_args": 2, "kind": "reduce"},
    {"name": "imin", "min_args": 1, "max_args": 2, "kind": "reduce"},
    {"name": "imax", "min_args": 1, "max_args": 2, "kind": "reduce"},
    {"name": "istd", "min_args": 1, "max_args": 2, "kind": "reduce"},
    {"name": "irms", "min_args": 1, "max_args": 2, "kind": "reduce"},
    {"name": "ivar", "min_args": 1, "max_args": 2, "kind": "reduce"},
    {"name": "iamax", "min_args": 1, "max_args": 2, "kind": "reduce"},
    {"name": "icount", "min_args": 1, "max_args": 1, "kind": "reduce"},
    {"name": "iargmax", "min_args": 1, "max_args": 1, "kind": "reduce"},
    {"name": "iargmin", "min_args": 1, "max_args": 1, "kind": "reduce"},
    {"name": "dot", "min_args": 2, "max_args": 2, "kind": "reduce"},
    {"name": "istats", "min_args": 1, "max_args": 1, "kind": "multi_red"},
    {"name": "icentroid", "min_args": 1, "max_args": 2, "kind": "multi_red"},
    {"name": "re", "min_args": 1, "max_args": 1, "kind": "complex"},
    {"name": "im", "min_args": 1, "max_args": 1, "kind": "complex"},
    {"name": "arg", "min_args": 1, "max_args": 1, "kind": "complex"},
    {"name": "conj", "min_args": 1, "max_args": 1, "kind": "complex"},
    {"name": "cplx", "min_args": 2, "max_args": 2, "kind": "complex"},
    {"name": "amph", "min_args": 2, "max_args": 2, "kind": "complex"},
]

SCALAR_VARS = ["x", "y", "n", "k", "t", "alpha", "_PI", "_e"]
SCALAR_LITS = ["0", "1", "2", "3", "-1", "0.5", "1.25", "2.5", "-0.75", "0.1", "10.0"]
ARITH_OPS = ["+", "-", "*", "/"]
INT_OPS = ["%", "&", "|"]
REL_OPS = ["<", "<=", ">", ">=", "==", "!="]
LOG_OPS = ["&&", "||"]


def load_functions(
    _unused: str | pathlib.Path | None = None,
) -> list[dict[str, object]]:
    """Return the calculator function table used for expression generation."""
    return list(FALLBACK_FUNCTIONS)


def _gen_scalar_expr(
    rng: random.Random,
    depth: int,
    unary_fns: list[str],
    binary_fns: list[str],
    sc_vars: list[str] | None = None,
) -> str:
    """Generate a scalar expression of depth at most @depth (depth <= 6)."""
    vpool = sc_vars if sc_vars is not None else SCALAR_VARS
    if depth <= 1 or rng.random() < 0.32:
        if sc_vars is None and rng.random() < 0.55:
            return rng.choice(SCALAR_LITS)
        return rng.choice(vpool)

    choice = rng.random()
    if choice < 0.15:
        op = rng.choice(["-", "!"])
        sub = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        return f"({op}({sub}))"
    if choice < 0.50:
        op = rng.choice(ARITH_OPS)
        lhs = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        rhs = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        if op == "/":
            return f"({lhs} / (abs({rhs}) + 0.5))"
        return f"({lhs} {op} {rhs})"
    if choice < 0.62:
        # Parenthesized binary relational comparison per §6.3
        op = rng.choice(REL_OPS)
        lhs = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        rhs = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        return f"({lhs} {op} {rhs})"
    if choice < 0.70:
        op = rng.choice(LOG_OPS)
        lhs = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        rhs = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        return f"(({lhs}) {op} ({rhs}))"
    if choice < 0.78:
        c1 = _gen_scalar_expr(rng, max(1, depth - 2), unary_fns, binary_fns, sc_vars)
        c2 = _gen_scalar_expr(rng, max(1, depth - 2), unary_fns, binary_fns, sc_vars)
        rop = rng.choice(REL_OPS)
        t_br = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        f_br = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        return f"(({c1} {rop} {c2}) ? ({t_br}) : ({f_br}))"
    if choice < 0.90 and unary_fns:
        fn = rng.choice(unary_fns)
        arg = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
        if fn in ("sqrt", "ln", "log", "log10", "log2"):
            return f"{fn}(abs({arg}) + 0.125)"
        if fn in ("asin", "acos"):
            return f"{fn}(clamp({arg}, -0.9, 0.9))"
        if fn in ("exp", "exp2", "sinh", "cosh"):
            return f"{fn}(clamp({arg}, -4.0, 4.0))"
        if fn in ("u8", "u16", "u32", "u64"):
            return f"{fn}(clamp(abs({arg}), 0.0, 100.0))"
        if fn in ("i8", "i16", "i32", "i64"):
            return f"{fn}(clamp({arg}, -100.0, 100.0))"
        return f"{fn}({arg})"

    fn = rng.choice(binary_fns) if binary_fns else "hypot"
    a1 = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
    a2 = _gen_scalar_expr(rng, depth - 1, unary_fns, binary_fns, sc_vars)
    if fn == "pow":
        exp_lit = rng.choice(["2", "3", "0.5", "1.5"])
        return f"pow(abs({a1}) + 0.1, {exp_lit})"
    if fn == "fmod":
        return f"fmod({a1}, abs({a2}) + 0.5)"
    if fn in ("clamp", "trunc"):
        return f"{fn}({a1}, -1.0, 1.0)"
    if fn == "soft_thresh":
        return f"soft_thresh({a1}, 0.25)"
    return f"{fn}({a1}, {a2})"


def _dataset_images(dataset: str) -> list[str]:
    """Return the primary image names available in @dataset."""
    if dataset == "int_limits":
        return ["A", "B", "im_u8", "im_i8", "im_u16", "im_i16", "im_u32", "im_i32"]
    if dataset == "complex":
        return ["A", "B", "Ad", "Bd"]
    if dataset == "masks":
        return ["A", "B", "G", "O", "M", "M0", "M1", "M_rand", "Mu8"]
    if dataset == "cubes":
        return ["cube", "cubeB"]
    if dataset == "broadcast":
        return ["A", "B", "row", "col", "sc1"]
    return ["A", "B", "D", "G", "O", "M"]


def _gen_image_expr(
    rng: random.Random,
    depth: int,
    imgs: list[str],
    unary_fns: list[str],
    binary_fns: list[str],
) -> str:
    """Generate an elementwise image expression of depth at most @depth (depth <= 6)."""
    if depth <= 1 or rng.random() < 0.35:
        if rng.random() < 0.78:
            return rng.choice(imgs)
        return rng.choice(["0.5", "1.5", "2.0", "-0.25", "x", "y", "t"])

    choice = rng.random()
    if choice < 0.12:
        sub = _gen_image_expr(rng, depth - 1, imgs, unary_fns, binary_fns)
        return f"(-({sub}))"
    if choice < 0.48:
        op = rng.choice(ARITH_OPS)
        lhs = _gen_image_expr(rng, depth - 1, imgs, unary_fns, binary_fns)
        rhs = _gen_image_expr(rng, depth - 1, imgs, unary_fns, binary_fns)
        if op == "/":
            return f"({lhs} / (abs({rhs}) + 0.5))"
        return f"({lhs} {op} {rhs})"
    if choice < 0.60:
        # Parenthesized binary relational comparison per §6.3
        op = rng.choice(REL_OPS)
        lhs = rng.choice(imgs)
        rhs = rng.choice(imgs + ["0.5", "0.25", "-0.25"])
        return f"({lhs} {op} {rhs})"
    if choice < 0.72:
        im1 = rng.choice(imgs)
        im2 = rng.choice(imgs)
        rop = rng.choice(REL_OPS)
        t_br = _gen_image_expr(rng, depth - 1, imgs, unary_fns, binary_fns)
        f_br = _gen_image_expr(rng, depth - 1, imgs, unary_fns, binary_fns)
        return f"where(({im1} {rop} {im2}), {t_br}, {f_br})"
    if choice < 0.86 and unary_fns:
        fn = rng.choice(unary_fns)
        arg = _gen_image_expr(rng, depth - 1, imgs, unary_fns, binary_fns)
        if fn in ("sqrt", "ln", "log", "log10", "log2"):
            return f"{fn}(abs({arg}) + 0.125)"
        if fn in ("asin", "acos"):
            return f"{fn}(clamp({arg}, -0.9, 0.9))"
        if fn in ("exp", "exp2", "sinh", "cosh"):
            return f"{fn}(clamp({arg}, -3.0, 3.0))"
        if fn in ("u8", "u16", "u32", "u64"):
            return f"{fn}(clamp(abs({arg}), 0.0, 100.0))"
        if fn in ("i8", "i16", "i32", "i64"):
            return f"{fn}(clamp({arg}, -100.0, 100.0))"
        return f"{fn}({arg})"
    if choice < 0.93:
        im0 = rng.choice(imgs)
        kind = rng.choice(
            ["ix", "iy", "ir", "itheta", "urand", "grand", "shift", "soft_thresh"]
        )
        if kind in ("ix", "iy", "ir", "itheta"):
            return f"({kind}({im0}) * 0.01)"
        if kind in ("urand", "grand"):
            seed = rng.randint(1, 50)
            return f"{kind}({im0}, {seed})"
        if kind == "shift":
            dx = rng.choice([-1, 0, 1])
            dy = rng.choice([-1, 0, 1])
            return f"shift({im0}, {dx}, {dy})"
        return f"soft_thresh({im0}, 0.25)"

    fn = rng.choice(binary_fns) if binary_fns else "hypot"
    a1 = _gen_image_expr(rng, depth - 1, imgs, unary_fns, binary_fns)
    a2 = _gen_image_expr(rng, depth - 1, imgs, unary_fns, binary_fns)
    if fn == "pow":
        exp_lit = rng.choice(["2", "3", "0.5"])
        return f"pow(abs({a1}) + 0.1, {exp_lit})"
    if fn == "fmod":
        return f"fmod({a1}, abs({a2}) + 0.5)"
    if fn in ("clamp", "trunc"):
        return f"{fn}({a1}, -0.5, 0.9)"
    return f"{fn}({a1}, {a2})"


def generate_cases(
    count: int,
    seed: int = 42,
    kind: str = "all",
    datasets: list[str] | None = None,
    include_errors: bool = True,
) -> Iterator[dict[str, object]]:
    """Yield @count seeded calculator test-case dicts."""
    rng = random.Random(seed)
    fns = [f for f in load_functions() if f.get("in_old", 1)]
    unary_fns = [
        str(f["name"])
        for f in fns
        if f["kind"] in ("elem", "cast") and f["min_args"] == 1 and f["max_args"] == 1
    ]
    binary_fns = [
        str(f["name"]) for f in fns if f["kind"] == "elem" and int(f["min_args"]) >= 2
    ] + ["soft_thresh"]
    red_fns = [
        str(f["name"]) for f in fns if f["kind"] == "reduce" and str(f["name"]) != "dot"
    ]

    ds_pool = datasets if datasets else all_dataset_ids(include_large=False)

    tpl_bank: list[str] = []
    tpl_bank2: list[str] = []
    sc_tpl_bank: list[str] = []
    bulk_schedule: list[str] = []
    if count > 2000 and kind in ("elementwise", "all", "scalar"):
        tpl_rng = random.Random(seed ^ 0xA5A5A5A5)
        slots = ["__IM0__", "__IM1__", "__IM2__"]
        for k in range(64):
            d = (k % 6) + 1
            tpl_bank.append(_gen_image_expr(tpl_rng, d, slots, unary_fns, binary_fns))
        for k in range(16):
            d = (k % 3) + 1
            tpl_bank2.append(_gen_image_expr(tpl_rng, d, slots, unary_fns, binary_fns))
        sc_slots = ["__SC0__", "__SC1__", "__SC2__"]
        for k in range(256):
            d = (k % 6) + 1
            sc_tpl_bank.append(
                _gen_scalar_expr(tpl_rng, d, unary_fns, binary_fns, sc_slots)
            )
        heavy_ds = {
            "offset_0",
            "offset_1e3",
            "offset_1e4",
            "offset_3e4",
            "size_255",
            "size_256",
            "size_257",
        }
        fast_tiny = ["size_1", "size_7", "int_limits", "nonsquare", "cubes"]
        fast_med = ["default", "random", "nan_special", "masks", "broadcast", "complex"]
        for dname in ds_pool:
            if dname in heavy_ds:
                bulk_schedule.extend([dname] * 120)
        rem = max(0, count - len(bulk_schedule))
        tiny_avail = [d for d in fast_tiny if d in ds_pool]
        med_avail = [d for d in fast_med if d in ds_pool]
        if tiny_avail and med_avail:
            tiny_tot = int(rem * 0.75)
            med_tot = rem - tiny_tot
            for idx_d, dname in enumerate(tiny_avail):
                n_d = tiny_tot // len(tiny_avail) + (
                    1 if idx_d < (tiny_tot % len(tiny_avail)) else 0
                )
                bulk_schedule.extend([dname] * n_d)
            for idx_d, dname in enumerate(med_avail):
                n_d = med_tot // len(med_avail) + (
                    1 if idx_d < (med_tot % len(med_avail)) else 0
                )
                bulk_schedule.extend([dname] * n_d)
        else:
            for idx_d, dname in enumerate(ds_pool):
                n_d = rem // len(ds_pool) + (1 if idx_d < (rem % len(ds_pool)) else 0)
                bulk_schedule.extend([dname] * n_d)

    invalid_templates = [
        "nosuchvar_xyz + 1",
        "sin(1, 2, 3)",
        "pow(1)",
        "where(A > B, A)",
        "C = (A + B",
        "x = A * 2",
    ]

    for i in range(count):
        if include_errors and (i % 40 == 39):
            yield {
                "dataset": "default",
                "setup": [],
                "stmt": rng.choice(invalid_templates),
                "is_test": False,
            }
            continue

        if kind == "scalar" or (kind == "all" and rng.random() < 0.35):
            if sc_tpl_bank:
                sc_pool = SCALAR_LITS + SCALAR_VARS
                s0 = rng.choice(sc_pool)
                s1 = rng.choice(sc_pool)
                s2 = rng.choice(sc_pool)
                expr = (
                    sc_tpl_bank[i % len(sc_tpl_bank)]
                    .replace("__SC0__", s0)
                    .replace("__SC1__", s1)
                    .replace("__SC2__", s2)
                )
            else:
                depth = rng.randint(1, 6)
                expr = _gen_scalar_expr(rng, depth, unary_fns, binary_fns)
                while len(expr) > 300:
                    depth = max(1, depth - 1)
                    expr = _gen_scalar_expr(rng, depth, unary_fns, binary_fns)
            r = rng.random()
            if r < 0.55:
                stmt = f"v = {expr}"
            elif r < 0.75:
                aop = rng.choice(["+=", "-=", "*="])
                stmt = f"v {aop} {expr}"
            elif r < 0.88:
                if sc_tpl_bank:
                    e2 = (
                        sc_tpl_bank[(i + 7) % len(sc_tpl_bank)]
                        .replace("__SC0__", s1)
                        .replace("__SC1__", s2)
                        .replace("__SC2__", s0)
                    )
                else:
                    e2 = _gen_scalar_expr(
                        rng, min(3, max(1, depth - 1)), unary_fns, binary_fns
                    )
                stmt = f"m, s = {expr}, {e2}"
            else:
                stmt = expr
            yield {
                "dataset": "default",
                "setup": [],
                "stmt": stmt,
                "is_test": False,
            }
            continue

        ds = bulk_schedule[i] if bulk_schedule else rng.choice(ds_pool)
        imgs = _dataset_images(ds)
        dst = "cubeC" if ds == "cubes" else "C"

        if kind == "reduction" or (kind == "all" and rng.random() < 0.25):
            im1 = rng.choice(imgs)
            im2 = rng.choice(imgs)
            r = rng.random()
            if r < 0.45:
                rfn = rng.choice(red_fns)
                stmt = f"v = {rfn}({im1})"
            elif r < 0.65 and ds not in ("cubes", "broadcast"):
                rfn = rng.choice(
                    ["isum", "imean", "imin", "imax", "istd", "irms", "ivar", "iamax"]
                )
                mask = "M1" if ds == "masks" else "M"
                stmt = f"v = {rfn}({im1}, {mask})"
            elif r < 0.78:
                axis = rng.choice([0, 1])
                stmt = f"o_ax = {im1} - imean({im1}, {axis})"
            elif r < 0.88 and ds != "cubes":
                stmt = f"m, s, lo, hi = istats({im1})"
            elif r < 0.94 and ds != "cubes":
                stmt = f"cx, cy, flux = icentroid(abs({im1}) + 0.1, 0.05)"
            else:
                stmt = f"v = dot({im1}, {im2})"
            yield {
                "dataset": ds,
                "setup": [],
                "stmt": stmt,
                "is_test": False,
            }
            continue

        # Elementwise image statement
        depth = rng.randint(1, 6)
        if ds == "complex" and rng.random() < 0.35:
            stmt = rng.choice(
                [
                    "zC = zA + zB",
                    "zC = zA * zB",
                    "zC = conj(zA)",
                    "C = re(zA) + im(zB)",
                    "C = abs(zA)",
                    "C = arg(zA)",
                    "zC = cplx(A, B)",
                    "zC = amph(abs(A) + 0.1, B)",
                ]
            )
        elif tpl_bank:
            if ds in ("int_limits", "complex"):
                t_idx = i % 8
                tpl = tpl_bank[t_idx]
                im0 = rng.choice(imgs)
                im1 = "A" if ds == "int_limits" else rng.choice(["A", "B"])
                im2 = "B"
            else:
                t_idx = rng.randrange(len(tpl_bank))
                tpl = tpl_bank[t_idx]
                if len(imgs) >= 3:
                    im0, im1, im2 = rng.sample(imgs, 3)
                else:
                    im0 = rng.choice(imgs)
                    im1 = rng.choice(imgs)
                    im2 = rng.choice(imgs)
            expr = (
                tpl.replace("__IM0__", im0)
                .replace("__IM1__", im1)
                .replace("__IM2__", im2)
            )
            r = rng.random()
            if r < 0.65:
                stmt = f"{dst} = {expr}"
            elif r < 0.80:
                aop = rng.choice(["+=", "-=", "*="])
                stmt = f"{dst} {aop} {expr}"
            elif r < 0.90 and ds not in ("cubes", "size_1", "int_limits", "complex"):
                tpl2 = tpl_bank2[t_idx % len(tpl_bank2)]
                e2 = (
                    tpl2.replace("__IM0__", im1)
                    .replace("__IM1__", im2)
                    .replace("__IM2__", im0)
                )
                stmt = f"U, V = {expr}, {e2}"
            elif r < 0.95 and ds in ("default", "random", "size_256"):
                c1 = rng.choice(["1.5", "0.5", "2.0", "1.25"])
                c2 = rng.choice(["0.5", "0.25", "0.75", "1.0"])
                stmt = f"C[0:H,0:H] = A[0:H,0:H] * {c1} - B[0:H,0:H] * {c2}"
            else:
                stmt = f"local tmp = {expr}; {dst} = tmp * 0.5 + tmp"
        else:
            expr = _gen_image_expr(rng, depth, imgs, unary_fns, binary_fns)
            r = rng.random()
            if r < 0.60:
                stmt = f"{dst} = {expr}"
            elif r < 0.75:
                aop = rng.choice(["+=", "-=", "*="])
                stmt = f"{dst} {aop} {expr}"
            elif r < 0.86 and ds not in ("cubes", "size_1"):
                e2 = _gen_image_expr(
                    rng, max(1, depth - 1), imgs, unary_fns, binary_fns
                )
                stmt = f"U, V = {expr}, {e2}"
            elif r < 0.94 and ds in ("default", "random", "size_256"):
                stmt = "C[0:H,0:H] = A[0:H,0:H] * 1.5 - B[0:H,0:H] * 0.5"
            else:
                stmt = f"local tmp = {expr}; {dst} = tmp * 0.5 + tmp"

        yield {
            "dataset": ds,
            "setup": [],
            "stmt": stmt,
            "is_test": False,
        }


def main() -> None:
    """CLI entry point to emit generated JSON lines to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--count", type=int, default=1000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument(
        "-k",
        "--kind",
        choices=["all", "scalar", "elementwise", "reduction"],
        default="all",
    )
    args = parser.parse_args()
    for item in generate_cases(count=args.count, seed=args.seed, kind=args.kind):
        print(json.dumps(item, separators=(",", ":")))


if __name__ == "__main__":
    main()
