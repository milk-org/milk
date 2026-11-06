"""Differential tests comparing calculator engines via `milk-calcdiff` (§5 Phase 3, §6)."""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.calcdiff.datasets import DATASETS, all_dataset_ids, offset_dataset_ids
from tests.cli.calcdiff.extract_corpus import collect_corpus
from tests.cli.calcdiff.gen_expr import generate_cases, load_functions
from tests.cli.conftest import _install_dir, milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(180),
]

ALLOWLIST_PATH = pathlib.Path(__file__).resolve().parent / "calc_diff_allowlist.json"


def _calcdiff_bin() -> pathlib.Path:
    """Return path to installed `milk-calcdiff` executable."""
    return _install_dir() / "bin" / "milk-calcdiff"


def _load_allowlist() -> dict[str, str]:
    """Load `calc_diff_allowlist.json` and return a map from statement text to its rule."""
    raw = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    allow: dict[str, str] = {}
    for entry in raw:
        stmt = entry.get("stmt", "")
        rule = entry.get("rule", "")
        assert stmt and rule, f"Allowlist entry missing stmt or rule: {entry!r}"
        allow[stmt] = rule
    return allow


def _run_calcdiff(
    json_lines: list[str],
    force_old_vs_old: bool = False,
    extra_args: list[str] | None = None,
) -> list[dict[str, str]]:
    """Run `milk-calcdiff` on @json_lines, falling back to `--old-vs-old` before Phase 4."""
    exe = _calcdiff_bin()
    assert exe.is_file(), f"milk-calcdiff not found at {exe}"
    payload = "\n".join(json_lines) + "\n"

    cmd = [str(exe)]
    if force_old_vs_old:
        cmd.append("--old-vs-old")
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd,
        input=payload,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if not force_old_vs_old and proc.returncode == 2 and "not available" in proc.stdout:
        proc = subprocess.run(
            [str(exe), "--old-vs-old"] + (extra_args or []),
            input=payload,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    assert (
        proc.returncode == 0
    ), f"milk-calcdiff failed (rc={proc.returncode}): {proc.stderr}"
    diffs: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            diffs.append(json.loads(line))
    return diffs


def test_allowlist_entries_have_guide_rules():
    """Every entry in `calc_diff_allowlist.json` names a statement and its guide rule."""
    allow = _load_allowlist()
    assert len(allow) >= 1


def test_datasets_and_dump_functions():
    """All §6.2 datasets are registered and `milk-calcdiff --dump-functions` returns the table."""
    ids = all_dataset_ids(include_large=True)
    for expected in (
        "default",
        "random",
        "offset_0",
        "offset_1e3",
        "offset_1e4",
        "offset_3e4",
        "nan_special",
        "int_limits",
        "complex",
        "masks",
        "cubes",
        "size_1",
        "size_7",
        "size_255",
        "size_256",
        "size_257",
        "size_1024",
        "size_2048",
        "nonsquare",
        "broadcast",
    ):
        assert expected in DATASETS
        assert expected in ids
    assert offset_dataset_ids() == [
        "offset_0",
        "offset_1e3",
        "offset_1e4",
        "offset_3e4",
    ]

    fns = load_functions(_calcdiff_bin())
    names = {f["name"] for f in fns}
    for required in (
        "sin",
        "cos",
        "log",
        "log10",
        "pow",
        "where",
        "imean",
        "istd",
        "istats",
    ):
        assert required in names


def test_corpus_diff():
    """Run the extracted corpus from `python/tests/` and `docs/guide/` through `milk-calcdiff`."""
    corpus = collect_corpus()
    assert len(corpus) >= 400, f"Expected >= 400 corpus statements, got {len(corpus)}"

    allow = _load_allowlist()
    lines = [e.to_json_line() for e in corpus]
    diffs = _run_calcdiff(lines)
    unexpected = [d for d in diffs if d.get("stmt") not in allow]
    assert not unexpected, (
        f"{len(unexpected)} unexpected diff(s) out of {len(corpus)} corpus statements:\n"
        + "\n".join(json.dumps(d) for d in unexpected[:20])
    )


def test_generated_expressions_diff():
    """Run a seeded batch of generated expressions across all §6.2 datasets."""
    cases = list(
        generate_cases(count=1000, seed=42, kind="all", calcdiff_bin=_calcdiff_bin())
    )
    lines = [json.dumps(c, separators=(",", ":")) for c in cases]
    diffs = _run_calcdiff(lines)
    assert (
        not diffs
    ), f"{len(diffs)} unexpected diff(s) in generated expressions:\n" + "\n".join(
        json.dumps(d) for d in diffs[:20]
    )


def test_generated_scalar_strict_diff():
    """Run generated scalar expressions with `--strict` (no fallback to old allowed)."""
    cases = list(
        generate_cases(count=5000, seed=42, kind="scalar", calcdiff_bin=_calcdiff_bin())
    )
    lines = [json.dumps(c, separators=(",", ":")) for c in cases]
    diffs = _run_calcdiff(lines, extra_args=["--strict"])
    assert (
        not diffs
    ), f"{len(diffs)} unexpected diff(s) in strict scalar expressions:\n" + "\n".join(
        json.dumps(d) for d in diffs[:20]
    )


def test_generated_elementwise_100k_diff():
    """Run 100,000 generated elementwise statements across all §6.2 datasets with `--strict`."""
    cases = list(
        generate_cases(
            count=100000, seed=42, kind="elementwise", calcdiff_bin=_calcdiff_bin()
        )
    )
    lines = [json.dumps(c, separators=(",", ":")) for c in cases]
    diffs = _run_calcdiff(lines, extra_args=["--strict"])
    assert not diffs, (
        f"{len(diffs)} unexpected diff(s) in 100,000 generated elementwise statements:\n"
        + "\n".join(json.dumps(d) for d in diffs[:20])
    )


def test_local_evaluator_parity():
    """Verify `mcalc_eval_local` matches `mcalc_eval` across corpus and scalar expressions."""
    corpus = collect_corpus()
    allow = _load_allowlist()
    lines = [e.to_json_line() for e in corpus]
    cases = list(
        generate_cases(count=2000, seed=42, kind="scalar", calcdiff_bin=_calcdiff_bin())
    )
    lines.extend(json.dumps(c, separators=(",", ":")) for c in cases)
    diffs = _run_calcdiff(lines, extra_args=["--test-local"])
    unexpected = [d for d in diffs if d.get("stmt") not in allow]
    assert not unexpected, (
        f"{len(unexpected)} unexpected diff(s) in local evaluator parity:\n"
        + "\n".join(json.dumps(d) for d in unexpected[:20])
    )


def test_offset_datasets_reduction_accuracy(tmp_path):
    """Verify `istd`, `istats` and z-score within 1e-6 of a double reference at offsets 0..3e4."""
    stmts = (
        "v = imean(A)",
        "v = istd(A)",
        "v = ivar(A)",
        "m, s, lo, hi = istats(A)",
        "C = (A - imean(A)) / istd(A)",
        "C = A - imean(A, 0)",
        "C = A - imean(A, 1)",
    )
    lines = [
        json.dumps({"dataset": ds, "stmt": st}, separators=(",", ":"))
        for ds in offset_dataset_ids()
        for st in stmts
    ]
    diffs = _run_calcdiff(lines, extra_args=["--strict"])
    assert not diffs, "Unexpected diff(s) on offset datasets:\n" + "\n".join(
        json.dumps(d) for d in diffs[:20]
    )

    script = (
        "mem.mk2Dim A 256 256 >/dev/null\n"
        "for off in 0 1000 10000 30000; do\n"
        '  @calc "A = $off + sin(0.37 * ix(A)) + cos(0.21 * iy(A)); Ad = f64(A)"\n'
        '  @calc "s32 = istd(A); s64 = istd(Ad); m32, st32, lo32, hi32 = istats(A); '
        'm64, st64, lo64, hi64 = istats(Ad)"\n'
        '  @calc "Z32 = (A - imean(A)) / istd(A); Z64 = f32((Ad - imean(Ad)) / istd(Ad)); '
        'zerr = imax(abs(Z32 - Z64)) / imax(abs(Z64))"\n'
        '  @calc "serr = abs(s32 - s64) / s64; sterr = abs(st32 - st64) / st64; '
        'merr = abs(m32 - m64) / max(abs(m64), 1.0)"\n'
        '  @calc -v serr,sterr,merr,zerr "serr; sterr; merr; zerr"\n'
        '  echo "off_$off=$serr,$sterr,$merr,$zerr"\n'
        "done\n"
    )
    res = milk_bash(script, tmp_path, extra_env={"MILK_CALC_ENGINE": "new"})
    assert res.returncode == 0, res.stdout + res.stderr
    for off in (0, 1000, 10000, 30000):
        vals = [float(x) for x in value(res.stdout, f"off_{off}").split(",")]
        assert len(vals) == 4
        for err_val in vals:
            assert (
                err_val <= 1e-6
            ), f"Offset {off} exceeded 1e-6 relative error vs double ref: {vals}"


def test_iargmax_iargmin_coord_unpack(tmp_path):
    """Verify 2-D and 3-D coordinate unpacking on `iargmax` and `iargmin` (§3.2 rule 11)."""
    script = (
        "mem.mk2Dim im2 16 12 >/dev/null\n"
        "mem.mk3Dim im3 8 6 4 >/dev/null\n"
        '@calc "im2 = ix(im2) + 10 * iy(im2); im2[5, 9] = 999.0; im2[11, 3] = -500.0"\n'
        '@calc "mx2, my2 = iargmax(im2); nx2, ny2 = iargmin(im2); flat2 = iargmax(im2)"\n'
        '@calc "im3 = ix(im3) + 10 * iy(im3); '
        'im3[3, 4, 2] = 9999.0; im3[6, 1, 3] = -9999.0"\n'
        '@calc "mx3, my3, mz3 = iargmax(im3); nx3, ny3, nz3 = iargmin(im3)"\n'
        '@calc "nan2 = im2 * nan; nanx, nany = iargmax(nan2)"\n'
        '@calc -v mx2,my2,nx2,ny2,flat2 "mx2; my2; nx2; ny2; flat2"\n'
        '@calc -v mx3,my3,mz3,nx3,ny3,nz3,nanx,nany "mx3; my3; mz3; nx3; ny3; nz3; nanx; nany"\n'
        'echo "c2=$mx2,$my2,$nx2,$ny2,$flat2"\n'
        'echo "c3=$mx3,$my3,$mz3,$nx3,$ny3,$nz3,$nanx,$nany"\n'
    )
    res = milk_bash(script, tmp_path, extra_env={"MILK_CALC_ENGINE": "new"})
    assert res.returncode == 0, res.stdout + res.stderr
    assert value(res.stdout, "c2") == "5,9,11,3,149"
    assert value(res.stdout, "c3") == "3,4,2,6,1,3,-1,-1"

    bad = milk_bash(
        'mem.mk2Dim im2 8 8 >/dev/null; @calc "px, py, pz = iargmax(im2)"',
        tmp_path,
        extra_env={"MILK_CALC_ENGINE": "new"},
    )
    assert bad.returncode != 0


def test_phase7_library_and_spatial_ops(tmp_path):
    """Verify Phase 7 library calls, spatial ops, and new §3.2 features (§5 Phase 7)."""
    diff_stmts = (
        "C = matmul(A, B)",
        "C = matmul(transpose(A), B)",
        "C = 0.5 * matmul(A, B) + 0.25 * C",
        "Mf = medfilt3(A)",
    )
    lines = [
        json.dumps({"dataset": "default", "stmt": st}, separators=(",", ":"))
        for st in diff_stmts
    ]
    diffs = _run_calcdiff(lines, extra_args=["--strict"])
    assert not diffs, "Unexpected diff(s) in Phase 7 library ops:\n" + "\n".join(
        json.dumps(d) for d in diffs[:20]
    )

    script = (
        "mem.mk2Dim A 8 8 >/dev/null\n"
        "mem.mk2Dim B 8 8 >/dev/null\n"
        '@calc "A = 1.0 + 0.1 * ix(A) + 0.2 * iy(A); B = 0.5 - 0.05 * ix(B) + 0.15 * iy(B)"\n'
        '@calc "M1 = A @ B; M2 = matmul(A, B); err_mm = imax(abs(M1 - M2))"\n'
        '@calc "T1 = A.T @ B; T2 = matmul(transpose(A), B); err_tr = imax(abs(T1 - T2))"\n'
        "@calc 'E1 = einsum(\"ij,jk->ik\", A, B); err_ein = imax(abs(E1 - M2))'\n"
        '@calc "U, S, Vt = svd(A); Sonly = svd(A); err_svd = imax(abs(S - Sonly))"\n'
        '@calc "H = A + A.T; W, V = eigh(H); Wonly = eigh(H); err_eigh = imax(abs(W - Wonly))"\n'
        '@calc "F = fft(A); R = real(ifft(F)); Fs = fftshift(F); err_fft = imax(abs(R - A))"\n'
        '@calc "Hs = hstack(A, B); Vs = vstack(A, B); Ds = dstack(A, B); Cs = cat(0, A, B)"\n'
        '@calc "err_cat = imax(abs(Hs - Cs)) + abs(icount(Hs) - 128) + abs(icount(Ds) - 128)"\n'
        '@calc "R4 = rot90(rot90(A, 1), 3); R2 = rot90(rot90(A, 2), 2); '
        'err_rot = imax(abs(R4 - A)) + imax(abs(R2 - A))"\n'
        "mem.mk2Dim K 3 3 >/dev/null\n"
        '@calc "K = 1.0 / 9.0; Cv = conv2(A, K); Bm = boxmean(A, 1); '
        'err_conv = imax(abs(Cv[1:7, 1:7] - Bm[1:7, 1:7]))"\n'
        '@calc "im = 10.0 + 0 * A; im[3, 4] = 50.0; im[6, 2] = 2.0"\n'
        '@calc "mn = imin3(im); mx = imax3(im); bm = boxmean(im, 1)"\n'
        '@calc "v_mx = mx[4, 4]; v_mn = mn[5, 2]; v_bm = bm[3, 4]"\n'
        "@calc -v err_mm,err_tr,err_ein,err_svd,err_eigh,err_fft,err_cat,err_rot,err_conv "
        '"err_mm; err_tr; err_ein; err_svd; err_eigh; err_fft; err_cat; err_rot; err_conv"\n'
        '@calc -v v_mx,v_mn,v_bm "v_mx; v_mn; v_bm"\n'
        'echo "errs=$err_mm,$err_tr,$err_ein,$err_svd,$err_eigh,$err_fft,$err_cat,$err_rot,$err_conv"\n'
        'echo "res=$v_mx,$v_mn,$v_bm"\n'
    )
    res = milk_bash(script, tmp_path, extra_env={"MILK_CALC_ENGINE": "new"})
    assert res.returncode == 0, res.stdout + res.stderr
    errs = [float(x) for x in value(res.stdout, "errs").split(",")]
    for e in errs:
        assert e <= 1e-5, f"Phase 7 library op error exceeded tolerance: {errs}"
    vals = [float(x) for x in value(res.stdout, "res").split(",")]
    assert vals[0] == pytest.approx(50.0, rel=1e-6)
    assert vals[1] == pytest.approx(2.0, rel=1e-6)
    assert vals[2] == pytest.approx((8 * 10.0 + 50.0) / 9.0, rel=1e-5)
