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
from tests.cli.conftest import _install_dir

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
