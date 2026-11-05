"""Extract calculator statements and preceding setup from python/tests/ and docs/guide/."""

from __future__ import annotations

import ast
import json
import pathlib
import re
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

# Regex patterns for @calc, @test, @assert, @{...}, and unit -e / -i options
_CALC_CMD_RE = re.compile(
    r"""@(calc|test|assert)\b"""
    r"""(?:\s+(?:-[v]\s+\S+|--quiet|-q|--bind|--pixels))*"""
    r"""\s+(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^']*)')"""
)
_AT_BRACE_RE = re.compile(r"""(?<![A-Za-z0-9_\\])@\{([^{}\n]+)\}""")
_UNIT_OPT_RE = re.compile(
    r"""(?:milk-fpsexec-milk-calc|milk\.calc)\b[^\n]*?\s-(?:e|i)\s+"""
    r"""(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^']*)')"""
)
_MEM_MK_RE = re.compile(
    r"""\b(?:mem\.)?(mk[23]Dim)\s+(?:"|')?(?:@[A-Za-z0-9]+:)?"""
    r"""([A-Za-z_][A-Za-z0-9_]*)(?:"|')?\s+(\d+(?:\s+\d+){1,2})\b"""
)
_MEM_RM_RE = re.compile(
    r"""\b(?:mem\.)?rm\s+(?:"|')?(?:@[A-Za-z0-9]+:)?([A-Za-z_][A-Za-z0-9_]*)(?:"|')?\b"""
)
_ENTITY_ONLY_RE = re.compile(r"""^[spfu]\.[A-Za-z0-9_.*:-]+$""")
_UNIT_ONLY_RE = re.compile(
    r"""(?:^\s*(?:if|else|end|for|skip|post)\b|\[\s*-\d+\s*\])"""
)


@dataclass
class CorpusEntry:
    """One extracted calculator statement with its preceding setup and source location."""

    stmt: str
    setup: list[str] = field(default_factory=list)
    dataset: str = "corpus"
    is_test: bool = False
    source: str = ""

    def to_json_line(self) -> str:
        """Serialize as a JSON line for milk-calcdiff."""
        return json.dumps(
            {
                "dataset": self.dataset,
                "setup": self.setup,
                "stmt": self.stmt,
                "is_test": self.is_test,
                "source": self.source,
            },
            separators=(",", ":"),
        )


def _is_usable_expr(expr: str) -> bool:
    """Return True if @expr is a concrete calculator expression without bash expansions."""
    s = expr.strip()
    if not s or len(s) > 1000:
        return False
    # Skip bash variable/command expansions, Python format placeholders, or doc ellipses
    if "$" in s or "`" in s or "…" in s or "..." in s:
        return False
    if "{0}" in s or "{1}" in s or "%s" in s or "%d" in s:
        return False
    if s.startswith("-") and s in ("--help", "--list", "--explain", "-h"):
        return False
    if _UNIT_ONLY_RE.search(s):
        return False
    return True


def _unescape_double_quoted(s: str) -> str:
    """Unescape bash double-quoted string escapes (`\"` -> `"`)."""
    return s.replace('\\"', '"').replace("\\\\", "\\")


def _extract_from_script_block(
    text: str,
    source: str,
    initial_setup: list[str] | None = None,
) -> tuple[list[CorpusEntry], list[str]]:
    """Extract CorpusEntry items and resulting setup in order from a shell/milk script snippet."""
    entries: list[CorpusEntry] = []
    setup: list[str] = list(initial_setup) if initial_setup else []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip lines that load external C extensions (-l *.c / *.so) or test --explain/--list
        if re.search(r"@calc\s+(?:-l\b|--list\b|--explain\b|-f\b)", line):
            continue

        events: list[tuple[int, str, re.Match[str]]] = []
        for m in _MEM_RM_RE.finditer(line):
            events.append((m.start(), "rm", m))
        for m in _MEM_MK_RE.finditer(line):
            events.append((m.start(), "mk", m))
        for m in _CALC_CMD_RE.finditer(line):
            events.append((m.start(), "calc", m))
        for m in _UNIT_OPT_RE.finditer(line):
            events.append((m.start(), "unit", m))
        for m in _AT_BRACE_RE.finditer(line):
            events.append((m.start(), "brace", m))
        events.sort(key=lambda t: t[0])

        for _, kind, m in events:
            if kind == "rm":
                if len(setup) < 64:
                    setup.append(f"@cmd:rm {m.group(1)}")
            elif kind == "mk":
                if len(setup) < 64:
                    setup.append(f"@cmd:{m.group(1)} {m.group(2)} {m.group(3)}")
            elif kind == "calc":
                kw = m.group(1)
                body = (
                    _unescape_double_quoted(m.group(2))
                    if m.group(2) is not None
                    else m.group(3)
                )
                if not _is_usable_expr(body):
                    continue
                if kw == "assert" and _ENTITY_ONLY_RE.match(body.strip()):
                    continue
                is_test = kw in ("test", "assert")
                tail = line[m.end() : m.end() + 24]
                expect_err = bool(re.match(r"^\s*2>", tail))
                entries.append(
                    CorpusEntry(
                        stmt=body,
                        setup=list(setup),
                        dataset="corpus",
                        is_test=is_test,
                        source=source,
                    )
                )
                if not is_test and not expect_err and len(setup) < 64:
                    setup.append(body)
            elif kind == "unit":
                body = (
                    _unescape_double_quoted(m.group(1))
                    if m.group(1) is not None
                    else m.group(2)
                )
                if not _is_usable_expr(body):
                    continue
                entries.append(
                    CorpusEntry(
                        stmt=body,
                        setup=list(setup),
                        dataset="corpus",
                        is_test=False,
                        source=source,
                    )
                )
                if len(setup) < 64:
                    setup.append(body)
            elif kind == "brace":
                body = m.group(1).strip()
                if not _is_usable_expr(body):
                    continue
                if re.match(r"^-?\d+$", body) or _ENTITY_ONLY_RE.match(body):
                    continue
                entries.append(
                    CorpusEntry(
                        stmt=body,
                        setup=list(setup),
                        dataset="corpus",
                        is_test=False,
                        source=source,
                    )
                )

    return entries, setup


def _string_constants_in_node(node: ast.AST) -> list[str]:
    """Collect string constants in AST order within @node."""
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.append(child.value)
    return out


def extract_from_python_file(path: pathlib.Path) -> list[CorpusEntry]:
    """Extract calculator statements and per-test setup from a pytest file."""
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, SyntaxError):
        return []

    rel = str(path.relative_to(REPO_ROOT))
    # Collect module-level string constants (like COMMON_SETUP / HELPERS) as shared initial setup
    mod_setup: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for s in _string_constants_in_node(stmt.value):
                _, mod_setup = _extract_from_script_block(
                    s, rel, initial_setup=mod_setup
                )

    entries: list[CorpusEntry] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            block_text = "\n".join(_string_constants_in_node(stmt))
            loc = f"{rel}:{stmt.name}"
            fn_entries, _ = _extract_from_script_block(
                block_text, loc, initial_setup=mod_setup
            )
            entries.extend(fn_entries)

    return entries


def extract_from_markdown_file(path: pathlib.Path) -> list[CorpusEntry]:
    """Extract calculator statements from fenced code blocks in a markdown guide."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return []

    rel = str(path.relative_to(REPO_ROOT))
    entries: list[CorpusEntry] = []
    in_fence = False
    fence_lines: list[str] = []
    fence_start = 1

    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lines = []
                fence_start = lineno
            else:
                in_fence = False
                block = "\n".join(fence_lines)
                blk_entries, _ = _extract_from_script_block(
                    block,
                    f"{rel}:{fence_start}",
                    initial_setup=[
                        "@cmd:mk2Dim cam 64 64",
                        "@cmd:mk2Dim dark 64 64",
                        "@cmd:mk2Dim im 64 64",
                    ],
                )
                entries.extend(blk_entries)
        elif in_fence:
            fence_lines.append(line)

    return entries


def collect_corpus() -> list[CorpusEntry]:
    """Collect the full calculator corpus from python/tests/ and docs/guide/."""
    entries: list[CorpusEntry] = []
    tests_dir = REPO_ROOT / "python" / "tests"
    guide_dir = REPO_ROOT / "docs" / "guide"

    skip_files = {"calc_diff_test.py", "calc_engine_bench_test.py"}
    for py_file in sorted(tests_dir.rglob("*.py")):
        if "calcdiff" in py_file.parts or py_file.name in skip_files:
            continue
        entries.extend(extract_from_python_file(py_file))

    for calc_file in sorted(tests_dir.rglob("*.calc")):
        rel = str(calc_file.relative_to(REPO_ROOT))
        calc_entries, _ = _extract_from_script_block(
            calc_file.read_text(encoding="utf-8"), rel
        )
        entries.extend(calc_entries)

    for md_file in sorted(guide_dir.rglob("*.md")):
        entries.extend(extract_from_markdown_file(md_file))

    # Deduplicate identical (dataset, tuple(setup), stmt, is_test) while preserving order
    seen: set[tuple[str, tuple[str, ...], str, bool]] = set()
    unique: list[CorpusEntry] = []
    for e in entries:
        key = (e.dataset, tuple(e.setup), e.stmt, e.is_test)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


if __name__ == "__main__":
    for item in collect_corpus():
        print(item.to_json_line())
