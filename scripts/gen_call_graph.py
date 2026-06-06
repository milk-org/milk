#!/usr/bin/env python3
"""
milk codebase function call dependency graph generator.

Scans ALL .c and .h files (treating all #ifdef branches as active).
Outputs:
  call_graph.dot  — GraphViz DOT source
  call_graph.svg  — rendered SVG

Node colouring:
  BLUE  — reachable from any main() or CLI command registration
  RED   — unreachable / dead code
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from collections import defaultdict, deque

# ── Configuration ──────────────────────────────────────────────────────────────

REPO_ROOT = Path("/home/vdeo/repos/milk-standalone")
SOURCE_DIRS = [REPO_ROOT / "src", REPO_ROOT / "plugins"]
OUTPUT_DOT = REPO_ROOT / "call_graph.dot"
OUTPUT_SVG = REPO_ROOT / "call_graph.svg"

# C keywords that syntactically look like function calls but are not
C_KEYWORDS = frozenset(
    {
        "if",
        "while",
        "for",
        "switch",
        "else",
        "do",
        "return",
        "sizeof",
        "typeof",
        "alignof",
        "__attribute__",
        "catch",
        "try",
        "case",
        "default",
        "goto",
        "struct",
        "union",
        "enum",
        "typedef",
        "static",
        "extern",
        "inline",
        "volatile",
        "const",
        "void",
        "__asm__",
        "asm",
        "register",
        "auto",
        "break",
        "continue",
        "defined",
        "NULL",
        "offsetof",
        "__builtin_expect",
        "__builtin_offsetof",
        "__typeof__",
    }
)

# Macro names that generate a main() and call the third argument (compute fn)
STANDALONE_MACRO_RE = re.compile(
    r"\bFPS_MAIN_STANDALONE_V2(?:_CONFCHECK)?\s*\("
    r"\s*\w+\s*,\s*\w+\s*,\s*(\w+)"  # third argument = compute function name
)

# CLIADDCMD_* calls — the callee name encodes the actual function name
CLIADDCMD_CALL_RE = re.compile(r"\bCLIADDCMD_(\w+)\s*\(")

# Function-like macro definitions: #define NAME(...) body (possibly multiline)
MACRO_DEF_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)

# ── Step 1: Collect source files ───────────────────────────────────────────────


def find_c_files(dirs):
    files = []
    for d in dirs:
        if not d.exists():
            continue
        for ext in ("*.c", "*.h"):
            files.extend(d.rglob(ext))
    return sorted(files)


# ── Step 2: Text pre-processing ────────────────────────────────────────────────

# Matches C block comments, line comments, string literals, char literals
# so we can replace them with whitespace before further analysis.
_COMMENT_STR_RE = re.compile(
    r"//[^\n]*"  # C99 line comment
    r"|/\*.*?\*/"  # block comment
    r'|"(?:[^"\\]|\\.)*"'  # string literal
    r"|'(?:[^'\\]|\\.)*'",  # char literal
    re.DOTALL,
)


def sanitize(code):
    """Replace comments and string/char literals with spaces (preserve newlines)."""

    def repl(m):
        s = m.group(0)
        # Preserve newlines for line-number tracking
        return re.sub(r"[^\n]", " ", s)

    return _COMMENT_STR_RE.sub(repl, code)


# ── Step 3: Extract function definitions ──────────────────────────────────────

# A function definition starts with a word followed by '(' and eventually '{'.
# We need to distinguish definitions from declarations (ending in ';').
# Strategy: find lines that look like function headers then check for '{'.

# Match: optional-storage-class type_stuff FUNCNAME ( params... ) { OR \n{
FUNC_HEADER_RE = re.compile(
    r"\b([A-Za-z_]\w*)"  # function name (captured)
    r"\s*\("  # opening paren
    r"([^;{}()]*?)"  # parameter list — no nested ()
    r"\)\s*(?:\w+\s*)?"  # optional trailing keyword (e.g. __attribute__)
    r"\{",  # opening brace of body
    re.DOTALL,
)


def extract_functions(sanitized_code, raw_code, filepath):
    """
    Returns list of (func_name, body_start, body_end, body_text).
    Uses brace-depth counting on the sanitized source.
    """
    results = []
    seen = set()

    for m in FUNC_HEADER_RE.finditer(sanitized_code):
        fname = m.group(1)
        if fname in C_KEYWORDS:
            continue
        if fname[0].isdigit():
            continue

        # The '{' is the last char of the match
        brace_start = m.end() - 1
        depth = 0
        pos = brace_start
        end = brace_start

        while pos < len(sanitized_code):
            ch = sanitized_code[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
            pos += 1

        if depth != 0:
            continue  # unbalanced — skip

        body = sanitized_code[brace_start + 1 : end]
        # Use fname + offset to allow same name in different files
        key = (str(filepath), fname)
        if key in seen:
            # Merge duplicate bodies (e.g. static inline in header included many times)
            continue
        seen.add(key)
        results.append((fname, brace_start, end, body))

    return results


# ── Step 4: Extract function calls from a body ────────────────────────────────

CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def extract_calls(body):
    return {m.group(1) for m in CALL_RE.finditer(body) if m.group(1) not in C_KEYWORDS}


# ── Step 5: Main analysis ──────────────────────────────────────────────────────


def main():
    print("Scanning source files …", flush=True)
    files = find_c_files(SOURCE_DIRS)
    print(f"  Found {len(files)} files", flush=True)

    # Maps: func_name → set of caller names  (i.e. callee ← callers)
    # Also: func_name → defining file (for clustering / labels)
    all_functions = {}  # fname → {"file": path, "callers": set(), "calls": set()}
    entry_points = set()  # functions that are roots (main / CLI)
    macro_functions = set()  # function-like macros defined in the codebase

    # ── Pass 1: collect all function and macro names ───────────────────────────
    print("Pass 1: collecting definitions …", flush=True)
    file_contents = {}  # filepath → (raw, sanitized)

    for fpath in files:
        try:
            raw = fpath.read_text(errors="replace")
        except Exception as e:
            print(f"  WARN: cannot read {fpath}: {e}", file=sys.stderr)
            continue

        san = sanitize(raw)
        file_contents[fpath] = (raw, san)

        # Function-like macro names
        for m in MACRO_DEF_RE.finditer(raw):
            mname = m.group(1)
            if mname not in C_KEYWORDS:
                macro_functions.add(mname)
                if mname not in all_functions:
                    all_functions[mname] = {
                        "file": str(fpath.relative_to(REPO_ROOT)),
                        "calls": set(),
                        "is_macro": True,
                    }

        # Regular functions
        for fname, bs, be, body in extract_functions(san, raw, fpath):
            if fname not in all_functions:
                all_functions[fname] = {
                    "file": str(fpath.relative_to(REPO_ROOT)),
                    "calls": set(),
                    "is_macro": False,
                }

    print(f"  {len(all_functions)} functions/macros found", flush=True)

    # ── Pass 2: build call edges ───────────────────────────────────────────────
    print("Pass 2: building call edges …", flush=True)

    for fpath, (raw, san) in file_contents.items():
        # Identify entry points from this file
        #  a) direct int main(
        if re.search(r"\bint\s+main\s*\(", san):
            # Find the actual main() function name
            entry_points.add("main")

        #  b) FPS_MAIN_STANDALONE_V2(INFO, PARAMS, COMPUTE_FN)
        for m in STANDALONE_MACRO_RE.finditer(raw):
            compute_fn = m.group(1)
            entry_points.add(compute_fn)
            # Also mark the generated main as reachable if it exists
            entry_points.add("main")

        #  c) CLIADDCMD_* calls — the function they register is considered reachable
        for m in CLIADDCMD_CALL_RE.finditer(san):
            registered = m.group(1)  # e.g. "COREMOD_iofits__loadfits"
            entry_points.add("CLIADDCMD_" + registered)
            entry_points.add(registered)

        # Build call edges for all functions defined in this file
        for fname, bs, be, body in extract_functions(san, raw, fpath):
            calls = extract_calls(body)
            # Also harvest CLIADDCMD_ calls from inside the body
            for cm in CLIADDCMD_CALL_RE.finditer(body):
                registered = cm.group(1)
                calls.add("CLIADDCMD_" + registered)
                calls.add(registered)
            # Register FPS_MAIN compute function references
            for sm in STANDALONE_MACRO_RE.finditer(raw[bs:be]):
                calls.add(sm.group(1))
            if fname in all_functions:
                all_functions[fname]["calls"].update(calls & all_functions.keys())

    print(f"  Entry points found: {len(entry_points)}", flush=True)

    # Ensure all entry-point names exist as nodes (even if body not parsed)
    for ep in list(entry_points):
        if ep not in all_functions:
            all_functions[ep] = {"file": "<entry>", "calls": set(), "is_macro": False}

    # ── Pass 3: reachability BFS ───────────────────────────────────────────────
    print("Pass 3: reachability BFS …", flush=True)
    reachable = set(entry_points)
    queue = deque(entry_points)

    while queue:
        fn = queue.popleft()
        if fn not in all_functions:
            continue
        for callee in all_functions[fn]["calls"]:
            if callee not in reachable:
                reachable.add(callee)
                queue.append(callee)

    dead = set(all_functions.keys()) - reachable
    print(f"  Reachable: {len(reachable)}  |  Dead code: {len(dead)}", flush=True)

    # ── Pass 4: emit DOT ───────────────────────────────────────────────────────
    print("Pass 4: writing DOT …", flush=True)

    # Group nodes by top-level module (first two path components after src/ or plugins/)
    def module_of(filepath):
        parts = Path(filepath).parts
        for i, p in enumerate(parts):
            if p in ("src", "plugins"):
                return (
                    "/".join(parts[i : i + 3])
                    if i + 2 < len(parts)
                    else "/".join(parts[i:])
                )
        return parts[0] if parts else "unknown"

    # Collect all edges (only between known functions)
    edges = set()
    for fname, info in all_functions.items():
        for callee in info["calls"]:
            if callee in all_functions:
                edges.add((fname, callee))

    # Remove nodes with no edges for readability (isolated functions)
    connected = set()
    for src, dst in edges:
        connected.add(src)
        connected.add(dst)
    # Also keep entry points even if they have no edges
    connected.update(entry_points & all_functions.keys())

    # Cluster by module
    clusters = defaultdict(list)
    for fname in connected:
        mod = module_of(all_functions[fname]["file"])
        clusters[mod].append(fname)

    def safe_id(name):
        """Return a DOT-safe node identifier."""
        return '"' + name.replace('"', '\\"') + '"'

    def node_color(fname):
        if fname in reachable:
            return "#4a90d9"  # blue
        return "#e05252"  # red

    def node_fontcolor(fname):
        return "white"

    lines = []
    lines.append("digraph call_graph {")
    lines.append(
        '  graph [rankdir=LR fontname="Helvetica" bgcolor="#1e1e1e"'
        " overlap=false splines=true nodesep=0.3 ranksep=1.0];"
    )
    lines.append(
        '  node  [shape=box style="filled,rounded" fontsize=9'
        ' fontname="Helvetica" margin="0.1,0.05"];'
    )
    lines.append('  edge  [arrowsize=0.5 color="#888888" penwidth=0.6];')
    lines.append("")

    cluster_idx = 0
    for mod, fnames in sorted(clusters.items()):
        safe_mod = re.sub(r"[^A-Za-z0-9_]", "_", mod)
        lines.append(f"  subgraph cluster_{cluster_idx}_{safe_mod} {{")
        lines.append(f"    label={safe_id(mod)};")
        lines.append(
            '    style=filled; color="#2a2a2a"; fontcolor="#aaaaaa";'
            ' fontsize=8; fontname="Helvetica";'
        )
        for fname in sorted(fnames):
            bg = node_color(fname)
            fc = node_fontcolor(fname)
            kind = "(M)" if all_functions[fname].get("is_macro") else ""
            lbl = fname + kind
            ep = "[EP] " if fname in entry_points else ""
            lbl = ep + lbl
            lines.append(
                f"    {safe_id(fname)} [label={safe_id(lbl)}"
                f' fillcolor="{bg}" fontcolor="{fc}"];'
            )
        lines.append("  }")
        lines.append("")
        cluster_idx += 1

    lines.append("")
    for src, dst in sorted(edges):
        if src in connected and dst in connected:
            col = "#4a90d9" if (src in reachable and dst in reachable) else "#888888"
            lines.append(f'  {safe_id(src)} -> {safe_id(dst)} [color="{col}"];')

    lines.append("}")

    OUTPUT_DOT.write_text("\n".join(lines))
    print(f"  Written {OUTPUT_DOT}", flush=True)
    print(f"  Nodes: {len(connected)}  Edges: {len(edges)}", flush=True)

    # ── Pass 5: render SVG ─────────────────────────────────────────────────────
    # For large graphs (>500 nodes) sfdp with overlap=prism is prohibitively slow.
    # Use sfdp with scale-based overlap removal, no expensive spline routing.
    print("Pass 5: rendering SVG with sfdp (fast mode) …", flush=True)
    result = subprocess.run(
        [
            "sfdp",
            "-Tsvg",
            "-Goverlap=scale",  # fast overlap removal
            "-Gsplines=false",  # skip expensive edge routing
            "-Gmaxiter=50",  # cap force-directed iterations
            str(OUTPUT_DOT),
            "-o",
            str(OUTPUT_SVG),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"  sfdp failed ({result.returncode}), trying neato …", flush=True)
        result = subprocess.run(
            [
                "neato",
                "-Tsvg",
                "-Goverlap=scale",
                "-Gsplines=false",
                str(OUTPUT_DOT),
                "-o",
                str(OUTPUT_SVG),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode == 0:
        size_mb = OUTPUT_SVG.stat().st_size / 1_048_576
        print(f"  Written {OUTPUT_SVG}  ({size_mb:.1f} MB)", flush=True)
    else:
        print(f"  ERROR: rendering failed:\n{result.stderr}", file=sys.stderr)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Total functions/macros : {len(all_functions)}")
    print(f"  Connected (with edges) : {len(connected)}")
    print(f"  Entry points           : {len(entry_points)}")
    print(f"  Reachable (blue)       : {len(reachable & connected)}")
    print(f"  Dead code (red)        : {len(dead & connected)}")
    print(f"  Edges                  : {len(edges)}")
    print(f"  DOT  → {OUTPUT_DOT}")
    print(f"  SVG  → {OUTPUT_SVG}")

    # Write a companion summary TSV
    tsv_path = REPO_ROOT / "call_graph_summary.tsv"
    with tsv_path.open("w") as f:
        f.write("function\tfile\treachable\tis_entry\tis_macro\n")
        for fname in sorted(all_functions):
            info = all_functions[fname]
            f.write(
                f"{fname}\t{info['file']}\t"
                f"{'yes' if fname in reachable else 'no'}\t"
                f"{'yes' if fname in entry_points else 'no'}\t"
                f"{'yes' if info.get('is_macro') else 'no'}\n"
            )
    print(f"  TSV  → {tsv_path}")


if __name__ == "__main__":
    main()
