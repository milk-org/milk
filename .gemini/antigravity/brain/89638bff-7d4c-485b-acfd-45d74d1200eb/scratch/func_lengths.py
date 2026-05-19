#!/usr/bin/env python3
"""Find longest C functions by brace-matching."""
import os
import sys
import re

root = sys.argv[1]
min_len = int(sys.argv[2]) if len(sys.argv) > 2 else 50

results = []

for dirpath, dirnames, filenames in os.walk(root):
    # Skip build dirs
    skip = False
    for skip_dir in ['_build', '/build/', 'CMakeFiles', 'treesitter']:
        if skip_dir in dirpath:
            skip = True
            break
    if skip:
        continue

    for fname in filenames:
        if not fname.endswith('.c'):
            continue
        fpath = os.path.join(dirpath, fname)
        try:
            with open(fpath, 'r', errors='replace') as f:
                lines = f.readlines()
        except:
            continue

        # Simple state machine to find top-level functions
        depth = 0
        func_start = 0
        func_name = "unknown"
        in_func = False
        candidate_name = ""

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Try to capture function name from lines before '{'
            # Look for pattern: word( or word (
            if depth == 0 and not in_func:
                m = re.search(r'([a-zA-Z_]\w*)\s*\(', stripped)
                if m and not stripped.startswith('#') \
                     and not stripped.startswith('//') \
                     and not stripped.startswith('/*') \
                     and not stripped.startswith('if') \
                     and not stripped.startswith('for') \
                     and not stripped.startswith('while') \
                     and not stripped.startswith('switch') \
                     and not stripped.startswith('return') \
                     and not stripped.startswith('else') \
                     and not stripped.startswith('typedef') \
                     and not stripped.startswith('struct ') \
                     and not stripped.startswith('enum '):
                    candidate_name = m.group(1)

            for ch in line:
                if ch == '{':
                    if depth == 0:
                        func_start = i
                        func_name = candidate_name
                        in_func = True
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and in_func:
                        length = i - func_start + 1
                        if length >= min_len:
                            rel = os.path.relpath(fpath, root)
                            results.append(
                                (length, func_name, rel, func_start + 1)
                            )
                        in_func = False

results.sort(reverse=True)
for length, name, path, line in results[:100]:
    print(f"{length:5d}  {name:50s}  {path}:{line}")
