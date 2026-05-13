import os
import re
from collections import defaultdict

# 1. Collect all function definitions
funcs = set()
for root, _, files in os.walk('.'):
    if 'old' in root or '.git' in root or '_build' in root: continue
    for f in files:
        if f.endswith('.c'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                # Simple regex for C function definitions (not declarations)
                # Assumes style: return_type \n func_name(args) {
                # or return_type func_name(args) {
                matches = re.finditer(r'^[A-Za-z0-9_]+\s+([A-Za-z0-9_]+)\([^)]*\)\s*\{', content, re.MULTILINE)
                for m in matches:
                    funcs.add(m.group(1))

# 2. Count occurrences of these function names across all source files
counts = defaultdict(int)
for root, _, files in os.walk('.'):
    if 'old' in root or '.git' in root or '_build' in root: continue
    for f in files:
        if f.endswith(('.c', '.h', '.cpp', '.hpp')):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                for func in funcs:
                    # count occurrences as whole words
                    counts[func] += len(re.findall(r'\b' + re.escape(func) + r'\b', content))

# 3. Print functions that appear exactly once
print("Functions appearing exactly once (likely dead code):")
for func in sorted(funcs):
    if counts[func] == 1:
        # Avoid main, and functions that might be constructor attributes, though we matched basic ones
        if func != 'main' and not func.startswith('_'):
            print(func)

