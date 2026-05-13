import os
import re

c_files = set()
for root, _, files in os.walk('src'):
    for f in files:
        if f.endswith('.c'):
            c_files.add(os.path.join(root, f))
for root, _, files in os.walk('plugins'):
    for f in files:
        if f.endswith('.c'):
            c_files.add(os.path.join(root, f))

compiled_files = set()
for root, _, files in os.walk('.'):
    if 'CMakeLists.txt' in files:
        with open(os.path.join(root, 'CMakeLists.txt'), 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Extract mentions of .c files
            matches = re.findall(r'([A-Za-z0-9_/-]+\.c)', content)
            for m in matches:
                # normalize path
                compiled_files.add(os.path.basename(m))

for c_file in sorted(c_files):
    if os.path.basename(c_file) not in compiled_files:
        # check if it's #included in another C file
        print(f"Potentially uncompiled: {c_file}")
