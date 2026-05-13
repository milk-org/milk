import os
import re

for root, _, files in os.walk('src'):
    for f in files:
        if not f.endswith(('.c', '.h')): continue
        path = os.path.join(root, f)
        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            # Find #if 0 ... #endif
            if0_blocks = re.findall(r'#if\s+0\s*?\n(.*?)\n#endif', content, re.DOTALL)
            for b in if0_blocks:
                if len(b.split('\n')) > 10:
                    print(f"{path}: Found #if 0 block with {len(b.split('\n'))} lines")

            # Find long /* ... */ blocks
            comment_blocks = re.findall(r'/\*(.*?)\*/', content, re.DOTALL)
            for b in comment_blocks:
                lines = b.split('\n')
                if len(lines) > 50 and not b.strip().startswith('*'):
                    # if it has code-like stuff
                    if ';' in b or '{' in b:
                        print(f"{path}: Found /* */ block with {len(lines)} lines")
