import subprocess
import os

# Find ALL .o files across the entire project
output = subprocess.check_output('find ../_build -name "*.o" ! -path "*/_build_lto_inspect/*" -type f', shell=True)
o_files = output.decode().strip().split('\n')

defs = {}
refs = set()

for o_file in o_files:
    if not o_file: continue
    try:
        nm_out = subprocess.check_output(['nm', '-g', o_file]).decode().split('\n')
        for line in nm_out:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[-2] in ['T', 'D', 'B', 'R', 'C']:
                sym = parts[-1]
                if sym not in defs:
                    defs[sym] = []
                defs[sym].append(o_file)
            elif len(parts) >= 2 and parts[-2] == 'U':
                refs.add(parts[-1])
    except:
        pass

static_candidates = []
for sym, locations in defs.items():
    if sym not in refs:
        # Check if it's an init_ module function, which are looked up via dlsym
        if sym.startswith("init_") or sym.startswith("cacao_") or sym.startswith("lib"):
            continue
        if sym == "main" or sym.startswith("_"):
            continue
        for loc in locations:
            # exclude fps executables, test executables etc
            if "fpsexec" in loc or "test" in loc or "perfbench" in loc or "cli-all" in loc:
                continue
            static_candidates.append(f"{sym} in {loc}")

print(f"Found {len(static_candidates)} candidates")
with open("candidates_all.txt", "w") as f:
    for c in sorted(static_candidates):
        f.write(c + "\n")
